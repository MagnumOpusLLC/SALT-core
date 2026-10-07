"""Maple transport only: native C owns tokenization, sampling, KV and commits.

Experimental loopback-only endpoint. No authentication layer is invented here;
non-loopback bind is refused. State export/import are explicit user operations.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from engine_config import load_engine_config

ROOT = Path(__file__).resolve().parents[2]


def identity(st):
    return {'dev':st.st_dev,'ino':st.st_ino,'size':st.st_size,
            'mtime_ns':st.st_mtime_ns,'ctime_ns':st.st_ctime_ns}


def integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in [{low}, {high}]')
    return value


class Engine:
    def __init__(self,args):
        if not args.maple_package or not args.maple_runner:
            raise ValueError('--maple-package and --maple-runner are required')
        if args.gpu_prefill or os.environ.get('SALT_GPU','0') != '0':
            raise ValueError('Maple accelerator modes are unsupported')
        self.package=Path(args.maple_package).resolve(strict=True)
        self.runner=Path(args.maple_runner).resolve(strict=True)
        self.lock=threading.Lock()
        self.closed=False
        self.position=0
        cfg=load_engine_config(model_dir=str(ROOT/'models/maple-preview'))
        self.context=integer(args.maple_context or int(cfg['MAPLE_CONTEXT']),'context',32,4096)
        self.batch=integer(args.maple_batch or int(cfg['MAPLE_PREFILL_B']),'batch',1,32)
        self.workers=integer(args.threads if args.threads is not None else int(cfg['MAPLE_WORKERS']),'workers',1,32)
        self.output_cap=int(cfg['MAPLE_OUTPUT_CAP'])
        receipt=json.loads((self.package/'auth.json').read_text())
        if receipt['schema']!='salt.maple.package.v1' or receipt['abi']!='salt-maple-int2-f32-v1':
            raise ValueError('unsupported package')
        expected={'trunk.bin','pool.bin','index.txt','tokenizer.json','tokenizer_config.json','config.json'}
        if set(receipt['files']) != expected:
            raise ValueError('package receipt file set mismatch')
        key=hashlib.sha256(json.dumps({'abi':receipt['abi'],'revision':receipt['source_revision'],
            'sha256':{k:v['sha256'] for k,v in receipt['files'].items()}},sort_keys=True).encode()).hexdigest()
        if key != receipt['package_id']:
            raise ValueError('package identity mismatch')
        build=json.loads(self.runner.with_name(self.runner.name+'.build.json').read_text())
        if hashlib.sha256(self.runner.read_bytes()).hexdigest()!=build['binary_sha256']:
            raise ValueError('binary receipt mismatch')
        for name,digest in build['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
                raise ValueError(f'source differs from built runner: {name}')
        fds={}
        try:
            for name,record in receipt['files'].items():
                fd=os.open(self.package/name,os.O_RDONLY|os.O_NOFOLLOW)
                fds[name]=fd
                if identity(os.fstat(fd))!=record['identity']:
                    raise ValueError(f'package identity drift; explicit reauthentication required: {name}')
            tokenizer_fd=fds['tokenizer.json']
            if cfg.get('MAPLE_TOKENIZER_LAYOUT') == 'compact-merges-v1':
                layout=json.loads((self.package/'tokenizer.runtime.auth.json').read_text())
                if (layout.get('schema') != 'salt.maple.tokenizer-layout.v1' or
                    layout.get('layout') != 'compact-merges-v1' or
                    layout.get('package_id') != key or
                    layout.get('source_sha256') != receipt['files']['tokenizer.json']['sha256'] or
                    layout.get('complete_document_round_trip') is not True):
                    raise ValueError('runtime tokenizer provenance mismatch')
                tokenizer_fd=os.open(self.package/'tokenizer.runtime.json',os.O_RDONLY|os.O_NOFOLLOW)
                fds['tokenizer.runtime.json']=tokenizer_fd
                if identity(os.fstat(tokenizer_fd)) != layout['identity']:
                    raise ValueError('runtime tokenizer identity drift')
            command=[str(self.runner),*[str(fds[k]) for k in ('trunk.bin','pool.bin','index.txt')],str(tokenizer_fd),
                     key,str(self.context),str(self.batch),str(self.workers),cfg['MAPLE_EXPERT_SLOTS'],
                     str(int(args.mem_limit_gb*1e9) if args.mem_limit_gb is not None else int(cfg['MAPLE_RSS_LIMIT_BYTES']))]
            self.proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                pass_fds=tuple(fds.values()),bufsize=0)
            if self.line()!='READY':
                self.close()
                raise RuntimeError('Maple native startup refused')
        finally:
            for fd in fds.values(): os.close(fd)
        self.states=self.package/'states'
        self.states.mkdir(exist_ok=True)

    def line(self):
        raw=self.proc.stdout.readline(1048576)
        if not raw or raw==b'FATAL\n':
            raise RuntimeError(f'native engine failed (exit={self.proc.poll()})')
        if not raw.endswith(b'\n'):
            raise RuntimeError('native protocol line exceeded bound')
        return raw.decode('ascii').rstrip('\n')

    def send(self,data):
        # FileIO.write may be partial even for a blocking pipe.
        view=memoryview(data)
        while view:
            n=self.proc.stdin.write(view)
            if not n: raise RuntimeError('native input pipe closed')
            view=view[n:]

    def close(self):
        if self.closed: return
        self.closed=True
        if self.proc.poll() is None:
            try: self.send(b'QUIT\n')
            except (BrokenPipeError,OSError): pass
            try: self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                self.proc.wait(timeout=10)
        self.proc.stdin.close()
        self.proc.stdout.close()

    def state(self,action,name=None):
        with self.lock:
            if action=='reset':
                self.send(b'RESET\n')
            else:
                if not isinstance(name,str) or not name or len(name)>100 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in name):
                    raise ValueError('state name must contain only letters, numbers, - or _')
                path=self.states/(name+'.kv')
                if action=='export' and path.exists(): raise ValueError('state already exists')
                if action=='import' and not path.is_file(): raise ValueError('state not found')
                self.send((('SAVE ' if action=='export' else 'LOAD ')+str(path)+'\n').encode())
            if self.line()!='OK': raise RuntimeError('native state operation refused')
            if action=='reset': self.position=0
            return {'action':action,'name':name,'status':'ok'}

    def complete(self,body,chat):
        supported={'model','prompt','messages','max_tokens','stream','temperature','seed','top_k','top_p','salt_proof_state','repetition_penalty'}
        if set(body)-supported: raise ValueError('unsupported request fields: '+','.join(sorted(set(body)-supported)))
        if body.get('model','maple-preview')!='maple-preview': raise ValueError('unknown model')
        if chat:
            messages=body.get('messages')
            if not isinstance(messages,list) or not messages: raise ValueError('messages required')
            parts=[]
            for m in messages:
                if not isinstance(m,dict) or set(m)!={'role','content'} or m['role'] not in ('system','user','assistant') or not isinstance(m['content'],str):
                    raise ValueError('only text system/user/assistant messages are supported')
                if any(t in m['content'] for t in ('<|im_start|>','<|im_end|>','<|endoftext|>')):
                    raise ValueError('reserved control token in message content')
                parts.append('<|im_start|>'+m['role']+'\n'+m['content']+'<|im_end|>\n')
            # Explicit ChatML adapter; checkpoint publishes control tokens but no
            # chat_template. Raw /completions is the source-neutral interface.
            prompt=''.join(parts)+'<|im_start|>assistant\n'
        else:
            prompt=body.get('prompt')
            if not isinstance(prompt,str): raise ValueError('prompt must be text')
        if not prompt or '\0' in prompt: raise ValueError('empty/NUL prompt')
        data=prompt.encode()
        if len(data)>=self.context*64: raise ValueError('prompt exceeds native input bound')
        output=integer(body.get('max_tokens',32),'max_tokens',1,self.output_cap)
        temp=body.get('temperature',0)
        if isinstance(temp,bool) or not isinstance(temp,(int,float)) or not math.isfinite(temp) or not 0<=temp<=5:
            raise ValueError('invalid temperature')
        seed=integer(body.get('seed',0),'seed',0,2**64-1)
        topk=integer(body.get('top_k',40),'top_k',1,256)
        topp=body.get('top_p',1.0)
        if isinstance(topp,bool) or not isinstance(topp,(int,float)) or not math.isfinite(topp) or not 0<topp<=1:
            raise ValueError('top_p must be finite and in (0, 1]')
        repetition=body.get('repetition_penalty',1.0)
        if isinstance(repetition,bool) or not isinstance(repetition,(int,float)) or not math.isfinite(repetition) or not 1<=repetition<=2:
            raise ValueError('repetition_penalty must be finite and between 1 and 2')
        if repetition>1 and temp==0:
            raise ValueError('repetition_penalty requires temperature > 0')
        proof=body.get('salt_proof_state',False)
        if type(proof) is not bool or type(body.get('stream',False)) is not bool:
            raise ValueError('proof/stream must be boolean')
        started=time.monotonic()
        with self.lock:
            self.send(f'RUN {len(data)} {output} {int(proof)} {temp} {seed} {topk} {repetition} {topp}\n'.encode()+data)
            tokens=[]
            while True:
                line=self.line()
                if line.startswith('TOKEN '): tokens.append(int(line[6:]))
                elif line.startswith('REFUSE '): raise ValueError(line)
                elif line.startswith('DONE '): break
                else: raise RuntimeError('unexpected native protocol response')
            fields=line.split(' ',9)
            before,prompt_count,completed,position,peak=map(int,fields[1:6])
            if len(tokens)!=completed: raise RuntimeError('native output count mismatch')
            self.position=position
            text=bytes.fromhex(fields[9]).decode('utf-8',errors='replace')
            choice={'index':0,'finish_reason':'stop' if tokens and tokens[-1]==151645 else 'length'}
            choice['message' if chat else 'text']={'role':'assistant','content':text} if chat else text
            return {'id':f'maple-{time.time_ns()}','object':'chat.completion' if chat else 'text_completion',
                'created':int(time.time()),'model':'maple-preview','choices':[choice],
                'usage':{'prompt_tokens':prompt_count,'completion_tokens':completed,'total_tokens':prompt_count+completed},
                'x_salt':{'backend':'cpu','position_before':before,'position':position,'output_ids':tokens,
                          'peak_rss_bytes':peak,'prefill_ms_per_token':float(fields[6]),
                          'decode_ms_per_token':float(fields[7]),'state_sha256':fields[8] if proof else None,
                          'elapsed_s':time.monotonic()-started}}


def serve_from_args(args):
    if args.host not in ('127.0.0.1','localhost','::1'):
        raise ValueError('experimental Maple serving is loopback-only')
    engine=Engine(args)
    class Handler(BaseHTTPRequestHandler):
        def reply(self,status,value):
            data=json.dumps(value,ensure_ascii=False).encode()
            self.send_response(status); self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            if self.path=='/health':
                alive=engine.proc.poll() is None
                self.reply(200 if alive else 503,{'status':'ready' if alive else 'failed',
                    'model':'maple-preview','backend':'cpu','native_pid':engine.proc.pid})
            elif self.path=='/v1/models': self.reply(200,{'object':'list','data':[{'id':'maple-preview','object':'model','owned_by':'local'}]})
            else: self.reply(404,{'error':'not found'})
        def do_POST(self):
            try:
                if self.headers.get('Transfer-Encoding') or len(self.headers.get_all('Content-Length',[]))!=1:
                    raise ValueError('one Content-Length required')
                n=int(self.headers['Content-Length'])
                if not 0<n<=1024*1024: raise ValueError('request size out of bounds')
                self.connection.settimeout(30)
                raw=self.rfile.read(n)
                if len(raw)!=n: raise ValueError('short request')
                body=json.loads(raw)
                if not isinstance(body,dict): raise ValueError('request must be an object')
                if self.path in ('/v1/completions','/v1/chat/completions'):
                    result=engine.complete(body,self.path.endswith('/chat/completions'))
                    if body.get('stream'):
                        # Buffered SSE presentation; the native transaction owns all work.
                        data=('data: '+json.dumps(result)+'\n\ndata: [DONE]\n\n').encode()
                        self.send_response(200); self.send_header('Content-Type','text/event-stream')
                        self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
                    else: self.reply(200,result)
                elif self.path in ('/v1/state/reset','/v1/state/export','/v1/state/import'):
                    self.reply(200,engine.state(self.path.rsplit('/',1)[1],body.get('name')))
                else: self.reply(404,{'error':'not found'})
            except (ValueError,KeyError,TypeError) as exc: self.reply(400,{'error':str(exc)})
            except (RuntimeError,OSError) as exc: self.reply(500,{'error':str(exc)})
    server=ThreadingHTTPServer((args.host,args.port),Handler)
    try:
        print(f'Maple CPU HTTP ready http://{args.host}:{args.port} native_pid={engine.proc.pid}',flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close(); engine.close()
    return 0
