#ifndef SALT_GEMMA4_INFERENCE_H
#define SALT_GEMMA4_INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

/* Run one persistent native Gemma server process. Python supplies the
 * authenticated resource binding and startup policy; this function owns the
 * complete C inference/session lifecycle until drain or fatal shutdown. */
int salt_gemma4_inference_main(int argc, char **argv);

#ifdef __cplusplus
}
#endif

#endif /* SALT_GEMMA4_INFERENCE_H */
