/* WHERE A FAILURE CAME FROM.
 *
 * A bare 'return -1' cannot be told from any other -1 in a file, which is why a refusal could not be
 * located.  Every failure return under rotation/ records its file, line and status here, and the engine's
 * dump emits the last one.  The value still propagates unchanged - only the origin is added, so no
 * control flow, ordering or arithmetic moves. */
#ifndef SALT_ROTATION_FAIL_H
#define SALT_ROTATION_FAIL_H

void salt_rot_fail_at(const char *file, int line, int status);
const char *salt_rot_fail_file(void);
int salt_rot_fail_line(void);
int salt_rot_fail_status(void);

#define SALT_TFAIL(st) (salt_rot_fail_at(__FILE__, __LINE__, (st)), (st))

#endif
