/* Gemma 4 native server executable entry point.
 *
 * Python owns authenticated resource-map selection, process supervision, and
 * HTTP/session control. The model-owned native inference/session lifecycle is
 * implemented behind the private inference interface.
 */

#include "inference.h"

int main(int argc, char **argv) {
    return salt_gemma4_inference_main(argc, argv);
}
