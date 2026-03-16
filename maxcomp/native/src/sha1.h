/* sha1.h — standalone SHA-1 */

#ifndef MAXCOMP_SHA1_H
#define MAXCOMP_SHA1_H

#include <stdint.h>
#include <stddef.h>

/* Compute SHA-1 digest of data[0..len-1].  digest must be at least 20 bytes. */
void sha1_hash(const uint8_t *data, size_t len, uint8_t digest[20]);

#endif /* MAXCOMP_SHA1_H */
