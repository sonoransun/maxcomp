/* sha256.h — standalone SHA-256 with optional ARM crypto extensions */

#ifndef MAXCOMP_SHA256_H
#define MAXCOMP_SHA256_H

#include <stdint.h>
#include <stddef.h>

/* Compute SHA-256 digest of data[0..len-1].  digest must be at least 32 bytes. */
void sha256_hash(const uint8_t *data, size_t len, uint8_t digest[32]);

#endif /* MAXCOMP_SHA256_H */
