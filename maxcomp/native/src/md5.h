/* md5.h — standalone MD5 (RFC 1321) */

#ifndef MAXCOMP_MD5_H
#define MAXCOMP_MD5_H

#include <stdint.h>
#include <stddef.h>

/* Compute MD5 digest of data[0..len-1].  digest must be at least 16 bytes. */
void md5_hash(const uint8_t *data, size_t len, uint8_t digest[16]);

#endif /* MAXCOMP_MD5_H */
