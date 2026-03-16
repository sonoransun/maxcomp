CC ?= clang
CFLAGS = -O3 -march=armv8.4-a+simd+crypto -fPIC -flto -Wall -Wextra -Wno-unused-parameter
LDFLAGS = -dynamiclib

METAL = xcrun -sdk macosx metal
METALLIB = xcrun -sdk macosx metallib

SRCS = src/maxcomp_native.c src/prng_kernels.c src/hash_kernels.c \
       src/ifs_kernels.c src/automata_kernels.c src/autocorrelation.c \
       src/bitstream_ops.c src/md5.c src/sha1.c src/sha256.c

HEADERS = src/maxcomp_native.h src/md5.h src/sha1.h src/sha256.h

.PHONY: all clean metal

all: _maxcomp_native.dylib

_maxcomp_native.dylib: $(SRCS) $(HEADERS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(SRCS)

metal: maxcomp_kernels.metallib

maxcomp_kernels.metallib: shaders/prng_kernels.metal shaders/hash_kernels.metal shaders/ifs_kernels.metal
	$(METAL) -c shaders/prng_kernels.metal -o /tmp/maxcomp_prng.air
	$(METAL) -c shaders/hash_kernels.metal -o /tmp/maxcomp_hash.air
	$(METAL) -c shaders/ifs_kernels.metal -o /tmp/maxcomp_ifs.air
	$(METALLIB) -o $@ /tmp/maxcomp_prng.air /tmp/maxcomp_hash.air /tmp/maxcomp_ifs.air

clean:
	rm -f _maxcomp_native.dylib maxcomp_kernels.metallib
	rm -f /tmp/maxcomp_*.air
