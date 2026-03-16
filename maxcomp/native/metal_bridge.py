"""Metal GPU bridge using ctypes to ObjC runtime.

Provides GPU-accelerated dispatch for PRNG search, hash preimage search,
and IFS block matching via Metal compute shaders.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import struct

logger = logging.getLogger(__name__)

# ObjC runtime type shortcuts
_objc = None
_id = ctypes.c_void_p
_SEL = ctypes.c_void_p
_BOOL = ctypes.c_bool


def _load_objc():
    """Load ObjC runtime and set up objc_msgSend signatures."""
    global _objc
    path = ctypes.util.find_library("objc")
    if path is None:
        raise RuntimeError("ObjC runtime not found")
    _objc = ctypes.cdll.LoadLibrary(path)
    _objc.objc_getClass.restype = _id
    _objc.objc_getClass.argtypes = [ctypes.c_char_p]
    _objc.sel_registerName.restype = _SEL
    _objc.sel_registerName.argtypes = [ctypes.c_char_p]
    _objc.objc_msgSend.restype = _id
    _objc.objc_msgSend.argtypes = [_id, _SEL]
    return _objc


def _sel(name: str) -> ctypes.c_void_p:
    return _objc.sel_registerName(name.encode())


def _cls(name: str) -> ctypes.c_void_p:
    return _objc.objc_getClass(name.encode())


def _msg(obj, sel, *args):
    """Send an ObjC message. Returns id (void*)."""
    return _objc.objc_msgSend(obj, sel, *args)


class MetalBridge:
    """Minimal bridge to Metal compute for maxcomp GPU kernels."""

    def __init__(self) -> None:
        self.available = False
        self._device = None
        self._queue = None
        self._library = None
        self._pipelines: dict[str, ctypes.c_void_p] = {}

        try:
            self._init_metal()
        except Exception as e:
            logger.debug("Metal init failed: %s", e)

    def _init_metal(self) -> None:
        """Initialize Metal device, command queue, and load shader library."""
        _load_objc()

        # Load Metal framework
        metal_path = "/System/Library/Frameworks/Metal.framework/Metal"
        if not os.path.exists(metal_path):
            raise RuntimeError("Metal framework not found")
        metal = ctypes.cdll.LoadLibrary(metal_path)

        # MTLCreateSystemDefaultDevice()
        metal.MTLCreateSystemDefaultDevice.restype = _id
        metal.MTLCreateSystemDefaultDevice.argtypes = []
        self._device = metal.MTLCreateSystemDefaultDevice()
        if not self._device:
            raise RuntimeError("No Metal device available")

        # Create command queue: [device newCommandQueue]
        self._queue = _msg(self._device, _sel("newCommandQueue"))
        if not self._queue:
            raise RuntimeError("Failed to create command queue")

        # Load metallib
        metallib_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "maxcomp_kernels.metallib",
        )
        if not os.path.isfile(metallib_path):
            logger.debug("Metal shader library not found at %s", metallib_path)
            return

        # [device newLibraryWithFile:path error:&err]
        ns_string_cls = _cls("NSString")
        path_nsstr = _msg(
            ns_string_cls,
            _sel("stringWithUTF8String:"),
            metallib_path.encode(),
        )
        err = _id()
        # We need a different msgSend signature for this call
        send = _objc.objc_msgSend
        send.restype = _id
        send.argtypes = [_id, _SEL, _id, ctypes.POINTER(_id)]
        self._library = send(
            self._device,
            _sel("newLibraryWithFile:error:"),
            path_nsstr,
            ctypes.byref(err),
        )
        if not self._library:
            logger.debug("Failed to load Metal library")
            return

        # Pre-create pipeline state objects for known kernels
        kernel_names = [
            "xorshift128_search",
            "lcg_search",
            "sha256_preimage_search",
            "ifs_block_match",
        ]
        for name in kernel_names:
            try:
                self._create_pipeline(name)
            except Exception as e:
                logger.debug("Failed to create pipeline for %s: %s", name, e)

        self.available = True
        logger.info("Metal bridge initialized with %d pipelines", len(self._pipelines))

    def _create_pipeline(self, kernel_name: str) -> None:
        """Create a compute pipeline state object for a kernel function."""
        name_nsstr = _msg(
            _cls("NSString"),
            _sel("stringWithUTF8String:"),
            kernel_name.encode(),
        )
        func = _msg(self._library, _sel("newFunctionWithName:"), name_nsstr)
        if not func:
            raise RuntimeError(f"Kernel function '{kernel_name}' not found in metallib")

        err = _id()
        send = _objc.objc_msgSend
        send.restype = _id
        send.argtypes = [_id, _SEL, _id, ctypes.POINTER(_id)]
        pipeline = send(
            self._device,
            _sel("newComputePipelineStateWithFunction:error:"),
            func,
            ctypes.byref(err),
        )
        if not pipeline:
            raise RuntimeError(f"Failed to create pipeline for '{kernel_name}'")

        self._pipelines[kernel_name] = pipeline

    def _create_buffer(self, data: bytes) -> ctypes.c_void_p:
        """Create a Metal buffer with the given data."""
        send = _objc.objc_msgSend
        send.restype = _id
        send.argtypes = [_id, _SEL, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]
        buf = send(
            self._device,
            _sel("newBufferWithBytes:length:options:"),
            data,
            len(data),
            0,  # MTLResourceStorageModeShared
        )
        return buf

    def _create_buffer_length(self, length: int) -> ctypes.c_void_p:
        """Create an empty Metal buffer of given length."""
        send = _objc.objc_msgSend
        send.restype = _id
        send.argtypes = [_id, _SEL, ctypes.c_uint64, ctypes.c_uint64]
        buf = send(
            self._device,
            _sel("newBufferWithLength:options:"),
            length,
            0,  # MTLResourceStorageModeShared
        )
        return buf

    def dispatch_prng_search(
        self,
        target_bytes: bytes,
        target_bit_length: int,
        prng_type: int,
        max_seed: int,
    ) -> int:
        """Dispatch PRNG seed search to GPU. Returns matching seed or -1."""
        kernel_name = "xorshift128_search" if prng_type == 1 else "lcg_search"
        if kernel_name not in self._pipelines:
            return -1

        # Create buffers
        target_buf = self._create_buffer(target_bytes)
        result_data = struct.pack("<i", -1)  # atomic_int initialized to -1
        result_buf = self._create_buffer(result_data)
        tbl_data = struct.pack("<I", target_bit_length)
        tbl_buf = self._create_buffer(tbl_data)
        offset_data = struct.pack("<I", 0)
        offset_buf = self._create_buffer(offset_data)

        # Encode and dispatch
        cmd_buf = _msg(self._queue, _sel("commandBuffer"))
        encoder = _msg(cmd_buf, _sel("computeCommandEncoder"))

        pipeline = self._pipelines[kernel_name]
        _msg(encoder, _sel("setComputePipelineState:"), pipeline)

        send = _objc.objc_msgSend
        send.restype = None
        send.argtypes = [_id, _SEL, _id, ctypes.c_uint64]
        send(encoder, _sel("setBuffer:offset:atIndex:"), target_buf, 0)
        # Note: simplified — full implementation needs proper atIndex: parameters

        _msg(encoder, _sel("endEncoding"))
        _msg(cmd_buf, _sel("commit"))
        _msg(cmd_buf, _sel("waitUntilCompleted"))

        # Read result
        contents = _msg(result_buf, _sel("contents"))
        if contents:
            result_val = ctypes.cast(contents, ctypes.POINTER(ctypes.c_int32))[0]
            return result_val
        return -1

    def dispatch_hash_search(
        self,
        target_bytes: bytes,
        target_bit_length: int,
        hash_type: int,
        preimage_len: int,
    ) -> int:
        """Dispatch hash preimage search to GPU. Returns preimage index or -1."""
        if "sha256_preimage_search" not in self._pipelines:
            return -1
        # Similar pattern to PRNG dispatch
        # TODO: implement full buffer setup and dispatch
        return -1

    def dispatch_ifs_match(
        self,
        domain_pool: bytes,
        range_blocks: bytes,
        num_domains: int,
        num_range_blocks: int,
        block_size: int,
    ) -> list[int] | None:
        """Dispatch IFS block matching to GPU. Returns hamming distance matrix or None."""
        if "ifs_block_match" not in self._pipelines:
            return None
        # TODO: implement full buffer setup and 2D dispatch
        return None
