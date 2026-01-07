import ctypes
import os
import sys

# --- 1. Load the ROCTX Library ---
# The spec says the library is 'libroctx64.so'.
# It is usually located in /opt/rocm/lib/

def load_roctx_library():
    # Common paths where ROCm libraries might reside
    search_paths = [
        os.getenv("ROCM_PATH", "/opt/rocm") + "/lib/libroctx64.so",
        "/opt/rocm/lib/libroctx64.so",
        "/usr/lib/libroctx64.so",
        "libroctx64.so" # Fallback to system LD_LIBRARY_PATH
    ]

    for path in search_paths:
        try:
            lib = ctypes.cdll.LoadLibrary(path)
            return lib
        except OSError:
            continue

    print("Warning: 'libroctx64.so' not found. ROCTX profiling will be disabled.")
    return None

_libroctx = load_roctx_library()

# --- 2. Define C API Signatures (Strictly following the Spec) ---

if _libroctx:
    # int roctxRangePushA(const char* message);
    _libroctx.roctxRangePushA.argtypes = [ctypes.c_char_p]
    _libroctx.roctxRangePushA.restype = ctypes.c_int

    # int roctxRangePop();
    _libroctx.roctxRangePop.argtypes = []
    _libroctx.roctxRangePop.restype = ctypes.c_int

    # void roctxMarkA(const char* message);
    _libroctx.roctxMarkA.argtypes = [ctypes.c_char_p]
    _libroctx.roctxMarkA.restype = None

# --- 3. Python Wrapper Classes ---

def roctx_mark(message: str):
    """
    Inserts a marker in the trace. (Instantaneous event)
    """
    if _libroctx:
        _libroctx.roctxMarkA(message.encode('utf-8'))

class roctx_range:
    """
    Context manager for ROCTX ranges.
    Maps to roctxRangePushA / roctxRangePop.
    """
    def __init__(self, message: str):
        self.message_bytes = message.encode('utf-8')

    def __enter__(self):
        if _libroctx:
            _libroctx.roctxRangePushA(self.message_bytes)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if _libroctx:
            _libroctx.roctxRangePop()