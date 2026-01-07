#ifndef SAUERKRAUT_CPYTHON_COMPAT_HH_INCLUDED
#define SAUERKRAUT_CPYTHON_COMPAT_HH_INCLUDED
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#if PY_VERSION_HEX >= 0x30D0000 && PY_VERSION_HEX < 0x30E0000
#    define SAUERKRAUT_PY313 1
#else
#    define SAUERKRAUT_PY313 0
#endif

#if PY_VERSION_HEX >= 0x30E0000 && PY_VERSION_HEX < 0x30F0000
#    define SAUERKRAUT_PY314 1
#else
#    define SAUERKRAUT_PY314 0
#endif

#if !SAUERKRAUT_PY313 && !SAUERKRAUT_PY314
#    error "Unsupported Python version"
#endif

namespace pycompat {
    constexpr size_t  CHUNK_ALLOC_MINIMUM_OVERHEAD = 1000;
    constexpr size_t DATA_STACK_CHUNK_SIZE = 16 * 1024;
}
#endif 