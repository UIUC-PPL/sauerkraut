#ifndef PY_STRUCTS_HH_INCLUDED
#define PY_STRUCTS_HH_INCLUDED
#include "sauerkraut_cpython_compat.h"

extern "C" {
typedef union _PyStackRef {
    uintptr_t bits;
} _PyStackRef;

#if SAUERKRAUT_PY314
#define Py_TAG_REFCNT 1
#define Py_INT_TAG 3
#define PyStackRef_IsTaggedInt(ref) ((((ref).bits) & Py_INT_TAG) == Py_INT_TAG)
#define PyStackRef_AsPyObjectBorrow(ref) ((PyObject*)(((ref).bits) & (~(uintptr_t)Py_TAG_REFCNT)))
#elif SAUERKRAUT_PY313
#define PyStackRef_IsTaggedInt(ref) (0)
#define PyStackRef_AsPyObjectBorrow(ref) ((PyObject*)((ref).bits))
#endif

// Dummy definition: real definition is in pycore_code.h
typedef struct _CodeUnit {
    uint8_t opcode;
    uint8_t oparg;
} _CodeUnit;

struct _frame {
    PyObject_HEAD
    PyFrameObject *f_back;      /* previous frame, or NULL */
    struct _PyInterpreterFrame *f_frame; /* points to the frame data */
    PyObject *f_trace;          /* Trace function */
    int f_lineno;               /* Current line number. Only valid if non-zero */
    char f_trace_lines;         /* Emit per-line trace events? */
    char f_trace_opcodes;       /* Emit per-opcode trace events? */
    PyObject *f_extra_locals;   /* Dict for locals set by users using f_locals, could be NULL */
    PyObject *f_locals_cache;   /* Backwards compatibility for PyEval_GetLocals */
    #if SAUERKRAUT_PY314
    PyObject *f_overwritten_fast_locals; /* Support for borrowed refs to fast locals */
    #endif
    PyObject *_f_frame_data[1]; /* Frame data if this frame object owns the frame */
};

struct _PyInterpreterFrame *
_PyThreadState_PushFrame(PyThreadState *tstate, size_t size);

typedef struct _PyInterpreterFrame {
    _PyStackRef f_executable; /* Deferred or strong reference (code object or None) */
    struct _PyInterpreterFrame *previous;
    #if SAUERKRAUT_PY314
    _PyStackRef f_funcobj; /* Deferred or strong reference. Only valid if not on C stack */
    #elif SAUERKRAUT_PY313
    PyObject *f_funcobj; /* Strong reference. Only valid if not on C stack */
    #endif
    PyObject *f_globals; /* Borrowed reference. Only valid if not on C stack */
    PyObject *f_builtins; /* Borrowed reference. Only valid if not on C stack */
    PyObject *f_locals; /* Strong reference, may be NULL. Only valid if not on C stack */
    PyFrameObject *frame_obj; /* Strong reference, may be NULL. Only valid if not on C stack */
    _CodeUnit *instr_ptr; /* Instruction currently executing (or about to begin) */
    #if SAUERKRAUT_PY314
    _PyStackRef *stackpointer;
    #elif SAUERKRAUT_PY313
    int stacktop;
    #endif
    uint16_t return_offset;  /* Only relevant during a function call */
    char owner;
    #if SAUERKRAUT_PY314
    uint8_t visited;
    #endif
    /* Locals and stack */
    _PyStackRef localsplus[1];
} _PyInterpreterFrame;

} // extern "C"

namespace sauerkraut {
    using PyInterpreterFrame = struct _PyInterpreterFrame;
    using PyFrame = struct _frame;
    using PyBitcodeInstruction = _CodeUnit;
}

#endif // PY_STRUCTS_HH_INCLUDED