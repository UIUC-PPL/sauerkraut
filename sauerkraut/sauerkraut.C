#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "sauerkraut_cpython_compat.h"
#include "greenlet_compat.h"
#include <stdbool.h>
#include <vector>
#include <memory>
#include "flatbuffers/flatbuffers.h"
#include "py_object_generated.h"
#include "utils.h"
#include "serdes.h"
#include "pyref.h" 
#include "py_structs.h"
#include <unordered_map>
#include <tuple>
#include <string>
#include <optional>

// The order of the tuple is: funcobj, code, globals
using PyCodeImmutables = std::tuple<pyobject_strongref, pyobject_strongref, pyobject_strongref>;
using PyCodeImmutableCache = std::unordered_map<std::string, PyCodeImmutables>;

class sauerkraut_modulestate {
    public:
        pyobject_strongref deepcopy;
        pyobject_strongref deepcopy_module;
        pyobject_strongref pickle_module;
        pyobject_strongref pickle_dumps;
        pyobject_strongref pickle_loads;
        pyobject_strongref dill_module;
        pyobject_strongref dill_dumps;
        pyobject_strongref dill_loads;
        pyobject_strongref liveness_module;
        pyobject_strongref get_dead_variables_at_offset;
        PyCodeImmutableCache code_immutable_cache;
        sauerkraut_modulestate() = default;

        bool init() {
            auto import_module = [](const char* name, pyobject_strongref& dest) -> bool {
                dest = PyImport_ImportModule(name);
                return static_cast<bool>(dest);
            };

            auto get_attr = [](pyobject_strongref& module, const char* attr, pyobject_strongref& dest) -> bool {
                dest = PyObject_GetAttrString(*module, attr);
                return static_cast<bool>(dest);
            };

            if (!import_module("copy", deepcopy_module) ||
                !get_attr(deepcopy_module, "deepcopy", deepcopy)) {
                return false;
            }

            if (!import_module("pickle", pickle_module) ||
                !get_attr(pickle_module, "dumps", pickle_dumps) ||
                !get_attr(pickle_module, "loads", pickle_loads)) {
                return false;
            }

            if (!import_module("dill", dill_module) ||
                !get_attr(dill_module, "dumps", dill_dumps) ||
                !get_attr(dill_module, "loads", dill_loads)) {
                return false;
            }

            if (!import_module("sauerkraut.liveness", liveness_module) ||
                !get_attr(liveness_module, "get_dead_variables_at_offset", get_dead_variables_at_offset)) {
                return false;
            }

            return true;
        }

        pyobject_strongref get_dead_variables(py_weakref<PyCodeObject> code, int offset) {
            pyobject_strongref args = pyobject_strongref::steal(Py_BuildValue("(Oi)", *code, offset));
            pyobject_strongref result = pyobject_strongref::steal(PyObject_CallObject(get_dead_variables_at_offset.borrow(), args.borrow()));
            return result;
        }

        void cache_code_immutables(py_weakref<PyFrameObject> frame) {
            pyobject_strongref code = pyobject_strongref::steal((PyObject*)PyFrame_GetCode(*frame));
            PyObject *name = ((PyCodeObject*)code.borrow())->co_name;
            std::string name_str = std::string(PyUnicode_AsUTF8(name));
            auto cached_invariants = code_immutable_cache.find(name_str);

            // it's already in the cache, so we can return
            if(cached_invariants != code_immutable_cache.end()) {
                return;
            }

            // it's not in the cache, so we need to compute the invariants
#if SAUERKRAUT_PY314
            auto funcobj = make_strongref(utils::py::stackref_as_pyobject(frame->f_frame->f_funcobj));
#else
            auto funcobj = make_strongref(frame->f_frame->f_funcobj);
#endif
            code_immutable_cache[name_str] = std::make_tuple(funcobj, code, frame->f_frame->f_globals);
        }

        std::optional<PyCodeImmutables> get_code_immutables(py_weakref<PyFrameObject> frame) {
            pycode_strongref code = pycode_strongref::steal(PyFrame_GetCode(*frame));
            PyObject *name = code->co_name;
            std::string name_str = std::string(PyUnicode_AsUTF8(name));
            auto cached_invariants = code_immutable_cache.find(name_str);
            if(cached_invariants != code_immutable_cache.end()) {
                return cached_invariants->second;
            }
            return std::nullopt;
        }
        std::optional<PyCodeImmutables> get_code_immutables(serdes::DeserializedPyInterpreterFrame &frame) {
            pyobject_weakref name = frame.f_executable.co_name.borrow();
            std::string name_str = std::string(PyUnicode_AsUTF8(*name));
            auto cached_invariants = code_immutable_cache.find(name_str);
            if(cached_invariants != code_immutable_cache.end()) {
                return cached_invariants->second;
            }
            return std::nullopt;
        }

        std::optional<PyCodeImmutables> get_code_immutables(serdes::DeserializedPyFrame &frame) {
            return get_code_immutables(frame.f_frame);
        }

        void clear() {
            // Clear the cache first - this decrefs Python objects while interpreter is still valid
            code_immutable_cache.clear();
            // Clear all module references
            deepcopy.reset();
            deepcopy_module.reset();
            pickle_module.reset();
            pickle_dumps.reset();
            pickle_loads.reset();
            dill_module.reset();
            dill_dumps.reset();
            dill_loads.reset();
            liveness_module.reset();
            get_dead_variables_at_offset.reset();
        }

};

class dumps_functor {
    pyobject_weakref pickle_dumps;
    pyobject_weakref _dill_dumps;
    public:
    dumps_functor(pyobject_weakref pickle_dumps, pyobject_weakref _dill_dumps) : pickle_dumps(pickle_dumps), _dill_dumps(_dill_dumps) {}

    pyobject_strongref operator()(PyObject *obj) {
        PyObject *result = PyObject_CallOneArg(*pickle_dumps, obj);
        return pyobject_strongref::steal(result);
    }

    pyobject_strongref dill_dumps(PyObject *obj) {
        PyObject *result = PyObject_CallOneArg(*_dill_dumps, obj);
        return pyobject_strongref::steal(result);
    }
};

class loads_functor {
    pyobject_weakref pickle_loads;
    pyobject_weakref _dill_loads;
    public:
    loads_functor(pyobject_weakref pickle_loads, pyobject_weakref _dill_loads) : pickle_loads(pickle_loads), _dill_loads(_dill_loads) {}

    pyobject_strongref operator()(PyObject *obj) {
        PyObject *result = PyObject_CallOneArg(*pickle_loads, obj);
        return pyobject_strongref::steal(result);
    }

    pyobject_strongref dill_loads(PyObject *obj) {
        PyObject *result = PyObject_CallOneArg(*_dill_loads, obj);
        return pyobject_strongref::steal(result);
    }
};


static sauerkraut_modulestate *sauerkraut_state;

extern "C" {

struct frame_copy_capsule;
static PyObject *_serialize_frame_direct_from_capsule(frame_copy_capsule *copy_capsule, serdes::SerializationArgs args);
static PyObject *_serialize_frame_from_capsule(PyObject *capsule, serdes::SerializationArgs args);

static inline _PyStackRef *_PyFrame_Stackbase(_PyInterpreterFrame *f) {
    return f->localsplus + ((PyCodeObject*)f->f_executable.bits)->co_nlocalsplus;
}


PyAPI_FUNC(PyFrameObject *) PyFrame_New(PyThreadState *, PyCodeObject *,
                                        PyObject *, PyObject *);

typedef struct serialized_obj {
    char *data;
    size_t size;
} serialized_obj;


static bool handle_exclude_locals(PyObject* exclude_locals, py_weakref<PyFrameObject> frame, serdes::SerializationArgs& ser_args) {
    if(exclude_locals != NULL) {
        auto bitmask = utils::py::exclude_locals(frame, exclude_locals);
        ser_args.set_exclude_locals(bitmask);
    }
    return true;
}

pyobject_strongref get_dead_locals_set(py_weakref<PyFrameObject> frame) {
    pycode_strongref code = pycode_strongref::steal(PyFrame_GetCode(*frame));
    auto offset = utils::py::get_instr_offset<utils::py::Units::Bytes>(frame);
    pyobject_strongref dead_vars = sauerkraut_state->get_dead_variables(code, offset);
    return dead_vars;
}

static bool handle_replace_locals(PyObject* replace_locals, py_weakref<PyFrameObject> frame) {
    if (replace_locals != NULL && replace_locals != Py_None) {
        if (!utils::py::check_dict(replace_locals)) {    
            PyErr_SetString(PyExc_TypeError, "replace_locals must be a dictionary");
            return false;
        }
        utils::py::replace_locals(frame, replace_locals);
    }
    return true;
}

PyObject *GetFrameLocalsFromFrame(py_weakref<PyObject> frame) {
    py_weakref<PyFrameObject> current_frame{(PyFrameObject *)*frame};
    
    PyObject *locals = PyFrame_GetLocals(*current_frame);
    if (locals == NULL) {
        return NULL;
    }


    if (PyFrameLocalsProxy_Check(locals)) {
        PyObject* ret = PyDict_New();
        if (ret == NULL) {
            Py_DECREF(locals);
            return NULL;
        }
        if (PyDict_Update(ret, locals) < 0) {
            Py_DECREF(ret);
            Py_DECREF(locals);
            return NULL;
        }
        Py_DECREF(locals);


        return ret;
    }

    assert(PyMapping_Check(locals));
    return locals;
}

PyObject *deepcopy_object(py_weakref<PyObject> obj) {
    if (*obj == NULL) {
        return NULL;
    }
    py_weakref<PyObject> deepcopy{*sauerkraut_state->deepcopy};
    PyObject *copy_obj = PyObject_CallFunction(*deepcopy, "O", *obj);
    return copy_obj;
}

static void cleanup_interpreter_frame(_PyInterpreterFrame *interp, int nlocalsplus, int stack_depth) {
    utils::py::stackref_decref(interp->f_executable);
    #if SAUERKRAUT_PY314
    utils::py::stackref_decref(interp->f_funcobj);
    #elif SAUERKRAUT_PY313
    Py_XDECREF(interp->f_funcobj);
    #endif
    Py_XDECREF(interp->f_locals);

    for (int i = 0; i < nlocalsplus; i++) {
        utils::py::stackref_decref(interp->localsplus[i]);
    }

    _PyStackRef *stack_base = interp->localsplus + nlocalsplus;
    for (int i = 0; i < stack_depth; i++) {
        utils::py::stackref_decref(stack_base[i]);
    }

    free(interp);
}

typedef struct frame_copy_capsule {
    // Strong reference
    PyFrameObject *frame;
    utils::py::StackState stack_state;
    bool owns_interpreter_frame;
    int nlocalsplus;   // For cleanup iteration
    int stack_depth;   // For stack cleanup

    ~frame_copy_capsule() {
        if (frame) {
            if (owns_interpreter_frame && frame->f_frame) {
                auto *interp = frame->f_frame;

                utils::py::stackref_decref(interp->f_executable);
                #if SAUERKRAUT_PY314
                utils::py::stackref_decref(interp->f_funcobj);
                #elif SAUERKRAUT_PY313
                Py_XDECREF(interp->f_funcobj);
                #endif
                Py_XDECREF(interp->f_locals);

                for (int i = 0; i < nlocalsplus; i++) {
                    utils::py::stackref_decref(interp->localsplus[i]);
                }

                _PyStackRef *stack_base = interp->localsplus + nlocalsplus;
                for (int i = 0; i < stack_depth; i++) {
                    utils::py::stackref_decref(stack_base[i]);
                }

                // f_globals, f_builtins are borrowed refs; frame_obj is weak (no Py_NewRef)

                free(interp);
                frame->f_frame = NULL;
            }
            Py_XDECREF(frame);
        }
    }
} frame_copy_capsule;

static char copy_frame_capsule_name[] = "Frame Capsule Object";

void frame_copy_capsule_destroy(PyObject *capsule) {
    struct frame_copy_capsule *copy_capsule = (struct frame_copy_capsule *)PyCapsule_GetPointer(capsule, copy_frame_capsule_name);
    delete copy_capsule;
}

frame_copy_capsule *frame_copy_capsule_create_direct(py_weakref<PyFrameObject> frame, utils::py::StackState stack_state, bool owns_interpreter_frame = false, int nlocalsplus = 0, int stack_depth = 0) {
    struct frame_copy_capsule *copy_capsule = new struct frame_copy_capsule;
    copy_capsule->frame = (PyFrameObject*)Py_NewRef(*frame);
    copy_capsule->stack_state = stack_state;
    copy_capsule->owns_interpreter_frame = owns_interpreter_frame;
    copy_capsule->nlocalsplus = nlocalsplus;
    copy_capsule->stack_depth = stack_depth;
    return copy_capsule;
}

PyObject *frame_copy_capsule_create(py_weakref<PyFrameObject> frame, utils::py::StackState stack_state, bool owns_interpreter_frame = false, int nlocalsplus = 0, int stack_depth = 0) {
    auto *copy_capsule = frame_copy_capsule_create_direct(frame, stack_state, owns_interpreter_frame, nlocalsplus, stack_depth);
    return PyCapsule_New(copy_capsule, copy_frame_capsule_name, frame_copy_capsule_destroy);
}

void copy_localsplus(py_weakref<sauerkraut::PyInterpreterFrame> to_copy, 
                    py_weakref<sauerkraut::PyInterpreterFrame> new_frame, 
                    int nlocals, int deepcopy) {
    if (deepcopy) {
        for (int i = 0; i < nlocals; i++) {
            _PyStackRef local_ref = to_copy->localsplus[i];
            auto local_obj = utils::py::stackref_to_object_for_serialization(local_ref);
            if (local_obj.obj == NULL) {
                new_frame->localsplus[i].bits = 0;
                continue;
            }
            py_weakref<PyObject> local{local_obj.obj};
            PyObject *local_copy = deepcopy_object(local);
            if (local_obj.owned) {
                Py_DECREF(local_obj.obj);
            }
            new_frame->localsplus[i].bits = (uintptr_t)local_copy;
        }
    } else {
        memcpy(new_frame->localsplus, to_copy->localsplus, nlocals * sizeof(_PyStackRef));
    }
}

void copy_stack(py_weakref<sauerkraut::PyInterpreterFrame> to_copy, 
               py_weakref<sauerkraut::PyInterpreterFrame> new_frame, 
               int stack_size, int deepcopy) {
    _PyStackRef *src_stack_base = utils::py::get_stack_base(*to_copy);
    _PyStackRef *dest_stack_base = utils::py::get_stack_base(*new_frame);

    if(deepcopy) {
        for(int i = 0; i < stack_size; i++) {
            _PyStackRef stack_ref = src_stack_base[i];
            auto stack_obj = utils::py::stackref_to_object_for_serialization(stack_ref);
            if (stack_obj.obj == NULL) {
                dest_stack_base[i].bits = 0;
                continue;
            }
            auto stack_weak = make_weakref(stack_obj.obj);
            PyObject *stack_obj_copy = deepcopy_object(stack_weak);
            if (stack_obj.owned) {
                Py_DECREF(stack_obj.obj);
            }
            dest_stack_base[i].bits = (uintptr_t) stack_obj_copy;
        }
    } else {
        memcpy(dest_stack_base, src_stack_base, stack_size * sizeof(_PyStackRef));
    }
}

static py_weakref<PyFrameObject> prepare_frame_for_execution(py_weakref<PyFrameObject> frame) {
    utils::py::skip_current_call_instruction(frame);
    return frame;
}

PyFrameObject *create_copied_frame(py_weakref<PyThreadState> tstate, 
                                 py_weakref<sauerkraut::PyInterpreterFrame> to_copy, 
                                 py_weakref<PyCodeObject> code_obj, 
                                 py_weakref<PyObject> LocalCopy,
                                 int push_frame, int deepcopy_localsplus, 
                                 int set_previous, int stack_size, 
                                 int copy_stack_flag) {
    int nlocals = code_obj->co_nlocalsplus;

    PyFrameObject *new_frame = PyFrame_New(*tstate, *code_obj, to_copy->f_globals, *LocalCopy);

    _PyInterpreterFrame *stack_frame;
    if (push_frame) {
        stack_frame = utils::py::AllocateFrame(*tstate, code_obj->co_framesize);
    } else {
        stack_frame = utils::py::AllocateFrame(code_obj->co_framesize);
    }

    if(stack_frame == NULL) {
        Py_DECREF(new_frame);
        PySys_WriteStderr("<Sauerkraut>: Could not allocate memory for new frame\n");
        return NULL;
    }

    new_frame->f_frame = stack_frame;
    py_weakref<sauerkraut::PyInterpreterFrame> new_frame_ref{new_frame->f_frame};

    new_frame_ref->owner = to_copy->owner;
    new_frame_ref->previous = set_previous ? *to_copy : NULL;
#if SAUERKRAUT_PY314
    new_frame_ref->f_funcobj.bits = (uintptr_t)deepcopy_object(make_weakref(utils::py::stackref_as_pyobject(to_copy->f_funcobj)));
#else
    new_frame_ref->f_funcobj = deepcopy_object(make_weakref(to_copy->f_funcobj));
#endif
    new_frame_ref->f_executable.bits = (uintptr_t)deepcopy_object(make_weakref((PyObject*)to_copy->f_executable.bits));
    new_frame_ref->f_globals = to_copy->f_globals;
    new_frame_ref->f_builtins = to_copy->f_builtins;
    new_frame_ref->f_locals = to_copy->f_locals;
    new_frame_ref->return_offset = to_copy->return_offset;
    new_frame_ref->frame_obj = new_frame;
    auto offset = utils::py::get_instr_offset<utils::py::Units::Bytes>(to_copy);
    new_frame->f_frame->instr_ptr = (_CodeUnit*) (code_obj->co_code_adaptive + offset);

    copy_localsplus(to_copy, new_frame_ref, nlocals, deepcopy_localsplus);
    copy_stack(to_copy, new_frame_ref, stack_size, 1);

    // Set stack position after copying stack
    #if SAUERKRAUT_PY314
    new_frame->f_frame->stackpointer = new_frame->f_frame->localsplus + nlocals + stack_size;
    new_frame->f_frame->visited = 0;
    #elif SAUERKRAUT_PY313
    new_frame->f_frame->stacktop = nlocals + stack_size;
    #endif

    if(push_frame) {
        return *prepare_frame_for_execution(new_frame);
    } else {
        return new_frame;
    }
}

PyFrameObject *push_frame_for_running(PyThreadState *tstate, _PyInterpreterFrame *to_push, PyCodeObject *code) {
    // what about ownership? I'm thinking this should steal everything from to_push
    // might create problems with the deallocation of the frame, though. Will have to see
    _PyInterpreterFrame *stack_frame = utils::py::ThreadState_PushFrame(tstate, code->co_framesize);
    py_weakref<PyFrameObject> pyframe_object = to_push->frame_obj;
    if(stack_frame == NULL) {
        PySys_WriteStderr("<Sauerkraut>: Could not allocate memory for new frame\n");
        PySys_WriteStderr("<Sauerkraut>: Tried to allocate frame of size %d\n", code->co_framesize);
        return NULL;
    }

    copy_localsplus(to_push, stack_frame, code->co_nlocalsplus, 0);
    auto offset = utils::py::get_instr_offset<utils::py::Units::Bytes>(to_push->frame_obj);
    
    stack_frame->owner = to_push->owner;
    // Set previous to NULL so that when the frame returns, it exits the interpreter loop
    // rather than trying to continue in some other frame
    stack_frame->previous = NULL;
    stack_frame->f_funcobj = to_push->f_funcobj;
    stack_frame->f_executable.bits = to_push->f_executable.bits;
    stack_frame->f_globals = to_push->f_globals;
    stack_frame->f_builtins = to_push->f_builtins;
    stack_frame->f_locals = to_push->f_locals;
    stack_frame->frame_obj = *pyframe_object;
    stack_frame->instr_ptr = (_CodeUnit*) (code->co_code_adaptive + (offset));
    auto stack_depth = utils::py::get_current_stack_depth(to_push);
    copy_stack(to_push, stack_frame, stack_depth, 0);
    #if SAUERKRAUT_PY314
    stack_frame->stackpointer = stack_frame->localsplus + code->co_nlocalsplus + stack_depth;
    stack_frame->visited = 0;
    #elif SAUERKRAUT_PY313
    stack_frame->stacktop = code->co_nlocalsplus + stack_depth;
    #endif
    stack_frame->return_offset = to_push->return_offset;

    pyframe_object->f_frame = stack_frame;
    return *prepare_frame_for_execution(pyframe_object);
}

struct SerializationOptions {
    bool serialize = false;
    pyobject_strongref exclude_locals;
    Py_ssize_t sizehint = 0;
    bool exclude_dead_locals = true;
    bool exclude_immutables = false;
    serdes::SerializationArgs to_ser_args() const {
        serdes::SerializationArgs args;
        if (sizehint > 0) {
            args.set_sizehint(sizehint);
        }
        args.set_exclude_immutables(exclude_immutables);
        return args;
    }
};

static pyobject_strongref combine_exclusions(py_weakref<PyFrameObject> frame, PyObject* exclude_locals, bool exclude_dead_locals) {
    pyobject_strongref excluded_vars;
    
    // Start with user-provided exclusions if any
    if (NULL != exclude_locals && exclude_locals != Py_None) {
        excluded_vars = pyobject_strongref::steal(PySet_New(exclude_locals));
        if (!excluded_vars) {
            PyErr_SetString(PyExc_TypeError, "exclude_locals must be a set-like object");
            return pyobject_strongref(NULL);
        }
    } else {
        excluded_vars = pyobject_strongref::steal(PySet_New(NULL));
    }
    
    // Add dead variables if requested
    if (exclude_dead_locals) {
        auto dead_locals = get_dead_locals_set(frame);
        if (dead_locals) {
            utils::py::set_update(excluded_vars.borrow(), dead_locals.borrow());
        }
    }
    
    return excluded_vars;
}

static bool apply_exclusions(py_weakref<PyFrameObject> frame, const SerializationOptions& options, 
                            serdes::SerializationArgs& ser_args) {
    auto excluded_vars = combine_exclusions(frame, options.exclude_locals.borrow(), options.exclude_dead_locals);
    if (!excluded_vars) {
        return false;
    }
    
    return handle_exclude_locals(excluded_vars.borrow(), frame, ser_args);
}

static PyObject *_copy_frame_object(py_weakref<PyFrameObject> frame, const SerializationOptions& options) {
    using namespace utils;
    serdes::SerializationArgs args = options.to_ser_args();
    
    if (!apply_exclusions(frame, options, args)) {
        return NULL;
    }
    
    _PyInterpreterFrame *to_copy = frame->f_frame;
    PyThreadState *tstate = PyThreadState_Get();
    pycode_strongref code = pycode_strongref::steal(PyFrame_GetCode(*frame));
    assert(code.borrow() != NULL);
    PyCodeObject *copy_code_obj = (PyCodeObject *)deepcopy_object((PyObject*)code.borrow());

    PyObject *FrameLocals = GetFrameLocalsFromFrame((PyObject*)*frame);

    // We want to copy these here because we want to "freeze" the locals
    // at this point; with a shallow copy, changes to locals will propagate to
    // the copied frame between its copy and serialization.
    PyObject *LocalCopy = deepcopy_object(FrameLocals);

    auto stack_state = utils::py::get_stack_state((PyObject*)*frame);
    PyFrameObject *new_frame = create_copied_frame(tstate, to_copy, copy_code_obj, LocalCopy, 0, 1, 0, stack_state.size(), 1);

    PyObject *capsule = frame_copy_capsule_create(new_frame, stack_state, true);
    Py_DECREF(new_frame);  // Drop our ref; capsule holds its own
    Py_DECREF(copy_code_obj);
    Py_DECREF(LocalCopy);
    Py_DECREF(FrameLocals);

    return capsule;
}


static PyObject *_copy_serialize_frame_object(py_weakref<PyFrameObject> frame, const SerializationOptions& options) {
    if(options.exclude_immutables) {
        sauerkraut_state->cache_code_immutables(frame);
    }

    // First copy the frame, then serialize from the copy
    // This ensures we have a consistent snapshot of the frame state
    PyObject *capsule = _copy_frame_object(frame, options);
    if (capsule == NULL) {
        return NULL;
    }

    serdes::SerializationArgs args = options.to_ser_args();
    if (!apply_exclusions(frame, options, args)) {
        Py_DECREF(capsule);
        return NULL;
    }
    PyObject *ret = _serialize_frame_from_capsule(capsule, args);
    Py_DECREF(capsule);  // Done with the capsule
    return ret;
}

static PyObject *_copy_current_frame(PyObject *self, PyObject *args, const SerializationOptions& options) {
    using namespace utils;
    PyFrameObject *frame = (PyFrameObject*) PyEval_GetFrame();
    return _copy_frame_object(make_weakref(frame), options);
}

static PyObject *_copy_serialize_current_frame(PyObject *self, PyObject *args, const SerializationOptions& options) {
    // here, we'll copy the frame "directly" into the serialized buffer
    using namespace utils;
    auto frame_ref = make_weakref(PyEval_GetFrame());
    return _copy_serialize_frame_object(frame_ref, options);
}



static bool parse_sizehint(PyObject* sizehint_obj, Py_ssize_t& sizehint) {
    if (sizehint_obj != NULL) {
        sizehint = PyLong_AsLong(sizehint_obj);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "sizehint must be an integer");
            return false;
        }
    }
    return true;
}

static bool parse_serialization_options(PyObject* args, PyObject* kwargs, SerializationOptions& options) {
    static char* kwlist[] = {"serialize", "exclude_locals", 
                             "exclude_immutables", "sizehint", 
                             "exclude_dead_locals", NULL};
    int serialize = 0;
    PyObject* sizehint_obj = NULL;
    PyObject* exclude_locals = NULL;
    int exclude_dead_locals = 1;
    int exclude_immutables = 0;
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|pOpOp", kwlist, 
                                    &serialize, &exclude_locals, 
                                    &exclude_immutables, &sizehint_obj, 
                                    &exclude_dead_locals)) {
        return false;
    }
    
    options.serialize = (serialize != 0);
    options.exclude_dead_locals = (exclude_dead_locals != 0);
    options.exclude_locals = pyobject_strongref(exclude_locals);
    options.exclude_immutables = (exclude_immutables != 0);
    return parse_sizehint(sizehint_obj, options.sizehint);
}

static PyObject *run_and_cleanup_frame(PyFrameObject *frame) {
    PyObject *res = PyEval_EvalFrame(frame);

    // The stack frame is automatically cleaned up by Python after PyEval_EvalFrame.
    // We just need to clear f_frame to avoid dangling pointer when the frame is GC'd.
    frame->f_frame = NULL;

    return res;
}

static PyObject *copy_current_frame(PyObject *self, PyObject *args, PyObject *kwargs) {
    SerializationOptions options;
    if (!parse_serialization_options(args, kwargs, options)) {
        return NULL;
    }

    if (options.serialize) {
        return _copy_serialize_current_frame(self, args, options);
    } else {
        return _copy_current_frame(self, args, options);
    }
}

static PyObject *copy_frame(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *frame = NULL;
    SerializationOptions options;

    static char *kwlist[] = {"frame", "exclude_locals", "sizehint",
                             "serialize", "exclude_dead_locals", "exclude_immutables", NULL};
    int serialize = 0;
    PyObject* sizehint_obj = NULL;
    PyObject* exclude_locals = NULL;
    int exclude_dead_locals = 1;
    int exclude_immutables = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OOppp", kwlist,
                                    &frame, &exclude_locals, &sizehint_obj, &serialize, &exclude_dead_locals, &exclude_immutables)) {
        return NULL;
    }

    options.serialize = (serialize != 0);
    options.exclude_dead_locals = (exclude_dead_locals != 0);
    options.exclude_immutables = (exclude_immutables != 0);
    options.exclude_locals = pyobject_strongref(exclude_locals);
    if (!parse_sizehint(sizehint_obj, options.sizehint)) {
        return NULL;
    }

    auto frame_back = py_strongref<PyFrameObject>::steal(PyFrame_GetBack((PyFrameObject*)frame));
    py_weakref<PyFrameObject> frame_ref{frame_back.borrow()};

    if (options.serialize) {
        return _copy_serialize_frame_object(frame_ref, options);
    } else {
        return _copy_frame_object(frame_ref, options);
    }
}

// static PyObject *_copy_run_frame_from_capsule(PyObject *capsule) {
//     if (PyErr_Occurred()) {
//         PyErr_Print();
//         return NULL;
//     }

//     struct frame_copy_capsule *copy_capsule = (struct frame_copy_capsule *)PyCapsule_GetPointer(capsule, copy_frame_capsule_name);
//     if (copy_capsule == NULL) {
//         return NULL;
//     }

//     PyFrameObject *frame = copy_capsule->frame;
//     _PyInterpreterFrame *to_copy = frame->f_frame;
//     (void) to_copy;
//     PyCodeObject *code = PyFrame_GetCode(frame);
//     assert(code != NULL);
//     PyCodeObject *copy_code_obj = (PyCodeObject *)deepcopy_object((PyObject*)code);
//     (void) copy_code_obj;

//     PyObject *FrameLocals = GetFrameLocalsFromFrame((PyObject*)frame);
//     (void) FrameLocals;
//     PyObject *LocalCopy = PyDict_Copy(FrameLocals);
//     (void) LocalCopy;

//     // PyFrameObject *new_frame = create_copied_frame(tstate, to_copy, copy_code_obj, LocalCopy, offset, 1, 0, 1, 0);
//     PyFrameObject *new_frame = NULL;

//     PyObject *res = PyEval_EvalFrame(new_frame);
//     Py_DECREF(copy_code_obj);
//     Py_DECREF(LocalCopy);
//     Py_DECREF(FrameLocals);

//     return res;
// }

// static PyObject *run_frame(PyObject *self, PyObject *args) {
//     PyObject *capsule;
//     if (!PyArg_ParseTuple(args, "O", &capsule)) {
//         return NULL;
//     }
//     return _copy_run_frame_from_capsule(capsule);
// }

static PyObject *_serialize_frame_direct_from_capsule(frame_copy_capsule *copy_capsule, serdes::SerializationArgs args) {
    loads_functor loads(sauerkraut_state->pickle_loads, sauerkraut_state->dill_loads);
    dumps_functor dumps(sauerkraut_state->pickle_dumps, sauerkraut_state->dill_dumps);

    flatbuffers::FlatBufferBuilder builder{args.sizehint};
    serdes::PyObjectSerdes po_serdes(loads, dumps);

    serdes::PyFrameSerdes frame_serdes{po_serdes};

    auto serialized_frame = frame_serdes.serialize(builder, *(static_cast<sauerkraut::PyFrame*>(copy_capsule->frame)), args);
    builder.Finish(serialized_frame);
    auto buf = builder.GetBufferPointer();
    auto size = builder.GetSize();
    PyObject *bytes = PyBytes_FromStringAndSize((const char *)buf, size);
    return bytes;
}

static PyObject* _serialize_frame_from_capsule(PyObject *capsule, serdes::SerializationArgs args) {
    if (PyErr_Occurred()) {
        PyErr_Print();
        return NULL;
    }

    struct frame_copy_capsule *copy_capsule = (struct frame_copy_capsule *)PyCapsule_GetPointer(capsule, copy_frame_capsule_name);
    if (copy_capsule == NULL) {
        return NULL;
    }

    return _serialize_frame_direct_from_capsule(copy_capsule, args);
}

static void init_code(PyCodeObject *obj, serdes::DeserializedCodeObject &code) {
    obj->co_consts = Py_NewRef(code.co_consts.borrow());
    obj->co_names = Py_NewRef(code.co_names.borrow());
    obj->co_exceptiontable = Py_NewRef(code.co_exceptiontable.borrow());

    obj->co_flags = code.co_flags;
    obj->co_argcount = code.co_argcount;
    obj->co_posonlyargcount = code.co_posonlyargcount;
    obj->co_kwonlyargcount = code.co_kwonlyargcount;
    obj->co_stacksize = code.co_stacksize;
    obj->co_firstlineno = code.co_firstlineno;

    obj->co_nlocalsplus = code.co_nlocalsplus;
    obj->co_framesize = code.co_framesize;
    obj->co_nlocals = code.co_nlocals;
    obj->co_ncellvars = code.co_ncellvars;
    obj->co_nfreevars = code.co_nfreevars;
    obj->co_version = code.co_version;

    obj->co_localsplusnames = Py_NewRef(code.co_localsplusnames.borrow());
    obj->co_localspluskinds = Py_NewRef(code.co_localspluskinds.borrow());

    obj->co_filename = Py_NewRef(code.co_filename.borrow());
    obj->co_name = Py_NewRef(code.co_name.borrow());
    obj->co_qualname = Py_NewRef(code.co_qualname.borrow());
    obj->co_linetable = Py_NewRef(code.co_linetable.borrow());

    memcpy(obj->co_code_adaptive, code.co_code_adaptive.data(), code.co_code_adaptive.size());

    // initialize the rest of the fields
    obj->co_weakreflist = NULL;
    obj->co_executors = NULL;
    obj->_co_cached = NULL;
    obj->_co_instrumentation_version = 0;
    obj->_co_monitoring = NULL;
    obj->_co_firsttraceable = 0;
    obj->co_extra = NULL;

    // optimization: cache the co_code_adaptive, which is a result
    // of PyCode_GetCode, and requires de-optimizing the code.
    // Here, we will pre-cache, without requiring another de-optimization.
    obj->_co_cached = PyMem_New(_PyCoCached, 1);
    std::memset(obj->_co_cached, 0, sizeof(_PyCoCached));
    obj->_co_cached->_co_code =  PyBytes_FromStringAndSize((const char *)code.co_code_adaptive.data(), code.co_code_adaptive.size());
}

static PyCodeObject *create_pycode_object(serdes::DeserializedCodeObject& code_obj) {
    auto code_size = static_cast<Py_ssize_t>(code_obj.co_code_adaptive.size())/2;
    // NOTE: We're not handling the necessary here when
    // Py_GIL_DISABLED is defined.
    PyCodeObject *code = PyObject_NewVar(PyCodeObject, &PyCode_Type, code_size*2);
    init_code(code, code_obj);

    return code;
}

// TODO
static void init_frame(PyFrameObject *frame, py_weakref<PyCodeObject> code, serdes::DeserializedPyFrame& frame_obj) {
    frame->f_back = NULL;
    frame->f_frame = NULL;
    frame->f_trace = NULL;
    frame->f_extra_locals = NULL;
    frame->f_locals_cache = NULL;

    frame->f_lineno = frame_obj.f_lineno;
    frame->f_trace_lines = frame_obj.f_trace_lines;
    frame->f_trace_opcodes = frame_obj.f_trace_opcodes;
    
    if(NULL != *frame_obj.f_trace) {
        frame->f_trace = Py_NewRef(frame_obj.f_trace.borrow());
    }
    if(NULL != *frame_obj.f_extra_locals) {
        frame->f_extra_locals = Py_NewRef(frame_obj.f_extra_locals.borrow());
    }
    if(NULL != *frame_obj.f_locals_cache) {
        frame->f_locals_cache = Py_NewRef(frame_obj.f_locals_cache.borrow());
    }
}

static PyFrameObject *create_pyframe_object(serdes::DeserializedPyFrame& frame_obj, py_weakref<PyCodeObject> code) {
    // TODO: Double-check that this is the correct size
    // TODO: What do we do when frame is owned by frame object?
    // can we just make it owned by thread by construction?
    // TODO: Should we just make it owned by the frame object?
    int slots = code->co_nlocalsplus + code->co_stacksize;
    PyFrameObject *frame = PyObject_GC_NewVar(PyFrameObject, &PyFrame_Type, slots);
    init_frame(frame, code, frame_obj);

    return frame;
}

static void init_pyinterpreterframe(sauerkraut::PyInterpreterFrame *interp_frame, 
                                   serdes::DeserializedPyInterpreterFrame& frame_obj,
                                   py_weakref<PyFrameObject> frame,
                                   py_weakref<PyCodeObject> code) {
    interp_frame->f_globals = NULL;
    interp_frame->f_builtins = NULL;
    interp_frame->f_locals = NULL;
    interp_frame->previous = NULL;

    interp_frame->f_executable.bits = (uintptr_t)Py_NewRef(code.borrow());
    if(frame_obj.f_executable.immutables_included()) {
#if SAUERKRAUT_PY314
        interp_frame->f_funcobj.bits = (uintptr_t)Py_NewRef(frame_obj.f_funcobj.value().borrow());
#else
        interp_frame->f_funcobj = Py_NewRef(frame_obj.f_funcobj.value().borrow());
#endif
        if(NULL != frame_obj.f_globals) {
            interp_frame->f_globals = frame_obj.f_globals.borrow();
        } else {
            interp_frame->f_globals = PyEval_GetFrameGlobals();
        }
    } else {
        auto invariants = sauerkraut_state->get_code_immutables(frame_obj);
        if(invariants) {
#if SAUERKRAUT_PY314
            interp_frame->f_funcobj.bits = (uintptr_t)Py_NewRef(std::get<0>(invariants.value()).borrow());
#else
            interp_frame->f_funcobj = Py_NewRef(std::get<0>(invariants.value()).borrow());
#endif
            interp_frame->f_globals = Py_NewRef(std::get<2>(invariants.value()).borrow());
        } else {
#if SAUERKRAUT_PY314
            interp_frame->f_funcobj.bits = 0;
#else
            interp_frame->f_funcobj = NULL;
#endif
            interp_frame->f_globals = NULL;
        }
    }

    if(NULL != *frame_obj.f_builtins) {
        interp_frame->f_builtins = frame_obj.f_builtins.borrow();
    } else {
        interp_frame->f_builtins = PyEval_GetFrameBuiltins();
    }
    
    // These are NOT fast locals, those come from localsplus
    if(NULL != *frame_obj.f_locals) {
        interp_frame->f_locals = Py_NewRef(frame_obj.f_locals.borrow());
    }

    // Here are the locals plus
    auto localsplus = frame_obj.localsplus;
    for(size_t i = 0; i < localsplus.size(); i++) {
        interp_frame->localsplus[i].bits = (intptr_t) Py_NewRef(localsplus[i].borrow());
    }
    auto stack = frame_obj.stack;
    _PyStackRef *frame_stack_base = utils::py::get_stack_base(interp_frame);
    for(size_t i = 0; i < stack.size(); i++) {
        frame_stack_base[i].bits = (intptr_t) Py_NewRef(stack[i].borrow());
    }
    for(size_t i = localsplus.size(); i < (size_t)code->co_nlocalsplus; i++) {
        interp_frame->localsplus[i].bits = 0;
    }
    interp_frame->instr_ptr = (sauerkraut::PyBitcodeInstruction*) 
        (utils::py::get_code_adaptive(code) + frame_obj.instr_offset/2);//utils::py::get_offset_for_skipping_call();
    interp_frame->return_offset = frame_obj.return_offset;
    #if SAUERKRAUT_PY314
    interp_frame->stackpointer = frame_stack_base + stack.size();
    #elif SAUERKRAUT_PY313
    interp_frame->stacktop = code->co_nlocalsplus + stack.size();
    #endif
    // TODO: Check what happens when we make the owner the frame object instead of the thread.
    // Might allow us to skip a copy when calling this frame
    interp_frame->owner = frame_obj.owner;
#if SAUERKRAUT_PY314
    interp_frame->visited = 0;
#endif
    // Weak ref to avoid circular reference with capsule
    interp_frame->frame_obj = *frame;
    frame->f_frame = interp_frame;
}

static sauerkraut::PyInterpreterFrame *create_pyinterpreterframe_object(serdes::DeserializedPyInterpreterFrame& frame_obj, 
                                                                      py_weakref<PyFrameObject> frame, 
                                                                      py_weakref<PyCodeObject> code,
                                                                      bool inplace=false) {
    sauerkraut::PyInterpreterFrame *interp_frame = NULL;
    if(inplace) {
        PyThreadState *tstate = PyThreadState_Get();
        interp_frame = utils::py::AllocateFrame(tstate, code->co_framesize);
    } else {
        interp_frame = utils::py::AllocateFrame(code->co_framesize);
    }
    init_pyinterpreterframe(interp_frame, frame_obj, frame, code);

    if(inplace) {
        prepare_frame_for_execution(frame);
    }
    return interp_frame;
}

static PyObject *_deserialize_frame(PyObject *bytes, bool inplace=false) {
    if(PyErr_Occurred()) {
        PyErr_Print();
        return NULL;
    }
    loads_functor loads(sauerkraut_state->pickle_loads, sauerkraut_state->dill_loads);
    dumps_functor dumps(sauerkraut_state->pickle_dumps, sauerkraut_state->dill_dumps);
    serdes::PyObjectSerdes po_serdes(loads, dumps);
    serdes::PyFrameSerdes frame_serdes{po_serdes};

    uint8_t *data = (uint8_t *)PyBytes_AsString(bytes);

    auto serframe = pyframe_buffer::GetPyFrame(data);
    auto deserframe = frame_serdes.deserialize(serframe);

    assert(deserframe.f_frame.owner == 0);
    pycode_strongref code;
    if(deserframe.f_frame.f_executable.immutables_included()) {
        code = pycode_strongref::steal(create_pycode_object(deserframe.f_frame.f_executable));
    } else {
        auto cached_invariants = sauerkraut_state->get_code_immutables(deserframe);
        if(cached_invariants) {
            code = make_strongref((PyCodeObject*)std::get<1>(cached_invariants.value()).borrow());
        } else {
            PyErr_SetString(PyExc_RuntimeError,
                "Cannot deserialize frame: immutables were excluded but cache lookup failed.");
            return NULL;
        }
    }

    PyFrameObject *frame = create_pyframe_object(deserframe, code.borrow());
    create_pyinterpreterframe_object(deserframe.f_frame, frame, code.borrow(), inplace);

    if (inplace) {
        return (PyObject*) frame;
    } else {
        // Wrap in capsule for proper cleanup of heap-allocated interpreter frame
        int nlocalsplus = code->co_nlocalsplus;
        int stack_depth = deserframe.f_frame.stack.size();
        utils::py::StackState stack_state;
        PyObject *capsule = frame_copy_capsule_create(frame, stack_state, true, nlocalsplus, stack_depth);
        Py_DECREF(frame);  // Drop our ref; capsule holds its own
        return capsule;
    }
}

static PyObject *run_frame_direct(py_weakref<PyFrameObject> frame) {
    PyThreadState *tstate = PyThreadState_Get();
    pycode_strongref code = pycode_strongref::steal(PyFrame_GetCode(*frame));
    _PyInterpreterFrame *heap_frame = frame->f_frame;

    // Allocate a new frame on the eval stack
    _PyInterpreterFrame *stack_frame = utils::py::ThreadState_PushFrame(tstate, code->co_framesize);
    if (stack_frame == NULL) {
        PySys_WriteStderr("<Sauerkraut>: failed to create frame on the framestack\n");
        return NULL;
    }

    // Copy all fields from the heap frame to the stack frame
    stack_frame->f_executable.bits = heap_frame->f_executable.bits;
    stack_frame->previous = NULL;  // No previous frame - we're the root
#if SAUERKRAUT_PY314
    stack_frame->f_funcobj.bits = heap_frame->f_funcobj.bits;
#else
    stack_frame->f_funcobj = heap_frame->f_funcobj;
#endif
    stack_frame->f_globals = heap_frame->f_globals;
    stack_frame->f_builtins = heap_frame->f_builtins;
    stack_frame->f_locals = heap_frame->f_locals;
    stack_frame->frame_obj = *frame;
    stack_frame->instr_ptr = heap_frame->instr_ptr;
    stack_frame->return_offset = heap_frame->return_offset;
    stack_frame->owner = heap_frame->owner;
#if SAUERKRAUT_PY314
    stack_frame->visited = 0;
#endif

    // Copy localsplus (shallow copy - no refcount changes needed as we're moving refs)
    int nlocalsplus = code->co_nlocalsplus;
    memcpy(stack_frame->localsplus, heap_frame->localsplus, nlocalsplus * sizeof(_PyStackRef));

    // Copy stack
    int stack_depth = utils::py::get_current_stack_depth(heap_frame);
    _PyStackRef *heap_stack = utils::py::get_stack_base(heap_frame);
    _PyStackRef *stack_stack = utils::py::get_stack_base(stack_frame);
    memcpy(stack_stack, heap_stack, stack_depth * sizeof(_PyStackRef));

    // Set stack pointer
#if SAUERKRAUT_PY314
    stack_frame->stackpointer = stack_frame->localsplus + nlocalsplus + stack_depth;
#elif SAUERKRAUT_PY313
    stack_frame->stacktop = nlocalsplus + stack_depth;
#endif

    // Update the frame object to point to the new stack frame
    frame->f_frame = stack_frame;

    // Skip past the CALL instruction
    prepare_frame_for_execution(frame);

    PyObject *res = run_and_cleanup_frame(*frame);
    return res;
}


static PyObject *deserialize_frame(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *bytes;
    int run = 0;  // Default to False
    PyObject *replace_locals = NULL;
    static char *kwlist[] = {"frame", "replace_locals", "run", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Op", kwlist, &bytes, &replace_locals, &run)) {
        return NULL;
    }

    PyObject *deser_result = _deserialize_frame(bytes, run);
    if (deser_result == NULL) {
        return NULL;
    }

    if (run) {
        PyFrameObject *frame = (PyFrameObject*)deser_result;
        if (!handle_replace_locals(replace_locals, frame)) {
            Py_DECREF(frame);
            return NULL;
        }
        PyObject *result = run_and_cleanup_frame(frame);
        Py_DECREF(frame);
        return result;
    } else {
        // replace_locals should be applied via run_frame
        return deser_result;
    }
}

static PyObject *run_frame(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *capsule_obj = NULL;
    PyObject *replace_locals = NULL;
    static char *kwlist[] = {"frame", "replace_locals", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwlist, &capsule_obj, &replace_locals)) {
        return NULL;
    }

    if (!PyCapsule_CheckExact(capsule_obj)) {
        PyErr_SetString(PyExc_TypeError, "frame must be a capsule from copy_current_frame, copy_frame, or deserialize_frame");
        return NULL;
    }

    frame_copy_capsule *capsule = (struct frame_copy_capsule *)PyCapsule_GetPointer(capsule_obj, copy_frame_capsule_name);
    if (capsule == NULL) {
        return NULL;
    }

    PyFrameObject *frame = capsule->frame;
    py_weakref<PyFrameObject> frame_ref = frame;

    if (!handle_replace_locals(replace_locals, frame_ref)) {
        return NULL;
    }

    // Save before run_frame_direct replaces f_frame with stack-allocated frame
    _PyInterpreterFrame *heap_interp_frame = frame->f_frame;

    PyObject *result = run_frame_direct(frame_ref);

    // Refs were shallow-copied to stack frame, so just free heap memory
    if (capsule->owns_interpreter_frame && heap_interp_frame) {
        free(heap_interp_frame);
        capsule->owns_interpreter_frame = false;
    }

    return result;
}

static PyObject *serialize_frame(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *capsule;
    PyObject *sizehint_obj = NULL;
    Py_ssize_t sizehint_val = 0; 

    static char *kwlist[] = {"frame", "sizehint", NULL};
    // Parse capsule and sizehint_obj (as PyObject*)
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwlist, &capsule, &sizehint_obj)) {
        return NULL;
    }

    if (!parse_sizehint(sizehint_obj, sizehint_val)) {
        return NULL;
    }

    serdes::SerializationArgs ser_args; 
    if (sizehint_val > 0) {
        ser_args.set_sizehint(sizehint_val);
    } else if (sizehint_obj != NULL) {
         PyErr_SetString(PyExc_ValueError, "sizehint must be a positive integer");
         return NULL;
    }
    return _serialize_frame_from_capsule(capsule, ser_args);
}

static PyObject *copy_frame_from_greenlet(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *greenlet = NULL;
    SerializationOptions options;

    static char *kwlist[] = {"greenlet", "exclude_locals", "sizehint", "serialize", "exclude_dead_locals", "exclude_immutables", NULL};
    int serialize = 0;
    PyObject* sizehint_obj = NULL;
    PyObject* exclude_locals = NULL;
    int exclude_dead_locals = 1;
    int exclude_immutables = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OOppp", kwlist,
                                    &greenlet, &exclude_locals,
                                    &sizehint_obj, &serialize, &exclude_dead_locals, &exclude_immutables)) {
        return NULL;
    }
    options.serialize = (serialize != 0);
    options.exclude_dead_locals = (exclude_dead_locals != 0);
    options.exclude_immutables = (exclude_immutables != 0);
    options.exclude_locals = pyobject_strongref(exclude_locals);
    if (!parse_sizehint(sizehint_obj, options.sizehint)) {
        return NULL;
    }

    assert(greenlet::is_greenlet(greenlet));
    auto frame = py_strongref<PyFrameObject>::steal(greenlet::getframe(greenlet));
    if (!frame) {
        PyErr_SetString(PyExc_ValueError, "Greenlet has no active frame");
        return NULL;
    }
    py_weakref<PyFrameObject> frame_ref(frame.borrow());

    if (options.serialize) {
        return _copy_serialize_frame_object(frame_ref, options);
    }
    return _copy_frame_object(frame_ref, options);
}

static PyObject *_resume_greenlet(py_weakref<PyFrameObject> frame) {
    return run_frame_direct(frame);
}

static PyObject *resume_greenlet(PyObject *self, PyObject *args) {
    PyObject *frame;
    if (!PyArg_ParseTuple(args, "O", &frame)) {
        return NULL;
    }
    py_weakref<PyFrameObject> frame_ref = (PyFrameObject*)frame;
    return _resume_greenlet(frame_ref);
}

static PyMethodDef MyMethods[] = {
    {"serialize_frame", (PyCFunction) serialize_frame, METH_VARARGS | METH_KEYWORDS, "Serialize the frame"},
    {"copy_frame", (PyCFunction) copy_frame, METH_VARARGS | METH_KEYWORDS, "Copy a given frame"},
    {"copy_current_frame", (PyCFunction) copy_current_frame, METH_VARARGS | METH_KEYWORDS, "Copy the current frame"},
    {"deserialize_frame", (PyCFunction) deserialize_frame, METH_VARARGS | METH_KEYWORDS, "Deserialize the frame"},
    {"run_frame", (PyCFunction) run_frame, METH_VARARGS | METH_KEYWORDS, "Run the frame"},
    {"resume_greenlet", (PyCFunction) resume_greenlet, METH_VARARGS, "Resume the frame from a greenlet"},
    {"copy_frame_from_greenlet", (PyCFunction) copy_frame_from_greenlet, METH_VARARGS | METH_KEYWORDS, "Copy the frame from a greenlet"},
    {NULL, NULL, 0, NULL}
};

static void sauerkraut_free(void *m) {
    if (sauerkraut_state) {
        delete sauerkraut_state;
        sauerkraut_state = nullptr;
    }
}

static struct PyModuleDef sauerkraut_mod = {
    PyModuleDef_HEAD_INIT,
    "sauerkraut",
    "A module that defines the 'abcd' function",
    -1,
    MyMethods,
    NULL, // slot definitions
    NULL, // traverse function for GC
    NULL, // clear function for GC
    sauerkraut_free // free function for GC
};

PyMODINIT_FUNC PyInit__sauerkraut(void) {
    sauerkraut_state = new sauerkraut_modulestate();
    if (!sauerkraut_state->init()) {
        delete sauerkraut_state;
        sauerkraut_state = NULL;
        return NULL;
    }
    greenlet::init_greenlet();

    return PyModule_Create(&sauerkraut_mod);
}

}
