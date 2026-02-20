#ifndef UTILS_HH_INCLUDED
#define UTILS_HH_INCLUDED
#include "sauerkraut_cpython_compat.h"
#include <opcode_ids.h>
#include <string>
#include <opcode_ids.h>
#include <map>
#include <iterator>
#include <stdexcept>

#include "py_structs.h"
#include "pyref.h"

namespace {

static _PyStackChunk*
allocate_chunk(int size_in_bytes, _PyStackChunk* previous)
{
    assert(size_in_bytes % sizeof(PyObject **) == 0);
    _PyStackChunk *res = (_PyStackChunk*) PyObject_Malloc(size_in_bytes);
    if (res == NULL) {
        return NULL;
    }
    res->previous = previous;
    res->size = size_in_bytes;
    res->top = 0;
    return res;
}

static PyObject **
    push_chunk(PyThreadState *tstate, int size)
    {
        int allocate_size = pycompat::DATA_STACK_CHUNK_SIZE;
        while (allocate_size < (int)sizeof(PyObject*)*(size + pycompat::CHUNK_ALLOC_MINIMUM_OVERHEAD)) {
            allocate_size *= 2;
        }
        _PyStackChunk *new_chunk = allocate_chunk(allocate_size, tstate->datastack_chunk);
        if (new_chunk == NULL) {
            return NULL;
        }
        if (tstate->datastack_chunk) {
            tstate->datastack_chunk->top = tstate->datastack_top -
                                           &tstate->datastack_chunk->data[0];
        }
        tstate->datastack_chunk = new_chunk;
        tstate->datastack_limit = (PyObject **)(((char *)new_chunk) + allocate_size);
        // When new is the "root" chunk (i.e. new->previous == NULL), we can keep
        // _PyThreadState_PopFrame from freeing it later by "skipping" over the
        // first element:
        PyObject **res = &new_chunk->data[new_chunk->previous == NULL];
        tstate->datastack_top = res + size;
        return res;
    }

}

namespace utils {
    namespace py
    {

        void print_object(PyObject *obj);
        class PyIterator
        {
        public:
            using iterator_category = std::input_iterator_tag;
            using value_type = PyObject *;
            using difference_type = std::ptrdiff_t;
            using pointer = PyObject **;
            using reference = PyObject *&;

            PyIterator(PyObject *iterable = nullptr, bool end = false) : current_item(nullptr)
            {
                if (end || !iterable)
                {
                    iterator = nullptr;
                    return;
                }

                iterator = PyObject_GetIter(iterable);
                if (!iterator)
                {
                    PyErr_Print();
                    throw std::runtime_error("Failed to get iterator from Python object");
                }

                ++(*this);
            }

            ~PyIterator()
            {
                Py_XDECREF(iterator);
                Py_XDECREF(current_item);
            }

            PyIterator(const PyIterator &other) : iterator(other.iterator), current_item(other.current_item)
            {
                Py_XINCREF(iterator);
                Py_XINCREF(current_item);
            }

            PyIterator(PyIterator &&other) noexcept : iterator(other.iterator), current_item(other.current_item)
            {
                other.iterator = nullptr;
                other.current_item = nullptr;
            }

            PyIterator &operator++()
            {
                Py_XDECREF(current_item);
                current_item = iterator ? PyIter_Next(iterator) : nullptr;

                // If we hit the end or an error occurred, clean up iterator
                if (!current_item && iterator)
                {
                    if (PyErr_Occurred())
                    {
                        PyErr_Print();
                    }
                    Py_DECREF(iterator);
                    iterator = nullptr;
                }

                return *this;
            }

            PyIterator operator++(int)
            {
                PyIterator tmp(*this);
                ++(*this);
                return tmp;
            }

            pyobject_weakref operator*() const
            {
                if (!current_item)
                {
                    throw std::runtime_error("Dereferencing end iterator");
                }
                return make_weakref(current_item);
            }

            bool operator==(const PyIterator &other) const
            {
                return (current_item == other.current_item);
            }

            bool operator!=(const PyIterator &other) const
            {
                return !(*this == other);
            }

        private:
            PyObject *iterator;
            PyObject *current_item;
        };

        class PyIterable
        {
        public:
            PyIterable(PyObject *obj) : iterable(obj)
            {
                if (!obj)
                {
                    throw std::invalid_argument("Null PyObject provided");
                }
                Py_INCREF(iterable);
            }

            ~PyIterable()
            {
                Py_DECREF(iterable);
            }

            PyIterator begin() const
            {
                return PyIterator(iterable);
            }

            PyIterator end() const
            {
                return PyIterator(nullptr, true);
            }

        private:
            PyObject *iterable;
        };

    }
}
namespace utils {
    namespace py {
       bool check_dict(PyObject *obj) {
           return PyDict_Check(obj);
       }

       struct StackRefObject {
           PyObject *obj;
           bool owned;
       };

       class ScopedStackRefObject {
           StackRefObject ref;
       public:
           ScopedStackRefObject(_PyStackRef stackref);
           ~ScopedStackRefObject() {
               if (ref.owned && ref.obj) {
                   Py_DECREF(ref.obj);
               }
           }
           ScopedStackRefObject(const ScopedStackRefObject&) = delete;
           ScopedStackRefObject& operator=(const ScopedStackRefObject&) = delete;
           PyObject* get() const { return ref.obj; }
           explicit operator bool() const { return ref.obj != nullptr; }
       };

#if SAUERKRAUT_PY314
       constexpr uintptr_t STACKREF_TAG_BITS = 0x3;
       constexpr uintptr_t STACKREF_TAG_INT = 0x3;
       constexpr uintptr_t STACKREF_TAG_REFCNT = 0x1;
       constexpr int STACKREF_TAGGED_SHIFT = 2;

       inline _PyStackRef stackref_null() {
           return (_PyStackRef){ .bits = STACKREF_TAG_REFCNT };
       }

       inline bool stackref_is_null(_PyStackRef ref) {
           // A zeroed stackref is invalid in 3.14 but may appear in older snapshots.
           return ref.bits == 0 || ref.bits == STACKREF_TAG_REFCNT;
       }

       inline bool stackref_is_tagged_int(_PyStackRef ref) {
           return (ref.bits & STACKREF_TAG_BITS) == STACKREF_TAG_INT;
       }

       inline bool stackref_is_deferred(_PyStackRef ref) {
           return (ref.bits & STACKREF_TAG_BITS) == STACKREF_TAG_REFCNT;
       }

       inline bool stackref_refcount_on_object(_PyStackRef ref) {
           return !stackref_is_null(ref) && !stackref_is_tagged_int(ref) && !stackref_is_deferred(ref);
       }

       inline intptr_t stackref_untag_int(_PyStackRef ref) {
           return ((intptr_t)ref.bits) >> STACKREF_TAGGED_SHIFT;
       }

       inline PyObject *stackref_as_pyobject(_PyStackRef ref) {
           if (stackref_is_null(ref) || stackref_is_tagged_int(ref)) {
               return NULL;
           }
           uintptr_t bits = ref.bits;
           if (stackref_is_deferred(ref)) {
               bits &= ~STACKREF_TAG_BITS;
           }
           return (PyObject *)bits;
       }

       inline _PyStackRef stackref_from_pyobject_steal(PyObject *obj) {
           if (obj == NULL) {
               return stackref_null();
           }
           if (_Py_IsImmortal(obj)) {
               return (_PyStackRef){ .bits = ((uintptr_t)obj) | STACKREF_TAG_REFCNT };
           }
           return (_PyStackRef){ .bits = (uintptr_t)obj };
       }

       inline _PyStackRef stackref_from_pyobject_new(PyObject *obj) {
           if (obj == NULL) {
               return stackref_null();
           }
           if (_Py_IsImmortal(obj)) {
               return (_PyStackRef){ .bits = ((uintptr_t)obj) | STACKREF_TAG_REFCNT };
           }
           return (_PyStackRef){ .bits = (uintptr_t)Py_NewRef(obj) };
       }

       inline StackRefObject stackref_to_object_for_serialization(_PyStackRef ref) {
           if (stackref_is_null(ref)) {
               return {NULL, false};
           }
           if (stackref_is_tagged_int(ref)) {
               auto value = (Py_ssize_t) stackref_untag_int(ref);
               return {PyLong_FromSsize_t(value), true};
           }
           return {stackref_as_pyobject(ref), false};
       }

       inline void stackref_decref(_PyStackRef ref) {
           if (!stackref_refcount_on_object(ref)) {
               return;
           }
           Py_DECREF(stackref_as_pyobject(ref));
       }

       inline PyObject *get_funcobj(sauerkraut::PyInterpreterFrame *frame) {
           return stackref_as_pyobject(frame->f_funcobj);
       }

       inline void set_funcobj(sauerkraut::PyInterpreterFrame *frame, PyObject *obj) {
           frame->f_funcobj = stackref_from_pyobject_steal(obj);
       }

       inline void set_stack_position(sauerkraut::PyInterpreterFrame *frame, int nlocalsplus, int stack_depth) {
           frame->stackpointer = frame->localsplus + nlocalsplus + stack_depth;
       }

       inline void init_frame_visited(sauerkraut::PyInterpreterFrame *frame) {
           frame->visited = 0;
       }
#else
       inline _PyStackRef stackref_null() {
           return (_PyStackRef){ .bits = 0 };
       }

       inline bool stackref_is_null(_PyStackRef ref) {
           return ref.bits == 0;
       }

       inline bool stackref_is_tagged_int(_PyStackRef) {
           return false;
       }

       inline PyObject *stackref_as_pyobject(_PyStackRef ref) {
           return (PyObject *) ref.bits;
       }

       inline _PyStackRef stackref_from_pyobject_steal(PyObject *obj) {
           return (_PyStackRef){ .bits = (uintptr_t)obj };
       }

       inline _PyStackRef stackref_from_pyobject_new(PyObject *obj) {
           if (obj == NULL) {
               return stackref_null();
           }
           return (_PyStackRef){ .bits = (uintptr_t)Py_NewRef(obj) };
       }

       inline StackRefObject stackref_to_object_for_serialization(_PyStackRef ref) {
           if (ref.bits == 0) {
               return {NULL, false};
           }
           return {(PyObject *) ref.bits, false};
       }

       inline void stackref_decref(_PyStackRef ref) {
           Py_XDECREF(stackref_as_pyobject(ref));
       }

       inline PyObject *get_funcobj(sauerkraut::PyInterpreterFrame *frame) {
           return frame->f_funcobj;
       }

       inline void set_funcobj(sauerkraut::PyInterpreterFrame *frame, PyObject *obj) {
           frame->f_funcobj = obj;
       }

       inline void set_stack_position(sauerkraut::PyInterpreterFrame *frame, int nlocalsplus, int stack_depth) {
           frame->stacktop = nlocalsplus + stack_depth;
       }

       inline void init_frame_visited(sauerkraut::PyInterpreterFrame *) {
           // No-op for Python 3.13
       }
#endif

       inline ScopedStackRefObject::ScopedStackRefObject(_PyStackRef stackref)
           : ref(stackref_to_object_for_serialization(stackref)) {}

       int get_code_stacksize(PyCodeObject *code) {
            return code->co_stacksize;
       }

       int get_code_nlocalsplus(PyCodeObject *code) {
           return code->co_nlocalsplus;
       }

       int get_code_nlocals(PyCodeObject *code) {
           return code->co_nlocals;
       }

       bool set_update(pyobject_weakref set, pyobject_weakref other) {
            for(auto item : PyIterable(*other)) {
                PySet_Add(*set, *item);
            }
            return true;
       }

     sauerkraut::PyBitcodeInstruction *get_code_adaptive(py_weakref<PyCodeObject> code) {
           return (sauerkraut::PyBitcodeInstruction*) code->co_code_adaptive;
       }

       int get_iframe_localsplus_size(sauerkraut::PyInterpreterFrame *iframe) {
           PyCodeObject *code = (PyCodeObject*) stackref_as_pyobject(iframe->f_executable);
           if(NULL == code) {
               return 0;
           }
           return get_code_nlocalsplus(code) + code->co_stacksize;
       }

        enum class Units { Bytes, Instructions };

        template <Units Unit>
        Py_ssize_t get_instr_offset(py_weakref<struct _frame> frame) {
            pycode_strongref code = pycode_strongref::steal(PyFrame_GetCode(*frame));
            Py_ssize_t first_instr_addr = (Py_ssize_t) code->co_code_adaptive;
            Py_ssize_t current_instr_addr = (Py_ssize_t) frame->f_frame->instr_ptr;
            Py_ssize_t offset = current_instr_addr - first_instr_addr;

            if constexpr (Unit == Units::Bytes) {
                return offset;
            } else if constexpr (Unit == Units::Instructions) {
                return offset / 2; // Assuming each instruction is 2 bytes
            }
        }
        char get_current_opcode(py_weakref<struct _frame> frame) {
            return frame->f_frame->instr_ptr->opcode;
        }
        char get_current_opcode(pycode_weakref code, int offset) {
            pyobject_strongref code_bytes = pyobject_strongref::steal(PyCode_GetCode(code.borrow()));
            char *bitcode = PyBytes_AsString(code_bytes.borrow());
            return ((sauerkraut::PyBitcodeInstruction*) (bitcode + offset))->opcode;
        }

        template <Units Unit>
        Py_ssize_t get_instr_offset(py_weakref<sauerkraut::PyInterpreterFrame> iframe) {
            PyCodeObject *code = (PyCodeObject*) stackref_as_pyobject(iframe->f_executable);
            Py_ssize_t first_instr_addr = (Py_ssize_t) code->co_code_adaptive;
            Py_ssize_t current_instr_addr = (Py_ssize_t) iframe->instr_ptr;
            Py_ssize_t offset = current_instr_addr - first_instr_addr;

            if constexpr (Unit == Units::Bytes) {
                return offset;
            } else if constexpr (Unit == Units::Instructions) {
                return offset / 2; // Assuming each instruction is 2 bytes
            }
        }

        bool ThreadState_HasStackSpace(py_weakref<PyThreadState> state, int size) {
            return state->datastack_top != NULL && size < state->datastack_limit - state->datastack_top;
        }

        _PyInterpreterFrame *ThreadState_PushFrame(py_weakref<PyThreadState> tstate, size_t size) {
            if(ThreadState_HasStackSpace(tstate, size)) {
                _PyInterpreterFrame *top = (_PyInterpreterFrame *)tstate->datastack_top;
                tstate->datastack_top += size;
                return top;
            }
            return (_PyInterpreterFrame*) push_chunk(*tstate, size);
        }

        _PyInterpreterFrame *AllocateFrame(size_t size) {
            return (_PyInterpreterFrame*) malloc(size * sizeof(PyObject*));
        }
        _PyInterpreterFrame *AllocateFrame(py_weakref<PyThreadState> tstate, size_t size) {
            return (_PyInterpreterFrame*) ThreadState_PushFrame(*tstate, size);
        }

        // TODO: This should use units
        Py_ssize_t get_offset_for_skipping_call(char opcode) {
            // return 2 * sizeof(_CodeUnit);
            #if SAUERKRAUT_PY314
            return 5 * sizeof(_CodeUnit);
            #elif SAUERKRAUT_PY313
            if(opcode == CALL) {
            return 5 * sizeof(_CodeUnit);
            } else if(opcode == CALL_KW) {
            return 2 * sizeof(_CodeUnit);
            } else {
                PySys_WriteStderr("(get_offset_for_skipping_call) Unknown opcode: %d\n", opcode);
                return 0;
            }
            #endif
        }

        void print_object(PyObject *obj) {
            if (obj == NULL) {
                printf("Error: NULL object passed\n");
                return;
            }
        
            PyObject *str = PyObject_Repr(obj);
            if (str == NULL) {
                PyErr_Print();
                return;
            }
        
            const char *c_str = PyUnicode_AsUTF8(str);
            if (c_str == NULL) {
                Py_DECREF(str);
                PyErr_Print();
                return;
            }
        
            printf("Object contents: %s\n", c_str);
            Py_DECREF(str);
        }
        
        void print_object_type_name(PyObject *obj) {
            if (obj == NULL) {
                printf("Error: NULL object\n");
                return;
            }
        
            PyObject *type = PyObject_Type(obj);
            if (type == NULL) {
                printf("Error: Could not get object type\n");
                PyErr_Print();
                return;
            }
        
            PyObject *type_name = PyObject_GetAttrString(type, "__name__");
            if (type_name == NULL) {
                printf("Error: Could not get type name\n");
                PyErr_Print();
                Py_DECREF(type);
                return;
            }
        
            const char *name = PyUnicode_AsUTF8(type_name);
            if (name == NULL) {
                printf("Error: Could not convert type name to string\n");
                PyErr_Print();
            } else {
                printf("Object type: %s\n", name);
            }
        
            Py_DECREF(type_name);
            Py_DECREF(type);
        }

        Py_ssize_t get_code_size(Py_ssize_t n_instructions) {
            return n_instructions * sizeof(_CodeUnit);
        }

        Py_ssize_t skip_current_call_instruction(py_weakref<PyFrameObject> frame) {
            pycode_strongref code = pycode_strongref::steal(PyFrame_GetCode(*frame));
            Py_ssize_t base_offset = get_instr_offset<Units::Bytes>(*frame);
            Py_ssize_t offset = get_instr_offset<Units::Bytes>(*frame) + get_offset_for_skipping_call(get_current_opcode(code, base_offset));
            frame->f_frame->instr_ptr = (_CodeUnit*) (code->co_code_adaptive + offset);
            return offset;
        }

        Py_ssize_t get_current_stack_depth(sauerkraut::PyInterpreterFrame *iframe) {
            // WARNING: The stack pointer is most often NULL when 
            // we stop a running Python function. Unless the function is stopped on a
            // yield instruction (which will not happen in this library, as we are stopped
            // on a CALL, then the stack pointer will be NULL.
            // This is NOT the method
            // you should use when trying to get the stack depth of a running frame.
            // Use get_stack_depth(PyObject *) instead.
            PyCodeObject *code = (PyCodeObject*) stackref_as_pyobject(iframe->f_executable);
            auto n_localsplus = get_code_nlocalsplus(code);
            #if SAUERKRAUT_PY314
            assert(NULL != iframe->stackpointer);
            assert(iframe->stackpointer >= iframe->localsplus);
            return (Py_ssize_t) (iframe->stackpointer - ( iframe->localsplus + n_localsplus));
            #elif SAUERKRAUT_PY313
            return (Py_ssize_t) iframe->stacktop - n_localsplus;
            #endif
        }

        Py_ssize_t get_stack_depth(PyObject *frame) {
            // we must analyze the code to determine the current stack depth.
            // iframe->stackpointer is rarely written to (e.g., with generators).
            // Therefore, we have to determine the stack depth by looking backwards
            // at the compiled bitcode.
            // TODO: This ignores the following constructs which affect the stack depth:
            // 1. Try/except blocks; 2. Calls.
            // We consider only loops.
            // sauerkraut::PyCodeObject *code = (sauerkraut::PyCodeObject*) iframe->f_executable.bits;
            sauerkraut::PyFrame *py_frame = (sauerkraut::PyFrame*) frame;
            pycode_strongref code = pycode_strongref::steal(PyFrame_GetCode((PyFrameObject*)py_frame));

            // divide by 2 to convert from bytes to instructions
            Py_ssize_t offset = get_instr_offset<Units::Instructions>(py_frame->f_frame);
            pyobject_strongref code_bytes = pyobject_strongref::steal(PyCode_GetCode(code.borrow()));
            char *bitcode = PyBytes_AsString(code_bytes.borrow());
            sauerkraut::PyBitcodeInstruction *first_instr = (sauerkraut::PyBitcodeInstruction*) bitcode;
            sauerkraut::PyBitcodeInstruction *instr = (sauerkraut::PyBitcodeInstruction*) first_instr + offset;

            Py_ssize_t num_for = 0;
            Py_ssize_t num_end = 0;
            while((intptr_t)instr >= (intptr_t)first_instr) {
                switch(instr->opcode) {
                    case FOR_ITER:
                        num_for++;
                        break;
                    case END_FOR:
                        num_end++;
                        break;
                    default:
                        break;
                }
                instr--;
            }
            return num_for - num_end;
        }

        _PyStackRef *get_stack_base(sauerkraut::PyInterpreterFrame *f) {
            return f->localsplus + ((PyCodeObject*)stackref_as_pyobject(f->f_executable))->co_nlocalsplus;
        }

        template<typename T>
        class _StackState {
            std::vector<T> state;

            public:
            _StackState(size_t max_size) {
                state.reserve(max_size);
            }
            _StackState() {
                state.reserve(10);
            }
            void add(T value) {
                state.push_back(value);
            }
            T& get(size_t index) {
                return state[index];
            }
            size_t size() {
                return state.size();
            }
        };

        using StackState = _StackState<PyObject*>;
        static StackState _get_stack_state_locals(struct _frame *frame, PyCodeObject *code, int stack_depth) {
            StackState state(stack_depth);
            _PyInterpreterFrame *iframe = (_PyInterpreterFrame*) frame->f_frame;

            #ifdef DEBUG
            _PyStackRef *frame_locals = iframe->localsplus;
            std::map<intptr_t, int> locals;
            for(int i = 0; i < code->co_nlocalsplus; i++) {
                PyObject *local = ((PyObject*) frame_locals[i].bits);
                if(NULL == local) {
                    continue;
                }
                locals[(intptr_t) local] = i;
            }
            #endif

            _PyStackRef *stack_pointer = iframe->localsplus + code->co_nlocalsplus;
            for(int i = 0; i < stack_depth; i++) {
                _PyStackRef stack_ref = stack_pointer[i];
                PyObject *stack_obj = stackref_as_pyobject(stack_ref);
                if (stack_obj == NULL && stackref_is_tagged_int(stack_ref)) {
                    stack_obj = Py_None;
                }
                #ifdef DEBUG
                assert(NULL != stack_obj);
                if(locals.find((intptr_t) stack_obj) != locals.end()) {
                    PySys_WriteStderr("ISSUE: We found a local on the stack. Currently, we assume that's not possible.\n");
                }
                #endif
                state.add(stack_obj);
            }


            return state;
        }

        StackState get_stack_state(pyobject_weakref frame) {
            py_weakref<struct _frame> frame_obj{(struct _frame*) *frame};
            _PyInterpreterFrame *iframe = (_PyInterpreterFrame*) frame_obj->f_frame;
            PyCodeObject *code = (PyCodeObject*) stackref_as_pyobject(iframe->f_executable);
            auto stack_depth = get_stack_depth((PyObject*)*frame);
            auto state = _get_stack_state_locals(*frame_obj, code, stack_depth);
            return state;
        }

        using LocalNameMap = std::map<std::string, int>;
        using LocalExclusionBitmask = std::vector<bool>;

        LocalNameMap get_local_name_map(py_weakref<PyFrameObject> frame) {
            _PyInterpreterFrame *iframe = (_PyInterpreterFrame*) frame->f_frame;
            pycode_strongref code = pycode_strongref::steal(PyFrame_GetCode((PyFrameObject*)*frame));
            // this is a tuple of the names of the localsplus
            PyObject *locals_plus_names = code->co_localsplusnames;
            LocalNameMap local_idx_map;

            for(int i = 0; i < code->co_nlocalsplus; i++) {
                PyObject *local = stackref_as_pyobject(iframe->localsplus[i]);
                std::string name = PyUnicode_AsUTF8(PyTuple_GetItem(locals_plus_names, i));
                local_idx_map[name] = i;
            }
            return local_idx_map;
        }

        LocalExclusionBitmask exclude_locals(py_weakref<PyFrameObject> frame, PyObject *exclude_locals) {
            _PyInterpreterFrame *iframe = (_PyInterpreterFrame*) frame->f_frame;
            pycode_strongref code = pycode_strongref::steal(PyFrame_GetCode(*frame));
            LocalExclusionBitmask bitmask(code->co_nlocalsplus);
            LocalNameMap local_idx_map = get_local_name_map(frame);

            PyIterable locals_to_exclude(exclude_locals);
            for(auto local : locals_to_exclude) {
                if(PyUnicode_Check(*local)) {
                    std::string local_name = PyUnicode_AsUTF8(*local);
                    auto it = local_idx_map.find(local_name);
                    if(it != local_idx_map.end()) {
                        bitmask[it->second] = true;
                    }
                } else if(PyLong_Check(*local)) {
                    int local_idx = PyLong_AsLong(*local);
                    if(local_idx >= 0 && local_idx < code->co_nlocalsplus) {
                        bitmask[local_idx] = true;
                    } else {
                        PyErr_SetString(PyExc_IndexError, "exclude_locals index out of range");
                        return LocalExclusionBitmask();
                    }
                } else {
                    PyErr_SetString(PyExc_TypeError, "exclude_locals must be an iterable of strings or integers");
                    return LocalExclusionBitmask();
                }
            }
            return bitmask;
        }

        void replace_locals(py_weakref<PyFrameObject> frame, PyObject *replace_locals) {
            _PyInterpreterFrame *iframe = (_PyInterpreterFrame*) frame->f_frame;
            LocalNameMap local_idx_map = get_local_name_map(frame);
            pycode_strongref code = pycode_strongref::steal(PyFrame_GetCode((PyFrameObject*)*frame));

            if(!PyDict_Check(replace_locals)) {
                PyErr_SetString(PyExc_TypeError, "replace_locals must be a dictionary");
                return;
            }

            PyObject *key, *value;
            Py_ssize_t pos = 0;
            
            while (PyDict_Next(replace_locals, &pos, &key, &value)) {
                int local_index = -1;
                
                if (PyUnicode_Check(key)) {
                    // Handle string keys (variable names)
                    std::string name = PyUnicode_AsUTF8(key);
                    auto it = local_idx_map.find(name);
                    if (it != local_idx_map.end()) {
                        local_index = it->second;
                    }
                } else if (PyLong_Check(key)) {
                    // Handle integer keys (direct indices)
                    local_index = PyLong_AsLong(key);
                    if (local_index < 0 || local_index >= code->co_nlocalsplus) {
                        PyErr_SetString(PyExc_IndexError, "replace_locals index out of range");
                        return;
                    }
                } else {
                    PyErr_SetString(PyExc_TypeError, "replace_locals key must be a string or integer");
                    return;
                }
                
                if (local_index >= 0) {
                    _PyStackRef old_ref = iframe->localsplus[local_index];

                    iframe->localsplus[local_index] = stackref_from_pyobject_new(value);

                    // Decrement reference count of old value
                    stackref_decref(old_ref);
                }
            }
        }
    }
}


#endif // UTILS_HH_INCLUDED
