#ifndef SERDES_HH_INCLUDED
#define SERDES_HH_INCLUDED
#include "sauerkraut_cpython_compat.h"
#include <iostream>
#include <optional>
#include <string>
#include "flatbuffers/flatbuffers.h"
#include "py_object_generated.h"
#include "py_var_object_head_generated.h"
#include "py_object_head_generated.h"
#include "py_frame_generated.h"
#include "offsets.h"
#include "pyref.h"
#include "py_structs.h"
#include "utils.h"
#include <optional>

namespace serdes {
    constexpr int SERIALIZATION_SIZEHINT_DEFAULT = 1024;
    class SerializationArgs {
        public:
        std::optional<utils::py::LocalExclusionBitmask> exclude_locals;
        bool exclude_immutables = false;
        bool capture_module_source = false;
        size_t sizehint;
        std::optional<std::string> module_name;
        std::optional<std::string> module_package;
        std::optional<std::string> module_filename;
        std::optional<std::vector<uint8_t>> module_source;

        SerializationArgs(std::optional<utils::py::LocalExclusionBitmask> exclude_locals, bool exclude_immutables, bool capture_module_source, size_t sizehint) :
            exclude_locals(exclude_locals), exclude_immutables(exclude_immutables), capture_module_source(capture_module_source), sizehint(sizehint) {}
        SerializationArgs() : exclude_locals(std::nullopt), exclude_immutables(false), capture_module_source(false), sizehint(SERIALIZATION_SIZEHINT_DEFAULT) {}
        SerializationArgs(size_t sizehint) : exclude_locals(std::nullopt), exclude_immutables(false), capture_module_source(false), sizehint(sizehint) {}

        void set_exclude_locals(std::optional<utils::py::LocalExclusionBitmask> exclude_locals) {
            this->exclude_locals = exclude_locals;
        }

        void set_exclude_immutables(bool exclude_immutables) {
            this->exclude_immutables = exclude_immutables;
        }

        void set_capture_module_source(bool capture_module_source) {
            this->capture_module_source = capture_module_source;
        }

        void set_sizehint(size_t sizehint) {
            this->sizehint = sizehint;
        }

        void set_module_name(std::optional<std::string> module_name) {
            this->module_name = std::move(module_name);
        }

        void set_module_package(std::optional<std::string> module_package) {
            this->module_package = std::move(module_package);
        }

        void set_module_filename(std::optional<std::string> module_filename) {
            this->module_filename = std::move(module_filename);
        }

        void set_module_source(std::optional<std::vector<uint8_t>> module_source) {
            this->module_source = std::move(module_source);
        }
    };
    
    template<typename Loads, typename Dumps>
    class PyObjectSerdes {
        Loads loads;
        Dumps dumps;
        public:
            PyObjectSerdes(Loads& loads, Dumps& dumps) :
                loads(loads), dumps(dumps) {
            }

            template<typename Builder>
            offsets::PyObjectOffset serialize(Builder &builder, PyObject *obj) {
                auto dumps_result = dumps(obj);
                if(NULL == dumps_result.borrow()) {
                    return 0;
                }
                Py_ssize_t size = 0;
                char *pickled_data;
                if(PyBytes_AsStringAndSize(*dumps_result, &pickled_data, &size) == -1) {
                    return 0;
                }
                auto bytes = builder.CreateVector((const uint8_t *)pickled_data, size);
                auto py_obj = pyframe_buffer::CreatePyObject(builder, bytes);

                return py_obj;
            }

            auto deserialize(const pyframe_buffer::PyObject *obj) -> decltype(loads(nullptr)) {
                if(NULL == obj) {
                    return NULL;
                }

                auto data = obj->data()->data();
                auto size = obj->data()->size();

                if(NULL == data) {
                    return NULL;
                }
                auto bytes = pyobject_strongref::steal(PyBytes_FromStringAndSize((const char*)data, size));
                auto retval = loads(bytes.borrow());
                return retval;
            }

            template<typename Builder>
            offsets::PyObjectOffset serialize_dill(Builder &builder, PyObject *obj) {
                auto dumps_result = dumps.dill_dumps(obj);
                if(NULL == dumps_result.borrow()) {
                    return 0;
                }
                Py_ssize_t size = 0;
                char *pickled_data;
                if(PyBytes_AsStringAndSize(*dumps_result, &pickled_data, &size) == -1) {
                    return 0;
                }
                
                auto bytes = builder.CreateVector((const uint8_t *)pickled_data, size);
                auto py_obj = pyframe_buffer::CreatePyObject(builder, bytes);

                return py_obj;
            }

            auto deserialize_dill(const pyframe_buffer::PyObject *obj) -> decltype(loads(nullptr)) {
                if(NULL == obj) {
                    return NULL;
                }

                auto data = obj->data()->data();
                auto size = obj->data()->size();

                if(NULL == data) {
                    return NULL;
                }
                auto bytes = pyobject_strongref::steal(PyBytes_FromStringAndSize((const char*)data, size));
                auto retval = loads.dill_loads(bytes.borrow());
                return retval;
            }

    };

    template<typename Loads, typename Dumps>
    PyObjectSerdes(Loads&, Dumps&) -> PyObjectSerdes<Loads, Dumps>;

    template<typename PyCodeObjectSerializer>
    class PyObjectHeadSerdes {
        PyCodeObjectSerializer serializer;
        public:
            PyObjectHeadSerdes(PyCodeObjectSerializer& serializer) : serializer(serializer) {}

            struct Head {
                pyobject_strongref obj;
            };

            template<typename Builder>
            offsets::PyObjectHeadOffset serialize(Builder &builder, PyObject *obj) {
                auto py_obj = serializer.serialize(builder, obj);
                auto head = pyframe_buffer::CreatePyObjectHead(builder, py_obj);
                return head;
            }

            Head deserialize(const pyframe_buffer::PyObjectHead *obj) {
                auto ob_base = obj->ob_base();
                auto py_obj = serializer.deserialize(ob_base);
                return {py_obj};
            }
    };

    template<typename PyCodeObjectSerializer>
    class PyVarObjectHeadSerdes {
        PyCodeObjectSerializer serializer;
        public:
            PyVarObjectHeadSerdes(PyCodeObjectSerializer& serializer) : serializer(serializer) {}

            struct VarHead {
                pyobject_strongref obj;
                size_t size;
            };

            template<typename Builder>
            offsets::PyVarObjectHeadOffset serialize(Builder &builder, PyObject *obj, size_t size) {
                auto py_obj = serializer.serialize(builder, obj);
                auto var_head = pyframe_buffer::CreatePyVarObjectHead(builder, py_obj, size);
                return var_head;
            }

            VarHead deserialize(const pyframe_buffer::PyVarObjectHead *obj) {
                auto py_obj = serializer.deserialize(obj->ob_base());
                size_t size = obj->ob_size();
                return {py_obj, size};
            }
    };

    class DeserializedCodeObject {
      public:
        pyobject_strongref co_consts{NULL};
        pyobject_strongref co_names{NULL};
        pyobject_strongref co_exceptiontable{NULL};

        int co_flags;

        int co_argcount;
        int co_posonlyargcount;
        int co_kwonlyargcount;
        int co_stacksize;
        int co_firstlineno;

        int co_nlocalsplus;
        int co_framesize;
        int co_nlocals;
        int co_ncellvars;
        int co_nfreevars;
        int co_version;

        pyobject_strongref co_localsplusnames;
        pyobject_strongref co_localspluskinds;

        pyobject_strongref co_filename;
        pyobject_strongref co_name;
        pyobject_strongref co_qualname;
        pyobject_strongref co_linetable;

        std::vector<unsigned char> co_code_adaptive;

        bool immutables_included() {
            if(co_consts.borrow()) {
                return true;
            }
            if(co_names.borrow()) {
                return true;
            }
            return false;
        }
 
    };

    template <typename PyCodeObjectSerializer>
    class PyCodeObjectSerdes {
        PyCodeObjectSerializer po_serializer;
        template<typename Builder>
        flatbuffers::Offset<flatbuffers::Vector<uint8_t>> serialize_bitcode(Builder &builder, PyCodeObject *code) {
            pyobject_strongref code_instrs = pyobject_strongref::steal(PyCode_GetCode(code));
            Py_ssize_t total_size_bytes = 0;
            char *bitcode = nullptr;
            PyBytes_AsStringAndSize(code_instrs.borrow(), &bitcode, &total_size_bytes);

            auto bytes = builder.CreateVector((const uint8_t*) bitcode, total_size_bytes);
            return bytes;
        }

        public:
        PyCodeObjectSerdes(PyCodeObjectSerializer& po_serializer) : 
            po_serializer(po_serializer) {}

        template<typename Builder>
        offsets::PyCodeObjectOffset serialize(Builder &builder, PyCodeObject *obj, serdes::SerializationArgs& ser_args) {
            // Always serialize co_name as it's used as a key for lookup
            auto co_name_ser = (NULL != obj->co_name) ?
                std::optional{po_serializer.serialize(builder, obj->co_name)} : std::nullopt;
            
            // Only serialize other fields if we're not excluding immutables
            std::optional<offsets::PyObjectOffset> co_consts_ser = std::nullopt;
            std::optional<offsets::PyObjectOffset> co_names_ser = std::nullopt;
            std::optional<offsets::PyObjectOffset> co_exceptiontable_ser = std::nullopt;
            std::optional<offsets::PyObjectOffset> co_localsplusnames_ser = std::nullopt;
            std::optional<offsets::PyObjectOffset> co_localspluskinds_ser = std::nullopt;
            std::optional<offsets::PyObjectOffset> co_filename_ser = std::nullopt;
            std::optional<offsets::PyObjectOffset> co_qualname_ser = std::nullopt;
            std::optional<offsets::PyObjectOffset> co_linetable_ser = std::nullopt;
            std::optional<flatbuffers::Offset<flatbuffers::Vector<uint8_t>>> co_code_adaptive_ser = std::nullopt;
            
            if (!ser_args.exclude_immutables) {
                co_consts_ser = (NULL != obj->co_consts) ? 
                    std::optional{po_serializer.serialize(builder, obj->co_consts)} : std::nullopt;

                co_names_ser = (NULL != obj->co_names) ?
                    std::optional{po_serializer.serialize(builder, obj->co_names)} : std::nullopt;

                co_exceptiontable_ser = (NULL != obj->co_exceptiontable) ?
                    std::optional{po_serializer.serialize(builder, obj->co_exceptiontable)} : std::nullopt;

                co_localsplusnames_ser = (NULL != obj->co_localsplusnames) ?
                    std::optional{po_serializer.serialize(builder, obj->co_localsplusnames)} : std::nullopt;

                co_localspluskinds_ser = (NULL != obj->co_localspluskinds) ?
                    std::optional{po_serializer.serialize(builder, obj->co_localspluskinds)} : std::nullopt;

                co_filename_ser = (NULL != obj->co_filename) ?
                    std::optional{po_serializer.serialize(builder, obj->co_filename)} : std::nullopt;

                co_qualname_ser = (NULL != obj->co_qualname) ?
                    std::optional{po_serializer.serialize(builder, obj->co_qualname)} : std::nullopt;

                co_linetable_ser = (NULL != obj->co_linetable) ?
                    std::optional{po_serializer.serialize(builder, obj->co_linetable)} : std::nullopt;
                
                // Only serialize bytecode if we're not excluding immutables
                co_code_adaptive_ser = serialize_bitcode(builder, obj);
            }

            pyframe_buffer::PyCodeObjectBuilder code_builder(builder);

            if (co_consts_ser) {
                code_builder.add_co_consts(co_consts_ser.value());
            }
            if (co_names_ser) {
                code_builder.add_co_names(co_names_ser.value());
            }
            if (co_exceptiontable_ser) {
                code_builder.add_co_exceptiontable(co_exceptiontable_ser.value());
            }

            // Only add flags and other numeric properties if not excluding immutables
            if (!ser_args.exclude_immutables) {
                code_builder.add_co_flags(obj->co_flags);

                code_builder.add_co_argcount(obj->co_argcount);
                code_builder.add_co_posonlyargcount(obj->co_posonlyargcount);
                code_builder.add_co_kwonlyargcount(obj->co_kwonlyargcount);
                code_builder.add_co_stacksize(obj->co_stacksize);
                code_builder.add_co_firstlineno(obj->co_firstlineno);

                code_builder.add_co_nlocalsplus(obj->co_nlocalsplus);
                code_builder.add_co_framesize(obj->co_framesize);
                code_builder.add_co_nlocals(obj->co_nlocals);
                code_builder.add_co_ncellvars(obj->co_ncellvars);
                code_builder.add_co_nfreevars(obj->co_nfreevars);
                code_builder.add_co_version(obj->co_version);
            }

            if (co_localsplusnames_ser) {
                code_builder.add_co_localsplusnames(co_localsplusnames_ser.value());
            }
            if (co_localspluskinds_ser) {
                code_builder.add_co_localspluskinds(co_localspluskinds_ser.value());
            }

            if (co_filename_ser) {
                code_builder.add_co_filename(co_filename_ser.value());
            }
            
            // Always add co_name as it's our lookup key
            if (co_name_ser) {
                code_builder.add_co_name(co_name_ser.value());
            }
            
            if (co_qualname_ser) {
                code_builder.add_co_qualname(co_qualname_ser.value());
            }
            if (co_linetable_ser) {
                code_builder.add_co_linetable(co_linetable_ser.value());
            }
            
            // Only add bytecode if we're not excluding immutables
            if (co_code_adaptive_ser) {
                code_builder.add_co_code_adaptive(co_code_adaptive_ser.value());
            }

            return code_builder.Finish();
        }

        DeserializedCodeObject deserialize(const pyframe_buffer::PyCodeObject *obj) {
            DeserializedCodeObject deser;
            deser.co_consts = po_serializer.deserialize(obj->co_consts());
            deser.co_names = po_serializer.deserialize(obj->co_names());
            deser.co_exceptiontable = po_serializer.deserialize(obj->co_exceptiontable());

            deser.co_flags = obj->co_flags();

            deser.co_argcount = obj->co_argcount();
            deser.co_posonlyargcount = obj->co_posonlyargcount();
            deser.co_kwonlyargcount = obj->co_kwonlyargcount();
            deser.co_stacksize = obj->co_stacksize();
            deser.co_firstlineno = obj->co_firstlineno();

            deser.co_nlocalsplus = obj->co_nlocalsplus();
            deser.co_framesize = obj->co_framesize();
            deser.co_nlocals = obj->co_nlocals();
            deser.co_ncellvars = obj->co_ncellvars();
            deser.co_nfreevars = obj->co_nfreevars();
            deser.co_version = obj->co_version();

            deser.co_localsplusnames = po_serializer.deserialize(obj->co_localsplusnames());
            deser.co_localspluskinds = po_serializer.deserialize(obj->co_localspluskinds());

            deser.co_filename = po_serializer.deserialize(obj->co_filename());
            deser.co_name = po_serializer.deserialize(obj->co_name());
            deser.co_qualname = po_serializer.deserialize(obj->co_qualname());
            deser.co_linetable = po_serializer.deserialize(obj->co_linetable());

            auto bitcode = obj->co_code_adaptive();
            if(bitcode) {
                deser.co_code_adaptive = std::vector<unsigned char>(bitcode->begin(), bitcode->end());
            }

            return deser;
        }

   };

    template<typename PyCodeObjectSerializer>
    PyVarObjectHeadSerdes(PyCodeObjectSerializer&) -> PyVarObjectHeadSerdes<PyCodeObjectSerializer>;

    class DeserializedPyInterpreterFrame {
      public:
        DeserializedCodeObject f_executable;
        std::optional<pyobject_strongref> f_funcobj;
        pyobject_strongref f_globals;
        pyobject_strongref f_builtins;
        pyobject_strongref f_locals;

        uint64_t instr_offset;
        uint16_t return_offset;

        uint8_t owner;

        std::vector<pyobject_strongref> localsplus;
        std::vector<pyobject_strongref> stack;
        std::optional<std::string> module_name;
        std::optional<std::string> module_package;
        std::optional<std::string> module_filename;

    };

    template<typename PyObjectSerializer>
    class PyInterpreterFrameSerdes {
        PyObjectSerializer po_serializer;
        PyCodeObjectSerdes<PyObjectSerializer> code_serializer;

        static bool set_dict_string(PyObject *dict, const char *key, const flatbuffers::String *value) {
            if (value == NULL) {
                return true;
            }
            auto py_value = pyobject_strongref::steal(
                PyUnicode_DecodeUTF8(value->c_str(), value->size(), NULL));
            if (!py_value) {
                return false;
            }
            return PyDict_SetItemString(dict, key, py_value.borrow()) == 0;
        }

        static bool bootstrap_module_globals(const pyframe_buffer::PyInterpreterFrame *obj) {
            auto source = obj->module_source();
            if (source == NULL) {
                return true;
            }

            auto module_name_field = obj->module_name();
            std::string module_name = "__sauerkraut_snapshot__";
            if (module_name_field != NULL) {
                module_name = std::string(module_name_field->c_str(), module_name_field->size());
            }

            auto module_filename_field = obj->module_filename();
            std::string compile_filename = "<sauerkraut_snapshot>";
            if (module_filename_field != NULL) {
                compile_filename = std::string(module_filename_field->c_str(), module_filename_field->size());
            } else if (module_name_field != NULL) {
                compile_filename = module_name;
            }

            pyobject_strongref globals_dict;
            if (module_name_field != NULL) {
                auto sys_module = pyobject_strongref::steal(PyImport_ImportModule("sys"));
                if (!sys_module) {
                    return false;
                }
                auto modules_dict = pyobject_strongref::steal(
                    PyObject_GetAttrString(sys_module.borrow(), "modules"));
                if (!modules_dict || !PyDict_Check(modules_dict.borrow())) {
                    PyErr_SetString(PyExc_RuntimeError, "Failed to access sys.modules during module reconstruction.");
                    return false;
                }

                auto module_obj = pyobject_strongref::steal(PyModule_New(module_name.c_str()));
                if (!module_obj) {
                    return false;
                }
                if (PyDict_SetItemString(modules_dict.borrow(), module_name.c_str(), module_obj.borrow()) < 0) {
                    return false;
                }
                globals_dict = pyobject_strongref(PyModule_GetDict(module_obj.borrow()));
            } else {
                globals_dict = pyobject_strongref::steal(PyDict_New());
            }

            if (!globals_dict) {
                return false;
            }

            if (PyDict_SetItemString(globals_dict.borrow(), "__builtins__", PyEval_GetBuiltins()) < 0) {
                return false;
            }

            if (module_name_field != NULL) {
                if (!set_dict_string(globals_dict.borrow(), "__name__", module_name_field)) {
                    return false;
                }
            } else {
                auto module_name_obj = pyobject_strongref::steal(PyUnicode_FromString(module_name.c_str()));
                if (!module_name_obj) {
                    return false;
                }
                if (PyDict_SetItemString(globals_dict.borrow(), "__name__", module_name_obj.borrow()) < 0) {
                    return false;
                }
            }

            if (!set_dict_string(globals_dict.borrow(), "__package__", obj->module_package())) {
                return false;
            }
            if (!set_dict_string(globals_dict.borrow(), "__file__", obj->module_filename())) {
                return false;
            }

            std::string source_text(reinterpret_cast<const char*>(source->data()), source->size());
            auto code_obj = pyobject_strongref::steal(
                Py_CompileString(source_text.c_str(), compile_filename.c_str(), Py_file_input));
            if (!code_obj) {
                return false;
            }

            auto eval_result = pyobject_strongref::steal(
                PyEval_EvalCode(code_obj.borrow(), globals_dict.borrow(), globals_dict.borrow()));
            if (!eval_result) {
                return false;
            }
            return true;
        }

        template <typename Builder>
        flatbuffers::Offset<flatbuffers::Vector<offsets::PyObjectOffset>> serialize_stack(Builder &builder, sauerkraut::PyInterpreterFrame &obj, int stack_depth) {
            std::vector<offsets::PyObjectOffset> stack;
            stack.reserve(stack_depth);

            _PyStackRef *stack_base = utils::py::get_stack_base(&obj);
            for(size_t i = 0; i < (size_t) stack_depth; i++) {
                auto stack_obj = utils::py::stackref_to_object_for_serialization(stack_base[i]);
                if (stack_obj.obj == NULL) {
                    continue;
                }
                auto stack_obj_ser = po_serializer.serialize(builder, stack_obj.obj);
                stack.push_back(stack_obj_ser);
                if (stack_obj.owned) {
                    Py_DECREF(stack_obj.obj);
                }
            }

            auto stack_offset = builder.CreateVector(stack);
            return stack_offset;
        }
        template<typename Builder>
        std::pair<flatbuffers::Offset<flatbuffers::Vector<offsets::PyObjectOffset>>, 
                  flatbuffers::Offset<flatbuffers::Vector<uint8_t>>> 
        serialize_fast_locals_plus(Builder &builder, sauerkraut::PyInterpreterFrame &obj, serdes::SerializationArgs& ser_args) {
            auto n_locals = utils::py::get_code_nlocals(
                (PyCodeObject*)utils::py::stackref_as_pyobject(obj.f_executable));
            auto exclude_local_bitmask = ser_args.exclude_locals.value_or(std::vector<bool>(n_locals, false));
            std::vector<offsets::PyObjectOffset> localsplus;

            std::vector<uint8_t> uint8_bitmask;
            uint8_bitmask.reserve(n_locals);
            int non_excluded_count = 0;
            for (int i = 0; i < n_locals; i++) {
                _PyStackRef local_ref = obj.localsplus[i];
                if(utils::py::stackref_is_null(local_ref) || exclude_local_bitmask[i]) {
                    // a local can be NULL if it has not been initialized for the first time
                    uint8_bitmask.push_back(1);
                } else {
                    uint8_bitmask.push_back(0);
                    non_excluded_count++;
                }
            }

            localsplus.reserve(non_excluded_count);

            // Only serialize non-excluded locals
            for(int i = 0; i < n_locals; i++) {
                auto local = obj.localsplus[i];
                auto local_pyobj = utils::py::stackref_to_object_for_serialization(local);

                if(NULL == local_pyobj.obj || exclude_local_bitmask[i]) {
                    if (local_pyobj.owned) {
                        Py_DECREF(local_pyobj.obj);
                    }
                    continue;
                }

                auto local_ser = po_serializer.serialize(builder, local_pyobj.obj);
                localsplus.push_back(local_ser);
                if (local_pyobj.owned) {
                    Py_DECREF(local_pyobj.obj);
                }
            }

            auto localsplus_offset = builder.CreateVector(localsplus);
            auto bitmask_offset = builder.CreateVector(uint8_bitmask);

            return std::make_pair(localsplus_offset, bitmask_offset);
        }

        public:
        PyInterpreterFrameSerdes(PyObjectSerializer& po_serializer) : 
            po_serializer(po_serializer),
            code_serializer(po_serializer) {}

        template<typename Builder>
        offsets::PyInterpreterFrameOffset serialize(Builder &builder, sauerkraut::PyInterpreterFrame &obj, int stack_depth, serdes::SerializationArgs& ser_args) {
            offsets::PyCodeObjectOffset f_executable_ser;
            offsets::PyObjectOffset f_func_obj_ser = 0;
            offsets::PyObjectOffset f_globals_ser = 0;
            bool has_f_funcobj = false;

            f_executable_ser = code_serializer.serialize(
                builder, (PyCodeObject*)utils::py::stackref_as_pyobject(obj.f_executable), ser_args);
            if(!ser_args.exclude_immutables) {
                PyObject *func_obj = utils::py::get_funcobj(&obj);
                if (func_obj != NULL) {
                    f_func_obj_ser = po_serializer.serialize(builder, func_obj);
                    has_f_funcobj = true;
                }
                f_globals_ser = po_serializer.serialize_dill(builder, obj.f_globals);
            }

            auto f_locals_ser = (NULL != obj.f_locals) ? 
                std::optional{po_serializer.serialize(builder, obj.f_locals)} : std::nullopt;

            auto fast_locals_result = serialize_fast_locals_plus(builder, obj, ser_args);
            auto stack_ser = serialize_stack(builder, obj, stack_depth);
            auto module_name_ser = ser_args.module_name ?
                std::optional{builder.CreateString(ser_args.module_name.value())} : std::nullopt;
            auto module_package_ser = ser_args.module_package ?
                std::optional{builder.CreateString(ser_args.module_package.value())} : std::nullopt;
            auto module_filename_ser = ser_args.module_filename ?
                std::optional{builder.CreateString(ser_args.module_filename.value())} : std::nullopt;
            auto module_source_ser = ser_args.module_source ?
                std::optional{builder.CreateVector(ser_args.module_source.value())} : std::nullopt;

            pyframe_buffer::PyInterpreterFrameBuilder frame_builder(builder);

            if(f_locals_ser) {
                frame_builder.add_f_locals(f_locals_ser.value());
            }
            if(!ser_args.exclude_immutables) {
                if (has_f_funcobj) {
                    frame_builder.add_f_funcobj(f_func_obj_ser);
                }
                frame_builder.add_f_globals(f_globals_ser);
            }

            frame_builder.add_f_executable(f_executable_ser);
            frame_builder.add_instr_offset(utils::py::get_instr_offset<utils::py::Units::Bytes>(obj.frame_obj));
            frame_builder.add_return_offset(obj.return_offset);
            frame_builder.add_owner(obj.owner);
            frame_builder.add_locals_plus(fast_locals_result.first);
            frame_builder.add_locals_exclusion_bitmask(fast_locals_result.second);
            frame_builder.add_stack(stack_ser);
            if (module_name_ser) {
                frame_builder.add_module_name(module_name_ser.value());
            }
            if (module_package_ser) {
                frame_builder.add_module_package(module_package_ser.value());
            }
            if (module_filename_ser) {
                frame_builder.add_module_filename(module_filename_ser.value());
            }
            if (module_source_ser) {
                frame_builder.add_module_source(module_source_ser.value());
            }

            return frame_builder.Finish();

        }

        DeserializedPyInterpreterFrame deserialize(const pyframe_buffer::PyInterpreterFrame *obj, bool reconstruct_module=true) {
            DeserializedPyInterpreterFrame deser;
            if (obj->module_name()) {
                deser.module_name = std::string(obj->module_name()->c_str(), obj->module_name()->size());
            }
            if (obj->module_package()) {
                deser.module_package = std::string(obj->module_package()->c_str(), obj->module_package()->size());
            }
            if (obj->module_filename()) {
                deser.module_filename = std::string(obj->module_filename()->c_str(), obj->module_filename()->size());
            }
            if (reconstruct_module && obj->module_source()) {
                if (!bootstrap_module_globals(obj)) {
                    if (!PyErr_Occurred()) {
                        PyErr_SetString(PyExc_RuntimeError, "Failed to reconstruct module source during frame deserialization.");
                    }
                    return deser;
                }
            }
            if(obj->f_executable()) {
                deser.f_executable = code_serializer.deserialize(obj->f_executable());
            }
            if(obj->f_funcobj()) {
                deser.f_funcobj = po_serializer.deserialize(obj->f_funcobj());
            }
            if(obj->f_globals()) {
                deser.f_globals = po_serializer.deserialize_dill(obj->f_globals());
            }
            deser.f_builtins = po_serializer.deserialize(obj->f_builtins());
            deser.f_locals = po_serializer.deserialize(obj->f_locals());

            deser.instr_offset = obj->instr_offset();
            deser.return_offset = obj->return_offset();
            deser.owner = obj->owner();

            auto localsplus = obj->locals_plus();
            auto exclusion_bitmask = obj->locals_exclusion_bitmask();
            if (localsplus == NULL || exclusion_bitmask == NULL) {
                PyErr_SetString(PyExc_RuntimeError, "Serialized frame is missing locals metadata.");
                return deser;
            }

            int total_locals = exclusion_bitmask->size();
            deser.localsplus.reserve(total_locals);

            int localsplus_idx = 0;
            for(int i = 0; i < total_locals; i++) {
                if(exclusion_bitmask->Get(i) != 0) {
                    // This local was excluded, use Py_None
                    deser.localsplus.push_back(Py_None);
                } else {
                    // This local was included, get it from the serialized data
                    deser.localsplus.push_back(po_serializer.deserialize(localsplus->Get(localsplus_idx++)));
                }
            }

            auto stack = obj->stack();
            if (stack == NULL) {
                PyErr_SetString(PyExc_RuntimeError, "Serialized frame is missing stack metadata.");
                return deser;
            }
            deser.stack.reserve(stack->size());
            for(auto stack_obj : *stack) {
                deser.stack.push_back(po_serializer.deserialize(stack_obj));
            }

            return deser;
        }
    };

    template<typename PyObjectSerializer>
    PyInterpreterFrameSerdes(PyObjectSerializer&) -> PyInterpreterFrameSerdes<PyObjectSerializer>;

    class DeserializedPyFrame {
      public:
        DeserializedPyInterpreterFrame f_frame;
        pyobject_strongref f_trace;
        int f_lineno;
        char f_trace_lines;
        char f_trace_opcodes;
        pyobject_strongref f_extra_locals;
        pyobject_strongref f_locals_cache;
    };

    template<typename PyObjectSerializer>
    class PyFrameSerdes {
        PyObjectSerializer po_serializer;
        PyObjectHeadSerdes<PyObjectSerializer> poh_serializer;
        public:
            PyFrameSerdes(PyObjectSerializer& po_serializer) : 
                          po_serializer(po_serializer),
                          poh_serializer(po_serializer) {}

            template<typename Builder>
            offsets::PyFrameOffset serialize(Builder &builder, sauerkraut::PyFrame &obj, serdes::SerializationArgs& ser_args) {
                PyInterpreterFrameSerdes interpreter_frame_serializer(po_serializer);
                auto stack_size = utils::py::get_stack_state((PyObject*)&obj).size();
                auto interp_frame_offset = interpreter_frame_serializer.serialize(builder, *obj.f_frame, stack_size, ser_args);

                pyframe_buffer::PyFrameBuilder frame_builder(builder);
                // Do NOT serialize the ob_base.
                // frame_builder.add_ob_base(poh_serializer.serialize(builder, &obj.ob_base));

                frame_builder.add_f_frame(interp_frame_offset);
                
                // TODO: These need to be changed to serialize BEFORE
                // creating the frame_builder.
                if(NULL != obj.f_trace) {
                    auto f_trace_ser = po_serializer.serialize(builder, obj.f_trace);
                    frame_builder.add_f_trace(f_trace_ser);
                }

                frame_builder.add_f_lineno(obj.f_lineno);
                frame_builder.add_f_trace_lines(obj.f_trace_lines);
                frame_builder.add_f_trace_opcodes(obj.f_trace_opcodes);

                if(NULL != obj.f_extra_locals) {
                    auto f_extra_locals_ser = po_serializer.serialize(builder, obj.f_extra_locals);
                    frame_builder.add_f_extra_locals(f_extra_locals_ser);
                }

                if(NULL != obj.f_locals_cache) {
                    auto f_locals_cache_ser = po_serializer.serialize(builder, obj.f_locals_cache);
                    frame_builder.add_f_locals_cache(f_locals_cache_ser);
                }

                
                return frame_builder.Finish();
            }

            DeserializedPyFrame deserialize(const pyframe_buffer::PyFrame *obj, bool reconstruct_module=true) {
                // auto ob_base_deser = poh_serializer.deserialize(obj->ob_base());
                DeserializedPyFrame deser;
                PyInterpreterFrameSerdes interpreter_frame_serializer(po_serializer);

                deser.f_frame = interpreter_frame_serializer.deserialize(obj->f_frame(), reconstruct_module);

                deser.f_trace = po_serializer.deserialize(obj->f_trace());

                deser.f_lineno = obj->f_lineno();
                deser.f_trace_lines = obj->f_trace_lines();
                deser.f_trace_opcodes = obj->f_trace_opcodes();

                deser.f_extra_locals = po_serializer.deserialize(obj->f_extra_locals());

                deser.f_locals_cache = po_serializer.deserialize(obj->f_locals_cache());
            return deser;
            }
    };

    template<typename PyObjectSerializer>
    PyFrameSerdes(PyObjectSerializer&) -> PyFrameSerdes<PyObjectSerializer>;
    
}

#endif // SERDES_HH_INCLUDED
