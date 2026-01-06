#include "greenlet_compat.h"

static PyObject* greenlet_type = NULL;

namespace greenlet {
    bool is_greenlet(PyObject *obj) {
        if (greenlet_type == NULL) {
            return false;
        }
        return PyObject_IsInstance(obj, greenlet_type) == 1;
    }

    PyFrameObject* getframe(PyObject* self) {
        PyObject* frame = PyObject_GetAttrString(self, "gr_frame");
        if (frame == NULL) {
            return NULL;
        }
        if (frame == Py_None) {
            Py_DECREF(frame);
            return NULL;
        }
        return (PyFrameObject*)frame;
    }

    void init_greenlet() {
        // Import greenlet module and cache the greenlet type
        PyObject* greenlet_module = PyImport_ImportModule("greenlet");
        if (greenlet_module == NULL) {
            PyErr_Print();
            return;
        }
        greenlet_type = PyObject_GetAttrString(greenlet_module, "greenlet");
        Py_DECREF(greenlet_module);
        if (greenlet_type == NULL) {
            PyErr_Print();
        }
    }
}

