/*  datetime360.h
 */

#ifndef DATETIME360_H
#define DATETIME360_H
#ifdef __cplusplus
extern "C" {
#endif

#include <datetime.h>

// Type checks to use static local 360 day version of PyDateTime_DateType and PyDateTime_DateTimeType
#define PyDate360_Check(op) PyObject_TypeCheck(op, &PyDateTime360_DateType)
#define PyDate360_CheckExact(op) (Py_TYPE(op) == &PyDateTime_DateType)

#define PyDateTime360_Check(op) PyObject_TypeCheck(op, &PyDateTime360_DateTimeType)
#define PyDateTime360_CheckExact(op) (Py_TYPE(op) == &PyDateTime360_DateTimeType)

// No CAPI capsule as yet
//#define PyDateTime360_CAPSULE_NAME "iris.datetime360.datetime360_CAPI"

/* Define global variable for the C API and a macro for setting it. */
//static PyDateTime_CAPI *PyDateTime360API = NULL;

//#define PyDateTime360_IMPORT PyDateTime360API = (PyDateTime_CAPI *)PyCapsule_Import(PyDateTime360_CAPSULE_NAME, 0)

#ifdef __cplusplus
}
#endif
#endif
