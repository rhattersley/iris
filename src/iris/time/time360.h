/* Defines the Time360 Python type.
 */

#ifndef TIME360_H
#define TIME360_H
#ifdef __cplusplus
extern "C" {
#endif

#include "cftime.h"

typedef struct {
    PyObject_HEAD
    datetime time;
} Time360;

PyTypeObject *register_Time360(PyObject *module);

#ifdef __cplusplus
}
#endif
#endif
