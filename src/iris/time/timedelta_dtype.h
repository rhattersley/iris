/* Defines the timedelta dtype. */

#ifndef TIMEDELTA_DTYPE_H
#define TIMEDELTA_DTYPE_H
#ifdef __cplusplus
extern "C" {
#endif

PyArray_Descr *register_timedelta_dtype(PyObject *module);

#ifdef __cplusplus
}
#endif
#endif
