/* Defines the time360 dtype.
 */

#ifndef TIME360_DTYPE_H
#define TIME360_DTYPE_H
#ifdef __cplusplus
extern "C" {
#endif

void register_time360_dtype(PyObject *module, PyTypeObject *time360Type,
                            PyArray_Descr *timedelta_dtype);

#ifdef __cplusplus
}
#endif
#endif
