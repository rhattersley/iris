/* Defines the time360 dtype.
 */

#ifndef TIME360_DTYPE_H
#define TIME360_DTYPE_H
#ifdef __cplusplus
extern "C" {
#endif

void register_time360_dtype(PyObject *module, PyTypeObject *Time360Type);

#ifdef __cplusplus
}
#endif
#endif
