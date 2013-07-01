/* Imports the NumPy array API for use in satellite files which don't
 * call `import_array()`.
 */

#ifndef USEARRAY_H
#define USEARRAY_H
#ifdef __cplusplus
extern "C" {
#endif

/* Ensure we have imported Python before trying to import NumPy. */
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#define PY_ARRAY_UNIQUE_SYMBOL time_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#ifdef __cplusplus
}
#endif
#endif
