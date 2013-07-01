#include <Python.h>

//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
//#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#define PY_ARRAY_UNIQUE_SYMBOL time_ARRAY_API
#include <numpy/arrayobject.h>

#define PY_UFUNC_UNIQUE_SYMBOL time_UFUNC_API
#include <numpy/ufuncobject.h>

#include "time360.h"
#include "time360_dtype.h"

/* XXX: For debug use only */
void
dump(char *prefix, PyObject *object)
{
    PyObject *str = PyObject_Str(object);

    printf("%s: %s\n", prefix, PyString_AsString(str));
    Py_DECREF(str);
}

PyMODINIT_FUNC
inittime(void)
{
    PyObject *module;
    PyTypeObject *Time360Type;

    module = Py_InitModule3("time", NULL,
                       "Support for CF-netCDF time values.");
    if (module == NULL)
        return;

    import_array();
    import_ufunc();

    /* TODO: Error handling. E.g. If Time360Type is NULL. */
    Time360Type = register_Time360(module);
    register_time360_dtype(module, Time360Type);
}
