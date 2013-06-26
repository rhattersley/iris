#include <Python.h>

#include <stddef.h>

/* Allow access to the `datetime` module's C API via the subsequent
 * use of PyDateTime_IMPORT.
 */
#include <datetime.h>

//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
//#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#define PY_ARRAY_UNIQUE_SYMBOL time_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#define PY_UFUNC_UNIQUE_SYMBOL time_UFUNC_API
#define NO_IMPORT_UFUNC
#include <numpy/ufuncobject.h>

#include "cftime.h"
#include "timedelta_dtype.h"


/*
 * Function's for timedelta dtype's PyArray_ArrFuncs
 */

static PyObject *
getitem(void *data, void *arr)
{
    PyObject *result;
    timedelta *delta;

    /* TODO: Consider supporting "misaligned and/or swapped" arrays */
    if (!PyArray_ISBEHAVED(arr)) {
        PyErr_SetString(PyExc_ValueError, "getitem on misbehaved array");
        return NULL;
    }

    delta = (timedelta *)data;
    result = PyDelta_FromDSU(delta->days, delta->seconds, delta->microseconds);
    return result;
}

static int
setitem(PyObject *item, void *data, void *arr)
{
    if (!PyDelta_Check(item)) {
        PyErr_SetString(PyExc_TypeError,
                        "assigned value must be datetime.timedelta instance");
        return -1;
    }

    if (arr == NULL || PyArray_ISBEHAVED(arr)) {
        memcpy(data, &((PyDateTime_Delta *)item)->days, sizeof(timedelta));
    } else {
        PyErr_SetString(PyExc_ValueError, "setitem on misbehaved array");
        return -1;
    }

    return 0;
}

static void
copyswapn(void *dest, npy_intp dstride, void *src, npy_intp sstride,
          npy_intp n, int swap, void *NPY_UNUSED(arr))
{
    timedelta *dest_item = (timedelta *)dest;
    timedelta *src_item = (timedelta *)src;

    PyArray_Descr *descr;
    descr = PyArray_DescrFromType(NPY_INT32);
    descr->f->copyswapn(&dest_item->days, dstride,
                        &src_item->days, sstride,
                        n, swap, NULL);
    descr->f->copyswapn(&dest_item->seconds, dstride,
                        &src_item->seconds, sstride,
                        n, swap, NULL);
    descr->f->copyswapn(&dest_item->microseconds, dstride,
                        &src_item->microseconds, sstride,
                        n, swap, NULL);
    Py_DECREF(descr);
}

static void
copyswap(void *dest, void *src, int swap, void *NPY_UNUSED(arr))
{
    timedelta *dest_item = (timedelta *)dest;
    timedelta *src_item = (timedelta *)src;
    PyArray_Descr *descr;

    descr = PyArray_DescrFromType(NPY_INT32);
    descr->f->copyswapn(dest_item, sizeof(int32_t), src_item, sizeof(int32_t),
                        3, swap, NULL);
    Py_DECREF(descr);
}

static int
compare(const timedelta *td1, const timedelta *td2, void *NPY_UNUSED(arr))
{
    return timedelta_compare(td1, td2);
}

static int
argmax(const timedelta *data, npy_intp n, npy_intp *max_ind,
       void *NPY_UNUSED(arr))
{
    const timedelta *max;
    npy_intp i;

    max = data++;
    *max_ind = 0;
    for (i = 1; i < n; i++, data++) {
        if (timedelta_compare(max, data) == -1) {
            max = data;
            *max_ind = i;
        }
    }
    return 0;
}

/*
 * ufuncs
 */

#define UNARY_GEN_UFUNC(array_type, func_name, arg_type, ret_type)\
static void array_type##_##func_name##_ufunc(\
    char** args, npy_intp* dimensions, npy_intp* steps, void* data)\
{\
    char *ip1 = args[0], *op1 = args[1];\
    npy_intp is1 = steps[0], os1 = steps[1];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    for (i = 0; i < n; i++, ip1 += is1, op1 += os1)\
    {\
        const arg_type in1 = *(arg_type *)ip1;\
        *((ret_type *)op1) = array_type##_##func_name(in1);\
    }\
}

#define BINARY_GEN_UFUNC(array_type, func_name, arg_type, ret_type)\
static void array_type##_##func_name##_ufunc(\
    char** args, npy_intp* dimensions, npy_intp* steps, void* data)\
{\
    char *ip1 = args[0], *ip2 = args[1], *op1 = args[2];\
    npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    for (i = 0; i < n; i++, ip1 += is1, ip2 += is2, op1 += os1)\
    {\
        const arg_type in1 = *(arg_type *)ip1;\
        const arg_type in2 = *(arg_type *)ip2;\
        *((ret_type *)op1) = array_type##_##func_name(in1, in2);\
    }\
}

BINARY_GEN_UFUNC(timedelta, equal_timedelta, timedelta, int8_t)
UNARY_GEN_UFUNC(timedelta, sign, timedelta, int8_t)

/*
 * Module initialisation
 */

typedef struct { char c; timedelta r; } align_test;

static PyArray_Descr *
create_dtype(void)
{
    PyArray_ArrFuncs *arrfuncs;
    PyArray_Descr *dtype;

    /* Define the standard array functions for our dtype. */
    arrfuncs = PyMem_New(PyArray_ArrFuncs, 1);
    PyArray_InitArrFuncs(arrfuncs);
    arrfuncs->getitem = getitem;
    arrfuncs->setitem = setitem;
    arrfuncs->copyswapn = copyswapn;
    arrfuncs->copyswap = copyswap;
    arrfuncs->compare = (PyArray_CompareFunc *)compare;
    arrfuncs->argmax = (PyArray_ArgFunc *)argmax;

    /* Must explicitly set all members or we risk a memory fault later. */
    dtype = PyObject_New(PyArray_Descr, &PyArrayDescr_Type);
    dtype->typeobj = PyDateTimeAPI->DeltaType;
    Py_INCREF(PyDateTimeAPI->DeltaType);
    dtype->kind = 't';
    dtype->type = 't';
    dtype->byteorder = '=';
    /* XXX Trying NPY_NEEDS_PYAPI to see if that helps raise errors more 
     * promptly/accurately.
     * Doesn't seem to help when errors are raised during setitem
     */
    dtype->flags = NPY_USE_GETITEM | NPY_USE_SETITEM | NPY_NEEDS_INIT | NPY_NEEDS_PYAPI;
    dtype->type_num = -1; /* Set when registered */
    dtype->elsize = sizeof(timedelta);
    dtype->alignment = offsetof(align_test, r);
    dtype->subarray = NULL;
    dtype->fields = NULL;
    dtype->names = NULL;
    dtype->f = arrfuncs;
    dtype->metadata = NULL;
    /*dtype->c_metadata = NULL; */

    /* TODO: This NumPy type number "should be stored and made available
     * by your module".
     */
    /* TODO: Clarify NumPy docs re. reference counting behaviour */
    PyArray_RegisterDataType(dtype);
    assert(dtype->type_num != -1);

    return dtype;
}

PyArray_Descr *
register_timedelta_dtype(PyObject *module)
{
    PyArray_Descr *dtype;
    int arg_types[3];

    PyDateTime_IMPORT;

    dtype = create_dtype();
    PyModule_AddObject(module, "timedelta_dtype", (PyObject *)dtype);

    // TODO: Move these variable declarations
    PyObject *numpy_module = PyImport_ImportModule("numpy");
    PyObject *numpy_dict = PyModule_GetDict(numpy_module);
    Py_DECREF(numpy_module);

    arg_types[0] = dtype->type_num;
    arg_types[1] = dtype->type_num;
    arg_types[2] = NPY_BOOL;
    PyUFunc_RegisterLoopForType(
        (PyUFuncObject *)PyDict_GetItemString(numpy_dict, "equal"),
        dtype->type_num, timedelta_equal_timedelta_ufunc, arg_types, NULL);

    arg_types[0] = dtype->type_num;
    arg_types[1] = NPY_INT8;
    PyUFunc_RegisterLoopForType(
        (PyUFuncObject *)PyDict_GetItemString(numpy_dict, "sign"),
        dtype->type_num, timedelta_sign_ufunc, arg_types, NULL);

    return dtype;
}
