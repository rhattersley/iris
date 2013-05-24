#include <Python.h>

#include <stddef.h>
#include <stdint.h>

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
#include "time360.h"

/*
 * PyArray_ArrFuncs
 */

static PyObject *
time360_getitem(void *data, void *arr)
{
    PyObject *result;
    datetime *item;
    PyTypeObject *scalar_type;

    /* TODO: Consider supporting "misaligned and/or swapped" arrays */
    if (!PyArray_ISBEHAVED(arr)) {
        PyErr_SetString(PyExc_ValueError, "getitem on misbehaved array");
        return NULL;
    }

    scalar_type = PyArray_DESCR((PyArrayObject *)arr)->typeobj;
    item = (datetime *)data;
    result = PyObject_CallFunction((PyObject *)scalar_type, "iiiiiii",
                                   item->year, item->month + 1, item->day + 1,
                                   item->hour, item->minute, item->second,
                                   item->microsecond);
    /* This catches an attempt to extract an item which is not a valid
     * value (e.g. the memory for this item hasn't been initialised and
     * contains an invalid month.)
     */
    if (result == NULL && PyErr_ExceptionMatches(PyExc_ValueError)) {
        PyErr_Clear();
        Py_RETURN_NONE;
    }
    return result;
}

static int
time360_setitem(PyObject *item, void *data, void *arr)
{
    PyTypeObject *scalar_type;

    scalar_type = PyArray_DESCR((PyArrayObject *)arr)->typeobj;
    int ok = PyObject_IsInstance(item, (PyObject *)scalar_type);
    if (ok == 0) {
        PyErr_SetNone(PyExc_TypeError);
    }
    if (ok != 1) {
        return -1;
    }

    if (arr == NULL || PyArray_ISBEHAVED(arr)) {
        *((datetime *)data) = ((Time360 *)item)->time;
    } else {
        PyErr_SetString(PyExc_ValueError, "setitem on misbehaved array");
        return -1;
    }

    return 0;
}

static void
time360_copyswapn(void *dest, npy_intp dstride, void *src, npy_intp sstride,
                  npy_intp n, int swap, void *NPY_UNUSED(arr))
{
    datetime *dest_item = (datetime *)dest;
    datetime *src_item = (datetime *)src;

    PyArray_Descr *descr;
    descr = PyArray_DescrFromType(NPY_INT32);
    descr->f->copyswapn(&dest_item->year, dstride,
                        &src_item->year, sstride,
                        n, swap, NULL);
    descr->f->copyswapn(&dest_item->microsecond, dstride,
                        &src_item->microsecond, sstride,
                        n, swap, NULL);
    Py_DECREF(descr);
    descr = PyArray_DescrFromType(NPY_UINT8);
    descr->f->copyswapn(&dest_item->month, dstride, &src_item->month, sstride,
                        n, swap, NULL);
    descr->f->copyswapn(&dest_item->day, dstride, &src_item->day, sstride,
                        n, swap, NULL);
    descr->f->copyswapn(&dest_item->hour, dstride, &src_item->hour, sstride,
                        n, swap, NULL);
    descr->f->copyswapn(&dest_item->minute, dstride, &src_item->minute, sstride,
                        n, swap, NULL);
    descr->f->copyswapn(&dest_item->second, dstride, &src_item->second, sstride,
                        n, swap, NULL);
    Py_DECREF(descr);
}

static void
time360_copyswap(void *dest, void *src, int swap, void *NPY_UNUSED(arr))
{
    datetime *dest_item = (datetime *)dest;
    datetime *src_item = (datetime *)src;

    PyArray_Descr *descr;
    descr = PyArray_DescrFromType(NPY_INT32);
    descr->f->copyswapn(dest_item, sizeof(int32_t), src_item, sizeof(int32_t),
                        1, swap, NULL);
    descr->f->copyswapn(&(dest_item->microsecond), sizeof(int32_t),
                        &(src_item->microsecond), sizeof(int32_t),
                        1, swap, NULL);
    Py_DECREF(descr);
    descr = PyArray_DescrFromType(NPY_UINT8);
    descr->f->copyswapn(&(dest_item->month), sizeof(uint8_t),
                        &(src_item->month), sizeof(uint8_t), 5, swap, NULL);
    Py_DECREF(descr);
}

/*
 * ufuncs
 */

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

//BINARY_GEN_UFUNC(time360, subtract, datetime, timedelta)
//
static void time360_subtract_time360_ufunc(
    char** args, npy_intp* dimensions, npy_intp* steps, void* data)
{
    char *ip1 = args[0], *ip2 = args[1], *op1 = args[2];
    npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2];
    npy_intp n = dimensions[0];
    npy_intp i;
    for (i = 0; i < n; i++, ip1 += is1, ip2 += is2, op1 += os1)
    {
        const datetime in1 = *(datetime *)ip1;
        const datetime in2 = *(datetime *)ip2;
        *((timedelta *)op1) = time360_subtract_time360(in1, in2);
    }
}

typedef struct { char c; datetime r; } align_test;

PyArray_Descr *create_dtype(PyTypeObject *time360Type)
{
    PyArray_Descr *dtype;

    /* Define the standard array functions for our dtype. */
    PyArray_ArrFuncs *arrfuncs = PyMem_New(PyArray_ArrFuncs, 1);
    PyArray_InitArrFuncs(arrfuncs);
    arrfuncs->getitem = time360_getitem;
    arrfuncs->setitem = time360_setitem;
    arrfuncs->copyswapn = time360_copyswapn;
    arrfuncs->copyswap = time360_copyswap;

    /* Must explicitly set all members or we risk a memory fault later. */
    dtype = PyObject_New(PyArray_Descr, &PyArrayDescr_Type);
    dtype->typeobj = time360Type;
    dtype->kind = 't';
    dtype->type = 't';
    dtype->byteorder = '=';
    /* XXX Trying NPY_NEEDS_PYAPI to see if that helps raise errors more
     * promptly/accurately.
     * Doesn't seem to help when errors are raised during setitem.
     */
    dtype->flags = NPY_USE_GETITEM | NPY_USE_SETITEM | NPY_NEEDS_INIT | NPY_NEEDS_PYAPI;
    dtype->type_num = -1; /* Set when registered */
    dtype->elsize = sizeof(datetime);
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

void
register_time360_dtype(PyObject *module, PyTypeObject *time360Type,
                       PyArray_Descr *timedelta_dtype)
{
    PyArray_Descr *dtype;
    int arg_types[3];

    dtype = create_dtype(time360Type);
    PyModule_AddObject(module, "time360", (PyObject *)dtype);

    // TODO: Move these variable declarations
    PyObject *numpy_module = PyImport_ImportModule("numpy");
    PyObject *numpy_dict = PyModule_GetDict(numpy_module);
    Py_DECREF(numpy_module);

    arg_types[0] = dtype->type_num;
    arg_types[1] = dtype->type_num;
    arg_types[2] = timedelta_dtype->type_num;
    PyUFunc_RegisterLoopForType(
        (PyUFuncObject *)PyDict_GetItemString(numpy_dict, "subtract"),
        dtype->type_num, time360_subtract_time360_ufunc, arg_types, NULL);
}
