#include <stddef.h>
#include <stdint.h>

#include <Python.h>

#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>


typedef struct {
    int32_t year;
    uint8_t month;
    uint8_t day;
} dt360;

static PyArray_ArrFuncs dt360_arrfuncs;

typedef struct { char c; dt360 r; } align_test;

static PyObject *date360_type;

// Should this be a static global?
static PyArray_Descr *dt360_descr;

// XXX: For debug use only
void dump(char *prefix, PyObject *object)
{
    PyObject *str = PyObject_Str(object);
    printf("%s: %s\n", prefix, PyString_AsString(str));
    Py_DECREF(str);
}

///////////////////////////////////////////////////////////
//
// PyArray_ArrFuncs
//

static PyObject *dt360_getitem(void *data, void *arr)
{
    // TODO: Consider "misaligned and/or swapped" arrays
    PyObject *result;
    dt360 *item = (dt360 *)data;

    result = PyObject_CallFunction(date360_type, "iii", item->year, item->month,
                                   item->day);
    return result;
}

static int dt360_setitem(PyObject *item, void *data, void *arr)
{
    dt360 d;

    d.year = PyInt_AsLong(PyObject_GetAttrString(item, "year"));
    d.month = PyInt_AsLong(PyObject_GetAttrString(item, "month"));
    d.day = PyInt_AsLong(PyObject_GetAttrString(item, "day"));

    if (arr == NULL || PyArray_ISBEHAVED(arr)) {
        *((dt360 *)data) = d;
    } else {
        PyErr_SetString(PyExc_ValueError, "setitem on misbehaved array");
        return -1;
    }

    return 0;
}

static void dt360_copyswap(char *dest, char *src, int swap, void *NPY_UNUSED(arr))
{
    PyArray_Descr *descr;
    descr = PyArray_DescrFromType(NPY_INT32);
    descr->f->copyswapn(dest, sizeof(int32_t), src, sizeof(int32_t),
                        1, swap, NULL);
    Py_DECREF(descr);
    descr = PyArray_DescrFromType(NPY_UINT8);
    descr->f->copyswapn(dest + sizeof(int32_t), sizeof(uint8_t),
                        src + sizeof(int32_t), sizeof(uint8_t), 2, swap, NULL);
    Py_DECREF(descr);
}

static void dt360_copyswapn(dt360 *dest, npy_intp dstride,
                      dt360 *src, npy_intp sstride,
                      npy_intp n, int swap, void *NPY_UNUSED(arr))
{
    PyArray_Descr *descr;
    descr = PyArray_DescrFromType(NPY_INT32);
    descr->f->copyswapn(&dest->year, dstride, &src->year, sstride,
                        n, swap, NULL);
    Py_DECREF(descr);
    descr = PyArray_DescrFromType(NPY_UINT8);
    descr->f->copyswapn(&dest->month, dstride, &src->month, sstride,
                        n, swap, NULL);
    descr->f->copyswapn(&dest->day, dstride, &src->day, sstride, n, swap, NULL);
    Py_DECREF(descr);
}

static npy_bool dt360_nonzero(dt360 *data, void *arr)
{
    return 1;
}

///////////////////////////////////////////////////////////
//
// ufuncs
//

dt360 dt360_subtract(dt360 d1, dt360 d2)
{
    return d1;  // TODO
}

#define BINARY_GEN_UFUNC(array_type, func_name, arg_type, ret_type)\
static void array_type##_##func_name##_ufunc(char** args, npy_intp* dimensions,\
                                   npy_intp* steps, void* data)\
{\
    char *ip1 = args[0], *ip2 = args[1], *op1 = args[2];\
    npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    for(i = 0; i < n; i++, ip1 += is1, ip2 += is2, op1 += os1)\
    {\
        const array_type in1 = *(array_type *)ip1;\
        const arg_type in2 = *(arg_type *)ip2;\
        *((ret_type *)op1) = array_type##_##func_name(in1, in2);\
    };\
};

#define BINARY_UFUNC(array_type, func_name, ret_type)\
    BINARY_GEN_UFUNC(array_type, func_name, dt360, ret_type)

BINARY_UFUNC(dt360, subtract, dt360)

///////////////////////////////////////////////////////////
//
// Module functions
//

static PyObject *
dt360_calendar_dtype(PyObject *module, PyObject *args, PyObject *kw)
{
    PyObject *month_lengths;
    int leap_year = 0;
    int leap_month = 0;     // This gives us no leap years by default.
    PyObject *metadata;
    PyArray_Descr *new_dtype;

    static char *kwlist[] = {"month_lengths", "leap_year", "leap_month", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kw, "O|ii", kwlist,
                                     &month_lengths, &leap_year, &leap_month))
        return NULL;

    metadata = Py_BuildValue("(Oii)", month_lengths, leap_year, leap_month);
    new_dtype = PyArray_DescrNewFromType(dt360_descr->type_num);
    new_dtype->metadata = metadata;
    return (PyObject *)new_dtype;
}

static PyMethodDef module_methods[] = {
    {"calendar_dtype", (PyCFunction)dt360_calendar_dtype, METH_VARARGS | METH_KEYWORDS, "Return dtype appropriate for given calendar definition."},
    {NULL, NULL}
};

///////////////////////////////////////////////////////////
//
// Module initialisation
//

PyMODINIT_FUNC initdt360(void)
{
    PyObject *m;        // a module object

    m = Py_InitModule3("dt360", module_methods,
                       "TODO ...");
    if (m == NULL)
        return;

    // TODO: How to handle error during this initialisation function?

    // Find our scalar class
    PyObject *date_module = PyImport_ImportModule("iris.date");
    date360_type = PyObject_GetAttrString(date_module, "date360");
    // TODO: Check it's really a type?
    Py_DECREF(date_module);

    // Ensure NumPy is initialised - otherwise things like
    // PyArray_InitArrFuncs will bomb out with a memory fault.
    import_array();
    import_umath();

    // Define the standard array functions for our dtype.
    PyArray_InitArrFuncs(&dt360_arrfuncs);
    dt360_arrfuncs.getitem = dt360_getitem;
    dt360_arrfuncs.setitem = dt360_setitem;
    dt360_arrfuncs.copyswapn = dt360_copyswapn;
    dt360_arrfuncs.copyswap = dt360_copyswap;
    dt360_arrfuncs.nonzero = dt360_nonzero;

    // Must explicitly set all members or we risk a memory fault later.
    dt360_descr = PyObject_New(PyArray_Descr, &PyArrayDescr_Type);
    dt360_descr->typeobj = (PyTypeObject *)date360_type;
    dt360_descr->kind = 't';
    dt360_descr->type = 't';
    dt360_descr->byteorder = '=';
    dt360_descr->flags = NPY_USE_GETITEM | NPY_USE_SETITEM | NPY_NEEDS_INIT;
    dt360_descr->type_num = -1; // Set when registered
    dt360_descr->elsize = sizeof(dt360);
    dt360_descr->alignment = offsetof(align_test, r);
    dt360_descr->subarray = NULL;
    dt360_descr->fields = NULL;
    dt360_descr->names = NULL;
    dt360_descr->f = &dt360_arrfuncs;
    dt360_descr->metadata = NULL; // TODO: Is this how we can parameterise?
    // XXX: In newer versions of NumPy (1.7?) there is a new c_metadata
    // member too.
    // In numpy/core/src/multiarray/datetime.c:
    //  - create_datetime_dtype()
    //      Uses PyArray_DescrNewFromType to make a new PyArray_Descr
    //      based on an existing one and then overwrites the metadata.
    //  - parse_dtype_from_datetime_typestr()
    //      Creates a dtype (a la PyArray_Descr) from type string.
    //  - parse_datetime_metadata_from_metastr()
    //      Parses a string into the metadata.

    // TODO: This NumPy type number "should be stored and made available
    // by your module".
    // TODO: Clarify NumPy docs re. reference counting behaviour
    PyArray_RegisterDataType(dt360_descr);
    assert(dt360_descr->type_num != -1);

    PyModule_AddIntConstant(m, "dt360_typenum", dt360_descr->type_num);
    PyModule_AddObject(m, "dt360_descr", (PyObject *)dt360_descr);

    PyObject *numpy_module = PyImport_ImportModule("numpy");
    PyObject *numpy_dict = PyModule_GetDict(numpy_module);
    Py_DECREF(numpy_module);

#define REGISTER_UFUNC(array_type, name)\
    PyUFunc_RegisterLoopForType((PyUFuncObject *)PyDict_GetItemString(numpy_dict, #name),\
            array_type##_descr->type_num, array_type##_##name##_ufunc, arg_types, NULL);

    int arg_types[3];
    arg_types[0] = dt360_descr->type_num;
    arg_types[1] = dt360_descr->type_num;
    arg_types[2] = dt360_descr->type_num;
    REGISTER_UFUNC(dt360, subtract);
}
