#include <Python.h>

/* Allow access to the `datetime` module's C API via the subsequent
 * use of PyDateTime_IMPORT.
 */
#include <datetime.h>

#include "cftime.h"

typedef struct {
    PyObject_HEAD
    datetime time;
} Time360;

static PyTypeObject Time360Type;

/*
 * Static Time360 definition
 */

#define YEAR(o) (o)->time.year
#define MONTH(o) (o)->time.month
#define DAY(o) (o)->time.day
#define HOUR(o) (o)->time.hour
#define MINUTE(o) (o)->time.minute
#define SECOND(o) (o)->time.second
#define MICROSECOND(o) (o)->time.microsecond

static PyObject *
new_Time360_from_datetime(datetime time)
{
    Time360 *self;

    self = (Time360 *)(Time360Type.tp_alloc(&Time360Type, 0));
    if (self != NULL) {
        YEAR(self) = time.year;
        MONTH(self) = time.month;
        DAY(self) = time.day;
        HOUR(self) = time.hour;
        MINUTE(self) = time.minute;
        SECOND(self) = time.second;
        MICROSECOND(self) = time.microsecond;
    }
    return (PyObject *)self;

}

static PyObject *
Time360_repr(Time360 *self)
{
    /* XXX: If the type name is too long this could overflow. */
    char buffer[1000];
    const char *type_name = Py_TYPE(self)->tp_name;

    if (MICROSECOND(self)) {
        PyOS_snprintf(buffer, sizeof(buffer),
                      "%s(%d, %d, %d, %d, %d, %d, %d)",
                      type_name,
                      YEAR(self), MONTH(self) + 1, DAY(self) + 1,
                      HOUR(self), MINUTE(self),
                      SECOND(self),
                      MICROSECOND(self));
    }
    else if (SECOND(self)) {
        PyOS_snprintf(buffer, sizeof(buffer),
                      "%s(%d, %d, %d, %d, %d, %d)",
                      type_name,
                      YEAR(self), MONTH(self) + 1, DAY(self) + 1,
                      HOUR(self), MINUTE(self),
                      SECOND(self));
    }
    else if (HOUR(self) || MINUTE(self)) {
        PyOS_snprintf(buffer, sizeof(buffer),
                      "%s(%d, %d, %d, %d, %d)",
                      type_name,
                      YEAR(self), MONTH(self) + 1, DAY(self) + 1,
                      HOUR(self), MINUTE(self));
    }
    else {
        PyOS_snprintf(buffer, sizeof(buffer),
                      "%s(%d, %d, %d)",
                      type_name,
                      YEAR(self), MONTH(self) + 1, DAY(self) + 1);
    }
    return PyString_FromString(buffer);
}

static PyObject *
Time360_str(Time360 *self)
{
    char buffer[32];

    if (MICROSECOND(self)) {
        PyOS_snprintf(buffer, sizeof(buffer),
                      "%04d-%02d-%02d %02d:%02d:%02d.%06d",
                      YEAR(self), MONTH(self) + 1, DAY(self) + 1,
                      HOUR(self), MINUTE(self),
                      SECOND(self), MICROSECOND(self));
    }
    else if (SECOND(self)) {
        PyOS_snprintf(buffer, sizeof(buffer), "%04d-%02d-%02d %02d:%02d:%02d",
                      YEAR(self), MONTH(self) + 1, DAY(self) + 1,
                      HOUR(self), MINUTE(self),
                      SECOND(self));
    }
    else {
        PyOS_snprintf(buffer, sizeof(buffer), "%04d-%02d-%02d %02d:%02d",
                      YEAR(self), MONTH(self) + 1, DAY(self) + 1,
                      HOUR(self), MINUTE(self));
    }
    return PyString_FromString(buffer);
}

static int
Time360_init(Time360 *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"year", "month", "day", "hour", "minute",
                             "second", "microsecond", NULL};

    /* Set default values for optional arguments */
    HOUR(self) = 0;
    MINUTE(self) = 0;
    SECOND(self) = 0;
    MICROSECOND(self) = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iii|iiii", kwlist,
                                     &YEAR(self), &MONTH(self), &DAY(self),
                                     &HOUR(self), &MINUTE(self), &SECOND(self),
                                     &MICROSECOND(self)))
        return -1;

    /* TODO: Validate arguments */
    if (MONTH(self) < 1 || MONTH(self) > 12) {
        PyErr_SetString(PyExc_ValueError, "month must be in range 1-12");
        return -1;
    }
    MONTH(self) -= 1;

    if (DAY(self) < 1 || DAY(self) > 30) {
        PyErr_SetString(PyExc_ValueError, "day must be in range 1-30");
        return -1;
    }
    DAY(self) -= 1;

    return 0;
}

static PyObject *
Time360_add(PyObject *o1, PyObject *o2)
{
    if (PyObject_TypeCheck(o1, &Time360Type)) {
        /* Time360 + ??? */
        if (PyDelta_Check(o2)) {
            /* Time360 + datetime.timedelta */
            datetime result;
            timedelta delta;

            memcpy(&delta, &((PyDateTime_Delta *)o2)->days, sizeof(delta));
            result = time360_add_timedelta(((Time360 *)o1)->time, delta);
            return new_Time360_from_datetime(result);
        }
    } else {
        /* ??? + Time360 */
        if (PyDelta_Check(o1)) {
            /* datetime.timedelta + Time360 */
            datetime result;
            timedelta delta;

            memcpy(&delta, &((PyDateTime_Delta *)o1)->days, sizeof(delta));
            result = time360_add_timedelta(((Time360 *)o2)->time, delta);
            return new_Time360_from_datetime(result);
        }
    }
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}

static PyObject *
Time360_subtract(PyObject *o1, PyObject *o2)
{
    PyObject *result = Py_NotImplemented;

    if (PyObject_TypeCheck(o1, &Time360Type)) {
        /* Time360 - ??? */
        if (PyObject_TypeCheck(o2, &Time360Type)) {
            /* Time360 - Time360 */
            Time360 *t1, *t2;
            int days, seconds, microseconds;

            t1 = (Time360 *)o1;
            t2 = (Time360 *)o2;

            /* TODO: Consider overflow of days from large year deltas. */
            days = (YEAR(t1) - YEAR(t2)) * 360;
            days += (MONTH(t1) - MONTH(t2)) * 30;
            days += DAY(t1) - DAY(t2);

            seconds = (HOUR(t1) - HOUR(t2)) * 3600;
            seconds += (MINUTE(t1) - MINUTE(t2)) * 60;
            seconds += (SECOND(t1) - SECOND(t2));

            microseconds = MICROSECOND(t1) - MICROSECOND(t2);

            result = PyDelta_FromDSU(days, seconds, microseconds);
        } else if (PyDelta_Check(o2)) {
            /* Time360 - datetime.timedelta */
            datetime dt;
            timedelta delta;

            memcpy(&delta, &((PyDateTime_Delta *)o2)->days, sizeof(delta));
            dt = time360_subtract_timedelta(((Time360 *)o1)->time, delta);
            result = new_Time360_from_datetime(dt);
        }
    }
    if (result == Py_NotImplemented)
        Py_INCREF(Py_NotImplemented);
    return result;
}

static int
Time360_Compare(Time360 *t1, Time360 *t2)
{
    return time360_compare(&t1->time, &t2->time);
}

static long
Time360_Hash(Time360 *t)
{
    PyObject *temp;
    long hash = -1;

    /* TODO: Caching (as for the datetime classes) */
    temp = PyString_FromStringAndSize((char *)&t->time, sizeof(datetime));
    if (temp != NULL) {
        hash = PyObject_Hash(temp);
        Py_DECREF(temp);
    }
    return hash;
}

static PyObject *
Time360_reduce(Time360 *self, PyObject *arg)
{
    PyObject *args;

    args = Py_BuildValue("(iiiiiii)",
                         YEAR(self), MONTH(self) + 1, DAY(self) + 1,
                         HOUR(self), MINUTE(self), SECOND(self),
                         MICROSECOND(self));
    return Py_BuildValue("(ON)", Py_TYPE(self), args);
}

static PyNumberMethods Time360_NumberMethods = {
    Time360_add,
    Time360_subtract,
};

static PyMethodDef Time360_methods[] = {
    {"__reduce__", (PyCFunction)Time360_reduce, METH_NOARGS, "TODO"},
    {NULL}
};

static PyTypeObject Time360Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "iris.time.Time360",                                /* tp_name */
    sizeof(Time360),                                    /* tp_basicsize */
    0,                                                  /* tp_itemsize */
    0,                                                  /* tp_dealloc */
    0,                                                  /* tp_print */
    0,                                                  /* tp_getattr */
    0,                                                  /* tp_setattr */
    (cmpfunc)Time360_Compare,                           /* tp_compare */
    (reprfunc)Time360_repr,                             /* tp_repr */
    &Time360_NumberMethods,                             /* tp_as_number */
    0,                                                  /* tp_as_sequence */
    0,                                                  /* tp_as_mapping */
    (hashfunc)Time360_Hash,                             /* tp_hash */
    0,                                                  /* tp_call */
    (reprfunc)Time360_str,                              /* tp_str */
    PyObject_GenericGetAttr,                            /* tp_getattro */
    0,                                                  /* tp_setattro */
    0,                                                  /* tp_as_buffer */
    //Py_TPFLAGS_DEFAULT | Py_TPFLAGS_CHECKTYPES | Py_TPFLAGS_BASETYPE,                                /* tp_flags */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_CHECKTYPES,         /* tp_flags */
    0, //date_doc,                                           /* tp_doc */
    0,                                                  /* tp_traverse */
    0,                                                  /* tp_clear */
    0, //(richcmpfunc)date_richcompare,                      /* tp_richcompare */
    0,                                                  /* tp_weaklistoffset */
    0,                                                  /* tp_iter */
    0,                                                  /* tp_iternext */
    Time360_methods,                                    /* tp_methods */
    0,                                                  /* tp_members */
    0, //date_getset,                                        /* tp_getset */
    0,                                                  /* tp_base */
    0,                                                  /* tp_dict */
    0,                                                  /* tp_descr_get */
    0,                                                  /* tp_descr_set */
    0,                                                  /* tp_dictoffset */
    (initproc)Time360_init,                             /* tp_init */
    0,                                                  /* tp_alloc */
    0, //date_new,                                           /* tp_new */
    0,                                                  /* tp_free */
};

/*
 * Module initialisation
 */

PyTypeObject *
register_Time360(PyObject *module)
{
    PyDateTime_IMPORT;

    Time360Type.tp_new = PyType_GenericNew;
    if (PyType_Ready(&Time360Type) < 0)
        return NULL;

    PyModule_AddObject(module, "Time360", (PyObject *)&Time360Type);

    return &Time360Type;
}
