/*  C implementation for the date/time type documented at
 *  http://www.zope.org/Members/fdrake/DateTimeWiki/FrontPage
 *  modified to use a 360 day calendar
 */

#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <modsupport.h>
#include <structmember.h>

#include <time.h>

#include <timefuncs.h>

#include "datetime360.h"

/* We require that C int be at least 32 bits, and use int virtually
 * everywhere.  In just a few cases we use a temp long, where a Python
 * API returns a C long.  In such cases, we have to ensure that the
 * final result fits in a C int (this can be an issue on 64-bit boxes).
 */
#if SIZEOF_INT < 4
#       error "datetime.c requires that C int have at least 32 bits"
#endif

#define MINYEAR 1
#define MAXYEAR 9999
#define MAXORDINAL 3599640 /* date(9999,12,30).toordinal() */

/* Nine decimal digits is easy to communicate, and leaves enough room
 * so that two delta days can be added w/o fear of overflowing a signed
 * 32-bit int, and with plenty of room left over to absorb any possible
 * carries from adding seconds.
 */
#define MAX_DELTA_DAYS 999999999

/* Rename the long macros in datetime.h to more reasonable short names. */
#define GET_YEAR                PyDateTime_GET_YEAR
#define GET_MONTH               PyDateTime_GET_MONTH
#define GET_DAY                 PyDateTime_GET_DAY
#define DATE_GET_HOUR           PyDateTime_DATE_GET_HOUR
#define DATE_GET_MINUTE         PyDateTime_DATE_GET_MINUTE
#define DATE_GET_SECOND         PyDateTime_DATE_GET_SECOND
#define DATE_GET_MICROSECOND    PyDateTime_DATE_GET_MICROSECOND

/* Date accessors for date and datetime. */
#define SET_YEAR(o, v)          (((o)->data[0] = ((v) & 0xff00) >> 8), \
                 ((o)->data[1] = ((v) & 0x00ff)))
#define SET_MONTH(o, v)         (PyDateTime_GET_MONTH(o) = (v))
#define SET_DAY(o, v)           (PyDateTime_GET_DAY(o) = (v))

/* Date/Time accessors for datetime. */
#define DATE_SET_HOUR(o, v)     (PyDateTime_DATE_GET_HOUR(o) = (v))
#define DATE_SET_MINUTE(o, v)   (PyDateTime_DATE_GET_MINUTE(o) = (v))
#define DATE_SET_SECOND(o, v)   (PyDateTime_DATE_GET_SECOND(o) = (v))
#define DATE_SET_MICROSECOND(o, v)      \
    (((o)->data[7] = ((v) & 0xff0000) >> 16), \
     ((o)->data[8] = ((v) & 0x00ff00) >> 8), \
     ((o)->data[9] = ((v) & 0x0000ff)))

/* Time accessors for time. */
#define TIME_GET_HOUR           PyDateTime_TIME_GET_HOUR
#define TIME_GET_MINUTE         PyDateTime_TIME_GET_MINUTE
#define TIME_GET_SECOND         PyDateTime_TIME_GET_SECOND
#define TIME_GET_MICROSECOND    PyDateTime_TIME_GET_MICROSECOND
#define TIME_SET_HOUR(o, v)     (PyDateTime_TIME_GET_HOUR(o) = (v))
#define TIME_SET_MINUTE(o, v)   (PyDateTime_TIME_GET_MINUTE(o) = (v))
#define TIME_SET_SECOND(o, v)   (PyDateTime_TIME_GET_SECOND(o) = (v))
#define TIME_SET_MICROSECOND(o, v)      \
    (((o)->data[3] = ((v) & 0xff0000) >> 16), \
     ((o)->data[4] = ((v) & 0x00ff00) >> 8), \
     ((o)->data[5] = ((v) & 0x0000ff)))

/* Delta accessors for timedelta. */
#define GET_TD_DAYS(o)          (((PyDateTime_Delta *)(o))->days)
#define GET_TD_SECONDS(o)       (((PyDateTime_Delta *)(o))->seconds)
#define GET_TD_MICROSECONDS(o)  (((PyDateTime_Delta *)(o))->microseconds)

#define SET_TD_DAYS(o, v)       ((o)->days = (v))
#define SET_TD_SECONDS(o, v)    ((o)->seconds = (v))
#define SET_TD_MICROSECONDS(o, v) ((o)->microseconds = (v))

/* p is a pointer to a time or a datetime object; HASTZINFO(p) returns
 * p->hastzinfo.
 */
#define HASTZINFO(p)            (((_PyDateTime_BaseTZInfo *)(p))->hastzinfo)

/* M is a char or int claiming to be a valid month.  The macro is equivalent
 * to the two-sided Python test
 *      1 <= M <= 12
 */
#define MONTH_IS_SANE(M) ((unsigned int)(M) - 1 < 12)

/* Forward declarations. */
static PyTypeObject PyDateTime360_DateType;
static PyTypeObject PyDateTime360_DateTimeType;

/* Pre-processor commands to access built-in datetime CAPI */
#define PyDateTime_DeltaType (*(PyDateTimeAPI->DeltaType)) 
#define PyDateTime_TimeType (*(PyDateTimeAPI->TimeType))
#define PyDateTime_TZInfoType (*(PyDateTimeAPI->TZInfoType))


/* ---------------------------------------------------------------------------
 * Math utilities.
 */

/* k = i+j overflows iff k differs in sign from both inputs,
 * iff k^i has sign bit set and k^j has sign bit set,
 * iff (k^i)&(k^j) has sign bit set.
 */
#define SIGNED_ADD_OVERFLOWED(RESULT, I, J) \
    ((((RESULT) ^ (I)) & ((RESULT) ^ (J))) < 0)

/* Compute Python divmod(x, y), returning the quotient and storing the
 * remainder into *r.  The quotient is the floor of x/y, and that's
 * the real point of this.  C will probably truncate instead (C99
 * requires truncation; C89 left it implementation-defined).
 * Simplification:  we *require* that y > 0 here.  That's appropriate
 * for all the uses made of it.  This simplifies the code and makes
 * the overflow case impossible (divmod(LONG_MIN, -1) is the only
 * overflow case).
 */
static int
divmod(int x, int y, int *r)
{
    int quo;

    assert(y > 0);
    quo = x / y;
    *r = x - quo * y;
    if (*r < 0) {
        --quo;
        *r += y;
    }
    assert(0 <= *r && *r < y);
    return quo;
}

/* ---------------------------------------------------------------------------
 * General calendrical helper functions
 */

static int
days_before_month(int year, int month)
{
    assert(month >= 1);
    assert(month <= 12);
    return (month-1)*30;
}

/* year -> number of days before January 1st of year.  Remember that we
 * start with year 1, so days_before_year(1) == 0.
 */
static int
days_before_year(int year)
{
    assert (year >= 0);
    return (year-1)*360;
}

/* ordinal -> year, month, day, considering 01-Jan-0001 as day 1. */
static void
ord_to_ymd(int ordinal, int *year, int *month, int *day)
{
    /* ordinal is a 1-based index, starting at 1-Jan-1. Life is much
     * clearer if we subtract 1 from ordinal first
     *
     *    D  M   Y            ordinal       n (ordinal-1)
     *    -- --- ----        ----------     ----------------
     *    30 Dec  000        -1             -2
     *    30 Dec  000         0             -1
     *      1 Jan  001         1              0          400-year boundary
     *     2 Jan  001         2              1
     *     3 Jan  001         3              2
     */

    int n, remainder;

    assert(ordinal >= 1);
    n = ordinal-1;

    *year = divmod(n, 360, &remainder) + 1;
    *month = divmod(remainder, 30, day) + 1;
    *day += 1;
}

/* year, month, day -> ordinal, considering 01-Jan-0001 as day 1. */
static int
ymd_to_ord(int year, int month, int day)
{
    return days_before_year(year) + days_before_month(year, month) + day;
}


/* ---------------------------------------------------------------------------
 * Range checkers.
 */

/* Check that date arguments are in range.  Return 0 if they are.  If they
 * aren't, raise ValueError and return -1.
 */
static int
check_date_args(int year, int month, int day)
{

    if (year < MINYEAR || year > MAXYEAR) {
        PyErr_SetString(PyExc_ValueError,
                        "year is out of range");
        return -1;
    }
    if (month < 1 || month > 12) {
        PyErr_SetString(PyExc_ValueError,
                        "month must be in 1..12");
        return -1;
    }
    if (day < 1 || day > 30) {
        PyErr_SetString(PyExc_ValueError,
                        "day must be in 1..30");
        return -1;
    }
    return 0;
}

/* Check that time arguments are in range.  Return 0 if they are.  If they
 * aren't, raise ValueError and return -1.
 */
static int
check_time_args(int h, int m, int s, int us)
{
    if (h < 0 || h > 23) {
        PyErr_SetString(PyExc_ValueError,
                        "hour must be in 0..23");
        return -1;
    }
    if (m < 0 || m > 59) {
        PyErr_SetString(PyExc_ValueError,
                        "minute must be in 0..59");
        return -1;
    }
    if (s < 0 || s > 59) {
        PyErr_SetString(PyExc_ValueError,
                        "second must be in 0..59");
        return -1;
    }
    if (us < 0 || us > 999999) {
        PyErr_SetString(PyExc_ValueError,
                        "microsecond must be in 0..999999");
        return -1;
    }
    return 0;
}

/* ---------------------------------------------------------------------------
 * Normalization utilities.
 */

/* One step of a mixed-radix conversion.  A "hi" unit is equivalent to
 * factor "lo" units.  factor must be > 0.  If *lo is less than 0, or
 * at least factor, enough of *lo is converted into "hi" units so that
 * 0 <= *lo < factor.  The input values must be such that int overflow
 * is impossible.
 */
static void
normalize_pair(int *hi, int *lo, int factor)
{
    assert(factor > 0);
    assert(lo != hi);
    if (*lo < 0 || *lo >= factor) {
        const int num_hi = divmod(*lo, factor, lo);
        const int new_hi = *hi + num_hi;
        assert(! SIGNED_ADD_OVERFLOWED(new_hi, *hi, num_hi));
        *hi = new_hi;
    }
    assert(0 <= *lo && *lo < factor);
}

/* Fiddle years (y), months (m), and days (d) so that
 *      1 <= *m <= 12
 *      1 <= *d <= 30
 * The input values must be such that the internals don't overflow.
 * The way this routine is used, we don't get close.
 */
static int
normalize_y_m_d(int *y, int *m, int *d)
{
    const int dim = 30;            /* # of days in month */

    if (*m < 1 || *m > 12) {
        --*m;
        normalize_pair(y, m, 12);
        ++*m;
        /* |y| can't be bigger than about
         * |original y| + |original m|/12 now.
         */
    }
    assert(1 <= *m && *m <= 12);

    /* Now only day can be out of bounds (year may also be out of bounds
     * for a datetime object, but we don't care about that here).
     * If day is out of bounds, what to do is arguable, but at least the
     * method here is principled and explainable.
     */
    if (*d < 1 || *d > dim) {
        /* Move day-1 days from the first of the month.  First try to
         * get off cheap if we're only one day out of range
         * (adjustments for timezone alone can't be worse than that).
         */
        if (*d == 0) {
            *d = 30;
            --*m;
            if (*m < 1){
                --*y;
                *m = 12;
            }
        }
        else if (*d == dim + 1) {
            /* move forward a day */
            ++*m;
            *d = 1;
            if (*m > 12) {
                *m = 1;
                ++*y;
            }
        }
        else {
            int ordinal = ymd_to_ord(*y, *m, *d);
            if (ordinal < 1 || ordinal > MAXORDINAL) {
                PyErr_SetString(PyExc_OverflowError,"date value out of range");
                return -1;
            } else {
                ord_to_ymd(ordinal, y, m, d);
            }
        }
    }
    assert(*m > 0);
    assert(*d > 0);
    if (MINYEAR <= *y && *y <= MAXYEAR)
        return 0;
    else
        PyErr_SetString(PyExc_OverflowError,"date value out of range");
        return -1;
}

/* Fiddle out-of-bounds months and days so that the result makes some kind
 * of sense.  The parameters are both inputs and outputs.  Returns < 0 on
 * failure, where failure means the adjusted year is out of bounds.
 */
static int
normalize_date(int *year, int *month, int *day)
{
    return normalize_y_m_d(year, month, day);
}

/* Force all the datetime fields into range.  The parameters are both
 * inputs and outputs.  Returns < 0 on error.
 */
static int
normalize_datetime(int *year, int *month, int *day,
                   int *hour, int *minute, int *second,
                   int *microsecond)
{
    normalize_pair(second, microsecond, 1000000);
    normalize_pair(minute, second, 60);
    normalize_pair(hour, minute, 60);
    normalize_pair(day, hour, 24);
    return normalize_date(year, month, day);
}

/* ---------------------------------------------------------------------------
 * Basic object allocation:  tp_alloc implementations.  These allocate
 * Python objects of the right size and type, and do the Python object-
 * initialization bit.  If there's not enough memory, they return NULL after
 * setting MemoryError.  All data members remain uninitialized trash.
 *
 * We abuse the tp_alloc "nitems" argument to communicate whether a tzinfo
 * member is needed.  This is ugly, imprecise, and possibly insecure.
 * tp_basicsize for the time and datetime types is set to the size of the
 * struct that has room for the tzinfo member, so subclasses in Python will
 * allocate enough space for a tzinfo member whether or not one is actually
 * needed.  That's the "ugly and imprecise" parts.  The "possibly insecure"
 * part is that PyType_GenericAlloc() (which subclasses in Python end up
 * using) just happens today to effectively ignore the nitems argument
 * when tp_itemsize is 0, which it is for these type objects.  If that
 * changes, perhaps the callers of tp_alloc slots in this file should
 * be changed to force a 0 nitems argument unless the type being allocated
 * is a base type implemented in this file (so that tp_alloc is time_alloc
 * or datetime_alloc below, which know about the nitems abuse).
 */

static PyObject *
datetime_alloc(PyTypeObject *type, Py_ssize_t aware)
{
    PyObject *self;

    self = (PyObject *)
        PyObject_MALLOC(aware ?
                        sizeof(PyDateTime_DateTime) :
                sizeof(_PyDateTime_BaseDateTime));
    if (self == NULL)
        return (PyObject *)PyErr_NoMemory();
    PyObject_INIT(self, type);
    return self;
}

/* ---------------------------------------------------------------------------
 * Helpers for setting object fields.  These work on pointers to the
 * appropriate base class.
 */

/* For date and datetime. */
static void
set_date_fields(PyDateTime_Date *self, int y, int m, int d)
{
    self->hashcode = -1;
    SET_YEAR(self, y);
    SET_MONTH(self, m);
    SET_DAY(self, d);
}

/* ---------------------------------------------------------------------------
 * Create various objects, mostly without range checking.
 */

/* Create a date instance with no range checking. */
static PyObject *
new_date_ex(int year, int month, int day, PyTypeObject *type)
{
    PyDateTime_Date *self;

    self = (PyDateTime_Date *) (type->tp_alloc(type, 0));
    if (self != NULL)
        set_date_fields(self, year, month, day);
    return (PyObject *) self;
}

#define new_date(year, month, day) \
    new_date_ex(year, month, day, &PyDateTime360_DateType)

/* Create a datetime instance with no range checking. */
static PyObject *
new_datetime_ex(int year, int month, int day, int hour, int minute,
             int second, int usecond, PyObject *tzinfo, PyTypeObject *type)
{
    PyDateTime_DateTime *self;
    char aware = tzinfo != Py_None;

    self = (PyDateTime_DateTime *) (type->tp_alloc(type, aware));
    if (self != NULL) {
        self->hastzinfo = aware;
        set_date_fields((PyDateTime_Date *)self, year, month, day);
        DATE_SET_HOUR(self, hour);
        DATE_SET_MINUTE(self, minute);
        DATE_SET_SECOND(self, second);
        DATE_SET_MICROSECOND(self, usecond);
        if (aware) {
            Py_INCREF(tzinfo);
            self->tzinfo = tzinfo;
        }
    }
    return (PyObject *)self;
}

#define new_datetime(y, m, d, hh, mm, ss, us, tzinfo)           \
    new_datetime_ex(y, m, d, hh, mm, ss, us, tzinfo,            \
                    &PyDateTime360_DateTimeType)

/* Create a time instance with no range checking. 
 *
 * Uses built-in new_time_ex via CAPI 
 * 
 * */

#define new_time(hh, mm, ss, us, tzinfo) (PyDateTimeAPI->Time_FromTime(hh, mm, ss, us, tzinfo, &PyDateTime_TimeType))


/* Create a timedelta instance.  Normalize the members iff normalize is
 * true.  Passing false is a speed optimization, if you know for sure
 * that seconds and microseconds are already in their proper ranges.  In any
 * case, raises OverflowError and returns NULL if the normalized days is out
 * of range). 
 *
 * Uses built-in new_delta_ex via CAPI.
 *
 */

#define new_delta(d, s, us, normalize) (PyDateTimeAPI->Delta_FromDelta(d, s, us, normalize, &PyDateTime_DeltaType))

/* ---------------------------------------------------------------------------
 * tzinfo helpers.
 */

/* Return tzinfo.methname(tzinfoarg), without any checking of results.
 * If tzinfo is None, returns None.
 */
static PyObject *
call_tzinfo_method(PyObject *tzinfo, char *methname, PyObject *tzinfoarg)
{
    PyObject *result;

    assert(tzinfo && methname && tzinfoarg);
    assert(check_tzinfo_subclass(tzinfo) >= 0);
    if (tzinfo == Py_None) {
        result = Py_None;
        Py_INCREF(result);
    }
    else
        result = PyObject_CallMethod(tzinfo, methname, "O", tzinfoarg);
    return result;
}

/* If self has a tzinfo member, return a BORROWED reference to it.  Else
 * return NULL, which is NOT AN ERROR.  There are no error returns here,
 * and the caller must not decref the result.
 */
static PyObject *
get_tzinfo_member(PyObject *self)
{
    PyObject *tzinfo = NULL;

    if (PyDateTime360_Check(self) && HASTZINFO(self))
        tzinfo = ((PyDateTime_DateTime *)self)->tzinfo;
    else if (PyTime_Check(self) && HASTZINFO(self))
        tzinfo = ((PyDateTime_Time *)self)->tzinfo;

    return tzinfo;
}

/* Call getattr(tzinfo, name)(tzinfoarg), and extract an int from the
 * result.  tzinfo must be an instance of the tzinfo class.  If the method
 * returns None, this returns 0 and sets *none to 1.  If the method doesn't
 * return None or timedelta, TypeError is raised and this returns -1.  If it
 * returnsa timedelta and the value is out of range or isn't a whole number
 * of minutes, ValueError is raised and this returns -1.
 * Else *none is set to 0 and the integer method result is returned.
 */
static int
call_utc_tzinfo_method(PyObject *tzinfo, char *name, PyObject *tzinfoarg,
                       int *none)
{
    PyObject *u;
    int result = -1;

    assert(tzinfo != NULL);
    assert(PyTZInfo360_Check(tzinfo));
    assert(tzinfoarg != NULL);

    *none = 0;
    u = call_tzinfo_method(tzinfo, name, tzinfoarg);
    if (u == NULL)
        return -1;

    else if (u == Py_None) {
        result = 0;
        *none = 1;
    }
    else if (PyDelta_Check(u)) {
        const int days = GET_TD_DAYS(u);
        if (days < -1 || days > 0)
            result = 24*60;             /* trigger ValueError below */
        else {
            /* next line can't overflow because we know days
             * is -1 or 0 now
             */
            int ss = days * 24 * 3600 + GET_TD_SECONDS(u);
            result = divmod(ss, 60, &ss);
            if (ss || GET_TD_MICROSECONDS(u)) {
                PyErr_Format(PyExc_ValueError,
                             "tzinfo.%s() must return a "
                             "whole number of minutes",
                             name);
                result = -1;
            }
        }
    }
    else {
        PyErr_Format(PyExc_TypeError,
                     "tzinfo.%s() must return None or "
                     "timedelta, not '%s'",
                     name, Py_TYPE(u)->tp_name);
    }

    Py_DECREF(u);
    if (result < -1439 || result > 1439) {
        PyErr_Format(PyExc_ValueError,
                     "tzinfo.%s() returned %d; must be in "
                     "-1439 .. 1439",
                     name, result);
        result = -1;
    }
    return result;
}

/* Call tzinfo.utcoffset(tzinfoarg), and extract an integer from the
 * result.  tzinfo must be an instance of the tzinfo class.  If utcoffset()
 * returns None, call_utcoffset returns 0 and sets *none to 1.  If uctoffset()
 * doesn't return None or timedelta, TypeError is raised and this returns -1.
 * If utcoffset() returns an invalid timedelta (out of range, or not a whole
 * # of minutes), ValueError is raised and this returns -1.  Else *none is
 * set to 0 and the offset is returned (as int # of minutes east of UTC).
 */
static int
call_utcoffset(PyObject *tzinfo, PyObject *tzinfoarg, int *none)
{
    return call_utc_tzinfo_method(tzinfo, "utcoffset", tzinfoarg, none);
}

/* Call tzinfo.tzname(tzinfoarg), and return the result.  tzinfo must be
 * an instance of the tzinfo class or None.  If tzinfo isn't None, and
 * tzname() doesn't return None or a string, TypeError is raised and this
 * returns NULL.
 */
static PyObject *
call_tzname(PyObject *tzinfo, PyObject *tzinfoarg)
{
    PyObject *result;

    assert(tzinfo != NULL);
    assert(check_tzinfo_subclass(tzinfo) >= 0);
    assert(tzinfoarg != NULL);

    if (tzinfo == Py_None) {
        result = Py_None;
        Py_INCREF(result);
    }
    else
        result = PyObject_CallMethod(tzinfo, "tzname", "O", tzinfoarg);

    if (result != NULL && result != Py_None && ! PyString_Check(result)) {
        PyErr_Format(PyExc_TypeError, "tzinfo.tzname() must "
                     "return None or a string, not '%s'",
                     Py_TYPE(result)->tp_name);
        Py_DECREF(result);
        result = NULL;
    }
    return result;
}

typedef enum {
              /* an exception has been set; the caller should pass it on */
          OFFSET_ERROR,

          /* type isn't date, datetime, or time subclass */
          OFFSET_UNKNOWN,

          /* date,
           * datetime with !hastzinfo
           * datetime with None tzinfo,
           * datetime where utcoffset() returns None
           * time with !hastzinfo
           * time with None tzinfo,
           * time where utcoffset() returns None
           */
          OFFSET_NAIVE,

          /* time or datetime where utcoffset() doesn't return None */
          OFFSET_AWARE
} naivety;

/* Classify an object as to whether it's naive or offset-aware.  See
 * the "naivety" typedef for details.  If the type is aware, *offset is set
 * to minutes east of UTC (as returned by the tzinfo.utcoffset() method).
 * If the type is offset-naive (or unknown, or error), *offset is set to 0.
 * tzinfoarg is the argument to pass to the tzinfo.utcoffset() method.
 */
static naivety
classify_utcoffset(PyObject *op, PyObject *tzinfoarg, int *offset)
{
    int none;
    PyObject *tzinfo;

    assert(tzinfoarg != NULL);
    *offset = 0;
    tzinfo = get_tzinfo_member(op);     /* NULL means no tzinfo, not error */
    if (tzinfo == Py_None)
        return OFFSET_NAIVE;
    if (tzinfo == NULL) {
        /* note that a datetime passes the PyDate360_Check test */
        return (PyTime_Check(op) || PyDate360_Check(op)) ?
               OFFSET_NAIVE : OFFSET_UNKNOWN;
    }
    *offset = call_utcoffset(tzinfo, tzinfoarg, &none);
    if (*offset == -1 && PyErr_Occurred())
        return OFFSET_ERROR;
    return none ? OFFSET_NAIVE : OFFSET_AWARE;
}

/* Classify two objects as to whether they're naive or offset-aware.
 * This isn't quite the same as calling classify_utcoffset() twice:  for
 * binary operations (comparison and subtraction), we generally want to
 * ignore the tzinfo members if they're identical.  This is by design,
 * so that results match "naive" expectations when mixing objects from a
 * single timezone.  So in that case, this sets both offsets to 0 and
 * both naiveties to OFFSET_NAIVE.
 * The function returns 0 if everything's OK, and -1 on error.
 */
static int
classify_two_utcoffsets(PyObject *o1, int *offset1, naivety *n1,
                        PyObject *tzinfoarg1,
                        PyObject *o2, int *offset2, naivety *n2,
                        PyObject *tzinfoarg2)
{
    if (get_tzinfo_member(o1) == get_tzinfo_member(o2)) {
        *offset1 = *offset2 = 0;
        *n1 = *n2 = OFFSET_NAIVE;
    }
    else {
        *n1 = classify_utcoffset(o1, tzinfoarg1, offset1);
        if (*n1 == OFFSET_ERROR)
            return -1;
        *n2 = classify_utcoffset(o2, tzinfoarg2, offset2);
        if (*n2 == OFFSET_ERROR)
            return -1;
    }
    return 0;
}

/* repr is like "someclass(arg1, arg2)".  If tzinfo isn't None,
 * stuff
 *     ", tzinfo=" + repr(tzinfo)
 * before the closing ")".
 */
static PyObject *
append_keyword_tzinfo(PyObject *repr, PyObject *tzinfo)
{
    PyObject *temp;

    assert(PyString_Check(repr));
    assert(tzinfo);
    if (tzinfo == Py_None)
        return repr;
    /* Get rid of the trailing ')'. */
    assert(PyString_AsString(repr)[PyString_Size(repr)-1] == ')');
    temp = PyString_FromStringAndSize(PyString_AsString(repr),
                                      PyString_Size(repr) - 1);
    Py_DECREF(repr);
    if (temp == NULL)
        return NULL;
    repr = temp;

    /* Append ", tzinfo=". */
    PyString_ConcatAndDel(&repr, PyString_FromString(", tzinfo="));

    /* Append repr(tzinfo). */
    PyString_ConcatAndDel(&repr, PyObject_Repr(tzinfo));

    /* Add a closing paren. */
    PyString_ConcatAndDel(&repr, PyString_FromString(")"));
    return repr;
}

/* ---------------------------------------------------------------------------
 * String format helpers.
 */

/* Add an hours & minutes UTC offset string to buf.  buf has no more than
 * buflen bytes remaining.  The UTC offset is gotten by calling
 * tzinfo.uctoffset(tzinfoarg).  If that returns None, \0 is stored into
 * *buf, and that's all.  Else the returned value is checked for sanity (an
 * integer in range), and if that's OK it's converted to an hours & minutes
 * string of the form
 *   sign HH sep MM
 * Returns 0 if everything is OK.  If the return value from utcoffset() is
 * bogus, an appropriate exception is set and -1 is returned.
 */
static int
format_utcoffset(char *buf, size_t buflen, const char *sep,
                PyObject *tzinfo, PyObject *tzinfoarg)
{
    int offset;
    int hours;
    int minutes;
    char sign;
    int none;

    assert(buflen >= 1);

    offset = call_utcoffset(tzinfo, tzinfoarg, &none);
    if (offset == -1 && PyErr_Occurred())
        return -1;
    if (none) {
        *buf = '\0';
        return 0;
    }
    sign = '+';
    if (offset < 0) {
        sign = '-';
        offset = - offset;
    }
    hours = divmod(offset, 60, &minutes);
    PyOS_snprintf(buf, buflen, "%c%02d%s%02d", sign, hours, sep, minutes);
    return 0;
}

static PyObject *
make_freplacement(PyObject *object)
{
    char freplacement[64];
    if (PyTime_Check(object))
        sprintf(freplacement, "%06d", TIME_GET_MICROSECOND(object));
    else if (PyDateTime360_Check(object))
        sprintf(freplacement, "%06d", DATE_GET_MICROSECOND(object));
    else
        sprintf(freplacement, "%06d", 0);

    return PyString_FromStringAndSize(freplacement, strlen(freplacement));
}

// TODO: Might be simplest to convert this to use named item format
// substitution, but it wouldn't be that efficient.
/* I sure don't want to reproduce the strftime code from the time module,
 * so this imports the module and calls it.  All the hair is due to
 * giving special meanings to the %z, %Z and %f format codes via a
 * preprocessing step on the format string.
 * tzinfoarg is the argument to pass to the object's tzinfo method, if
 * needed.
 */
static PyObject *
wrap_strftime(PyObject *object, const char *format, size_t format_len,
                PyObject *timetuple, PyObject *tzinfoarg)
{
    PyObject *result = NULL;            /* guilty until proved innocent */

    PyObject *zreplacement = NULL;      /* py string, replacement for %z */
    PyObject *Zreplacement = NULL;      /* py string, replacement for %Z */
    PyObject *freplacement = NULL;      /* py string, replacement for %f */

    const char *pin;            /* pointer to next char in input format */
    char ch;                    /* next char in input format */

    PyObject *newfmt = NULL;            /* py string, the output format */
    char *pnew;         /* pointer to available byte in output format */
    size_t totalnew;            /* number bytes total in output format buffer,
                               exclusive of trailing \0 */
    size_t usednew;     /* number bytes used so far in output format buffer */

    const char *ptoappend;      /* ptr to string to append to output buffer */
    size_t ntoappend;           /* # of bytes to append to output buffer */

    assert(object && format && timetuple);

    /* Give up if the year is before 1900.
     * Python strftime() plays games with the year, and different
     * games depending on whether envar PYTHON2K is set.  This makes
     * years before 1900 a nightmare, even if the platform strftime
     * supports them (and not all do).
     * We could get a lot farther here by avoiding Python's strftime
     * wrapper and calling the C strftime() directly, but that isn't
     * an option in the Python implementation of this module.
     */
    {
        long year;
        PyObject *pyyear = PySequence_GetItem(timetuple, 0);
        if (pyyear == NULL) return NULL;
        assert(PyInt_Check(pyyear));
        year = PyInt_AsLong(pyyear);
        Py_DECREF(pyyear);
        if (year < 1900) {
            PyErr_Format(PyExc_ValueError, "year=%ld is before "
                         "1900; the datetime strftime() "
                         "methods require year >= 1900",
                         year);
            return NULL;
        }
    }

    /* Scan the input format, looking for %z/%Z/%f escapes, building
     * a new format.  Since computing the replacements for those codes
     * is expensive, don't unless they're actually used.
     */
    if (format_len > INT_MAX - 1) {
        PyErr_NoMemory();
        goto Done;
    }

    totalnew = format_len + 1;          /* realistic if no %z/%Z/%f */
    newfmt = PyString_FromStringAndSize(NULL, totalnew);
    if (newfmt == NULL) goto Done;
    pnew = PyString_AsString(newfmt);
    usednew = 0;

    pin = format;
    while ((ch = *pin++) != '\0') {
        if (ch != '%') {
            ptoappend = pin - 1;
            ntoappend = 1;
        }
        else if ((ch = *pin++) == '\0') {
            /* There's a lone trailing %; doesn't make sense. */
            PyErr_SetString(PyExc_ValueError, "strftime format "
                            "ends with raw %");
            goto Done;
        }
        /* A % has been seen and ch is the character after it. */
        else if (ch == 'z') {
            if (zreplacement == NULL) {
                /* format utcoffset */
                char buf[100];
                PyObject *tzinfo = get_tzinfo_member(object);
                zreplacement = PyString_FromString("");
                if (zreplacement == NULL) goto Done;
                if (tzinfo != Py_None && tzinfo != NULL) {
                    assert(tzinfoarg != NULL);
                    if (format_utcoffset(buf,
                                         sizeof(buf),
                                         "",
                                         tzinfo,
                                         tzinfoarg) < 0)
                        goto Done;
                    Py_DECREF(zreplacement);
                    zreplacement = PyString_FromString(buf);
                    if (zreplacement == NULL) goto Done;
                }
            }
            assert(zreplacement != NULL);
            ptoappend = PyString_AS_STRING(zreplacement);
            ntoappend = PyString_GET_SIZE(zreplacement);
        }
        else if (ch == 'Z') {
            /* format tzname */
            if (Zreplacement == NULL) {
                PyObject *tzinfo = get_tzinfo_member(object);
                Zreplacement = PyString_FromString("");
                if (Zreplacement == NULL) goto Done;
                if (tzinfo != Py_None && tzinfo != NULL) {
                    PyObject *temp;
                    assert(tzinfoarg != NULL);
                    temp = call_tzname(tzinfo, tzinfoarg);
                    if (temp == NULL) goto Done;
                    if (temp != Py_None) {
                        assert(PyString_Check(temp));
                        /* Since the tzname is getting
                         * stuffed into the format, we
                         * have to double any % signs
                         * so that strftime doesn't
                         * treat them as format codes.
                         */
                        Py_DECREF(Zreplacement);
                        Zreplacement = PyObject_CallMethod(
                            temp, "replace",
                            "ss", "%", "%%");
                        Py_DECREF(temp);
                        if (Zreplacement == NULL)
                            goto Done;
                        if (!PyString_Check(Zreplacement)) {
                            PyErr_SetString(PyExc_TypeError, "tzname.replace() did not return a string");
                            goto Done;
                        }
                    }
                    else
                        Py_DECREF(temp);
                }
            }
            assert(Zreplacement != NULL);
            ptoappend = PyString_AS_STRING(Zreplacement);
            ntoappend = PyString_GET_SIZE(Zreplacement);
        }
        else if (ch == 'f') {
            /* format microseconds */
            if (freplacement == NULL) {
                freplacement = make_freplacement(object);
                if (freplacement == NULL)
                    goto Done;
            }
            assert(freplacement != NULL);
            assert(PyString_Check(freplacement));
            ptoappend = PyString_AS_STRING(freplacement);
            ntoappend = PyString_GET_SIZE(freplacement);
        }
        else {
            /* percent followed by neither z nor Z */
            ptoappend = pin - 2;
            ntoappend = 2;
        }

        /* Append the ntoappend chars starting at ptoappend to
         * the new format.
         */
        assert(ptoappend != NULL);
        assert(ntoappend >= 0);
        if (ntoappend == 0)
            continue;
        while (usednew + ntoappend > totalnew) {
            size_t bigger = totalnew << 1;
            if ((bigger >> 1) != totalnew) { /* overflow */
                PyErr_NoMemory();
                goto Done;
            }
            if (_PyString_Resize(&newfmt, bigger) < 0)
                goto Done;
            totalnew = bigger;
            pnew = PyString_AsString(newfmt) + usednew;
        }
        memcpy(pnew, ptoappend, ntoappend);
        pnew += ntoappend;
        usednew += ntoappend;
        assert(usednew <= totalnew);
    }  /* end while() */

    if (_PyString_Resize(&newfmt, usednew) < 0)
        goto Done;
    {
        PyObject *time = PyImport_ImportModuleNoBlock("time");
        if (time == NULL)
            goto Done;
        result = PyObject_CallMethod(time, "strftime", "OO",
                                     newfmt, timetuple);
        Py_DECREF(time);
    }
 Done:
    Py_XDECREF(freplacement);
    Py_XDECREF(zreplacement);
    Py_XDECREF(Zreplacement);
    Py_XDECREF(newfmt);
    return result;
}

/* ---------------------------------------------------------------------------
 * Miscellaneous helpers.
 */

/* For obscure reasons, we need to use tp_richcompare instead of tp_compare.
 * The comparisons here all most naturally compute a cmp()-like result.
 * This little helper turns that into a bool result for rich comparisons.
 */
static PyObject *
diff_to_bool(int diff, int op)
{
    PyObject *result;
    int istrue;

    switch (op) {
        case Py_EQ: istrue = diff == 0; break;
        case Py_NE: istrue = diff != 0; break;
        case Py_LE: istrue = diff <= 0; break;
        case Py_GE: istrue = diff >= 0; break;
        case Py_LT: istrue = diff < 0; break;
        case Py_GT: istrue = diff > 0; break;
        default:
            assert(! "op unknown");
            istrue = 0; /* To shut up compiler */
    }
    result = istrue ? Py_True : Py_False;
    Py_INCREF(result);
    return result;
}

/* Raises a "can't compare" TypeError and returns NULL. */
static PyObject *
cmperror(PyObject *a, PyObject *b)
{
    PyErr_Format(PyExc_TypeError,
                 "can't compare %s to %s",
                 Py_TYPE(a)->tp_name, Py_TYPE(b)->tp_name);
    return NULL;
}

/* ---------------------------------------------------------------------------
 * Class implementations.
 */

/*
 * PyDateTime360_DateType implementation. Use built-in C PyDateTime_Date struct.
 */

/* Accessor properties. */

static PyObject *
date_year(PyDateTime_Date *self, void *unused)
{
    return PyInt_FromLong(GET_YEAR(self));
}

static PyObject *
date_month(PyDateTime_Date *self, void *unused)
{
    return PyInt_FromLong(GET_MONTH(self));
}

static PyObject *
date_day(PyDateTime_Date *self, void *unused)
{
    return PyInt_FromLong(GET_DAY(self));
}

static PyGetSetDef date_getset[] = {
    {"year",        (getter)date_year},
    {"month",       (getter)date_month},
    {"day",         (getter)date_day},
    {NULL}
};

/* Constructors. */

static char *date_kws[] = {"year", "month", "day", NULL};

static PyObject *
date_new(PyTypeObject *type, PyObject *args, PyObject *kw)
{
    PyObject *self = NULL;
    PyObject *state;
    int year;
    int month;
    int day;

    /* Check for invocation from pickle with __getstate__ state */
    if (PyTuple_GET_SIZE(args) == 1 &&
        PyString_Check(state = PyTuple_GET_ITEM(args, 0)) &&
        PyString_GET_SIZE(state) == _PyDateTime_DATE_DATASIZE &&
        MONTH_IS_SANE(PyString_AS_STRING(state)[2]))
    {
        PyDateTime_Date *me;

        me = (PyDateTime_Date *) (type->tp_alloc(type, 0));
        if (me != NULL) {
            char *pdata = PyString_AS_STRING(state);
            memcpy(me->data, pdata, _PyDateTime_DATE_DATASIZE);
            me->hashcode = -1;
        }
        return (PyObject *)me;
    }

    if (PyArg_ParseTupleAndKeywords(args, kw, "iii", date_kws,
                                    &year, &month, &day)) {
        if (check_date_args(year, month, day) < 0)
            return NULL;
        self = new_date_ex(year, month, day, type);
    }
    return self;
}

/*
 * Date arithmetic.
 */

/* date + timedelta -> date.  If arg negate is true, subtract the timedelta
 * instead.
 */
static PyObject *
add_date_timedelta(PyDateTime_Date *date, PyDateTime_Delta *delta, int negate)
{
    PyObject *result = NULL;
    int year = GET_YEAR(date);
    int month = GET_MONTH(date);
    int deltadays = GET_TD_DAYS(delta);
    /* C-level overflow is impossible because |deltadays| < 1e9. */
    int day = GET_DAY(date) + (negate ? -deltadays : deltadays);

    if (normalize_date(&year, &month, &day) >= 0)
        result = new_date(year, month, day);
    return result;
}

static PyObject *
date_add(PyObject *left, PyObject *right)
{
    if (PyDateTime360_Check(left) || PyDateTime360_Check(right)) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    if (PyDate360_Check(left)) {
        /* date + ??? */
        if (PyDelta_Check(right))
            /* date + delta */
            return add_date_timedelta((PyDateTime_Date *) left,
                                      (PyDateTime_Delta *) right,
                                      0);
    }
    else {
        /* ??? + date
         * 'right' must be one of us, or we wouldn't have been called
         */
        if (PyDelta_Check(left))
            /* delta + date */
            return add_date_timedelta((PyDateTime_Date *) right,
                                      (PyDateTime_Delta *) left,
                                      0);
    }
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}

static PyObject *
date_subtract(PyObject *left, PyObject *right)
{
    if (PyDateTime360_Check(left) || PyDateTime360_Check(right)) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    if (PyDate360_Check(left)) {
        if (PyDate360_Check(right)) {
            /* date - date */
            int left_ord = ymd_to_ord(GET_YEAR(left),
                                      GET_MONTH(left),
                                      GET_DAY(left));
            int right_ord = ymd_to_ord(GET_YEAR(right),
                                       GET_MONTH(right),
                                       GET_DAY(right));
            return new_delta(left_ord - right_ord, 0, 0, 0);
        }
        if (PyDelta_Check(right)) {
            /* date - delta */
            return add_date_timedelta((PyDateTime_Date *) left,
                                      (PyDateTime_Delta *) right,
                                      1);
        }
    }
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}


/* Various ways to turn a date into a string. */

static PyObject *
date_repr(PyDateTime_Date *self)
{
    char buffer[1028];
    const char *type_name;

    type_name = Py_TYPE(self)->tp_name;
    PyOS_snprintf(buffer, sizeof(buffer), "%s(%d, %d, %d)",
                  type_name,
                  GET_YEAR(self), GET_MONTH(self), GET_DAY(self));

    return PyString_FromString(buffer);
}

static PyObject *
date_str(PyDateTime_Date *self)
{
    char buffer[1028];

    PyOS_snprintf(buffer, sizeof(buffer), "%04d-%02d-%02d",
                  GET_YEAR(self), GET_MONTH(self), GET_DAY(self));

    return PyString_FromString(buffer);
}


static PyObject *
date_strftime(PyDateTime_Date *self, PyObject *args, PyObject *kw)
{
    /* This method can be inherited, and needs to call the
     * timetuple() method appropriate to self's class.
     */
    PyObject *result;
    PyObject *tuple;
    const char *format;
    Py_ssize_t format_len;
    static char *keywords[] = {"format", NULL};

    if (! PyArg_ParseTupleAndKeywords(args, kw, "s#:strftime", keywords,
                                      &format, &format_len))
        return NULL;

    // TODO: Replace the use of `self.timetuple` with an equivalent
    // private C function which has dummy values for the unsupported
    // fields? But arguably we'd want to ignore unsupported fields
    // instead of having them replaced by zeros.
    PyErr_SetString(PyExc_RuntimeError, "Not yet supported");
    return NULL;

    tuple = PyObject_CallMethod((PyObject *)self, "timetuple", "()");
    if (tuple == NULL)
        return NULL;
    result = wrap_strftime((PyObject *)self, format, format_len, tuple,
                           (PyObject *)self);
    Py_DECREF(tuple);
    return result;
}

static PyObject *
date_format(PyDateTime_Date *self, PyObject *args)
{
    PyObject *format;

    if (!PyArg_ParseTuple(args, "O:__format__", &format))
        return NULL;

    /* Check for str or unicode */
    if (PyString_Check(format)) {
        /* If format is zero length, return str(self) */
        if (PyString_GET_SIZE(format) == 0)
            return PyObject_Str((PyObject *)self);
    } else if (PyUnicode_Check(format)) {
        /* If format is zero length, return str(self) */
        if (PyUnicode_GET_SIZE(format) == 0)
            return PyObject_Unicode((PyObject *)self);
    } else {
        PyErr_Format(PyExc_ValueError,
                     "__format__ expects str or unicode, not %.200s",
                     Py_TYPE(format)->tp_name);
        return NULL;
    }
    return PyObject_CallMethod((PyObject *)self, "strftime", "O", format);
}

/* Miscellaneous methods. */

/* This is more natural as a tp_compare, but doesn't work then:  for whatever
 * reason, Python's try_3way_compare ignores tp_compare unless
 * PyInstance_Check returns true, but these aren't old-style classes.
 */
static PyObject *
date_richcompare(PyDateTime_Date *self, PyObject *other, int op)
{
    int diff = 42;      /* nonsense */

    if(PyDate360_Check(other))
        diff = memcmp(self->data, ((PyDateTime_Date *)other)->data,
                      _PyDateTime_DATE_DATASIZE);
    /* Removed in an attempt to prevent datetime360.date objects being compared to datetime.date ojects.
     * Note that comparisons like datetime.date() > datetime360.date() will not be caught 
     * as these will use the date_richcompare() of the built-in datetime.
     */  
//    else if (PyObject_HasAttrString(other, "timetuple")) {
//        /* A hook for other kinds of date objects. */
//        Py_INCREF(Py_NotImplemented);
//        return Py_NotImplemented;
//    }
    else if (op == Py_EQ || op == Py_NE)
        diff = 1;               /* any non-zero value will do */
    else /* stop this from falling back to address comparison */
        return cmperror((PyObject *)self, other);

    return diff_to_bool(diff, op);
}

static PyObject *
date_replace(PyDateTime_Date *self, PyObject *args, PyObject *kw)
{
    PyObject *clone;
    PyObject *tuple;
    int year = GET_YEAR(self);
    int month = GET_MONTH(self);
    int day = GET_DAY(self);

    if (! PyArg_ParseTupleAndKeywords(args, kw, "|iii:replace", date_kws,
                                      &year, &month, &day))
        return NULL;
    tuple = Py_BuildValue("iii", year, month, day);
    if (tuple == NULL)
        return NULL;
    clone = date_new(Py_TYPE(self), tuple, NULL);
    Py_DECREF(tuple);
    return clone;
}

static PyObject *date_getstate(PyDateTime_Date *self);

static long
date_hash(PyDateTime_Date *self)
{
    if (self->hashcode == -1) {
        PyObject *temp = date_getstate(self);
        if (temp != NULL) {
            self->hashcode = PyObject_Hash(temp);
            Py_DECREF(temp);
        }
    }
    return self->hashcode;
}

/* Pickle support, a simple use of __reduce__. */

/* __getstate__ isn't exposed */
static PyObject *
date_getstate(PyDateTime_Date *self)
{
    return Py_BuildValue(
        "(N)",
        PyString_FromStringAndSize((char *)self->data,
                                   _PyDateTime_DATE_DATASIZE));
}

static PyObject *
date_reduce(PyDateTime_Date *self, PyObject *arg)
{
    return Py_BuildValue("(ON)", Py_TYPE(self), date_getstate(self));
}

static PyMethodDef date_methods[] = {

    /* Class methods: */

    /* Instance methods: */

    {"strftime",        (PyCFunction)date_strftime,     METH_VARARGS | METH_KEYWORDS,
     PyDoc_STR("format -> strftime() style string.")},

    {"__format__",      (PyCFunction)date_format,       METH_VARARGS,
     PyDoc_STR("Formats self with strftime.")},

    {"replace",     (PyCFunction)date_replace,      METH_VARARGS | METH_KEYWORDS,
     PyDoc_STR("Return date with new specified fields.")},

    {"__reduce__", (PyCFunction)date_reduce,        METH_NOARGS,
     PyDoc_STR("__reduce__() -> (cls, state)")},

    {NULL,      NULL}
};

static PyNumberMethods date_as_number = {
    date_add,                                           /* nb_add */
    date_subtract,                                      /* nb_subtract */
    0,                                                  /* nb_multiply */
    0,                                                  /* nb_divide */
    0,                                                  /* nb_remainder */
    0,                                                  /* nb_divmod */
    0,                                                  /* nb_power */
    0,                                                  /* nb_negative */
    0,                                                  /* nb_positive */
    0,                                                  /* nb_absolute */
    0,                                                  /* nb_nonzero */
};

static PyTypeObject PyDateTime360_DateType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "iris.datetime360.date",                                    /* tp_name */
    sizeof(PyDateTime_Date),                            /* tp_basicsize */
    0,                                                  /* tp_itemsize */
    0,                                                  /* tp_dealloc */
    0,                                                  /* tp_print */
    0,                                                  /* tp_getattr */
    0,                                                  /* tp_setattr */
    0,                                                  /* tp_compare */
    //(cmpfunc)date_compare,                                                  /* tp_compare */
    (reprfunc)date_repr,                                /* tp_repr */
    &date_as_number,                                    /* tp_as_number */
    0,                                                  /* tp_as_sequence */
    0,                                                  /* tp_as_mapping */
    (hashfunc)date_hash,                                /* tp_hash */
    0,                                                  /* tp_call */
    (reprfunc)date_str,                                 /* tp_str */
    PyObject_GenericGetAttr,                            /* tp_getattro */
    0,                                                  /* tp_setattro */
    0,                                                  /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_CHECKTYPES |
    Py_TPFLAGS_BASETYPE,                                /* tp_flags */
    date_doc,                                           /* tp_doc */
    0,                                                  /* tp_traverse */
    0,                                                  /* tp_clear */
    (richcmpfunc)date_richcompare,                      /* tp_richcompare */
    0,                                                  /* tp_weaklistoffset */
    0,                                                  /* tp_iter */
    0,                                                  /* tp_iternext */
    date_methods,                                       /* tp_methods */
    0,                                                  /* tp_members */
    date_getset,                                        /* tp_getset */
    0,                                                  /* tp_base */
    0,                                                  /* tp_dict */
    0,                                                  /* tp_descr_get */
    0,                                                  /* tp_descr_set */
    0,                                                  /* tp_dictoffset */
    0,                                                  /* tp_init */
    0,                                                  /* tp_alloc */
    date_new,                                           /* tp_new */
    0,                                                  /* tp_free */
};


/*
 * PyDateTime_DateTimeType implementation. Uses built-in C PyDateTime_DateTime struct.
 */

/* Accessor properties.  Properties for day, month, and year are inherited
 * from date.
 */

static PyObject *
datetime_hour(PyDateTime_DateTime *self, void *unused)
{
    return PyInt_FromLong(DATE_GET_HOUR(self));
}

static PyObject *
datetime_minute(PyDateTime_DateTime *self, void *unused)
{
    return PyInt_FromLong(DATE_GET_MINUTE(self));
}

static PyObject *
datetime_second(PyDateTime_DateTime *self, void *unused)
{
    return PyInt_FromLong(DATE_GET_SECOND(self));
}

static PyObject *
datetime_microsecond(PyDateTime_DateTime *self, void *unused)
{
    return PyInt_FromLong(DATE_GET_MICROSECOND(self));
}

static PyObject *
datetime_tzinfo(PyDateTime_DateTime *self, void *unused)
{
    PyObject *result = HASTZINFO(self) ? self->tzinfo : Py_None;
    Py_INCREF(result);
    return result;
}

static PyGetSetDef datetime_getset[] = {
    {"hour",        (getter)datetime_hour},
    {"minute",      (getter)datetime_minute},
    {"second",      (getter)datetime_second},
    {"microsecond", (getter)datetime_microsecond},
    // TODO: Get rid of this!
    {"tzinfo",          (getter)datetime_tzinfo},
    {NULL}
};

/*
 * Constructors.
 */

static char *datetime_kws[] = {
    "year", "month", "day", "hour", "minute", "second",
    "microsecond", "tzinfo", NULL
};

static PyObject *
datetime_new(PyTypeObject *type, PyObject *args, PyObject *kw)
{
    PyObject *self = NULL;
    PyObject *state;
    int year;
    int month;
    int day;
    int hour = 0;
    int minute = 0;
    int second = 0;
    int usecond = 0;
    PyObject *tzinfo = Py_None;

    /* Check for invocation from pickle with __getstate__ state */
    if (PyTuple_GET_SIZE(args) >= 1 &&
        PyTuple_GET_SIZE(args) <= 2 &&
        PyString_Check(state = PyTuple_GET_ITEM(args, 0)) &&
        PyString_GET_SIZE(state) == _PyDateTime_DATETIME_DATASIZE &&
        MONTH_IS_SANE(PyString_AS_STRING(state)[2]))
    {
        PyDateTime_DateTime *me;
        char aware;

        if (PyTuple_GET_SIZE(args) == 2) {
            tzinfo = PyTuple_GET_ITEM(args, 1);
            if (check_tzinfo_subclass(tzinfo) < 0) {
                PyErr_SetString(PyExc_TypeError, "bad "
                    "tzinfo state arg");
                return NULL;
            }
        }
        aware = (char)(tzinfo != Py_None);
        me = (PyDateTime_DateTime *) (type->tp_alloc(type , aware));
        if (me != NULL) {
            char *pdata = PyString_AS_STRING(state);

            memcpy(me->data, pdata, _PyDateTime_DATETIME_DATASIZE);
            me->hashcode = -1;
            me->hastzinfo = aware;
            if (aware) {
                Py_INCREF(tzinfo);
                me->tzinfo = tzinfo;
            }
        }
        return (PyObject *)me;
    }

    if (PyArg_ParseTupleAndKeywords(args, kw, "iii|iiiiO", datetime_kws,
                                    &year, &month, &day, &hour, &minute,
                                    &second, &usecond, &tzinfo)) {
        if (check_date_args(year, month, day) < 0)
            return NULL;
        if (check_time_args(hour, minute, second, usecond) < 0)
            return NULL;
        if (check_tzinfo_subclass(tzinfo) < 0)
            return NULL;
        self = new_datetime_ex(year, month, day,
                                hour, minute, second, usecond,
                                tzinfo, type);
    }
    return self;
}

/* TM_FUNC is the shared type of localtime() and gmtime(). */
typedef struct tm *(*TM_FUNC)(const time_t *timer);

/* Return new datetime from time.strptime(). */
static PyObject *
datetime_strptime(PyObject *cls, PyObject *args)
{
    PyErr_SetString(PyExc_RuntimeError, "Not yet supported");
    return NULL;

    static PyObject *module = NULL;
    PyObject *result = NULL, *obj, *st = NULL, *frac = NULL;
    const char *string, *format;

    if (!PyArg_ParseTuple(args, "ss:strptime", &string, &format))
        return NULL;

    if (module == NULL &&
        (module = PyImport_ImportModuleNoBlock("_strptime")) == NULL)
        return NULL;

    /* _strptime._strptime returns a two-element tuple.  The first
       element is a time.struct_time object.  The second is the
       microseconds (which are not defined for time.struct_time). */
    obj = PyObject_CallMethod(module, "_strptime", "ss", string, format);
    if (obj != NULL) {
        int i, good_timetuple = 1;
        long int ia[7];
        if (PySequence_Check(obj) && PySequence_Size(obj) == 2) {
            st = PySequence_GetItem(obj, 0);
            frac = PySequence_GetItem(obj, 1);
            if (st == NULL || frac == NULL)
                good_timetuple = 0;
            /* copy y/m/d/h/m/s values out of the
               time.struct_time */
            if (good_timetuple &&
                PySequence_Check(st) &&
                PySequence_Size(st) >= 6) {
                for (i=0; i < 6; i++) {
                    PyObject *p = PySequence_GetItem(st, i);
                    if (p == NULL) {
                        good_timetuple = 0;
                        break;
                    }
                    if (PyInt_Check(p))
                        ia[i] = PyInt_AsLong(p);
                    else
                        good_timetuple = 0;
                    Py_DECREF(p);
                }
            }
            else
                good_timetuple = 0;
            /* follow that up with a little dose of microseconds */
            if (good_timetuple && PyInt_Check(frac))
                ia[6] = PyInt_AsLong(frac);
            else
                good_timetuple = 0;
        }
        else
            good_timetuple = 0;
        if (good_timetuple)
            result = PyObject_CallFunction(cls, "iiiiiii",
                                           ia[0], ia[1], ia[2],
                                           ia[3], ia[4], ia[5],
                                           ia[6]);
        else
            PyErr_SetString(PyExc_ValueError,
                "unexpected value from _strptime._strptime");
    }
    Py_XDECREF(obj);
    Py_XDECREF(st);
    Py_XDECREF(frac);
    return result;
}

/* Return new datetime from date/datetime and time arguments. */
static PyObject *
datetime_combine(PyObject *cls, PyObject *args, PyObject *kw)
{
    static char *keywords[] = {"date", "time", NULL};
    PyObject *date;
    PyObject *time;
    PyObject *result = NULL;

    if (PyArg_ParseTupleAndKeywords(args, kw, "O!O!:combine", keywords,
                                    &PyDateTime360_DateType, &date,
                                    &PyDateTime_TimeType, &time)) {
        PyObject *tzinfo = Py_None;

        if (HASTZINFO(time))
            tzinfo = ((PyDateTime_Time *)time)->tzinfo;
        result = PyObject_CallFunction(cls, "iiiiiiiO",
                                        GET_YEAR(date),
                                        GET_MONTH(date),
                                        GET_DAY(date),
                                        TIME_GET_HOUR(time),
                                        TIME_GET_MINUTE(time),
                                        TIME_GET_SECOND(time),
                                        TIME_GET_MICROSECOND(time),
                                        tzinfo);
    }
    return result;
}

/*
 * Destructor.
 */

static void
datetime_dealloc(PyDateTime_DateTime *self)
{
    if (HASTZINFO(self)) {
        Py_XDECREF(self->tzinfo);
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/*
 * datetime arithmetic.
 */

/* factor must be 1 (to add) or -1 (to subtract).  The result inherits
 * the tzinfo state of date.
 */
static PyObject *
add_datetime_timedelta(PyDateTime_DateTime *date, PyDateTime_Delta *delta,
                       int factor)
{
    /* Note that the C-level additions can't overflow, because of
     * invariant bounds on the member values.
     */
    int year = GET_YEAR(date);
    int month = GET_MONTH(date);
    int day = GET_DAY(date) + GET_TD_DAYS(delta) * factor;
    int hour = DATE_GET_HOUR(date);
    int minute = DATE_GET_MINUTE(date);
    int second = DATE_GET_SECOND(date) + GET_TD_SECONDS(delta) * factor;
    int microsecond = DATE_GET_MICROSECOND(date) +
                      GET_TD_MICROSECONDS(delta) * factor;

    assert(factor == 1 || factor == -1);
    if (normalize_datetime(&year, &month, &day,
                           &hour, &minute, &second, &microsecond) < 0)
        return NULL;
    else
        return new_datetime(year, month, day,
                            hour, minute, second, microsecond,
                            HASTZINFO(date) ? date->tzinfo : Py_None);
}

static PyObject *
datetime_add(PyObject *left, PyObject *right)
{
    if (PyDateTime360_Check(left)) {
        /* datetime + ??? */
        if (PyDelta_Check(right))
            /* datetime + delta */
            return add_datetime_timedelta(
                            (PyDateTime_DateTime *)left,
                            (PyDateTime_Delta *)right,
                            1);
    }
    else if (PyDelta_Check(left)) {
        /* delta + datetime */
        return add_datetime_timedelta((PyDateTime_DateTime *) right,
                                      (PyDateTime_Delta *) left,
                                      1);
    }
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}

static PyObject *
datetime_subtract(PyObject *left, PyObject *right)
{
    PyObject *result = Py_NotImplemented;

    if (PyDateTime360_Check(left)) {
        /* datetime - ??? */
        if (PyDateTime360_Check(right)) {
            /* datetime - datetime */
            naivety n1, n2;
            int offset1, offset2;
            int delta_d, delta_s, delta_us;

            if (classify_two_utcoffsets(left, &offset1, &n1, left,
                                        right, &offset2, &n2,
                                        right) < 0)
                return NULL;
            assert(n1 != OFFSET_UNKNOWN && n2 != OFFSET_UNKNOWN);
            if (n1 != n2) {
                PyErr_SetString(PyExc_TypeError,
                    "can't subtract offset-naive and "
                    "offset-aware datetimes");
                return NULL;
            }
            delta_d = ymd_to_ord(GET_YEAR(left),
                                 GET_MONTH(left),
                                 GET_DAY(left)) -
                      ymd_to_ord(GET_YEAR(right),
                                 GET_MONTH(right),
                                 GET_DAY(right));
            /* These can't overflow, since the values are
             * normalized.  At most this gives the number of
             * seconds in one day.
             */
            delta_s = (DATE_GET_HOUR(left) -
                       DATE_GET_HOUR(right)) * 3600 +
                      (DATE_GET_MINUTE(left) -
                       DATE_GET_MINUTE(right)) * 60 +
                      (DATE_GET_SECOND(left) -
                       DATE_GET_SECOND(right));
            delta_us = DATE_GET_MICROSECOND(left) -
                       DATE_GET_MICROSECOND(right);
            /* (left - offset1) - (right - offset2) =
             * (left - right) + (offset2 - offset1)
             */
            delta_s += (offset2 - offset1) * 60;
            result = new_delta(delta_d, delta_s, delta_us, 1);
        }
        else if (PyDelta_Check(right)) {
            /* datetime - delta */
            result = add_datetime_timedelta(
                            (PyDateTime_DateTime *)left,
                            (PyDateTime_Delta *)right,
                            -1);
        }
    }

    if (result == Py_NotImplemented)
        Py_INCREF(result);
    return result;
}

/* Various ways to turn a datetime into a string. */

static PyObject *
datetime_repr(PyDateTime_DateTime *self)
{
    char buffer[1000];
    const char *type_name = Py_TYPE(self)->tp_name;
    PyObject *baserepr;

    if (DATE_GET_MICROSECOND(self)) {
        PyOS_snprintf(buffer, sizeof(buffer),
                      "%s(%d, %d, %d, %d, %d, %d, %d)",
                      type_name,
                      GET_YEAR(self), GET_MONTH(self), GET_DAY(self),
                      DATE_GET_HOUR(self), DATE_GET_MINUTE(self),
                      DATE_GET_SECOND(self),
                      DATE_GET_MICROSECOND(self));
    }
    else if (DATE_GET_SECOND(self)) {
        PyOS_snprintf(buffer, sizeof(buffer),
                      "%s(%d, %d, %d, %d, %d, %d)",
                      type_name,
                      GET_YEAR(self), GET_MONTH(self), GET_DAY(self),
                      DATE_GET_HOUR(self), DATE_GET_MINUTE(self),
                      DATE_GET_SECOND(self));
    }
    else {
        PyOS_snprintf(buffer, sizeof(buffer),
                      "%s(%d, %d, %d, %d, %d)",
                      type_name,
                      GET_YEAR(self), GET_MONTH(self), GET_DAY(self),
                      DATE_GET_HOUR(self), DATE_GET_MINUTE(self));
    }
    baserepr = PyString_FromString(buffer);
    if (baserepr == NULL || ! HASTZINFO(self))
        return baserepr;
    return append_keyword_tzinfo(baserepr, self->tzinfo);
}

static PyObject *
datetime_str(PyDateTime_DateTime *self)
{
    char buffer[1028];

    if (DATE_GET_MICROSECOND(self)) {
        PyOS_snprintf(buffer, sizeof(buffer),
                      "%04d-%02d-%02dT%02d:%02d:%02d.%06d",
                      GET_YEAR(self), GET_MONTH(self), GET_DAY(self),
                      DATE_GET_HOUR(self), DATE_GET_MINUTE(self),
                      DATE_GET_SECOND(self), DATE_GET_MICROSECOND(self));
    }
    else {
        PyOS_snprintf(buffer, sizeof(buffer), "%04d-%02d-%02dT%02d:%02d:%02d",
                      GET_YEAR(self), GET_MONTH(self), GET_DAY(self),
                      DATE_GET_HOUR(self), DATE_GET_MINUTE(self),
                      DATE_GET_SECOND(self));
    }

    return PyString_FromString(buffer);
}

/* Miscellaneous methods. */

/* This is more natural as a tp_compare, but doesn't work then:  for whatever
 * reason, Python's try_3way_compare ignores tp_compare unless
 * PyInstance_Check returns true, but these aren't old-style classes.
 */
static PyObject *
datetime_richcompare(PyDateTime_DateTime *self, PyObject *other, int op)
{
    int diff;
    naivety n1, n2;
    int offset1, offset2;

    if (! PyDateTime360_Check(other)) {
        /* If other has a "timetuple" attr, that's an advertised
         * hook for other classes to ask to get comparison control.
         * However, date instances have a timetuple attr, and we
         * don't want to allow that comparison.  Because datetime
         * is a subclass of date, when mixing date and datetime
         * in a comparison, Python gives datetime the first shot
         * (it's the more specific subtype).  So we can stop that
         * combination here reliably.
         */
        /* Removed in an attempt to prevent datetime360.datetime objects being compared to datetime ojects.
         * Note that comparisons like datetime.datetime() > datetime360.datetime() will not be caught 
         * as these will use the datetime_richcompare() of the built-in datetime.
         */ 
        //if (PyObject_HasAttrString(other, "timetuple") &&
        //    ! PyDate360_Check(other)) {
        //    /* A hook for other kinds of datetime objects. */
        //    Py_INCREF(Py_NotImplemented);
        //    return Py_NotImplemented;
        //}
        if (op == Py_EQ || op == Py_NE) {
            PyObject *result = op == Py_EQ ? Py_False : Py_True;
            Py_INCREF(result);
            return result;
        }
        /* Stop this from falling back to address comparison. */
        return cmperror((PyObject *)self, other);
    }

    if (classify_two_utcoffsets((PyObject *)self, &offset1, &n1,
                                (PyObject *)self,
                                 other, &offset2, &n2,
                                 other) < 0)
        return NULL;
    assert(n1 != OFFSET_UNKNOWN && n2 != OFFSET_UNKNOWN);
    /* If they're both naive, or both aware and have the same offsets,
     * we get off cheap.  Note that if they're both naive, offset1 ==
     * offset2 == 0 at this point.
     */
    if (n1 == n2 && offset1 == offset2) {
        diff = memcmp(self->data, ((PyDateTime_DateTime *)other)->data,
                      _PyDateTime_DATETIME_DATASIZE);
        return diff_to_bool(diff, op);
    }

    if (n1 == OFFSET_AWARE && n2 == OFFSET_AWARE) {
        PyDateTime_Delta *delta;

        assert(offset1 != offset2);             /* else last "if" handled it */
        delta = (PyDateTime_Delta *)datetime_subtract((PyObject *)self,
                                                       other);
        if (delta == NULL)
            return NULL;
        diff = GET_TD_DAYS(delta);
        if (diff == 0)
            diff = GET_TD_SECONDS(delta) |
                   GET_TD_MICROSECONDS(delta);
        Py_DECREF(delta);
        return diff_to_bool(diff, op);
    }

    assert(n1 != n2);
    PyErr_SetString(PyExc_TypeError,
                    "can't compare offset-naive and "
                    "offset-aware datetimes");
    return NULL;
}

static long
datetime_hash(PyDateTime_DateTime *self)
{
    if (self->hashcode == -1) {
        naivety n;
        int offset;
        PyObject *temp;

        n = classify_utcoffset((PyObject *)self, (PyObject *)self,
                               &offset);
        assert(n != OFFSET_UNKNOWN);
        if (n == OFFSET_ERROR)
            return -1;

        /* Reduce this to a hash of another object. */
        if (n == OFFSET_NAIVE)
            temp = PyString_FromStringAndSize(
                            (char *)self->data,
                            _PyDateTime_DATETIME_DATASIZE);
        else {
            int days;
            int seconds;

            assert(n == OFFSET_AWARE);
            assert(HASTZINFO(self));
            days = ymd_to_ord(GET_YEAR(self),
                              GET_MONTH(self),
                              GET_DAY(self));
            seconds = DATE_GET_HOUR(self) * 3600 +
                      (DATE_GET_MINUTE(self) - offset) * 60 +
                      DATE_GET_SECOND(self);
            temp = new_delta(days,
                             seconds,
                             DATE_GET_MICROSECOND(self),
                             1);
        }
        if (temp != NULL) {
            self->hashcode = PyObject_Hash(temp);
            Py_DECREF(temp);
        }
    }
    return self->hashcode;
}

static PyObject *
datetime_replace(PyDateTime_DateTime *self, PyObject *args, PyObject *kw)
{
    PyObject *clone;
    PyObject *tuple;
    int y = GET_YEAR(self);
    int m = GET_MONTH(self);
    int d = GET_DAY(self);
    int hh = DATE_GET_HOUR(self);
    int mm = DATE_GET_MINUTE(self);
    int ss = DATE_GET_SECOND(self);
    int us = DATE_GET_MICROSECOND(self);
    PyObject *tzinfo = HASTZINFO(self) ? self->tzinfo : Py_None;

    if (! PyArg_ParseTupleAndKeywords(args, kw, "|iiiiiiiO:replace",
                                      datetime_kws,
                                      &y, &m, &d, &hh, &mm, &ss, &us,
                                      &tzinfo))
        return NULL;
    tuple = Py_BuildValue("iiiiiiiO", y, m, d, hh, mm, ss, us, tzinfo);
    if (tuple == NULL)
        return NULL;
    clone = datetime_new(Py_TYPE(self), tuple, NULL);
    Py_DECREF(tuple);
    return clone;
}

static PyObject *
datetime_getdate(PyDateTime_DateTime *self)
{
    return new_date(GET_YEAR(self),
                    GET_MONTH(self),
                    GET_DAY(self));
}

static PyObject *
datetime_gettime(PyDateTime_DateTime *self)
{
    return new_time(DATE_GET_HOUR(self),
                    DATE_GET_MINUTE(self),
                    DATE_GET_SECOND(self),
                    DATE_GET_MICROSECOND(self),
                    Py_None);
}

/* Pickle support, a simple use of __reduce__. */

/* Let basestate be the non-tzinfo data string.
 * If tzinfo is None, this returns (basestate,), else (basestate, tzinfo).
 * So it's a tuple in any (non-error) case.
 * __getstate__ isn't exposed.
 */
static PyObject *
datetime_getstate(PyDateTime_DateTime *self)
{
    PyObject *basestate;
    PyObject *result = NULL;

    basestate = PyString_FromStringAndSize((char *)self->data,
                                      _PyDateTime_DATETIME_DATASIZE);
    if (basestate != NULL) {
        if (! HASTZINFO(self) || self->tzinfo == Py_None)
            result = PyTuple_Pack(1, basestate);
        else
            result = PyTuple_Pack(2, basestate, self->tzinfo);
        Py_DECREF(basestate);
    }
    return result;
}

static PyObject *
datetime_reduce(PyDateTime_DateTime *self, PyObject *arg)
{
    return Py_BuildValue("(ON)", Py_TYPE(self), datetime_getstate(self));
}

static PyMethodDef datetime_methods[] = {

    /* Class methods: */

    {"strptime", (PyCFunction)datetime_strptime,
     METH_VARARGS | METH_CLASS,
     PyDoc_STR("string, format -> new datetime parsed from a string "
               "(like time.strptime()).")},

    {"combine", (PyCFunction)datetime_combine,
     METH_VARARGS | METH_KEYWORDS | METH_CLASS,
     PyDoc_STR("date, time -> datetime with same date and time fields")},

    /* Instance methods: */

    {"date",   (PyCFunction)datetime_getdate, METH_NOARGS,
     PyDoc_STR("Return date object with same year, month and day.")},

    {"time",   (PyCFunction)datetime_gettime, METH_NOARGS,
     PyDoc_STR("Return time object with same time but with tzinfo=None.")},

    {"replace",     (PyCFunction)datetime_replace,      METH_VARARGS | METH_KEYWORDS,
     PyDoc_STR("Return datetime with new specified fields.")},

    {"__reduce__", (PyCFunction)datetime_reduce,     METH_NOARGS,
     PyDoc_STR("__reduce__() -> (cls, state)")},

    {NULL,      NULL}
};

static char datetime_doc[] =
PyDoc_STR("datetime(year, month, day[, hour[, minute[, second[, microsecond[,tzinfo]]]]])\n\
\n\
The year, month and day arguments are required. tzinfo may be None, or an\n\
instance of a tzinfo subclass. The remaining arguments may be ints or longs.\n");

static PyNumberMethods datetime_as_number = {
    datetime_add,                               /* nb_add */
    datetime_subtract,                          /* nb_subtract */
    0,                                          /* nb_multiply */
    0,                                          /* nb_divide */
    0,                                          /* nb_remainder */
    0,                                          /* nb_divmod */
    0,                                          /* nb_power */
    0,                                          /* nb_negative */
    0,                                          /* nb_positive */
    0,                                          /* nb_absolute */
    0,                                          /* nb_nonzero */
};

statichere PyTypeObject PyDateTime360_DateTimeType = {
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
    "iris.datetime360.datetime",                        /* tp_name */
    sizeof(PyDateTime_DateTime),                /* tp_basicsize */
    0,                                          /* tp_itemsize */
    (destructor)datetime_dealloc,               /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_compare */
    (reprfunc)datetime_repr,                    /* tp_repr */
    &datetime_as_number,                        /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    (hashfunc)datetime_hash,                    /* tp_hash */
    0,                                          /* tp_call */
    (reprfunc)datetime_str,                     /* tp_str */
    PyObject_GenericGetAttr,                    /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_CHECKTYPES |
    Py_TPFLAGS_BASETYPE,                        /* tp_flags */
    datetime_doc,                               /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    (richcmpfunc)datetime_richcompare,          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    datetime_methods,                           /* tp_methods */
    0,                                          /* tp_members */
    datetime_getset,                            /* tp_getset */
    &PyDateTime360_DateType,                       /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    datetime_alloc,                             /* tp_alloc */
    datetime_new,                               /* tp_new */
    0,                                          /* tp_free */
};

/* ---------------------------------------------------------------------------
 * Module methods and initialization.
 */

static PyObject *
datetime360_foobar(PyObject *self, PyObject *args)
{
    const char *str;
    size_t len;

    if (!PyArg_ParseTuple(args, "s", &str))
        return NULL;

    len = strlen(str);
    printf("Length of '%s' is %lu\n", str, len);

    return Py_BuildValue("n", len);
}

static PyObject *
datetime360_check_CAPI(PyObject *self, PyObject *args)
{
    //Check access to built-in datetime API
    if(PyDateTimeAPI == NULL)
        fprintf(stderr, "PyDateTime CAPI pointer is NULL\n");
    else  
        printf("PyDateTimeAPI = %p\n", PyDateTimeAPI);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyMethodDef module_methods[] = {
    {"foobar", datetime360_foobar, METH_VARARGS, "Simple C function"},
    {"check_API", datetime360_check_CAPI, METH_NOARGS, "Function to check access to the PyDateTime CAPI"},
    {NULL, NULL}
};

/* C API.  Clients get at this via PyDateTime_IMPORT, defined in
 * datetime.h.
 */
//static PyDateTime360_CAPI CAPI = {
//    &PyDateTime360_DateType,
//    &PyDateTime360_DateTimeType,
//    new_date_ex,
//    new_datetime_ex,
//};


PyMODINIT_FUNC
initdatetime360(void)
{
    PyObject *m;        /* a module object */
    PyObject *d;        /* its dict */
    PyObject *x;

    m = Py_InitModule3("datetime360", module_methods,
                       "Fast implementation of the datetime type with a 360 day calendar.");
    if (m == NULL)
        return;

    // Important - Obtain access to time and timedelta (and associated new_???_ex() C contructor functions) from 
    // built-in datetime module through CAPI (PyDateTime_CAPI *PyDateTimeAPI)
    PyDateTime_IMPORT;

    if (PyType_Ready(&PyDateTime360_DateType) < 0)
        return;
    if (PyType_Ready(&PyDateTime360_DateTimeType) < 0)
        return;
    
    /* date values */
    d = PyDateTime360_DateType.tp_dict;

    x = new_date(1, 1, 1);
    if (x == NULL || PyDict_SetItemString(d, "min", x) < 0)
        return;
    Py_DECREF(x);

    x = new_date(MAXYEAR, 12, 30);
    if (x == NULL || PyDict_SetItemString(d, "max", x) < 0)
        return;
    Py_DECREF(x);

    x = new_delta(1, 0, 0, 0);
    if (x == NULL || PyDict_SetItemString(d, "resolution", x) < 0)
        return;
    Py_DECREF(x);

    /* datetime values */
    d = PyDateTime360_DateTimeType.tp_dict;

    x = new_datetime(1, 1, 1, 0, 0, 0, 0, Py_None);
    if (x == NULL || PyDict_SetItemString(d, "min", x) < 0)
        return;
    Py_DECREF(x);

    x = new_datetime(MAXYEAR, 12, 30, 23, 59, 59, 999999, Py_None);
    if (x == NULL || PyDict_SetItemString(d, "max", x) < 0)
        return;
    Py_DECREF(x);

    x = new_delta(0, 0, 1, 0);
    if (x == NULL || PyDict_SetItemString(d, "resolution", x) < 0)
        return;
    Py_DECREF(x);

    /* module initialization */
    PyModule_AddIntConstant(m, "MINYEAR", MINYEAR);
    PyModule_AddIntConstant(m, "MAXYEAR", MAXYEAR);

    Py_INCREF(&PyDateTime360_DateType);
    PyModule_AddObject(m, "date", (PyObject *) &PyDateTime360_DateType);

    Py_INCREF(&PyDateTime360_DateTimeType);
    PyModule_AddObject(m, "datetime",
                       (PyObject *)&PyDateTime360_DateTimeType);

    // Add time, timedelta and tzinfo from built-in datetime module to ease access
    // and avoid the need to import both. These will still be datetime.??? objects.
    Py_INCREF(&PyDateTime_TimeType);
    PyModule_AddObject(m, "time", (PyObject *) &PyDateTime_TimeType);

    Py_INCREF(&PyDateTime_DeltaType);
    PyModule_AddObject(m, "timedelta", (PyObject *) &PyDateTime_DeltaType);

}

