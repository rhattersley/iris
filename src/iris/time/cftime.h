/* Defines the underlying time/time-delta C structures and the routines
 * to manipulate them.
 */

#ifndef CFTIME_H
#define CFTIME_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/* TODO: Consider whether to pack the microsecond component into three
 * bytes so that the whole structure can fit in 12 bytes.
 */
typedef struct {
    int32_t year;   /* -2147483648 to 2147483647 */
    int8_t month;   /* 0 to 11 */
    int8_t day;     /* 0 to 29 */
    int8_t hour;    /* 0 to 23 */
    int8_t minute;  /* 0 to 59 */
    int8_t second;  /* 0 to 59 */
    int32_t microsecond;    /* 0 to 999999 */
} datetime;

/* This is the non-Python-object part of the built-in PyDateTime_Delta. */
typedef struct {
    int days;                   /* -MAX_DELTA_DAYS <= days <= MAX_DELTA_DAYS */
    int seconds;                /* 0 <= seconds < 24*3600 is invariant */
    int microseconds;           /* 0 <= microseconds < 1000000 is invariant */
} timedelta;

/*
 * `time360` functions
 */

int time360_compare(const datetime *t1, const datetime *t2);
datetime time360_add_timedelta(datetime t, timedelta delta);
datetime time360_subtract_timedelta(datetime t, timedelta delta);
timedelta time360_subtract_time360(datetime t1, datetime t2);
int timedelta_compare(const timedelta *td1, const timedelta *td2);
int8_t timedelta_equal_timedelta(timedelta td1, timedelta td2);
int8_t timedelta_sign(timedelta td1);

#ifdef __cplusplus
}
#endif
#endif
