/* Defines the routines to manipulate the underlying time/time-delta C
 * structures.
 */

#include "cftime.h"

int
time360_compare(const datetime *t1, const datetime *t2)
{
    int result = 0;

    result = (t1->year < t2->year) ? -1 : (t1->year > t2->year);
    if (result == 0)
        result = (t1->month < t2->month) ? -1 : (t1->month > t2->month);
    if (result == 0)
        result = (t1->day < t2->day) ? -1 : (t1->day > t2->day);
    if (result == 0)
        result = (t1->hour < t2->hour) ? -1 : (t1->hour > t2->hour);
    if (result == 0)
        result = (t1->minute < t2->minute) ? -1 :
                        (t1->minute > t2->minute);
    if (result == 0)
        result = (t1->second < t2->second) ? -1 :
                        (t1->second > t2->second);
    if (result == 0)
        result = (t1->microsecond < t2->microsecond) ? -1 :
                        (t1->microsecond > t2->microsecond);
    return result;
}

datetime
time360_add_timedelta(datetime t, timedelta delta)
{
    int year = t.year;
    int month = t.month;
    int day = t.day;
    int hour = t.hour;
    int minute = t.minute;
    int second = t.second;
    int microsecond = t.microsecond;

    microsecond += delta.microseconds;
    if (microsecond >= 1000000) {
        microsecond -= 1000000;
        second += 1;
    }
    second += delta.seconds % 60;
    if (second >= 60) {
        second -= 60;
        minute += 1;
    }
    minute += (delta.seconds / 60) % 60;
    if (minute >= 60) {
        minute -= 60;
        hour += 1;
    }
    hour += delta.seconds / 3600;
    if (hour >= 24) {
        hour -= 24;
        day += 1;
    }
    day += delta.days % 30;
    if (day >= 30) {
        day -= 30;
        month += 1;
    }
    month += (delta.days / 30) % 12;
    if (month >= 12) {
        month -= 12;
        year += 1;
    }
    year += delta.days / 360;
    return (datetime) {year, month, day, hour, minute, second, microsecond};
}

datetime
time360_subtract_timedelta(datetime t, timedelta delta)
{
    int year = t.year;
    int month = t.month;
    int day = t.day;
    int hour = t.hour;
    int minute = t.minute;
    int second = t.second;
    int microsecond = t.microsecond;

    microsecond -= delta.microseconds;
    if (microsecond < 0) {
        microsecond += 1000000;
        second -= 1;
    }
    second -= (delta.seconds % 60);
    if (second < 0) {
        second += 60;
        minute -= 1;
    }
    minute -= (delta.seconds / 60) % 60;
    if (minute < 0) {
        minute += 60;
        hour -= 1;
    }
    hour -= (delta.seconds / 3600);
    if (hour < 0) {
        hour += 24;
        day -= 1;
    }
    day -= delta.days % 30;
    if (day < 0) {
        day += 30;
        month -= 1;
    }
    month -= (delta.days / 30) % 12;
    if (month < 0) {
        month += 12;
        year -= 1;
    }
    year -= delta.days / 360;
    return (datetime) {year, month, day, hour, minute, second, microsecond};
}

timedelta
time360_subtract_time360(datetime t1, datetime t2)
{
    timedelta delta;

    /* TODO: Consider overflow of days from large year deltas. */
    delta.days = (t1.year - t2.year) * 360;
    delta.days += (t1.month - t2.month) * 30;
    delta.days += t1.day - t2.day;

    delta.seconds = (t1.hour - t2.hour) * 3600;
    delta.seconds += (t1.minute - t2.minute) * 60;
    delta.seconds += t1.second - t2.second;

    delta.microseconds = t1.microsecond - t2.microsecond;

    return delta;
}

int
timedelta_compare(const timedelta *td1, const timedelta *td2)
{
    int result;

    result = (td1->days < td2->days) ? -1 : (td1->days > td2->days);
    if (result == 0)
        result = (td1->seconds < td2->seconds) ? -1 :
                        (td1->seconds > td2->seconds);
    if (result == 0)
        result = (td1->microseconds < td2->microseconds) ? -1 :
                        (td1->microseconds > td2->microseconds);
    return result;
}

int8_t
timedelta_equal_timedelta(timedelta td1, timedelta td2)
{
    return td1.days == td2.days && td1.seconds == td2.seconds &&
           td1.microseconds == td2.microseconds;
}

int8_t
timedelta_sign(timedelta td)
{
    int8_t sign = 0;
    if (td.days < 0)
        sign = -1;
    else if (td.days > 0 || td.seconds > 0 || td.microseconds > 0)
        sign = 1;
    return sign;
}
