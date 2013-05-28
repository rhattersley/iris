from datetime import timedelta


# The pure-Python scalar type for our new dtype
class date360(object):
    def __init__(self, year, month, day):
        assert 1 <= month <= 12
        assert 1 <= day <= 30
        self.year = year
        self.month = month
        self.day = day

    def to_ordinal(self):
        return 360 * self.year + 30 * (self.month - 1) + self.day - 1

    def __repr__(self):
        return 'date360' + str((self.year, self.month, self.day))

    def __sub__(self, other):
        result = NotImplemented
        if isinstance(other, date360):
            days = (self.year - other.year) * 360
            days += (self.month - other.month) * 30
            days += self.day - other.day
            result = timedelta(days)
        return result
