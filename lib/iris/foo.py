# The pure-Python scalar type for our new dtype
class Foo(object):
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day

    def wibble(self):
        return 360 * self.year + 30 * self.month + self.day

    def __repr__(self):
        return 'Foo' + str((self.year, self.month, self.day))
