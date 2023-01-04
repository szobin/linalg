from enum import Enum
from collections import MutableSequence


class SegType(Enum):
    EMPTY = 0
    OPEN = 1
    OPEN_CLOSE = 2
    CLOSE_OPEN = 3
    CLOSE = 4


def get_seg_type(a_open, b_open):
    if a_open and b_open:
        return SegType.OPEN
    if a_open and (not b_open):
        return SegType.OPEN_CLOSE
    if (not a_open) and b_open:
        return SegType.CLOSE_OPEN
    return SegType.CLOSE


class Segment:

    def __init__(self, seg_type: SegType, *args):
        self.type = seg_type

        if len(args) not in (0, 2):
            raise Exception("wrong argument count")

        if len(args) < 2:
            self.start = None
            self.finish = None
        else:
            self.start = args[0]
            self.finish = args[1]

    def __str__(self):
        if self.type == SegType.EMPTY:
            return "[]"
        if self.type == SegType.OPEN:
            return "]{};{}[".format(self.start, self.finish)
        if self.type == SegType.CLOSE:
            return "[{};{}]".format(self.start, self.finish)
        if self.type == SegType.OPEN_CLOSE:
            return "]{};{}]".format(self.start, self.finish)
        if self.type == SegType.CLOSE_OPEN:
            return "[{};{}[".format(self.start, self.finish)
        return "N/A"

    def __add__(self, other):
        if type(other) is not Segment:
            raise Exception("Cannot join B")
        if other.type == SegType.EMPTY:
            return Segment(self.type, self.start, self.finish)
        if self.type == SegType.EMPTY:
            return Segment(other.type, other.start, other.finish)

        if self.start < other.start:
            a = self.start
            a_open = self.type in [SegType.OPEN, SegType.OPEN_CLOSE]
        else:
            a = other.start
            a_open = other.type in [SegType.OPEN, SegType.OPEN_CLOSE]

        if self.finish > other.finish:
            b = self.finish
            b_open = self.type in [SegType.OPEN, SegType.CLOSE_OPEN]
        else:
            b = other.finish
            b_open = other.type in [SegType.OPEN, SegType.CLOSE_OPEN]

        return Segment(get_seg_type(a_open, b_open), a, b)


    def __div__(self, other):
        pass


class SegmentSet(MutableSequence):

    def __init__(self, data=None):
        self._list = list(data)

    def __repr__(self):
        return "<{0} {1}>".format(self.__class__.__name__, self._list)

    def __len__(self):
        """List length"""
        return len(self._list)

    def __getitem__(self, ii):
        """Get a list item"""
        return self._list[ii]

    def __delitem__(self, ii):
        """Delete an item"""
        del self._list[ii]

    def __setitem__(self, ii, val):
        # optional: self._acl_check(val)
        self._list[ii] = val

    def __str__(self):
        return str(self._list)

    def insert(self, ii, val):
        # optional: self._acl_check(val)
        self._list.insert(ii, val)

    def append(self, val):
        self.insert(len(self._list), val)
