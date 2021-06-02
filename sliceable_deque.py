# -*- coding: utf-8 -*-
from collections import deque
import itertools

class sliceable_deque(deque):
    def __getitem__(self, index):
        if isinstance(index, slice):
            start = index.start if (index.start is None or index.start > 0) else len(self) + index.start
            stop = index.stop if (index.stop is None or index.stop > 0) else len(self) + index.stop
            return itertools.islice(self, start, index.stop, index.step) if self else iter([])
        return deque.__getitem__(self, index)