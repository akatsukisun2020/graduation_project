#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from data import produce_signals
from data import generate_batch


signals = produce_signals()
# for a in signals:
#     print(a)

x_batches, y_batches = generate_batch(10, signals)
for i in range(10):
    print("batch %d", i)
    print(x_batches[i])
    print(y_batches[i])


# tmp = [1, 2, 3, 4, 5, 6]
# xdata = np.zeros((1, 5))
# ydata = np.zeros((1, 5))
# print(tmp)
# xdata = tmp[: -1]
# ydata = tmp[: 1]
# print(xdata)
# print(ydata)
