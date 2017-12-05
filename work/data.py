#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pulse_produce import produce_one_signal

signal_numbers = 10000


def getkey(x):
    return x[0]


def produce_signals():
    # produce signal
    signal1 = produce_one_signal(35, 3, 600, 0.5, 10, 50, 1)
    signal2 = produce_one_signal(21, 90, 860, 6.0, 10, 50, 2)
    signal3 = produce_one_signal(69, 45, 2100, 23, 10, 50, 3)

    # 合并所有的信号list
    signals = []
    signals += signal1
    signals += signal2
    signals += signal3
    # 依据到达时间进行排序
    signals.sort(key=getkey)

    return signals


def generate_batch(batch_size, signals):
    n_chunk = len(signals) // batch_size
    x_batches = []
    y_batches = []

    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size

        batches = signals[start_index:end_index]

        x_data = np.zeros((batch_size, 5))
        y_data = np.zeros((batch_size, 1), dtype=int)
        for row in range(batch_size):
            batch = batches[row]
            x_data[row, :] = batch[: -1]  # 实际上就是取数据， 然后生成对应的batch
            y_data[row, :] = batch[-1]

        x_batches.append(x_data)
        y_batches.append(y_data)

    return x_batches, y_batches
