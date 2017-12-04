#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


# times : 采样时间总和
def produce_one_signal(pri, doa, cf, pw, pa, times, signal_id):
    # list 的初始化与dict的初始化貌似是不同的 !!
    signal = []
    # 起始时间设置在1us之内
    # start_time = np.random.uniform(0.0, 1.0)
    start_time = 0.0

    # 时间的变化量， 总和不能超过采样时间
    time = 0
    while time < times:
        pluse = {}
        # produce toa
        jitter = (1 - 2 * np.random.random()) * (10 ** -3)
        delta_pri = pri + pri * jitter
        time += 1   # 这是按照次数问题去产生个数
#        time = time + delta_pri # TODO
        start_time += delta_pri
        pluse['toa'] = start_time

        # produce doa
        jitter = (1 - 2 * np.random.random()) * (10 ** -3)
        pluse['doa'] = doa + doa * jitter

        # produce cf
        jitter = (1 - 2 * np.random.random()) * (10 ** -3)
        pluse['cf'] = cf + cf * jitter

        # produce pw
        jitter = (1 - 2 * np.random.random()) * (10 ** -3)
        pluse['pw'] = pw + pw * jitter

        # produce pa
        jitter = (1 - 2 * np.random.random()) * (10 ** -3)
        pluse['pa'] = pa + pa * jitter

        # 数据以列表的形式返回
        result = [pluse['toa'], pluse['doa'], pluse['cf'], pluse['pw'], pluse['pa'], signal_id]

        signal.append(result)

    return signal
