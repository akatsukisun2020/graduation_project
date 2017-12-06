#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import tensorflow as tf
#
#
# label = tf.constant([1, 2, 3, 4, 5, 6])
# sess = tf.Session()
# print('label : ', sess.run(label))
# b = tf.one_hot(label, 7)
# sess.run(tf.global_variables_initializer())
# print('one_hot : ', sess.run(b))

con = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
length = len(con) // 2
a = con[: length]
b = con[length:]
print("legth : ", length)
print("a : ", a)
print("b : ", b)
