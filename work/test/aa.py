#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf


label = tf.constant([1, 2, 3, 4, 5, 6])
sess = tf.Session()
print('label : ', sess.run(label))
b = tf.one_hot(label, 7)
sess.run(tf.global_variables_initializer())
print('one_hot : ', sess.run(b))
