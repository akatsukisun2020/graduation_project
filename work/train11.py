import os
import numpy as np
import tensorflow as tf
from data import produce_signals
from data import generate_file
from data import get_training_data
from model import rnn_model

tf.app.flags.DEFINE_integer('vocab_size', 3, 'vocab size.')
tf.app.flags.DEFINE_integer('batch_size', 5, 'batch size.')
tf.app.flags.DEFINE_integer('time_steps', 10, 'time size.')
tf.app.flags.DEFINE_integer('pulse_size', 5, 'pulse size.')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate.')
tf.app.flags.DEFINE_string('model_dir', os.path.abspath('./model'), 'model save path.')
tf.app.flags.DEFINE_string('file_path', os.path.abspath('./data/poems.txt'), 'file name of poems.')
tf.app.flags.DEFINE_string('model_prefix', 'poems', 'model save prefix.')
tf.app.flags.DEFINE_integer('epochs', 3, 'train how many epochs.')

FLAGS = tf.app.flags.FLAGS

data = [[1,2,3,1,2,3], [4,5,6,4,5,6], [7,8,9,7,8,9], [10,11,12,10,11,12], [13,14,15,13,14,15], [16,17,18,16,17,18]]

def run_training():
    print(data)
    print(float(data))

    # signals = produce_signals()
#     generate_file(data)
#     print("training_file data :")
#     x, y = get_training_data(1)
#     print(x)
#     print(y)


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
