import os
import numpy as np
import tensorflow as tf
from data import produce_signals
from data import generate_batch
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


def run_training():
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)

    # poems_vector, word_to_int, vocabularies = process_poems(FLAGS.file_path)
    # batches_inputs, batches_outputs = generate_batch(FLAGS.batch_size, poems_vector, word_to_int)

    signals = produce_signals()
    x_batches, y_batches = generate_batch(FLAGS.batch_size * FLAGS.time_steps, signals)
    # x_batches = tf.reshape(x_batches_,[-1, FLAGS.time_steps, FLAGS.pulse_size])  # reshape input_data

## add by sun
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        start_epoch = 0
        try:
            for epoch in range(start_epoch, FLAGS.epochs):
                n = 0
                n_chunk = len(x_batches) ## how many batch
                for batch in range(n_chunk):
                    # process input data
                    input_batch = tf.reshape(x_batches[n],[-1, FLAGS.time_steps, FLAGS.pulse_size])  # reshape input_data
                    output_batch = tf.reshape(y_batches[n],[-1, FLAGS.time_steps, 1])  # reshape input_data 这里的第3维度为 1
                    print(sess.run(input_batch))
                    print(sess.run(output_batch))

                    n += 1
        except KeyboardInterrupt:
            print('## Interrupt manually, try saving checkpoint for now...')


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
