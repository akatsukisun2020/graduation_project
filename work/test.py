import os
import numpy as np
import tensorflow as tf
from data import get_testing_data
from model import rnn_model

tf.app.flags.DEFINE_integer('vocab_size', 3, 'vocab size.')
tf.app.flags.DEFINE_integer('batch_size', 50, 'batch size.')
tf.app.flags.DEFINE_integer('time_steps', 100, 'time size.')
tf.app.flags.DEFINE_integer('pulse_size', 5, 'pulse size.')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate.')
tf.app.flags.DEFINE_string('model_dir', os.path.abspath('./model'), 'model save path.')
tf.app.flags.DEFINE_string('file_path', os.path.abspath('./data/poems.txt'), 'file name of poems.')
tf.app.flags.DEFINE_string('model_prefix', 'signal', 'model save prefix.')
tf.app.flags.DEFINE_integer('epochs', 10, 'train how many epochs.')

FLAGS = tf.app.flags.FLAGS


def run_testing(test_x_batches, test_y_batches):
    input_data = tf.placeholder(tf.float32, [None, FLAGS.time_steps, FLAGS.pulse_size])
    # output_targets = tf.placeholder(tf.int32, [None, FLAGS.time_steps, 1])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=None, vocab_size=FLAGS.vocab_size,
            rnn_size=128, num_layers=2, batch_size=FLAGS.batch_size, learning_rate=FLAGS.learning_rate)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        checkpoint = tf.train.latest_checkpoint(FLAGS.model_dir)
        saver.restore(sess, checkpoint)

        n = 0
        n_chunk = len(test_x_batches)
        for batch in range(n_chunk):
            test_input_batch = np.reshape(test_x_batches[n], [-1, FLAGS.time_steps, FLAGS.pulse_size])  # reshape input_data
            test_output_batch = np.reshape(test_y_batches[n], [-1, FLAGS.time_steps, 1])  # reshape input_data 这里的第3维度为 1
            # print(input_data)
            # print(test_input_batch.shape)
            [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                    feed_dict={input_data: test_input_batch})

            correct_prediction = tf.equal(tf.argmax(predict, 1), tf.argmax(test_output_batch, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("test batch id : ", n, "  test accuracy : ", sess.run(accuracy))
            n += 1


def main(_):
    test_x_batches, test_y_batches = get_testing_data(FLAGS.batch_size * FLAGS.time_steps)
    run_testing(test_x_batches, test_y_batches)


if __name__ == '__main__':
    tf.app.run()
