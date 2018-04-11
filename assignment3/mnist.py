# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple MNIST classifier which displays summaries in TensorBoard.
This is an unimpressive MNIST model, but it is a good example of using
tf.name_scope to make a graph legible in the TensorBoard graph explorer, and of
naming summary tags so that they are grouped meaningfully in TensorBoard.
It demonstrates the functionality of every TensorBoard dashboard.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


def train():

  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir,
                            one_hot=True,
                            fake_data=FLAGS.fake_data)
  if FLAGS.cpu:
    config = tf.ConfigProto(
      device_count={'GPU': 0})
    sess = tf.Session(config=config)
  else:
    sess = tf.Session()
  # saver = tf.train.Saver()
  # saver = tf.train.import_meta_graph(FLAGS.model_dir + "/mnist.meta")


  # Create a multilayer model.

  # Input placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

  with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

  def deepnn(x):
    """deepnn builds the graph for a deep net for classifying digits.
    Args:
      x: an input tensor with the dimensions (N_examples, 784), where 784 is the
      number of pixels in a standard MNIST image.
    Returns:
      A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
      equal to the logits of classifying the digit into one of 10 classes (the
      digits 0-9). keep_prob is a scalar placeholder for the probability of
      dropout.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob, h_pool2, h_pool1, h_conv1

  def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

  def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  y, keep_prob, h_pool2, h_pool1, h_conv1 = deepnn(x)

  with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('total'):
      cross_entropy = tf.reduce_mean(diff)
  tf.summary.scalar('cross_entropy', cross_entropy)

  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
      cross_entropy)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

  # Merge all the summaries and write them out to
  # /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')


  saver = tf.train.Saver(max_to_keep=50)
  sess.run(tf.global_variables_initializer())
  checkpoint = tf.train.latest_checkpoint(FLAGS.model_dir)
  if checkpoint:
    saver.restore(sess, checkpoint)

  # Train the model, and also write summaries.
  # Every 10th step, measure test-set accuracy, and write test summaries
  # All other steps, run train_step on training data, & add training summaries

  def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train or FLAGS.fake_data:
      xs, ys = mnist.train.next_batch(50, fake_data=FLAGS.fake_data)
      k = FLAGS.dropout
    else:
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

  if FLAGS.visualize:
    # reinitalize
    sess.run(tf.global_variables_initializer())
    results = sess.run([h_pool2, h_pool1, h_conv1], feed_dict(False))
    h_pool2_np, h_pool1_np, h_conv1_np = results
    h_pool2_np.dump("h_pool2_i.np")
    h_pool1_np.dump("h_pool1_i.np")
    h_conv1_np.dump("h_conv1_i.np")

    # # predict
    # results = sess.run([h_pool2, h_pool1, h_conv1], feed_dict(False))
    # h_pool2_np, h_pool1_np, h_conv1_np = results
    # h_pool2_np.dump("h_pool2.np")
    # h_pool1_np.dump("h_pool1.np")
    # h_conv1_np.dump("h_conv1.np")
    return

  train_loss_list = []
  train_acc_list = []
  test_acc_list = []
  max_acc = 0
  start_time = time.time()
  for i in range(FLAGS.max_steps):
    if i % 10 == 0:  # Record summaries and test-set accuracy
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (i, acc))
      train_loss, train_acc = sess.run([cross_entropy, accuracy], feed_dict(True))
      test_acc, = sess.run([accuracy], feed_dict(False))
      print(train_loss, train_acc, test_acc)
      train_loss_list.append(train_loss)
      train_acc_list.append(train_acc)
      test_acc_list.append(test_acc)
      # saver.save(sess, FLAGS.model_dir + "/mnist", global_step=i, write_meta_graph=False)
      if acc > max_acc:
        saver.save(sess, FLAGS.model_dir + "/mnist", global_step=i)
        max_acc = acc

    # else:  # Record train set summaries, and train
    #   if i % 100 == 1:  # Record execution stats
    #     run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    #     run_metadata = tf.RunMetadata()
    #     summary, _ = sess.run([merged, train_step],
    #                           feed_dict=feed_dict(True),
    #                           options=run_options,
    #                           run_metadata=run_metadata)
    #     train_loss, train_acc = sess.run([cross_entropy, accuracy], feed_dict(True))
    #     test_acc = sess.run([accuracy], feed_dict(False))
    #     print(train_loss, train_acc, test_acc)
    #     train_loss_list.append(train_loss)
    #     train_acc_list.append(train_acc)
    #     test_acc_list.append(test_acc)
    #     train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
    #     train_writer.add_summary(summary, i)
    #     saver.save(sess, FLAGS.model_dir + "/mnist", global_step=i, write_meta_graph=False)
    #     print('Adding run metadata for', i)
    #   else:  # Record a summary
    summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
    train_writer.add_summary(summary, i)

  end_time = time.time()
  print("training time: {}".format(end_time - start_time))
  train_writer.close()
  test_writer.close()

  # dump data
  np.asarray(train_loss_list).dump("train_loss_list.np")
  np.asarray(train_acc_list).dump("train_acc_list.np")
  np.asarray(test_acc_list).dump("test_acc_list.np")

  # Matlotlib code to plot the loss and accuracies
  eval_indices = range(0, FLAGS.max_steps, 10)
  # Plot loss over time
  plt.plot(eval_indices, train_loss_list, 'k-')
  plt.title('Softmax Loss per Generation')
  plt.xlabel('Generation')
  plt.ylabel('Softmax Loss')
  plt.savefig("loss.pdf")

  # Plot train and test accuracy
  plt.plot(eval_indices, train_acc_list, 'k-', label='Train Set Accuracy')
  plt.plot(eval_indices, test_acc_list, 'r--', label='Test Set Accuracy')
  plt.title('Train and Test Accuracy')
  plt.xlabel('Generation')
  plt.ylabel('Accuracy')
  plt.legend(loc='lower right')
  plt.savefig("accuracy.pdf")

def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--max_steps', type=int, default=20000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--visualize', dest="visualize", action='store_true',
                      help='If true, run visualize code')
  parser.add_argument('--no-visualize', dest="visualize", action='store_false',
                      help='If true, run visualize code')
  parser.set_defaults(visualize=True)

  parser.add_argument('--cpu', dest="cpu", action='store_true',
                      help='If true, use cpu')
  parser.add_argument('--gpu', dest="cpu", action='store_false')
  parser.set_defaults(cpu=True)
  parser.add_argument('--dropout', type=float, default=0.5,
                      help='Keep probability for training dropout.')
  parser.add_argument(
    '--data_dir',
    type=str,
    default='/tmp/tensorflow/mnist/input_data',
    help='Directory for storing input data')
  parser.add_argument(
    '--log_dir',
    type=str,
    default='/tmp/tensorflow/mnist/logs/mnist_with_summaries',
    help='Summaries log directory')
  parser.add_argument(
      '--model_dir',
      type=str,
      default='checkpoint',
      help='model checkpoint directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
