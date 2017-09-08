from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf
import my_cifar10_input

from six.moves import urllib
import tarfile

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', './cifar10_data',
                           """Path to the CIFAR-10 data directory.""")

IMAGE_SIZE = my_cifar10_input.IMAGE_SIZE
NUM_CLASSES = my_cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = my_cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = my_cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

def test_input():
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    images, labels = my_cifar10_input.test_input(data_dir, FLAGS.batch_size)
    return images, labels

def distorted_inputs():
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    images, labels = my_cifar10_input.distorted_inputs(data_dir=data_dir,
                                                       batch_size=FLAGS.batch_size)
    return images, labels

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = x.op.name
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))

def print_activations(t):
  print(t.op.name, ' ', t.get_shape().as_list())

def inference(images):
    '''
    Build AlexNet model
    :param images:
    :return: logit
    '''
    parameters = []
    with tf.variable_scope('conv1') as scope:
        # get_variable: allow easy reuse
        kernel = tf.get_variable(name='weights', shape=[5,5,3,64],
                                 dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(images, kernel, [1,1,1,1], padding='SAME')
        biases = tf.get_variable(name='biases', shape=[64],
                                 dtype=tf.float32, initializer=tf.constant_initializer(value=0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        parameters += [kernel, biases]
        _activation_summary(conv1)
    print_activations(conv1)

    with tf.variable_scope('lrn1') as scope:
        lrn1 = tf.nn.local_response_normalization(conv1,
                                                  alpha=1e-4,
                                                  beta=0.75,
                                                  depth_radius=2,
                                                  bias=2.0)

    pool1 = tf.nn.max_pool(lrn1,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool1'
                           )
    print_activations(pool1)

    with tf.variable_scope('conv2') as scope:
        kernel = tf.get_variable(name='weights', shape=[4,4,64,192],
                                 dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(name='biases', shape=[192],
                                 dtype=tf.float32, initializer=tf.constant_initializer(value=0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        parameters += [kernel, biases]
        _activation_summary(conv2)
    print_activations(conv2)

    with tf.variable_scope('lrn2') as scope:
        lrn2 = tf.nn.local_response_normalization(conv2,
                                              alpha=1e-4,
                                              beta=0.75,
                                              depth_radius=2,
                                              bias=2.0)

    pool2 = tf.nn.max_pool(lrn2,
                         ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='pool2')
    print_activations(pool2)

    with tf.variable_scope('conv3') as scope:
        kernel = tf.get_variable(name = 'weights', shape = [3, 3, 192, 384],
                                 dtype = tf.float32, initializer = tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(name = 'biases', shape = [384],
                                 dtype= tf.float32, initializer=tf.constant_initializer(value=0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope.name)
        parameters += [kernel, biases]
        _activation_summary(conv3)
    print_activations(conv3)

    with tf.variable_scope('conv4') as scope:
        kernel = tf.get_variable(name='weights', shape=[3, 3, 384, 256],
                                 dtype = tf.float32, initializer = tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(name = 'biases', shape=[256],
                                 dtype=tf.float32, initializer=tf.constant_initializer(value=0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope.name)
        parameters += [kernel, biases]
        _activation_summary(conv4)
    print_activations(conv4)

    with tf.variable_scope('conv5') as scope:
        kernel = tf.get_variable(name = 'weights', shape=[3, 3, 256, 256],
                                 dtype = tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(name='biases', shape=[256],
                                 dtype=tf.float32, initializer=tf.constant_initializer(value=0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope.name)
        parameters += [kernel, biases]
        _activation_summary(conv5)
    print_activations(conv5)

    #[-1, 6, 6, 256]
    pool5 = tf.nn.max_pool(conv5,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool5'
                           )
    print_activations(pool5)

    with tf.variable_scope('fc1') as scope:
        #[-1, 6*6*256]
        pool5_reshape = tf.reshape(pool5, [FLAGS.batch_size, 3 * 3 * 256])
        weights = tf.get_variable(name='weights', shape=[3 * 3 * 256, 1024],
                                  dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable(name='biases', shape=[1024],
                                 dtype=tf.float32, initializer=tf.constant_initializer(value=0.0))
        #[-1, 1024]
        fc1 = tf.nn.relu(tf.matmul(pool5_reshape, weights) + biases, name=scope.name)
        parameters += [weights, biases]
        _activation_summary(fc1)
    print_activations(fc1)

    with tf.variable_scope('dp1') as scope:
        dp1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('fc2') as scope:
        weights = tf.get_variable(name='weights', shape=[1024, 200],
                                  dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable(name='biases', shape=[200],
                                 dtype=tf.float32, initializer=tf.constant_initializer(value=0.0))
        fc2 = tf.nn.relu(tf.matmul(dp1, weights) + biases, name=scope.name)
        parameters += [weights, biases]
        _activation_summary(fc2)
    print_activations(fc2)

    with tf.variable_scope('dp2') as scope:
        dp2 = tf.nn.dropout(fc2, 0.5)

    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable(name='weights', shape=[200, 10],
                                  dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable(name='biases', shape=[10],
                                 dtype=tf.float32, initializer=tf.constant_initializer(value=0.0))
        logit = tf.add(tf.matmul(dp2, weights), biases, name=scope.name)
        parameters += [weights, biases]
        _activation_summary(logit)
    print_activations(logit)

    return logit, parameters

def loss(logit, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logit, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return cross_entropy_mean

def train(loss, global_step):
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply([loss])
    tf.summary.scalar(loss.op.name + ' (raw)', loss)
    tf.summary.scalar(loss.op.name, loss_averages.average(loss))

    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, variable_averages.average(var))

    opt = tf.train.AdamOptimizer(learning_rate=lr)
    grads = opt.compute_gradients(loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.group(loss_averages_op, variables_averages_op)

    return train_op

def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = FLAGS.data_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
  if not os.path.exists(extracted_dir_path):
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)