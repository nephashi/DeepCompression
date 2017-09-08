import tensorflow as tf
import numpy as np
import argparse
import config
import sys

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('H:/data/', one_hot=True)

def test(predict_logit):
    correct_prediction = tf.equal(tf.arg_max(predict_logit,1), tf.arg_max(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = 0
    for i in range(20):
        batch = mnist.test.next_batch(500)
        result = result + sess.run(accuracy, feed_dict={x:batch[0], y_:batch[1], keep_prob : 1.0})
    result = result / 20.0
    return result

sess = tf.Session()
saver = tf.train.import_meta_graph("./model_ckpt_sparse_retrained.meta")
saver.restore(sess, "./model_ckpt_sparse_retrained")

x = tf.get_collection("x_placeholder")[0]
y_ = tf.placeholder(tf.float32, [None, 10], name="y_")
keep_prob = tf.placeholder(tf.float32, name="keep_prob")

logit = tf.get_collection("sp_logit")[0]
test_acc_sp = test(logit)
print(test_acc_sp)
