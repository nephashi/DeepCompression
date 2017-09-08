import tensorflow as tf

sess = tf.Session()

def test(predict_logit):
    correct_prediction = tf.equal(tf.arg_max(predict_logit,1), tf.arg_max(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = 0
    for i in range(20):
        batch = mnist.test.next_batch(500)
        result = result + sess.run(accuracy, feed_dict={x:batch[0], y_:batch[1], keep_prob : 1.0})
    result = result / 20.0
    return result

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('H:/data/', one_hot=True)

dense_w = {
    "w_conv1":tf.Variable(tf.truncated_normal([5,5,1,32], stddev=0.1), name="w_conv1"),
    "b_conv1":tf.Variable(tf.constant(0.1, shape=[32]), name="b_conv1"),
    "w_conv2":tf.Variable(tf.truncated_normal([5,5,32,64], stddev=0.1), name="w_conv2"),
    "b_conv2":tf.Variable(tf.constant(0.1, shape=[64]), name="b_conv2"),
    "w_fc1":tf.Variable(tf.truncated_normal([7*7*64,1024], stddev=0.1), name="w_fc1"),
    "b_fc1":tf.Variable(tf.constant(0.1, shape=[1024]), name="b_fc1"),
    "w_fc2":tf.Variable(tf.truncated_normal([1024,10], stddev=0.1), name="w_fc2"),
    "b_fc2":tf.Variable(tf.constant(0.1, shape=[10]), name="b_fc2")
}

def dense_cnn_model(image, weights, keep_prob):
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
    x_image = tf.reshape(image, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, weights["w_conv1"]) + weights["b_conv1"])
    #[-1,14,14,32]
    h_pool1 = max_pool_2x2(h_conv1)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, weights["w_conv2"]) + weights["b_conv2"])
    #[-1,7,7,64]
    h_pool2 = max_pool_2x2(h_conv2)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, weights["w_fc1"]) + weights["b_fc1"])
    h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob=keep_prob)
    #[-1,10]
    logit = tf.matmul(h_fc1_dropout, weights["w_fc2"]) + weights["b_fc2"]
    return logit

def dense_conv_model(image, weights):
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
    x_image = tf.reshape(image, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, weights["w_conv1"]) + weights["b_conv1"])
    #[-1,14,14,32]
    h_pool1 = max_pool_2x2(h_conv1)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, weights["w_conv2"]) + weights["b_conv2"])
    #[-1,7,7,64]
    h_pool2 = max_pool_2x2(h_conv2)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    return h_pool2_flat

saver = tf.train.import_meta_graph("./model_ckpt_dense.meta")
saver.restore(sess, "./model_ckpt_dense")

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)
for var in tf.all_variables():
    if sess.run(tf.is_variable_initialized(var)) == False:
        sess.run(tf.initialize_variables([var]))

logit = dense_cnn_model(x, dense_w, keep_prob)

test_acc = test(logit)
print(test_acc)

result = 0
for i in range(20):
    batch = mnist.test.next_batch(1)
    correct_prediction = tf.equal(tf.arg_max(logit, 1), tf.arg_max(y_, 1))
    accuracy = sess.run(correct_prediction, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
    print(1)



# pool2_flat = dense_conv_model(x, dense_w)

# for i in range(20):
#     batch = mnist.test.next_batch(1)
#     tmp_pool2_flat = sess.run(pool2_flat, feed_dict={x:batch[0]})
#     tmp_pool2_flat_variable = tf.Variable(tmp_pool2_flat, dtype=tf.float32)
#     sess.run(tf.initialize_variables([tmp_pool2_flat_variable]))
#     weight_fc1 = dense_w["w_fc1"]
#     tmp_fc1 = sess.run(tf.matmul(tmp_pool2_flat_variable, weight_fc1) + dense_w["b_fc1"])
#     tmp_fc1_variable = tf.Variable(tmp_fc1, dtype=tf.float32)
#     sess.run(tf.initialize_variables([tmp_fc1_variable]))
#     h_fc1 = tf.nn.relu(tmp_fc1_variable)
#     tmp_h_fc1 = sess.run(h_fc1)
#     tmp_h_fc1_variable = tf.Variable(tmp_h_fc1, dtype=tf.float32)
#     sess.run(tf.initialize_variables([tmp_h_fc1_variable]))
#     weight_fc2 = dense_w["w_fc2"]
#     tmp_fc2 = sess.run(tf.matmul(tmp_h_fc1_variable, weight_fc2) + dense_w["b_fc2"])
#     tmp_fc2_variable = tf.Variable(tmp_fc2, dtype=tf.float32)
#     sess.run(tf.initialize_variables([tmp_fc2_variable]))
#
#     if_correct = tf.equal(tf.argmax(tmp_fc2_variable, 1), tf.argmax(y_, 1))
#     acc = sess.run(if_correct,feed_dict={y_:batch[1]})
#     print(1)


