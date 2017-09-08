import sys
import tensorflow as tf
import numpy as np
import argparse
from sparse_op import sparse_dense_matmul_b
import read_image
from tensorflow.examples.tutorials.mnist import input_data
import time

argparser = argparse.ArgumentParser()
argparser.add_argument("-1", "--train", action="store_true",
    help="train dense MNIST model with 20000 iterations")
argparser.add_argument("-2", "--prune", action="store_true",
    help="prune model and retrain, iteratively")
argparser.add_argument("-3", "--sparse", action="store_true",
    help="transform model to a sparse format and save it")
argparser.add_argument("-m", "--checkpoint", default="./models/model_ckpt_dense",
    help="Target checkpoint model file for 2nd and 3rd round")
argparser.add_argument("-i", "--iteration", default="20",
    help="iteration number")
argparser.add_argument("-p", "--percentage", default="0.7",
    help="pruning percentage in every iteration")
args = argparser.parse_args()
if(args.train or args.prune or args.sparse) == False:
    argparser.print_help()
    sys.exit()
mnist = input_data.read_data_sets('H:/data/', one_hot=True)
sess = tf.Session()
file_obj = open("log.txt", "w")

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
    return h_pool2_flat, h_fc1, logit

def test(predict_logit):
    correct_prediction = tf.equal(tf.argmax(predict_logit,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = 0
    for i in range(20):
        batch = mnist.test.next_batch(500)
        result = result + sess.run(accuracy, feed_dict={x:batch[0], y_:batch[1], keep_prob : 1.0})
    result = result / 20.0
    return result

def prune(weights, th):
    '''
    :param weights: weight Variable
    :param th: float value, weight under th will be pruned
    :return: sparse_weight: weight matrix after pruning, which is a 2d numpy array
              ~under_threshold: boolean matrix in same shape with sparse_weight indicating whether corresponding element is zero
    '''
    shape = weights.shape
    weight_arr = sess.run(weights)
    under_threshold = abs(weight_arr) < th
    weight_arr[under_threshold] = 0
    tmp = weight_arr
    #set last matrix elemet to a small number, I have to do that since the drawback of tensorflow sparse matrix support
    #hope it would have less impact on model
    for i in range(len(shape) - 1):
        tmp = tmp[-1]
    if(tmp[-1] == 0):
        tmp[-1] = 0.01
    count = np.sum(under_threshold)
    #print ("None-zero element: %s" % (weight_arr.size - count))
    return weight_arr, ~under_threshold

def get_th(weight, percentage=0.8):
    flat = tf.reshape(weight, [-1])
    flat_list = sorted(map(abs,sess.run(flat)))
    return flat_list[int(len(flat_list) * percentage)]

def delete_none_grads(grads):
    count = 0
    length = len(grads)
    while(count < length):
        if(grads[count][0] == None):
            del grads[count]
            length -= 1
        else:
            count += 1

def transfer_to_sparse(weight):
    weight_arr = sess.run(weight)
    values = weight_arr[weight_arr != 0]
    indices = np.transpose(np.nonzero(weight_arr))
    shape = list(weight_arr.shape)
    return [indices, values, shape]

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

if(args.train == True):
    x = tf.placeholder(tf.float32, [None, 784], name="x")
    y_ = tf.placeholder(tf.float32, [None, 10], name="y_")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    useless1, useless2, logit = dense_cnn_model(x, dense_w, keep_prob)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y_)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.arg_max(logit,1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())

    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_acc = sess.run(accuracy, feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
            print("step %d, training acc %g" % (i , train_acc))
        sess.run(train_step, feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

    start = time.clock()
    test_acc = test(logit)
    end = time.clock()
    print("test acc %g" % test_acc + ", inference time:" + str(end - start))
    saver = tf.train.Saver()
    saver.save(sess, "./models/model_ckpt_dense")

if(args.prune == True):
    print("total pruning iteration: %d. pruning percentage each iter: %g" % (int(args.iteration), float(args.percentage)))
    file_obj.write("total pruning iteration: %d. pruning percentage each iter: %g\n" % (int(args.iteration), float(args.percentage)))
    saver = tf.train.Saver()
    saver.restore(sess, args.checkpoint)
    p = 1.0
    for i in range(int(args.iteration)):
        p = p * float(args.percentage)
        print("\033[0;31miteration %d, p=%g\033[0m" % (i, p))
        file_obj.write("iteration %d, p=%g\n" % (i, p))
        file_obj.flush()
        th_fc1 = get_th(dense_w["w_fc1"], percentage=(1.0 - p))
        th_fc2 = get_th(dense_w["w_fc2"], percentage=(1.0 - p))
        sp_w_fc1, idx_fc1 = prune(dense_w["w_fc1"], th_fc1)
        sp_w_fc2, idx_fc2 = prune(dense_w["w_fc2"], th_fc2)
        sess.run(tf.assign(dense_w["w_fc1"], sp_w_fc1))
        sess.run(tf.assign(dense_w["w_fc2"], sp_w_fc2))

        array_wfc1 = sess.run(dense_w["w_fc1"])
        array_wfc2 = sess.run(dense_w["w_fc2"])

        print("none-zero in fc1 :%d" % np.sum(array_wfc1 != 0))
        print("none-zero in fc2 :%d" % np.sum(array_wfc2 != 0))

        for var in tf.global_variables():
            if sess.run(tf.is_variable_initialized(var)) == False:
                sess.run(var.initializer)

        x = tf.placeholder(tf.float32, [None, 784], name="x")
        y_ = tf.placeholder(tf.float32, [None, 10], name="y_")
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        useless1, useless2, logit = dense_cnn_model(x, dense_w, keep_prob)
        test_acc = test(logit)
        print("\033[0;31mtest acc after iteration %d pruning: %g\033[0m" % (i, test_acc))
        file_obj.write("test acc after iteration %d pruning: %g\n" % (i, test_acc))

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y_)
        trainer = tf.train.AdamOptimizer(1e-4)
        grads = trainer.compute_gradients(cross_entropy)
        delete_none_grads(grads)

        count = 0
        for grad, var in grads:
            if (var.name == "w_fc1:0"):
                idx_in1 = tf.cast(tf.constant(idx_fc1), tf.float32)
                grads[count] = (tf.multiply(idx_in1, grad), var)
            if (var.name == "w_fc2:0"):
                idx_in2 = tf.cast(tf.constant(idx_fc2), tf.float32)
                grads[count] = (tf.multiply(idx_in2, grad), var)
            count += 1
        train_step = trainer.apply_gradients(grads)

        correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        for var in tf.global_variables():
            if sess.run(tf.is_variable_initialized(var)) == False:
                sess.run(var.initializer)

        for j in range(10000):
            batch = mnist.train.next_batch(50)
            idx_in1_value = sess.run(idx_in1)
            grads_fc1_value = sess.run(grads, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            if (j % 3000 == 0 or j == 100 or j == 500 or j == 1000 or j == 1500 or j == 9999):
                train_acc = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
                print("retraining step %d, acc %g" % (j, train_acc))
            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        array_wfc1 = sess.run(dense_w["w_fc1"])
        array_wfc2 = sess.run(dense_w["w_fc2"])

        print("none-zero in fc1 after retrain:%d" % np.sum(array_wfc1 != 0))
        print("none-zero in fc2 after retrain:%d" % np.sum(array_wfc2 != 0))

        test_acc = test(logit)
        print("\033[0;31mtest acc after interation %d pruning and retraining: %g\033[0m" % (i, test_acc))
        file_obj.write("test acc after interation %d pruning and retraining: %g\n" % (i, test_acc))

        saver = tf.train.Saver(dense_w)
        saver.save(sess, "./models/model_ckpt_dense_retrained", global_step=i)

    saver = tf.train.Saver(dense_w)
    saver.save(sess, "./models/model_ckpt_dense_retrained")

if(args.sparse == True):
    saver = tf.train.Saver()
    saver.restore(sess, "./models/model_ckpt_dense_retrained-16")

    sparse_w = {
        "w_conv1": tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1)),
        "b_conv1": tf.Variable(tf.constant(0.1, shape=[32])),
        "w_conv2": tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1)),
        "b_conv2": tf.Variable(tf.constant(0.1, shape=[64])),
        "w_fc1": tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1)),
        "b_fc1": tf.Variable(tf.constant(0.1, shape=[1024])),
        "w_fc2": tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1)),
        "b_fc2": tf.Variable(tf.constant(0.1, shape=[10]))
    }

    def sparse_cnn_model(image, sparse_weight):
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        x_image = tf.reshape(image, [-1, 28, 28, 1])
        h_conv1 = tf.nn.relu(conv2d(x_image, sparse_weight["w_conv1"]) + sparse_weight["b_conv1"])
        # [-1,14,14,32]
        h_pool1 = max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(conv2d(h_pool1, sparse_weight["w_conv2"]) + sparse_weight["b_conv2"])
        # [-1,7,7,64]
        h_pool2 = max_pool_2x2(h_conv2)
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        ndarray_w_fc1_idx = sess.run(sparse_weight["w_fc1_idx"])
        ndarray_w_fc1 = sess.run(sparse_weight["w_fc1"])
        ndarray_w_fc1_shape = sess.run(sparse_weight["w_fc1_shape"])
        h_fc1 = tf.nn.relu(sparse_dense_matmul_b(ndarray_w_fc1_idx, ndarray_w_fc1, ndarray_w_fc1_shape, h_pool2_flat, True) + sparse_weight["b_fc1"])
        ndarray_w_fc2_idx = sess.run(sparse_weight["w_fc2_idx"])
        ndarray_w_fc2 = sess.run(sparse_weight["w_fc2"])
        ndarray_w_fc2_shape = sess.run(sparse_weight["w_fc2_shape"])
        logit = sparse_dense_matmul_b(ndarray_w_fc2_idx, ndarray_w_fc2, ndarray_w_fc2_shape, h_fc1, True) + sparse_weight["b_fc2"]
        return h_pool2_flat, h_fc1, logit

    copy_ops = []
    for key, value in dense_w.items():
        copy_ops.append(sparse_w[key].assign(value))
    for e in copy_ops:
        sess.run(e)

    fc1_sparse_tmp = transfer_to_sparse(dense_w["w_fc1"])
    sparse_w["w_fc1_idx"] = tf.Variable(tf.constant(fc1_sparse_tmp[0], dtype=tf.int64)\
                                        , name="w_fc1_idx")
    sparse_w["w_fc1"] = tf.Variable(tf.constant(fc1_sparse_tmp[1], dtype=tf.float32)\
                                    , name="w_fc1")
    sparse_w["w_fc1_shape"] = tf.Variable(tf.constant(fc1_sparse_tmp[2], dtype=tf.int64)\
                                          , name="w_fc1_shape")
    fc2_sparse_tmp = transfer_to_sparse(dense_w["w_fc2"])
    sparse_w["w_fc2_idx"] = tf.Variable(tf.constant(fc2_sparse_tmp[0], dtype=tf.int64)\
                                        , name="w_fc2_idx")
    sparse_w["w_fc2"] = tf.Variable(tf.constant(fc2_sparse_tmp[1], dtype=tf.float32)\
                                    , name="w_fc2")
    sparse_w["w_fc2_shape"] = tf.Variable(tf.constant(fc2_sparse_tmp[2], dtype=tf.int64)\
                                          , name="w_fc2_shape")

    for var in tf.global_variables():
        if sess.run(tf.is_variable_initialized(var)) == False:
            sess.run(var.initializer)

    x = tf.placeholder(tf.float32, [None, 784], name="x")
    y_ = tf.placeholder(tf.float32, [None, 10], name="y_")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    tf.add_to_collection("x_placeholder", x)

    #dense_prediction
    dense_pool2_flat, dense_h_fc1, dense_logit = dense_cnn_model(x, dense_w, keep_prob)
    #sparse prediction
    sp_pool2_flat, sp_h_fc1, sp_logit = sparse_cnn_model(x, sparse_w)

    start = time.clock()
    test_acc_dense = test(dense_logit)
    end = time.clock()
    print("dense acc:" + str(test_acc_dense) + ", inference time:" + str(end - start))

    start = time.clock()
    test_acc_sp = test(sp_logit)
    end = time.clock()
    print("sp acc" + str(test_acc_sp) + ", inference time:" + str(end - start))

    tf.add_to_collection("sp_logit", sp_logit)

    sparse_saver = tf.train.Saver(sparse_w)
    sparse_saver.save(sess, "./models/model_ckpt_sparse_retrained")

file_obj.close()