import tensorflow as tf
import numpy as np

def printtt():
    pass

def sparse_dense_matmul_b(sp_indices, sp_value, sp_shape, b, swap = False):
    '''
    multiplication of a sparse matrix and a dense matrix
    first three params is exactly the three parameter of SparseTensor constructor
    a * b if swap==False else b * sp_a
    shape error happens in extremely sparse case, where right-bottom margin elements are all zero
    I do not know how to fix it, seldom happens in practice
    set last number of sparse matrix to a very small number is recommanded
    :param sp_indices:
    :param sp_value:
    :param sp_shape:
    :param b:
    :param swap:
    :return:
    '''
    if(not swap):
        sp_a = tf.SparseTensor(sp_indices, sp_value, sp_shape)
        return tf.sparse_tensor_dense_matmul(sp_a, b)
    else:
        b = tf.transpose(b)
        sp_indices = np.array(sp_indices)
        internal_sp_indices = sp_indices[:,1]
        tmp = []
        for c in internal_sp_indices:
            tmp.append([c, 0])
        internal_sp_indices = tmp
        sp_indice_value = sp_indices[:, 0]
        sp_value = np.array(sp_value).astype(float)
        tmp1 = tf.sparse_reorder(tf.SparseTensor(indices=internal_sp_indices, values=sp_indice_value, dense_shape=sp_shape))
        tmp2 = tf.sparse_reorder(tf.SparseTensor(indices=internal_sp_indices, values=sp_value, dense_shape=sp_shape))
        y = tf.transpose(tf.nn.embedding_lookup_sparse(b, tmp1, tmp2, combiner="sum"))
        return y

def sparse_dense_matmul(sp_a, b, swap=False):
    '''
    multiplication of a sparse matrix and a dense matrix
    sp_a * b if swap==False else b * sp_a
    shape error happens in extremely sparse case, where right-bottom margin elements are all zero
    I do not know how to fix it, seldom happens in practice
    set last number of sparse matrix to a very small number is recommanded
    :param sp_a: SparseTensor
    :param b: 2dTensor
    :param swap: Boolean
    :return: Tensor
    '''
    # if(type(b[0][0]) is not type(0.1)):
    #     b = np.array(b).astype(float).tolist()
    if(not swap):
        return tf.sparse_tensor_dense_matmul(sp_a, b)
    else:
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        b = tf.transpose(b)
        sp_indices = sess.run(sp_a.indices)[:,1]
        tmp = []
        for c in sp_indices:
            tmp.append([c, 0])
        sp_indices = tmp
        sp_indice_value = sess.run(sp_a.indices)[:,0]
        sp_value = np.array(sess.run(sp_a.values)).astype(float)
        sp_shape = sess.run(sp_a.dense_shape)
        sess.close()
        tmp1 = tf.sparse_reorder(tf.SparseTensor(indices=sp_indices, values=sp_indice_value, dense_shape=sp_shape))
        tmp2 = tf.sparse_reorder(tf.SparseTensor(indices=sp_indices, values=sp_value, dense_shape=sp_shape))
        y = tf.transpose(tf.nn.embedding_lookup_sparse(b, tmp1, tmp2, combiner="sum"))
        tmp = y.shape
        return y

if __name__ == '__main__':
    a = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])

    X = tf.placeholder(tf.float32, shape=[2,3])

    x = np.array([[1,2,3],[2,4,6]], dtype=np.float32)

    mul = sparse_dense_matmul(a, X, True)
    mul2 = sparse_dense_matmul_b(sp_indices=[[0, 0], [1, 2]], sp_value=[1, 2], sp_shape=[3,4], b=X, swap=True)

    sess = tf.Session()
    print(sess.run(mul, feed_dict={X:x}))
    print(sess.run(mul2, feed_dict={X: x}))