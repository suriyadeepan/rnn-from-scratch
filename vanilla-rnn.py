import tensorflow as tf
import numpy as np

import data
import utils


# step operation
def step(hprev, x):
    #hprev (shape=[hsize], dtype=tf.float32)
    #x (shape=[isize], dtype=tf.float32)
    # reshape vectors to matrices
    hprev = tf.reshape(hprev, [1, hsize])
    x = tf.reshape(x, [1, isize])
    # initializer
    xav_init = tf.contrib.layers.xavier_initializer()
    # params
    W = tf.get_variable('W', shape=[hsize, hsize], initializer=xav_init)
    U = tf.get_variable('U', shape=[isize, hsize], initializer=xav_init)
    b = tf.get_variable('b', shape=[hsize], initializer=tf.constant_initializer(0.))
    # current hidden state
    h = tf.tanh(tf.matmul(hprev, W) + tf.matmul(x,U) + b)
    h = tf.reshape(h, [hsize])
    return h

if __name__ == '__main__':
    # build graph
    tf.reset_default_graph()
    # inputs
    xs_ = tf.placeholder(shape=[None, isize], dtype=tf.float32)
    ys_ = tf.placeholder(shape=[None, osize], dtype=tf.float32)
    #
    # initial hidden state
    init_state = tf.zeros([hsize])
    #
    # here comes the scan operation; wake up!
    states = tf.scan(step, xs_, initializer=init_state) # tf.scan(fn, elems, initializer)
    #
    # predictions
    V = tf.get_variable('V', shape=[hsize, osize], 
                        initializer=tf.contrib.layers.xavier_initializer())
    bo = tf.get_variable('bo', shape=[osize], initializer=tf.constant_initializer(0.))
    preds = tf.matmul(states,V) + bo
    #
    # optimization
    loss = tf.reduce_mean(tf.square(preds - ys_))
    train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
    # 
    # get training set
    X, Y, idx2ch, ch2idx = data.load_data()
    trainset = utils.rand_batch_gen(X, Y, batch_size=1)
    #
    # training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        xs, ys = trainset.__next__()
        train_loss = 0
        for i in range(epochs):
            _, train_loss_ = sess.run([train_op, loss], feed_dict = {
                    xs_ : xs,
                    ys_ : ys
                })
            train_loss += train_loss_
            if(i%100 == 0 and i):
                print('[{}] loss : {}'.format(i,train_loss/100))
                train_loss = 0
