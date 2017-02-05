import tensorflow as tf
import numpy as np

import data
import utils


###
# get data
X, Y, idx2w, w2idx, seqlen = data.load_data('data/shakespeare/')
#
# params
hsize = 128
num_classes = len(idx2w)
state_size = hsize


# step operation
def step(hprev, x):
    # reshape vectors to matrices
    hprev = tf.reshape(hprev, [1, hsize])
    x = tf.reshape(x, [1,state_size])
    # initializer
    xav_init = tf.contrib.layers.xavier_initializer
    # params
    W = tf.get_variable('W', shape=[hsize, hsize], initializer=xav_init())
    U = tf.get_variable('U', shape=[state_size, hsize], initializer=xav_init())
    b = tf.get_variable('b', shape=[hsize], initializer=tf.constant_initializer(0.))
    # current hidden state
    h = tf.tanh(tf.matmul(hprev, W) + tf.matmul(x,U) + b)
    h = tf.reshape(h, [hsize])
    return h


if __name__ == '__main__':
    # build graph
    tf.reset_default_graph()
    # inputs
    xs_ = tf.placeholder(shape=[None], dtype=tf.int32)
    ys_ = tf.placeholder(shape=[None], dtype=tf.int32)
    #
    # embeddings
    embs = tf.get_variable('emb', [num_classes, state_size])
    rnn_inputs = tf.nn.embedding_lookup(embs, xs_)
    #
    # initial hidden state
    init_state = tf.zeros([hsize])
    #
    # here comes the scan operation; wake up!
    states = tf.scan(step, rnn_inputs, initializer=init_state) # tf.scan(fn, elems, initializer)
    #
    # predictions
    V = tf.get_variable('V', shape=[hsize, num_classes], 
                        initializer=tf.contrib.layers.xavier_initializer())
    bo = tf.get_variable('bo', shape=[num_classes], 
                         initializer=tf.constant_initializer(0.))
    logits = tf.matmul(states,V) + bo
    #
    # optimization
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, ys_)
    loss = tf.reduce_mean(losses)
    train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
    # 
    # setup batches for training
    epochs = 10
    train_set = utils.rand_batch_gen(X,Y,batch_size=1)
    #
    # training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        xs, ys = train_set.__next__()
        train_loss = 0
        try:
            for i in range(epochs):
                for j in range(1000):
                    _, train_loss_ = sess.run([train_op, loss], feed_dict = {
                            xs_ : xs.reshape([seqlen]),
                            ys_ : ys.reshape([seqlen])
                        })
                    train_loss += train_loss_
                print('[{}] loss : {}'.format(i,train_loss/1000))
                train_loss = 0
        except KeyboardInterrupt:
            print('interrupted by user at ' + str(i))
