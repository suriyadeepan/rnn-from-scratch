import tensorflow as tf
import numpy as np

import data
import utils

import argparse
import random

####
# disable logs
tf.logging.set_verbosity(tf.logging.ERROR)
#
# checkpoint
ckpt_path = 'ckpt/vanilla1/'
#
###
# get data
X, Y, idx2ch, ch2idx = data.load_data('data/paulg/')
#
# params
hsize = 256
num_classes = len(idx2ch)
seqlen = X.shape[1]
state_size = hsize
BATCH_SIZE = 128


# step operation
def step(hprev, x):
    # initializer
    xav_init = tf.contrib.layers.xavier_initializer
    # params
    W = tf.get_variable('W', shape=[state_size, state_size], initializer=xav_init())
    U = tf.get_variable('U', shape=[state_size, state_size], initializer=xav_init())
    b = tf.get_variable('b', shape=[state_size], initializer=tf.constant_initializer(0.))
    # current hidden state
    h = tf.tanh(tf.matmul(hprev, W) + tf.matmul(x,U) + b)
    return h

# parse arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description='Vanilla Recurrent Neural Network for Text Hallucination, built with tf.scan')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-g', '--generate', action='store_true',
                        help='generate text')
    group.add_argument('-t', '--train', action='store_true',
                        help='train model')
    parser.add_argument('-n', '--num_words', required=False, type=int,
                        help='number of words to generate')
    args = vars(parser.parse_args())
    return args

 
if __name__ == '__main__':
    #
    # parse arguments
    args = parse_args()
    #
    # build graph
    tf.reset_default_graph()
    # inputs
    xs_ = tf.placeholder(shape=[None, None], dtype=tf.int32)
    ys_ = tf.placeholder(shape=[None], dtype=tf.int32)
    #
    # embeddings
    embs = tf.get_variable('emb', [num_classes, state_size])
    rnn_inputs = tf.nn.embedding_lookup(embs, xs_)
    #
    # initial hidden state
    init_state = tf.placeholder(shape=[None, state_size], dtype=tf.float32, name='initial_state')
    #
    # here comes the scan operation; wake up!
    #   tf.scan(fn, elems, initializer)
    states = tf.scan(step, 
            tf.transpose(rnn_inputs, [1,0,2]),
            initializer=init_state) 
    ###
    # set last state
    last_state = states[-1]
    states = tf.transpose(states, [1,0,2])
    #
    # predictions
    V = tf.get_variable('V', shape=[state_size, num_classes], 
                        initializer=tf.contrib.layers.xavier_initializer())
    bo = tf.get_variable('bo', shape=[num_classes], 
                         initializer=tf.constant_initializer(0.))
    #
    # flatten states to 2d matrix for matmult with V
    states_reshaped = tf.reshape(states, [-1, state_size])
    logits = tf.matmul(states_reshaped, V) + bo
    predictions = tf.nn.softmax(logits)
    #
    # optimization
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, ys_)
    loss = tf.reduce_mean(losses)
    train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
    # 
    # to generate or to train - that is the question.
    if args['train']:
        # 
        # training
        #  setup batches for training
        epochs = 50
        #
        # set batch size
        batch_size = BATCH_SIZE
        train_set = utils.rand_batch_gen(X,Y,batch_size=batch_size)
        # training session
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_loss = 0
            try:
                for i in range(epochs):
                    for j in range(1000):
                        xs, ys = train_set.__next__()
                        _, train_loss_ = sess.run([train_op, loss], feed_dict = {
                                xs_ : xs,
                                ys_ : ys.reshape([batch_size*seqlen]),
                                init_state : np.zeros([batch_size, state_size])
                            })
                        train_loss += train_loss_
                    print('[{}] loss : {}'.format(i,train_loss/1000))
                    train_loss = 0
            except KeyboardInterrupt:
                print('interrupted by user at ' + str(i))
                #
                # training ends here; 
                #  save checkpoint
                saver = tf.train.Saver()
                saver.save(sess, ckpt_path + 'vanilla1.ckpt', global_step=i)
    elif args['generate']:
        #
        # generate text
        random_init_word = random.choice(idx2ch)
        current_word = ch2idx[random_init_word]
        #
        # start session
        with tf.Session() as sess:
            # init session
            sess.run(tf.global_variables_initializer())
            #
            # restore session
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
            saver = tf.train.Saver()
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            # generate operation
            words = [current_word]
            state = None
            # set batch_size to 1
            batch_size = 1
            num_words = args['num_words'] if args['num_words'] else 111
            # enter the loop
            for i in range(num_words):
                if state:
                    feed_dict = { xs_ : np.array(current_word).reshape([1, 1]), 
                            init_state : state_ }
                else:
                    feed_dict = { xs_ : np.array(current_word).reshape([1,1])
                            , init_state : np.zeros([batch_size, state_size]) }
                #
                # forward propagation
                preds, state_ = sess.run([predictions, last_state], feed_dict=feed_dict)
                # 
                # set flag to true
                state = True
                # 
                # set new word
                current_word = np.random.choice(preds.shape[-1], 1, p=np.squeeze(preds))[0]
                # add to list of words
                words.append(current_word)
        #########
        # text generation complete
        #
        print('______Generated Text_______')
        print(''.join([idx2ch[w] for w in words]))
        print('___________________________')
