import tensorflow as tf
import numpy as np

import utils
import data

import random
import argparse
import sys

BATCH_SIZE = 128

class GRU_rnn():

    def __init__(self, state_size, num_classes, num_layers,
            ckpt_path='ckpt/gru2/',
            model_name='gru2'):

        self.state_size = state_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.ckpt_path = ckpt_path
        self.model_name = model_name

        # build graph ops
        def __graph__():
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
            init_state = tf.placeholder(shape=[num_layers, None, state_size], 
                    dtype=tf.float32, name='initial_state')
            # initializer
            xav_init = tf.contrib.layers.xavier_initializer
            # params
            W = tf.get_variable('W', 
                    shape=[num_layers, 3, self.state_size, self.state_size], 
                    initializer=xav_init())
            U = tf.get_variable('U', 
                    shape=[num_layers, 3, self.state_size, self.state_size], 
                    initializer=xav_init())
            b = tf.get_variable('b', 
                    shape=[num_layers, self.state_size], 
                    initializer=tf.constant_initializer(0.))
            ####
            # step - GRU
            def step(st_1, x):
                # iterate through layers
                st = []
                inp = x
                for i in range(num_layers):
                    ####
                    # GATES
                    #
                    #  update gate
                    z = tf.sigmoid(tf.matmul(inp,U[i][0]) + tf.matmul(st_1[i],W[i][0]))
                    #  reset gate
                    r = tf.sigmoid(tf.matmul(inp,U[i][1]) + tf.matmul(st_1[i],W[i][1]))
                    #  intermediate
                    h = tf.tanh(tf.matmul(inp,U[i][2]) + tf.matmul( (r*st_1[i]),W[i][2]))
                    ###
                    # new state
                    st_i = (1-z)*h + (z*st_1[i])
                    inp = st_i
                    st.append(st_i)
                return tf.pack(st)
            ###
            # here comes the scan operation; wake up!
            #   tf.scan(fn, elems, initializer)
            states = tf.scan(step, 
                    tf.transpose(rnn_inputs, [1,0,2]),
                    initializer=init_state)
            #### 
            # get last state before reshape
            last_state = states[-1]
            #
            # predictions
            V = tf.get_variable('V', shape=[state_size, num_classes], 
                                initializer=xav_init())
            bo = tf.get_variable('bo', shape=[num_classes], 
                                 initializer=tf.constant_initializer(0.))
            ####
            # transpose
            states = tf.transpose(states, [1,2,0,3])[-1]
            # flatten states to 2d matrix for matmult with V
            states_reshaped = tf.reshape(states, [-1, state_size])
            logits = tf.matmul(states_reshaped, V) + bo
            # predictions
            predictions = tf.nn.softmax(logits) 
            #
            # optimization
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, ys_)
            loss = tf.reduce_mean(losses)
            train_op = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(loss)
            #
            # expose symbols
            self.xs_ = xs_
            self.ys_ = ys_
            self.loss = loss
            self.train_op = train_op
            self.predictions = predictions
            self.last_state = last_state
            self.init_state = init_state
        ##### 
        # build graph
        sys.stdout.write('\n<log> Building Graph...')
        __graph__()
        sys.stdout.write('</log>\n')

    ####
    # training
    def train(self, train_set, epochs=100):
        # training session
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_loss = 0
            try:
                for i in range(epochs):
                    for j in range(300):
                        xs, ys = train_set.__next__()
                        batch_size = xs.shape[0]
                        _, train_loss_ = sess.run([self.train_op, self.loss], feed_dict = {
                                self.xs_ : xs,
                                self.ys_ : ys.flatten(),
                                self.init_state : np.zeros([self.num_layers, batch_size, self.state_size])
                            })
                        train_loss += train_loss_
                    print('[{}] loss : {}'.format(i,train_loss/300))
                    train_loss = 0
            except KeyboardInterrupt:
                print('interrupted by user at ' + str(i))
            #
            # training ends here; 
            #  save checkpoint
            saver = tf.train.Saver()
            saver.save(sess, self.ckpt_path + self.model_name, global_step=i)
    ####
    # generate characters
    def generate(self, idx2w, w2idx, num_words=100, div=' '):
        #
        # generate text
        random_init_word = random.choice(idx2w)
        current_word = w2idx[random_init_word]
        #
        # start session
        with tf.Session() as sess:
            # init session
            sess.run(tf.global_variables_initializer())
            #
            # restore session
            ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
            saver = tf.train.Saver()
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            # generate operation
            words = [current_word]
            state = None
            # enter the loop
            for i in range(num_words):
                if state:
                    feed_dict = {self.xs_ : np.array([current_word]).reshape([1,1]),
                            self.init_state : state_}
                else:
                    feed_dict = {self.xs_ : np.array([current_word]).reshape([1,1]),
                            self.init_state : np.zeros([self.num_layers, 1, self.state_size])}
                #
                # forward propagation
                preds, state_ = sess.run([self.predictions, self.last_state], feed_dict=feed_dict)
                # 
                # set flag to true
                state = True
                # 
                # set new word
                current_word = np.random.choice(preds.shape[-1], 1, p=np.squeeze(preds))[0]
                # add to list of words
                words.append(current_word)
        ########
        # return the list of words as string
        return div.join([idx2w[w] for w in words])

### 
# parse arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description='Stacked Gated Recurrent Unit RNN for Text Hallucination, built with tf.scan')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-g', '--generate', action='store_true',
                        help='generate text')
    group.add_argument('-t', '--train', action='store_true',
                        help='train model')
    parser.add_argument('-n', '--num_words', required=False, type=int,
                        help='number of words to generate')
    args = vars(parser.parse_args())
    return args


###
# main function
if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    #
    # fetch data
    X, Y, idx2w, w2idx = data.load_data('data/paulg/')
    seqlen = X.shape[0]
    #
    # create the model
    model = GRU_rnn(state_size = 1024, num_classes=len(idx2w), num_layers=3)
    # to train or to generate?
    if args['train']:
        # get train set
        train_set = utils.rand_batch_gen(X, Y ,batch_size=BATCH_SIZE)
        #
        # start training
        model.train(train_set)
    elif args['generate']:
        # call generate method
        text = model.generate(idx2w, w2idx, 
                num_words=args['num_words'] if args['num_words'] else 100,
                div='')
        #########
        # text generation complete
        #
        print('______Generated Text_______')
        print(text)
        print('___________________________')
