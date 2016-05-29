#load
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time

import tensorflow.python.platform

import numpy as np
import tensorflow as tf
import sys
sys.path.append('..')

import rnn_cell 
from tensorflow.models.rnn import seq2seq
import reader

class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    lstm_cell = rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
    if is_training and config.keep_prob < 1:
      lstm_cell = rnn_cell.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)
    cell = rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

    self._initial_state = cell.zero_state(batch_size, tf.float32)

    with tf.device("/cpu:0"):
      embedding = tf.get_variable("embedding", [vocab_size, size])
      print ('make embedding with %s and name: %s' %([vocab_size, size], embedding.name))
      inputs = tf.split(
          1, num_steps, tf.nn.embedding_lookup(embedding, self._input_data))
      print('inputs[0] shape: %s'  % inputs[0].get_shape())
      inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
      print('new inputs[0] shape: %s'  % inputs[0].get_shape())
    if is_training and config.keep_prob < 1:
      inputs = [tf.nn.dropout(input_, config.keep_prob) for input_ in inputs]

    # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # from tensorflow.models.rnn import rnn
    # outputs, states = rnn.rnn(cell, inputs, initial_state=self._initial_state)
    outputs = []
    states = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step, input_ in enumerate(inputs):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(input_, state)
        outputs.append(cell_output)
        states.append(state)

    output = tf.reshape(tf.concat(1, outputs), [-1, size])
    print ('softmax_w: %s' %[size, vocab_size])
    print ('softmax_b: %s' %[vocab_size])
    logits = tf.nn.xw_plus_b(output,
                             tf.get_variable("softmax_w", [size, vocab_size]),
                             tf.get_variable("softmax_b", [vocab_size]))
    loss = seq2seq.sequence_loss_by_example([logits],
                                            [tf.reshape(self._targets, [-1])],
                                            [tf.ones([batch_size * num_steps])],
                                            vocab_size)
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = states[-1]

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

  def assign_lr(self, session, lr_value):
    session.run(tf.assign(self.lr, lr_value))

  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


  
flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None, "data_path")

FLAGS = flags.FLAGS

FLAGS.data_path='/Users/marting/scratch/tensorflow/simple-examples/data/'    
raw_data = reader.ptb_raw_data(FLAGS.data_path)
train_data, valid_data, test_data, _ = raw_data
class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000

config = SmallConfig()
eval_config = SmallConfig()
eval_config.batch_size = 1
eval_config.num_steps = 1
size = 200
vocab_size = 10000
num_layers = 2
batch_size = 20
      # W0 = tf.get_variable("RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix", [400, 800])
      # b0 = tf.get_variable("RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias", [800])
      # W0 = session.run(W0) 
      # b0 = session.run(b0)
      # W1 = tf.get_variable("RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix", [400, 800])
      # b1 = tf.get_variable("RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias", [800])
      # W1 = session.run(W1) 
      # b1 = session.run(b1)

import reader
voc = reader._build_vocab('/Users/marting/scratch/tensorflow/simple-examples/data/ptb.train.txt')
cov = {v:k for k,v in voc.items()}

with tf.Graph().as_default(), tf.Session() as session:
  with tf.variable_scope("model", reuse=None, initializer=None):
     m = PTBModel(is_training=False, config=config)
  saver = tf.train.Saver()
  saver.restore(session, "/Users/marting/scratch/tensorflow/model.ckpt")
  with tf.variable_scope("model", reuse=True):
      print("Model restored.")
      embedding = tf.get_variable("embedding", [vocab_size, size])
      softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
      softmax_b = tf.get_variable("softmax_b", [vocab_size])
      softmax_w = session.run(softmax_w)
      softmax_b = session.run(softmax_b)
      embedding = session.run(embedding)
      nextword = 'food'
      wordvec=embedding[voc[nextword]]
      print (wordvec)
      lstm_cell = rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
      cell = rnn_cell.MultiRNNCell([lstm_cell] * num_layers)
      state = cell.zero_state(1, tf.float32)
      input = tf.convert_to_tensor(wordvec)
      with tf.variable_scope("RNN"):
          #state = tf.reshape(state, [1,800])
          input = tf.reshape(input, [1, 200])
          (cell_output, state) = cell(input, state)
          cell_output=cell_output.eval()
          print (cell_output.shape)
          nextword = cov[np.argmax((cell_output.dot(softmax_w)+softmax_b))]
          print (nextword)


    

with tf.Graph().as_default(), tf.Session() as session:
  with tf.variable_scope("model", reuse=True):
      m = PTBModel(is_training=False, config=config)
      saver = tf.train.Saver()
      saver.restore(session, "/Users/marting/scratch/tensorflow/model.ckpt")
      print("Model restored.")
      embedding = tf.get_variable("embedding", [vocab_size, size])
      softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
      softmax_b = tf.get_variable("softmax_b", [vocab_size])
      softmax_w = session.run(softmax_w)
      softmax_b = session.run(softmax_b)
      embedding = session.run(embedding)
      lstm_cell = rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
      cell = rnn_cell.MultiRNNCell([lstm_cell] * num_layers)
      state = cell.zero_state(1, tf.float32)
      nextword = 'table'
      with tf.variable_scope("RNN"):
          state = tf.reshape(state, [1, 800])
          print (nextword)
          input = tf.convert_to_tensor(embedding[voc[nextword]])
          input = tf.reshape(input, [1, 200])
          (cell_output, state) = cell(input, state)
          cell_output = cell_output.eval()


      with tf.variable_scope("RNN"):
          #state = tf.reshape(state, [1,800])
          for i in range(20):
              print (nextword)
              input = tf.convert_to_tensor(embedding[voc[nextword]])
              input = tf.reshape(input, [1, 200])
              (cell_output, state) = cell(input, state)
              cell_output = cell_output.eval()
              ix = np.argmax((cell_output.dot(softmax_w)+softmax_b))
              print (ix)
              nextword = cov[ix]
    


#just need to multiplly the input word embedding by the weight matrix and adding bias term if rnn
#output = tf.tanh(linear.linear([inputs, state], self._num_units, True))
#if LSTM then need to do more

# multi RNN so the outputs of layer become the inputs of the next layer.
#output=state=output
#each output is linearly mapped and then softmaxed
#    logits = tf.nn.xw_plus_b(output,                             tf.get_variable("softmax_w", [size, vocab_size]),                             tf.get_variable("softmax_b", [vocab_size]))
