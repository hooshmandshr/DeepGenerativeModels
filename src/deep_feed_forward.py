"""TensorFlow tools for deep feedforward networks."""

import tensorflow as tf
import numpy as np


class MultiLayerPerceptron(object):

  def __init__(self, units_per_layer, input_tensor=None,
      target_tensor=None):
    """Sets up TensorFlow graph for an MLP."""
    # Number of hidden layers in the MLP.
    self.depth = len(units_per_layer) - 1
    self.layers = []
    if input_tensor is not None:
      self.graph = input_tensor.graph
      self.layers.append(input_tensor)
    else:
      # Define the stand alone graph with place holders for input & output.
      self.graph = tf.Graph()
      self.feat_tensor_name = 'features'
      self.target_tensor_name = 'targets'
      input_dim = units_per_layer[0]
      with self.graph.as_default():
        input_tensor = tf.placeholder(
            dtype=tf.float32,shape=[None, input_dim],
            name=self.feat_tensor_name)
        target_tensor = tf.placeholder(
            dtype=tf.float32, shape=[None], name=self.target_tensor_name)
        self.layers.append(input_tensor)

    with self.graph.as_default():
      #layers = [input_tensor]
      for n_units in units_per_layer[1:]:
        # Add a ReLU hidden layer with n_units hidden units.
        self.layers.append(
            tf.contrib.layers.fully_connected(self.layers[-1], n_units))
      # Logistic regression output layer.
      output = tf.contrib.layers.fully_connected(
          self.layers[-1], 1, activation_fn=None)
      prediction = tf.reshape(tf.nn.sigmoid(output), shape=[-1])
      # Set up cross entropy as loss.
      loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
          labels=target_tensor, logits=prediction), name='loss')
      # Optimizer
      train_op = tf.train.GradientDescentOptimizer(
        learning_rate=0.1).minimize(loss, name='train')

  def initialize(self):
    """Initializes the weights and biases of the hidden layers."""
    with self.graph.as_default():
      self.session = tf.Session()
      init_op = tf.global_variables_initializer()
      self.session.run(init_op)

  def get_loss(self, batch_features, batch_targets):
    """Trains the MLP with a single training step.

    Args:
      batch_features: a numpy array with shape (?, dim), where dim is the
      dimensionality
      batch_targets: {0, 1} numpy array of length dim.
      of input.

    Returs:
      The loss of the network on the given batch.
    """
    with self.graph.as_default():
      loss = self.session.run(
          ['loss:0'],
          feed_dict={
            '{}:0'.format(self.feat_tensor_name): batch_features,
            '{}:0'.format(self.target_tensor_name): batch_targets})
      return loss

  def train_batch(self, batch_features, batch_targets):
    """Trains the MLP with a single training step.

    Args:
      batch_features: a numpy array with shape (?, dim), where dim is the
      dimensionality
      batch_targets: {0, 1} numpy array of length dim.
      of input.

    Returs:
      The loss of the network on the given batch before training.
    """
    with self.graph.as_default():
      loss, _ = self.session.run(
          ['loss:0', 'train'],
          feed_dict={
            '{}:0'.format(self.feat_tensor_name): batch_features,
            '{}:0'.format(self.target_tensor_name): batch_targets})
      return loss

  def close_session(self):
    '''Closes any existing tensorflow session for the MLP graph.'''
    try:
      self.session.close()
    except Exception as e:
      raise e

  def get_layer_weights(self):
    '''Gets the current value of the hidden layer weights

    Returns:
      Tuple of size self.depth + 1. Each element is a numpy array corresponding
      to the hidden layer weights at that depth. For instance, element 0 is
      the weights of hidden layer 1, element 1 is the weights of hidden layer 2
      , and so forth. The last element of the tuple is the weights of the
      output layer.
    '''
    with self.graph.as_default():
      weight_tensor_name_list = []
      for i in range(self.depth + 1):
        extension = ''
        if i > 0:
          extension = '_{}'.format(i)
        weight_tensor_name_list.append(
            'fully_connected{}/weights:0'.format(extension))
      return self.session.run(weight_tensor_name_list)


class MultiLayerPerceptronWithProbes(MultiLayerPerceptron):

  def __init__(self, units_per_layer):
    super(MultiLayerPerceptronWithProbes, self).__init__(units_per_layer)
    self.probe_loss_tensor_names = []
    self.probe_train_op_names = []
    with self.graph.as_default():
      target_tensor = self.graph.get_tensor_by_name(
          '{}:0'.format(self.target_tensor_name))
      for i, layer in enumerate(self.layers):
        output = tf.contrib.layers.fully_connected(
            tf.stop_gradient(layer), 1, activation_fn=None)
        prediction = tf.reshape(tf.nn.sigmoid(output), shape=[-1])
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
          labels=target_tensor, logits=prediction), name='loss_{}'.format(i))
        self.probe_loss_tensor_names.append('loss_{}:0'.format(i))
        train_op = tf.train.GradientDescentOptimizer(
            learning_rate=0.1).minimize(loss, name='train_{}'.format(i))
        self.probe_train_op_names.append('train_{}'.format(i))

  def train_batch(self, batch_features, batch_targets, mode='all'):
    """Trains the MLP with a single training step.

    Args:
      batch_features: a numpy array with shape (?, dim), where dim is the
      dimensionality
      batch_targets: {0, 1} numpy array of length dim.
      of input.
      mode: one of 'core', 'probe', and 'all'.

    Returs:
      The loss of the network on the given batch before training.
    """
    if mode == 'core':
      all_ops = ['loss:0', 'train']
    elif mode == 'probe':
      all_ops = self.probe_loss_tensor_names + ['loss:0'] 
      all_ops +=  self.probe_train_op_names 
    elif mode == 'all':
      all_ops = self.probe_loss_tensor_names + ['loss:0'] 
      all_ops +=  self.probe_train_op_names + ['train']
    with self.graph.as_default():
      loss = self.session.run(
          all_ops,
          feed_dict={
            '{}:0'.format(self.feat_tensor_name): batch_features,
            '{}:0'.format(self.target_tensor_name): batch_targets})
      return loss


