"""Class for implementing dynamical systems models according to Model design.

The goal of this library is to provide necessary dynamical systems models
to easily sample from, compute density, etc.
"""

import numpy as np
import tensorflow as tf

from model import Model

class MarkovLatentDynamics(object):

    def __init__(self, transition, emmision, time_steps):
        """Sets up the necessary networks for the markov latent dynamics.

        params:
        -------
        transition: Transform
            Transformation that expresses the transition function
        """
        self.time_steps = time_steps
        self.emmision = emmision
        self.transition = transition

        self.init_loc = tf.Variable(np.zeros(self.transition.dim_x))
        self.init_noise = tf.nn.softplus(
                tf.Variable(np.ones(self.transition.dim_x)))
        self.init_state_p = tf.distributions.Normal(
            loc=self.init_loc, scale=self.init_noise)

    def log_prob(self, z, x):
        """Log evidence of observations and latent states.
        
        Parameters:
        -----------
        z: numpy.ndarray
        Samples from the hidden states of shape (S, T x d)
        x: numpy.ndarray
        Samples from the observations of shape (S, T x D)
        """
        n_samples = z.shape[0].value
        log_prob = 0
        for i in range(self.time_steps):
            lat_dim = self.transition.dim_x
            obs_dim = self.emmision.dim_x
            # Current time step latent state and observation
            z_t = tf.slice(z, [0, i * lat_dim], [-1, lat_dim])
            x_t = tf.slice(x, [0, i * obs_dim], [-1, obs_dim])
            if i == 0:
                log_prob += tf.reduce_sum(
                    self.init_state_p.log_prob(z_t), axis=1)
            else:
                log_prob += self.transition.log_prob(z_t, z_tminus)
            log_prob += self.emmision.log_prob(x_t, z_t)
            # Update the previous latent state for next iteration
            z_tminus = z_t
        return log_prob

    def sample(self, n_samples, time_steps=None, z_not=None):
        """Samples from the latent dynamics model.

        Returns:
        --------
        numpy.ndarray of shape
        """
        if time_steps is None:
            time_steps = self.time_steps
        if z_not is None: 
            z = [self.init_state_p.sample(n_samples)]
        else:
            z = [z_not]
        x = []
        for i in range(time_steps):
            if not i == 0:
                z.append(self.transition.sample(z[-1]))
            x.append(self.emmision.sample(z[-1]))
        return tf.concat(z, axis=1), tf.concat(x, axis=1)

    def sample_observations(self, n_samples, z):
        """Samples from the latent dynamics model."""
        lat_dim = self.transition.dim_x
        x = []
        for i in range(self.time_steps):
            z_t = tf.slice(z, [0, i * lat_dim], [-1, lat_dim]) 
            x.append(self.emmision.sample(z_t))
        return tf.concat(x, axis=1)


class MarkovDynamics(object):

    def __init__(self, transition, time_steps):
        """Sets up the necessary networks for the markov latent dynamics."""
        self.time_steps = time_steps
        self.transition = transition

        self.init_loc = tf.Variable(np.zeros(self.transition.dim_x)) + 1.
        self.init_noise = tf.nn.softplus(tf.Variable(np.ones(self.transition.dim_x)))
        self.init_state_p = tf.distributions.Normal(
            loc=self.init_loc, scale=self.init_noise)

    def log_prob(self, x):
        """Log evidence of observations and latent states.
        
        Parameters:
        -----------
        z: numpy.ndarray
        Samples from the hidden states of shape (S, T x d)
        x: numpy.ndarray
        Samples from the observations of shape (S, T x D)
        """
        n_samples = x.shape[0].value
        log_prob = 0
        for i in range(self.time_steps):
            obs_dim = self.transition.dim_x
            # Current time step latent state and observation
            x_t = tf.slice(x, [0, i * obs_dim], [-1, obs_dim])
            if i == 0:
                log_prob += tf.reduce_sum(
                    self.init_state_p.log_prob(x_t), axis=1)
            else:
                log_prob += self.transition.log_prob(x_t, x_tminus)
            # Update the previous latent state for next iteration
            x_tminus = x_t
        return log_prob

    def sample(self, n_samples, time_steps=None, x_not=None):
        """Samples from the latent dynamics model.

        Returns:
        --------
        numpy.ndarray of shape
        """
        if time_steps is None:
            time_steps = self.time_steps
        if x_not is None: 
            x = [self.init_state_p.sample(n_samples)]
        else:
            x = [x_not]
        for i in range(time_steps):
            if not i == 0:
                x.append(self.transition.sample(x[-1]))
        return tf.concat(x, axis=1)

