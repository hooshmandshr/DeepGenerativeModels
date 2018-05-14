
import numpy as np
import tensorflow as tf

from normalizing_flow import PlanarFlow


class Transform(object):

    def __init__(self, in_dim, out_dim, initial_value=None):
        if initial_value is None:
            initial_value = np.random.normal(
                0, 1, [in_dim, out_dim])
        self.lin_trans = tf.Variable(initial_value)
        self.bias = tf.Variable(
            np.random.normal(0, 1, [1, out_dim]))

    def operator(self, x):
        return tf.matmul(x, self.lin_trans) + self.bias


class QTransform(object):

    def __init__(self, in_dim, out_dim, initial_value=None):
        if initial_value is None:
            self.f = []
            for i in range(4):
                self.f.append(PlanarFlow(dim=in_dim))

    def operator(self, x):
        out = x
        for flow in self.f:
            out = flow.transform(out)
        return out


class QTransform(object):

    def __init__(self, in_dim, out_dim, initial_value=None):
        if initial_value is None:
            self.f = []
            for i in range(4):
                self.f.append(PlanarFlow(dim=in_dim))

    def operator(self, x):
        out = x
        for flow in self.f:
            out = flow.transform(out)
        return out


class ConditionalRandomVariable(object):
    """Conditional probability model P(x|y)."""

    def __init__(self, dim_x, dim_y, transform, noise=1):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.noise = noise
        # Currently only normal initializer.
        # Change to inlucde other schemes as well.
        self.noise = tf.Variable(
            np.ones(dim_x) * noise)
        self.transform = transform

    def make_distribution(self, y):
        mu = self.transform.operator(y)
        return tf.distributions.Normal(
            loc=mu, scale=self.noise)

    def log_prob(self, x, y):
        dist = self.make_distribution(y)
        return tf.reduce_sum(
            dist.log_prob(x), axis=1)

    def sample(self, y, n_samples=1):
        dist = self.make_distribution(y)
        return tf.squeeze(dist.sample(n_samples))


class MarkovLatentDynamics(object):

    def __init__(self, transition, emmision, time_steps):
        """Sets up the necessary networks for the markov latent dynamics."""
        self.time_steps = time_steps
        self.emmision = emmision
        self.transition = transition

        self.init_loc = tf.Variable(np.zeros(self.transition.dim_x))
        self.init_noise = tf.Variable(np.ones(self.transition.dim_x))
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

    def sample(self, n_samples, z_not=None):
        """Samples from the latent dynamics model.

        Returns:
        --------
        numpy.ndarray of shape
        """
        if z_not is None: 
            z = [self.init_state_p.sample(n_samples)]
        else:
            z = [z_not]
        x = []
        for i in range(self.time_steps):
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