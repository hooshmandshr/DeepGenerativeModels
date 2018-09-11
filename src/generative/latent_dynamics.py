
import numpy as np
import tensorflow as tf

from normalizing_flow import PlanarFlow
from variational import MultiLayerPerceptron


class Transform(object):

    def __init__(self, in_dim, out_dim, initial_value=None):
        pass

    def operator(self, x):
        """Gives the tensorflow operation for transforming a tensor."""
        pass



class LinearTransform(Transform):

    def __init__(self, in_dim, out_dim, initial_value=None):
        if initial_value is None:
            initial_value = np.random.normal(
                0, 1, [in_dim, out_dim])
        self.lin_trans = tf.Variable(initial_value)
        self.bias = tf.Variable(
            np.random.normal(0, 1, [1, out_dim]))

    def operator(self, x):
        return tf.matmul(x, self.lin_trans) + self.bias


class LorentzTransform(Transform):

    def __init__(self, initial_value=None):
        """Sets up Lorentz transformation variables."""
        if initial_value is None:
            initial_value = np.random.normal(
                0, 1, 4)
        self.var = tf.Variable(initial_value)
        self.sigma = self.var[0]
        self.rho = self.var[1]
        self.beta = self.var[2]
        self.time_delta = tf.nn.sigmoid(self.var[3]) * 0.05

    def operator(self, x):
        if not(x.shape[-1].value == 3):
            raise ValueError('Dimension of variable should be 3')
        x_ = tf.slice(x, [0, 0], [-1, 1])
        y_ = tf.slice(x, [0, 1], [-1, 1])
        z_ = tf.slice(x, [0, 2], [-1, 1])
        return x + self.time_delta * tf.concat([
            self.sigma * (y_ - x_),
            x_ * (self.rho - z_) - y_,
            x_ * y_ - self.beta * z_], axis=1)


class QTransform(Transform):

    def __init__(self, in_dim, out_dim, initial_value=None):
        if initial_value is None:
            self.f = []
            for i in range(10):
                self.f.append(PlanarFlow(dim=in_dim))

    def operator(self, x):
        out = x
        for flow in self.f:
            out = flow.transform(out)
        return out


class MLPTransform(Transform):

    def __init__(self, in_dim, out_dim, layers = [64, 64], activation=tf.nn.relu):
        self.activation = activation
        layers = [in_dim] + layers + [out_dim]
        self.trans = []
        self.bias = []
        for i in range(len(layers) - 1):
            in_dim = layers[i]
            out_dim = layers[i + 1]
            init_val = np.random.normal(
                    0, 1, [in_dim, out_dim])
            self.trans.append(tf.Variable(init_val))
            self.bias.append(tf.Variable(
                    np.random.normal(0, 1, [1, out_dim])))

    def operator(self, x):
        for trans, bias in zip(self.trans, self.bias):
            x = self.activation(tf.matmul(x, trans) + bias)
        return x


class ConditionalRandomVariable(object):
    """Conditional probability model P(x|y)."""

    def __init__(self, dim_x, dim_y, transform, noise=1):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.noise = noise
        # Currently only normal initializer.
        # Change to inlucde other schemes as well.
        self.noise = tf.Variable(
            np.zeros(dim_x) - noise) 
        self.noise = tf.nn.softplus(self.noise)
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


class ConditionalPoissonVariable(object):
    """Conditional probability model P(x|y)."""

    def __init__(self, dim_x, dim_y, transform):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.transform = transform

    def make_distribution(self, y):
        lambda_ = tf.nn.softplus(self.transform.operator(y))
        return tf.contrib.distributions.Poisson(
            rate=lambda_)

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
