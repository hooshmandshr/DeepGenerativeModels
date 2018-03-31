class PlanarFlow(object):
    """Class for defining operations for a normalizing flow."""

    def __init__(self, dim, w=None, b=None, u=None):
        self.dim = dim
        if w is None or b is None or u is None:
            self.create_flow_variables()
        else:
            self.w = w
            self.b = b
            self.u = u
        # Enforcing reversibility.
        self.u_bar = self.reversible_constraint(self.u, self.w)

    def create_flow_variables(self):
        self.w = tf.Variable(np.random.normal(0, 1, [self.dim, 1]))
        self.b = tf.Variable(np.random.normal(0, 1, 1))
        self.u = tf.Variable(np.random.normal(0, 1, [1, self.dim]))

    def reversible_constraint(self, u, w):
        dot = tf.squeeze(tf.matmul(u, w))
        scalar = - 1 + tf.nn.softplus(dot) - dot
        norm_squared = tf.reduce_sum(w * w)
        comp = scalar * tf.transpose(w) / norm_squared
        return u + comp

    def transform(self, inputs):
        dialation = tf.matmul(inputs, self.w) + self.b
        return inputs + self.u_bar * tf.tanh(dialation)

    def log_det_jacobian(self, inputs):
        dialation = tf.matmul(inputs, self.w) + self.b
        psi = 1.0 - tf.pow(tf.tanh(dialation), 2)
        det_jac = tf.matmul(self.u_bar, self.w) * psi
        return - tf.squeeze(tf.log(tf.abs(1 + det_jac)))

class FlowRandomVariable(object):

    def __init__(self, dim, num_layers, base_dist=None):
        if base_dist is None:
            base_dist = tf.distributions.Normal(
                loc=np.zeros(dim), scale=np.ones(dim))
        self.dim = dim
        self.num_layers = num_layers
        self.base_dist = base_dist
        self.flows = []
        for i in range(self.num_layers):
            self.flows.append(PlanarFlow(dim))

    def sample_log_prob(self, n_samples):
        """Provide samples from the flow distribution and its log prob."""
        samples = self.base_dist.sample(n_samples)
        log_prob = tf.reduce_sum(
            self.base_dist.log_prob(samples), axis=1)
        for flow in self.flows:
            log_prob += flow.log_det_jacobian(samples)
            samples = flow.transform(samples)
        return samples, log_prob

    def transform(self, x):
        for flow in self.flows:
            x = flow.transform(x)
        return x