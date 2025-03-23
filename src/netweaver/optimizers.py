from netweaver.utils import LayerDense, np


class OptimizerSGD:
    """
    #### What
        - Stochastic Gradient Descent is the simplest optimizer.
        - SGD with momentum is usually one of 2 main choices for an optimizer in practice next to the Adam optimizer.
        - It comes with learning rate decay and GD with momentum. Both are crucial for faster convergence.
        - weight_update = Wn + Wn-1*M^(n-1) + Wn-2*M^(n-2) + ... + W1*M^0. M is the momentum. compare it with EWMA β, (1 - β)
    #### Improve
        - calculate the no of iterations before learning rate decays to near zero.
        - how about implementing a decayer for momentum?
    #### Flow
        - [init -> (pre_update_params -> update_params -> post_update_params)]
    """

    def __init__(
        self, learning_rate: float = 1.0, decay: float = 0.0, momentum: float = 0.0
    ) -> None:
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.iterations)
            )  # 1/x graph

    def update_params(self, Layer: LayerDense) -> None:
        if self.momentum:
            if not hasattr(Layer, "weight_momentums"):
                Layer.weight_momentums = np.zeros_like(Layer.weights)
                Layer.bias_momentums = np.zeros_like(Layer.biases)
            weight_updates = (
                self.momentum * Layer.weight_momentums
                - self.current_learning_rate * Layer.dweights
            )
            Layer.weight_momentums = weight_updates
            bias_updates = (
                self.momentum * Layer.bias_momentums
                - self.current_learning_rate * Layer.dbiases
            )
            Layer.bias_momentums = bias_updates
        else:  # Vanilla SGD updates (as before momentum update)
            weight_updates = -self.current_learning_rate * Layer.dweights
            bias_updates = -self.current_learning_rate * Layer.dbiases
        Layer.weights += weight_updates
        Layer.biases += bias_updates

    def post_update_params(self) -> None:
        self.iterations += 1


class OptimizerAdagrad:
    """
    #### What
        - Adagrad optimizer is an adaptive learning rate optimizer. It is not widely used.
        - learning does stall (1/X) x tends to 0.
        - The idea here is to normalize updates made to the features.
    #### Improve
    #### Flow
        - [init -> (pre_update_params -> update_params -> post_update_params)]
        - formula: lr = lr / (sqrt(cache) + epsilon)
    """

    def __init__(
        self, learning_rate: float = 1.0, decay: float = 0.0, epsilon: float = 1e-7
    ) -> None:
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update_params(self) -> None:
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.iterations)
            )

    def update_params(self, Layer: LayerDense) -> None:
        if not hasattr(Layer, "weight_cache"):
            Layer.weight_cache = np.zeros_like(Layer.weights)
            Layer.bias_cache = np.zeros_like(Layer.biases)
        Layer.weight_cache += Layer.dweights**2
        Layer.bias_cache += Layer.dbiases**2
        Layer.weights += (
            -self.current_learning_rate
            * Layer.dweights
            / (np.sqrt(Layer.weight_cache) + self.epsilon)
        )
        Layer.biases += (
            -self.current_learning_rate
            * Layer.dbiases
            / (np.sqrt(Layer.bias_cache) + self.epsilon)
        )

    def post_update_params(self) -> None:
        self.iterations += 1


class OptimizerRMSprop:
    """
    #### What
        - RMSprop optimizer(Adagrad variant) is an adaptive learning rate optimizer. learning does not stall.
        - Implements decaying caching mechanism, which prevents the learning rate from becoming too small.
    #### Improve
    #### Flow
        - [init -> (pre_update_params -> update_params -> post_update_params)]
        - formula: lr = lr / (sqrt(cache) + epsilon)
        - cache = cache * beta + (1 - beta) * gradient^2 **EMWA discounting past gradients**
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        decay: float = 0.0,
        epsilon: float = 1e-7,
        beta: float = 0.9,
    ) -> None:
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta = beta

    def pre_update_params(self) -> None:
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.iterations)
            )

    def update_params(self, Layer: LayerDense) -> None:
        if not hasattr(Layer, "weight_cache"):
            Layer.weight_cache = np.zeros_like(Layer.weights)
            Layer.bias_cache = np.zeros_like(Layer.biases)
        Layer.weight_cache = (
            self.beta * Layer.weight_cache + (1 - self.beta) * Layer.dweights**2
        )
        Layer.bias_cache += (
            self.beta * Layer.bias_cache + (1 - self.beta) * Layer.dbiases**2
        )
        Layer.weights += (
            -self.current_learning_rate
            * Layer.dweights
            / (np.sqrt(Layer.weight_cache) + self.epsilon)
        )
        Layer.biases += (
            -self.current_learning_rate
            * Layer.dbiases
            / (np.sqrt(Layer.bias_cache) + self.epsilon)
        )

    def post_update_params(self) -> None:
        self.iterations += 1


class OptimizerAdam:
    """
    #### What
        - Adaptive Moment Estimation optimizer is an adaptive learning rate optimizer. (Momentum L1 moment) + RMSprop (L2 moment)
        - It is robust to noisy gradients and sparse gradients
    #### Flow
        - [init -> (pre_update_params -> update_params -> post_update_params)]
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        decay: float = 0.0,
        epsilon: float = 1e-7,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
    ) -> None:
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self) -> None:
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.iterations)
            )

    def update_params(self, Layer: LayerDense) -> None:
        """
        #### Note
            - first moment:  m_t = beta_1 * m_t-1 + (1 - beta_1) * gradient
            - second moment: v_t = beta_2 * v_t-1 + (1 - beta_2) * gradient^2
            - params update: w_t = w_t-1 - lr * m_t / (sqrt(v_t) + epsilon)
            - bias correction in inital stages, the first and second moments are biased towards zero.
        """
        if not hasattr(Layer, "weight_cache"):
            Layer.weight_momentums = np.zeros_like(Layer.weights)
            Layer.bias_momentums = np.zeros_like(Layer.biases)
            Layer.weight_cache = np.zeros_like(Layer.weights)
            Layer.bias_cache = np.zeros_like(Layer.biases)
        # first moment
        Layer.weight_momentums = (
            self.beta_1 * Layer.weight_momentums + (1 - self.beta_1) * Layer.dweights
        )
        Layer.bias_momentums = (
            self.beta_1 * Layer.bias_momentums + (1 - self.beta_1) * Layer.dbiases
        )
        weight_momentums_corrected = Layer.weight_momentums / (
            1 - self.beta_1 ** (self.iterations + 1)
        )  # bias correction
        bias_momentums_corrected = Layer.bias_momentums / (
            1 - self.beta_1 ** (self.iterations + 1)
        )
        # second moment
        Layer.weight_cache = (
            self.beta_2 * Layer.weight_cache + (1 - self.beta_2) * Layer.dweights**2
        )
        Layer.bias_cache = (
            self.beta_2 * Layer.bias_cache + (1 - self.beta_2) * Layer.dbiases**2
        )
        weight_cache_corrected = Layer.weight_cache / (
            1 - self.beta_2 ** (self.iterations + 1)
        )  # bias correction
        bias_cache_corrected = Layer.bias_cache / (
            1 - self.beta_2 ** (self.iterations + 1)
        )
        # params updates
        Layer.weights += (
            -self.current_learning_rate
            * weight_momentums_corrected
            / (np.sqrt(weight_cache_corrected) + self.epsilon)
        )
        Layer.biases += (
            -self.current_learning_rate
            * bias_momentums_corrected
            / (np.sqrt(bias_cache_corrected) + self.epsilon)
        )

    def post_update_params(self) -> None:
        self.iterations += 1
