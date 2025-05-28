from typing import Union

import numpy as np

from .layers import LayerDense


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

    def __init__(self, learning_rate: float = 1.0, decay: float = 0.0, momentum: float = 0.0) -> None:
        """
        Initializes the SGD optimizer with optional learning rate decay and momentum.

        Parameters
        ----------
        learning_rate : float, optional
            The initial learning rate, by default 1.0.
        decay : float, optional
            The learning rate decay factor, by default 0.0.
        momentum : float, optional
            The momentum factor, by default 0.0.

        Returns
        -------
        None
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        """
        Updates the current learning rate based on the decay parameter.

        If decay is set, the learning rate is adjusted according to the number of iterations.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))  # 1/x graph

    def update_params(self, layer: LayerDense) -> None:
        """
        Updates the parameters of the given layer using SGD with optional momentum.

        This method applies either vanilla SGD updates or momentum-based updates to the layer's weights and biases.

        Parameters
        ----------
        layer : LayerDense
            The layer whose parameters will be updated.

        Returns
        -------
        None
        """
        if self.momentum:
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:  # Vanilla SGD updates (as before momentum update)
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self) -> None:
        """
        Increments the iteration counter after parameter updates.

        This method should be called after updating parameters in each training step.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.iterations += 1

    def __str__(self):
        """
        Returns a string representation of the OptimizerSGD object.

        Returns
        -------
        str
            The string representation of the object.
        """
        return f"Optimizer_SGD(): Learning_Rate: {self.learning_rate}| Decay_rate: {self.decay}| Momentum: {self.momentum}"


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

    def __init__(self, learning_rate: float = 1.0, decay: float = 0.0, epsilon: float = 1e-7) -> None:
        """
        Initializes the Adagrad optimizer with optional learning rate decay and epsilon for numerical stability.

        Parameters
        ----------
        learning_rate : float, optional
            The initial learning rate, by default 1.0.
        decay : float, optional
            The learning rate decay factor, by default 0.0.
        epsilon : float, optional
            A small value to prevent division by zero, by default 1e-7.

        Returns
        -------
        None
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update_params(self) -> None:
        """
        Updates the current learning rate based on the decay parameter.

        If decay is set, the learning rate is adjusted according to the number of iterations.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

    def update_params(self, layer: LayerDense) -> None:
        """
        Updates the parameters of the given layer using the Adagrad optimization algorithm.

        This method adapts the learning rate for each parameter based on the sum of the squares of past gradients.

        Parameters
        ----------
        layer : LayerDense
            The layer whose parameters will be updated.

        Returns
        -------
        None
        """
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self) -> None:
        """
        Increments the iteration counter after parameter updates.

        This method should be called after updating parameters in each training step.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.iterations += 1

    def __str__(self):
        """
        Returns a string representation of the OptimizerAdagrad object.

        Parameters
        ----------
        None

        Returns
        -------
        str
            The string representation of the object.
        """
        return f"Optimizer_Adagrad(): Learning_Rate: {self.learning_rate}| Decay_rate: {self.decay}"


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
        """
        Updates the current learning rate based on the decay parameter.

        If decay is set, the learning rate is adjusted according to the number of iterations.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

    def update_params(self, layer: LayerDense) -> None:
        """
        Updates the parameters of the given layer using the RMSprop optimization algorithm.

        This method applies a decaying average of squared gradients to adapt the learning rate for each parameter.

        Parameters
        ----------
        layer : LayerDense
            The layer whose parameters will be updated.

        Returns
        -------
        None
        """
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        layer.weight_cache += self.beta * layer.weight_cache + (1 - self.beta) * layer.dweights**2
        layer.bias_cache += self.beta * layer.bias_cache + (1 - self.beta) * layer.dbiases**2
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self) -> None:
        """
        Increments the iteration counter after parameter updates.

        This method should be called after updating parameters in each training step.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.iterations += 1

    def __str__(self):
        """
        Returns a string representation of the OptimizerAdagrad object.

        Parameters
        ----------
        None

        Returns
        -------
        str
            The string representation of the object.
        """
        return f"Optimizer_RMSprop(): Learning_Rate: {self.learning_rate}| Decay_rate: {self.decay}| Beta: {self.beta}"


class OptimizerAdam:
    """Implements the Adam optimization algorithm.

    Adam combines the benefits of both momentum and RMSprop.
    It uses moving averages of the gradient and its squared value to adapt the learning rate for each parameter.
    Adam is known for its efficiency and effectiveness in a wide range of optimization problems.\
            It is less prone to getting stuck in local minima and handles noisy/sparse gradients well.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        decay: float = 0.0,
        epsilon: float = 1e-7,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
    ) -> None:
        """
        Initializes the Adam optimizer with optional learning rate decay, epsilon for numerical stability, and beta parameters for moment estimates.

        Parameters
        ----------
        learning_rate : float, optional
            The initial learning rate, by default 0.001.
        decay : float, optional
            The learning rate decay factor, by default 0.0.
        epsilon : float, optional
            A small value to prevent division by zero, by default 1e-7.
        beta_1 : float, optional
            The exponential decay rate for the first moment estimates, by default 0.9.
        beta_2 : float, optional
            The exponential decay rate for the second moment estimates, by default 0.999.

        Returns
        -------
        None
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2  # consider 1e-6 learning rate gives insignificant impact on gradient update
        self.count_valued_steps = f"{(((self.learning_rate / 1e-6) - 1) / self.decay):,}" if self.decay else "decay rate is 0"

    def pre_update_params(self) -> None:
        """
        Updates the current learning rate based on the decay parameter.

        If decay is set, the learning rate is adjusted according to the number of iterations.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

    def update_params(self, layer: LayerDense) -> None:
        """Updates the layer's parameters using the Adam algorithm.

        Uses moving averages of the gradient (first moment) and its squared value (second moment)
        to adapt the learning rate for each parameter.  Bias correction is applied to the
        moment estimates to account for their initialization at zero.

        - first moment:  m_t = beta_1 * m_t-1 + (1 - beta_1) * gradient
        - second moment: v_t = beta_2 * v_t-1 + (1 - beta_2) * gradient^2
        - params update: w_t = w_t-1 - lr * m_t / (sqrt(v_t) + epsilon)

        Parameters
        ----------
        layer : LayerDense
            The layer whose parameters to update.

        Returns
        -------
        None
        """

        if not hasattr(layer, "weight_cache"):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        # first moment
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))  # bias correction
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        # second moment
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))  # bias correction
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
        weight_updates = -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        bias_updates = -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
        # params updates
        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self) -> None:
        """
        Increments the iteration counter after parameter updates.

        This method should be called after updating parameters in each training step.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.iterations += 1

    def __str__(self):
        """
        Returns a string representation of the OptimizerAdagrad object.

        Parameters
        ----------
        None

        Returns
        -------
        str
            The string representation of the object.
        """
        return (
            f"Optimizer_Adam(): Learning_Rate: {self.learning_rate}| Decay_rate: {self.decay}| Beta_1: {self.beta_1}| Beta_2: {self.beta_2}|"
            + f" valued steps: {self.count_valued_steps}"
        )


OptimizerTypes = Union[OptimizerAdagrad, OptimizerAdam, OptimizerRMSprop, OptimizerSGD]
