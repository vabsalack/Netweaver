import numpy as np

# Dense layer
class Layer_Dense:

    # Layer initialization
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_L1=0, weight_regularizer_L2=0,
                 bias_regularizer_L1=0, bias_regularizer_L2=0):
        # the weight's initialization significantly impacts convergence
        # (T) he initialization. variance scaled by number of input connections. σ2 = 2 / n_in
        # below weights matrix initialization, notice dim is ninputs x nneurons, make easier for mat mul. 
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) 
        self.biases = np.zeros((1, n_neurons))

        # set regularization strength
        # weight regu..
        self.weight_regularizer_L1 = weight_regularizer_L1
        self.weight_regularizer_L2 = weight_regularizer_L2
        # bias regu..
        self.bias_regularizer_L1 = bias_regularizer_L1
        self.bias_regularizer_L2 = bias_regularizer_L2

    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
        # In output matrix, rows constitues each sample and column constitues each neurons in a layer.
        self.output = np.dot(inputs, self.weights) + self.biases
        
    # Backward pass
    def backward(self, dvalues):
        # Gradients on parameters
        # dvalues are backpropogated gradient from next layer
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_L1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_L1 * dL1
        # L2 on weights
        if self.weight_regularizer_L2 > 0:
            self.dweights += 2 * self.weight_regularizer_L2 * self.weights
        
        # L1 on biases
        if self.bias_regularizer_L1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_L1 * dL1
        # L2 on biases
        if self.bias_regularizer_L2 > 0:
            self.dbiases += 2 * self.bias_regularizer_L2 * self.biases

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

# dropout
class Layer_Dropout:
    def __init__(self, rate):
        #invert it here rate is q (p = 1 - q)
        self.rate = 1 - rate
    # forward pass
    def forward(self, inputs):
        self.inputs = inputs
        # save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size=self.inputs.shape) / self.rate
        # apply mask to output values
        self.output = self.inputs * self.binary_mask
    # backward pass
    def backward(self, dvalues):
        # gradient on values
        self.dinputs = dvalues * self.binary_mask


# ReLU activation
class Activation_ReLU:

    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify original variable,
        # let's make a copy of values first
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0


# Softmax activation
class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        # Get unnormalized probabilities
        # exponentiation graph is rapid, avoid overflow by making maximum to zero. get the idea
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

        # Backward pass
        # Backward pass

    # Backward pass

    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output and
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

# sigmoid activation
class Activation_Sigmoid:
    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
    
    #backward pass
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output

# Common loss class
class Loss:
    # Regularization loss calculation
    def regularization_loss(self,layer):
        
        # 0 by default
        regularization_loss = 0

        # L1 and L2 penalties for weights
        if layer.weight_regularizer_L1 > 0:
            regularization_loss += layer.weight_regularizer_L1 * np.sum(np.abs(layer.weights))
        if layer.weight_regularizer_L2 > 0:
            regularization_loss += layer.weight_regularizer_L2 * np.sum(layer.weights * layer.weights)
        # L1 and L2 penalties for biases
        if layer.bias_regularizer_L1 > 0:
            regularization_loss += layer.bias_regularizer_L1 * np.sum(np.abs(layer.biases))
        if layer.bias_regularizer_L2 > 0:
            regularization_loss += layer.bias_regularizer_L2 * np.sum(layer.biases * layer.biases)

        return regularization_loss

    # Calculates the data and regularization losses
    def calculate(self, output, y):
        # sample losses
        sample_losses = self.forward(output, y)
        # mean loss
        data_loss = np.mean(sample_losses)
        
        return data_loss
    
# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = len(y_pred)
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true,axis=1)
        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
        
    # Backward pass
    def backward(self, dvalues, y_true):
        # davalues are softmax outputs (predicted outputs)
        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])
        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():
    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)
    
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient, true lables are 1
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples

class Loss_BinaryCrossentropy(Loss):

    def forward(self, y_pred, y_true):
        # np.log(1e-323) = -inf
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # calculate sample-wise loss
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 -y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        return sample_losses
    # backward pass
    def backward(self, dvalues, y_true):
        # here dvalues are outputs from sigmoid
        samples = len(dvalues)
        # number of output in every sample
        outputs = len(dvalues[0])

        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true)/(1 - clipped_dvalues)) / outputs
        # normalize gradient
        self.dinputs /= samples

class Optimizer_SGD:
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            # adding 1 ensure value decreases
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # update parameters
    def update_params(self, Layer):

        # If we use momentum
        if self.momentum:
            if not hasattr(Layer, "weight_momentums"):
                Layer.weight_momentums = np.zeros_like(Layer.weights)
                Layer.bias_momentums = np.zeros_like(Layer.biases)

            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and update with
            # current gradients
            weight_updates = self.momentum * Layer.weight_momentums - self.current_learning_rate * Layer.dweights
            Layer.weight_momentums = weight_updates
            # Build bias updates
            bias_updates = self.momentum * Layer.bias_momentums - self.current_learning_rate * Layer.dbiases
            Layer.bias_momentums = bias_updates
            # Vanilla SGD updates (as before momentum update)
        else:
            weight_updates = -self.current_learning_rate * Layer.dweights
            bias_updates = -self.current_learning_rate * Layer.dbiases

                    
        Layer.weights += weight_updates
        Layer.biases += bias_updates

    # call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adagrad:
    # learning rate of 1. is default for this optimizer
    # epsilon for numerical stability
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    # call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            # adding 1 ensure value < 0
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # update parameters
    def update_params(self, Layer):

        if not hasattr(Layer, "weight_cache"):
            Layer.weight_cache = np.zeros_like(Layer.weights)
            Layer.bias_cache = np.zeros_like(Layer.biases)
        
        # update cache with squared current gradients
        Layer.weight_cache += Layer.dweights**2
        Layer.bias_cache += Layer.dbiases**2

        # vanilla + normalization with square rooted cache
        Layer.weights += -self.current_learning_rate * Layer.dweights / (np.sqrt(Layer.weight_cache) + self.epsilon)        
        Layer.biases += -self.current_learning_rate * Layer.dbiases / (np.sqrt(Layer.bias_cache) + self.epsilon)        


    # call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

class Optimizer_RMSprop:
    # learning rate of 0.001 is default for this optimizer
    # epsilon for numerical stability
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta = beta

    # call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            # adding 1 ensure value < 0
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # update parameters
    def update_params(self, Layer):

        if not hasattr(Layer, "weight_cache"):
            Layer.weight_cache = np.zeros_like(Layer.weights)
            Layer.bias_cache = np.zeros_like(Layer.biases)
        
        # update cache with squared current gradients
        # exponentially discounts past gradients
        # EWMA (Exponentially Weighted Moving Average)
        Layer.weight_cache = self.beta * Layer.weight_cache + (1 - self.beta) * Layer.dweights**2
        Layer.bias_cache += self.beta * Layer.bias_cache + (1- self.beta) * Layer.dbiases**2

        # vanilla + normalization with square rooted cache
        Layer.weights += -self.current_learning_rate * Layer.dweights / (np.sqrt(Layer.weight_cache) + self.epsilon)        
        Layer.biases += -self.current_learning_rate * Layer.dbiases / (np.sqrt(Layer.bias_cache) + self.epsilon)        


    # call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adam:
    # learning rate, beta_1 and beta_2 are in default values
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1        
        self.beta_2 = beta_2     

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, Layer):
        
        if not hasattr(Layer, "weight_cache"):
            Layer.weight_momentums = np.zeros_like(Layer.weights)
            Layer.bias_momentums = np.zeros_like(Layer.biases)
            Layer.weight_cache = np.zeros_like(Layer.weights)
            Layer.bias_cache = np.zeros_like(Layer.biases)

        # scaled momentums (EWMA)
        Layer.weight_momentums = self.beta_1 * Layer.weight_momentums + (1 - self.beta_1) * Layer.dweights
        Layer.bias_momentums = self.beta_1 * Layer.bias_momentums + (1 - self.beta_1) * Layer.dbiases
        # bias correction
        weight_momentums_corrected = Layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = Layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        # scaled gradients (EWMA)
        Layer.weight_cache = self.beta_2 * Layer.weight_cache + (1 - self.beta_2) * Layer.dweights**2
        Layer.bias_cache = self.beta_2 * Layer.bias_cache + (1 - self.beta_2) * Layer.dbiases**2
        # bias correction
        weight_cache_corrected = Layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = Layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
        
        # params updates
        Layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        Layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)


    def post_update_params(self):
        self.iterations += 1








           


    

