from cneural import *
from typing import Union

class Model:
    def __init__(self):
        self.layers = []
        self.softmax_classifier_output = None
    
    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer: Union[OptimizerAdagrad], accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy
    
    def finalize(self):
        self.input_layer = LayerInput()
        self.trainable_layers = []
        layer_count = len(self.layers)

        for i in range(layer_count):
            self.layers[i].prev = self.input_layer if i == 0 else self.layers[i - 1]
            self.layers[i].next = self.loss if i == layer_count - 1 else self.layers[i + 1]
            if hasattr(self.layers[i], "weights"):
                self.trainable_layers.append(self.layers[i])

        self.output_layer_activation = self.layers[layer_count - 1]
        self.loss.remember_trainable_layers(self.trainable_layers)

        if isinstance(self.layers[-1], ActivationSoftmax) and isinstance(self.loss, LossCategoricalCrossentropy):
            self.softmax_classifier_output = ActivationSoftmaxLossCategoricalCrossentropy()

    def train(self, X, y, *, epochs=1, print_every=1, validation_data=None):

        for epoch in range(1, epochs+1):
            output = self.forward(X, training=True)

            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)
            data_loss, regularization_loss = self.loss.calculate(output, y, include_regularization=True)
            loss = data_loss + regularization_loss

            self.backward(output, y)

            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            if not epoch % print_every:
                print(
                    f"Epoch: {epoch}, " +
                    f"accuracy: {accuracy: .3f}, " +
                    f"loss: {loss}, " +
                    f"data_loss: {data_loss}, " +
                    f"reg_loss: {regularization_loss}, " +
                    f"lr: {self.optimizer.current_learning_rate}"   
                    )
            
            if validation_data is not None: 
                X_val, y_val = validation_data
                output = self.forward(X_val, training=False)
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, y_val)
                loss = self.loss.calculate(output, y_val, include_regularization=False)
                print(
                    f"Validation, " +
                    f"acc: {accuracy: .3f}, " +
                    f"loss: {loss: .3f}"
                    )

    
    def forward(self, X, training):
        
        self.input_layer.forward(X, training)
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        
        return layer.output
    
    def backward(self, output, y):
        
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return

        self.loss.backward(output, y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
    

        