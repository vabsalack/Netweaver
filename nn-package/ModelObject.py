class Model:
    def __init__(self):
        self.layers = []
    
    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer
    
    def train(self, X, y, *, epochs=1, print_every=1):
        for epoch in range(1, epochs+1):
            self.forward()
    
    def finalize(self):
        self.input_layer = Layer_Input()
        layer_count = len(self.layers)

        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]
            if i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
    
    def forward(self, X):
        
        self.input_layer.forward(X)
        for layer in self.layers:
            layer.forward(layer.prev.output)
        
        return layer.output
    

        