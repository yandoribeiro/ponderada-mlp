class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, parameters, gradients):
        for parameter, gradient in zip(parameters, gradients):
            parameter -= self.learning_rate * gradient

