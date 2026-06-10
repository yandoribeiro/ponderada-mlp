import numpy as np

from mlp.activations import relu, relu_derivative, softmax
from mlp.losses import cross_entropy
from mlp.optimizers import SGD
from mlp.utils import one_hot


class MLP:
    def __init__(self, layer_sizes, activation="relu", learning_rate=0.01, random_seed=None):
        self._validate_layer_sizes(layer_sizes)

        if activation != "relu":
            raise ValueError("Por enquanto, a unica ativacao suportada e 'relu'.")

        self.layer_sizes = list(layer_sizes)
        self.activation_name = activation
        self.rng = np.random.default_rng(random_seed)
        self.weights = []
        self.biases = []
        self.cache = {}
        self.gradients = {}
        self.optimizer = SGD(learning_rate)

        self._initialize_parameters()

    @staticmethod
    def _validate_layer_sizes(layer_sizes):
        if len(layer_sizes) < 2:
            raise ValueError("A rede precisa ter ao menos camada de entrada e saida.")

        if any(size <= 0 for size in layer_sizes):
            raise ValueError("Todas as camadas precisam ter tamanho positivo.")

    def _initialize_parameters(self):
        for fan_in, fan_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            scale = np.sqrt(2.0 / fan_in)
            weight = self.rng.normal(0.0, scale, size=(fan_in, fan_out))
            bias = np.zeros((1, fan_out))

            self.weights.append(weight)
            self.biases.append(bias)

    def forward(self, x):
        activation = self._prepare_input(x)
        activations = [activation]
        pre_activations = []

        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            z = activation @ weight + bias
            activation = relu(z)

            pre_activations.append(z)
            activations.append(activation)

        logits = activation @ self.weights[-1] + self.biases[-1]
        probabilities = softmax(logits)

        pre_activations.append(logits)
        activations.append(probabilities)

        self.cache = {
            "pre_activations": pre_activations,
            "activations": activations,
        }

        return probabilities

    def loss(self, x, y_true):
        probabilities = self.forward(x)
        y_true = self._prepare_targets(y_true, probabilities.shape[0])
        return cross_entropy(y_true, probabilities)

    def backward(self, y_true):
        if not self.cache:
            raise ValueError("Execute forward antes de backward.")

        activations = self.cache["activations"]
        pre_activations = self.cache["pre_activations"]
        probabilities = activations[-1]
        y_true = self._prepare_targets(y_true, probabilities.shape[0])
        batch_size = y_true.shape[0]

        delta = (probabilities - y_true) / batch_size
        weight_gradients = []
        bias_gradients = []

        for layer_index in reversed(range(len(self.weights))):
            weight_gradient = activations[layer_index].T @ delta
            bias_gradient = np.sum(delta, axis=0, keepdims=True)

            weight_gradients.insert(0, weight_gradient)
            bias_gradients.insert(0, bias_gradient)

            if layer_index > 0:
                delta = delta @ self.weights[layer_index].T
                delta *= relu_derivative(pre_activations[layer_index - 1])

        self.gradients = {
            "weights": weight_gradients,
            "biases": bias_gradients,
        }

        return self.gradients

    def update_parameters(self):
        if not self.gradients:
            raise ValueError("Execute backward antes de atualizar os parametros.")

        self.optimizer.step(self.weights, self.gradients["weights"])
        self.optimizer.step(self.biases, self.gradients["biases"])

    def train_batch(self, x, y_true):
        probabilities = self.forward(x)
        y_true = self._prepare_targets(y_true, probabilities.shape[0])
        loss = cross_entropy(y_true, probabilities)
        self.backward(y_true)
        self.update_parameters()
        return loss

    def train(self, x_train, y_train, epochs=10, batch_size=64, shuffle=True, validation_data=None):
        x_train = self._prepare_input(x_train)
        y_train = self._prepare_targets(y_train, x_train.shape[0])
        history = {
            "loss": [],
            "accuracy": [],
        }

        if validation_data is not None:
            history["val_loss"] = []
            history["val_accuracy"] = []

        for _ in range(epochs):
            if shuffle:
                indices = self.rng.permutation(x_train.shape[0])
                x_epoch = x_train[indices]
                y_epoch = y_train[indices]
            else:
                x_epoch = x_train
                y_epoch = y_train

            batch_losses = []

            for start in range(0, x_epoch.shape[0], batch_size):
                end = start + batch_size
                batch_loss = self.train_batch(x_epoch[start:end], y_epoch[start:end])
                batch_losses.append(batch_loss)

            history["loss"].append(float(np.mean(batch_losses)))
            history["accuracy"].append(self.accuracy(x_train, y_train))

            if validation_data is not None:
                x_val, y_val = validation_data
                history["val_loss"].append(self.loss(x_val, y_val))
                history["val_accuracy"].append(self.accuracy(x_val, y_val))

        return history

    def train_batch_with_updates(self, x, y_true):
        before_weights = [weight.copy() for weight in self.weights]
        before_biases = [bias.copy() for bias in self.biases]
        loss = self.train_batch(x, y_true)

        weight_updates = []
        bias_updates = []

        for before, after in zip(before_weights, self.weights):
            weight_updates.append({
                "before": before,
                "after": after.copy(),
                "difference": after - before,
            })

        for before, after in zip(before_biases, self.biases):
            bias_updates.append({
                "before": before,
                "after": after.copy(),
                "difference": after - before,
            })

        return {
            "loss": loss,
            "weights": weight_updates,
            "biases": bias_updates,
        }

    def predict(self, x):
        probabilities = self.forward(x)
        return np.argmax(probabilities, axis=1)

    def accuracy(self, x, y_true):
        predictions = self.predict(x)
        targets = self._target_classes(y_true, predictions.shape[0])
        return float(np.mean(predictions == targets))

    def _prepare_input(self, x):
        x = np.asarray(x, dtype=float)

        if x.ndim == 1:
            x = x.reshape(1, -1)

        if x.ndim != 2:
            raise ValueError("A entrada precisa ser um vetor ou uma matriz 2D.")

        if x.shape[1] != self.layer_sizes[0]:
            raise ValueError(
                f"Entrada com {x.shape[1]} atributos, mas a rede espera "
                f"{self.layer_sizes[0]}."
            )

        return x

    def _prepare_targets(self, y_true, batch_size):
        y_true = np.asarray(y_true)

        if y_true.ndim == 1:
            if batch_size == 1 and y_true.shape[0] == self.layer_sizes[-1]:
                y_true = y_true.reshape(1, -1)
            elif y_true.shape[0] == batch_size:
                y_true = one_hot(y_true, self.layer_sizes[-1])
            else:
                raise ValueError("Formato de rotulos incompativel com o batch.")

        if y_true.ndim != 2:
            raise ValueError("Os rotulos precisam estar em formato one-hot 2D.")

        if y_true.shape[0] != batch_size:
            raise ValueError("Quantidade de entradas e rotulos nao corresponde.")

        if y_true.shape[1] != self.layer_sizes[-1]:
            raise ValueError(
                f"Rotulos com {y_true.shape[1]} classes, mas a rede espera "
                f"{self.layer_sizes[-1]}."
            )

        return y_true.astype(float)

    def _target_classes(self, y_true, batch_size):
        y_true = np.asarray(y_true)

        if y_true.ndim == 1:
            if batch_size == 1 and y_true.shape[0] == self.layer_sizes[-1]:
                return np.argmax(y_true.reshape(1, -1), axis=1)

            if y_true.shape[0] == batch_size:
                return y_true.astype(int)

        y_true = self._prepare_targets(y_true, batch_size)
        return np.argmax(y_true, axis=1)
