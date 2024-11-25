import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical

# Load and preprocess data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create the LeNet model
model = Sequential([
    Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
    AveragePooling2D(),
    Conv2D(16, kernel_size=(5, 5), activation='relu'),
    AveragePooling2D(),
    Flatten(),
    Dense(120, activation='relu'),
    Dense(84, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

class NeuralNetworkDFO:
    def __init__(self, model, swarm_size=30, generations=1000, delta=0.1):
        self.model = model
        self.swarm_size = swarm_size
        self.generations = generations
        self.delta = delta
        self.global_best_accuracies = []

    def dfo_optimize(self, X, y):
        flies = [(self.model.get_weights(), self.compute_fitness(X, y)) for _ in range(self.swarm_size)]
        best_fly = max(flies, key=lambda x: x[1])
        global_best_position = best_fly[0]
        global_best_accuracy = best_fly[1]

        for itr in range(self.generations):
            for i in range(self.swarm_size):
                weights = flies[i][0]
                fitness = self.compute_fitness_with_params(X, y, weights)
                flies[i] = (flies[i][0], fitness)
                if fitness > global_best_accuracy:
                    global_best_position = flies[i][0]
                    global_best_accuracy = fitness

            self.global_best_accuracies.append(global_best_accuracy)

            for i in range(self.swarm_size):
                if flies[i][1] == global_best_accuracy:
                    continue  # Elitist strategy

                left = (i - 1) % self.swarm_size
                right = (i + 1) % self.swarm_size
                b_neighbour = right if flies[right][1] > flies[left][1] else left

                for j in range(len(flies[i][0])):  # Update weights
                    if np.random.rand() < self.delta:
                        flies[i][0][j] = np.random.randn(*flies[i][0][j].shape)
                    else:
                        u = np.random.rand()
                        flies[i][0][j] += u * (global_best_position[j] - flies[i][0][j])

        self.model.set_weights(global_best_position)
        print(f"Final best fitness: {global_best_accuracy}")
        print(f"Best fly position: {global_best_position}")

        plt.plot(self.global_best_accuracies)
        plt.title("Global Best Accuracy Over Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.show()

    def compute_fitness_with_params(self, X, y, weights):
        original_weights = self.model.get_weights()
        self.model.set_weights(weights)
        fitness = self.compute_fitness(X, y)
        self.model.set_weights(original_weights)
        return fitness

    def compute_fitness(self, X, y):
        loss, accuracy = self.model.evaluate(X, y, verbose=0)
        return accuracy

    def train(self, X, y):
        self.dfo_optimize(X, y)
        self.prune()
        self.quantize()

    def prune(self, threshold=0.01):
        weights = self.model.get_weights()
        for i in range(len(weights)):
            weights[i][np.abs(weights[i]) < threshold] = 0
        self.model.set_weights(weights)

    def quantize(self):
        weights = self.model.get_weights()
        for i in range(len(weights)):
            weights[i] = np.round(weights[i] * 127).astype(np.int8)
        self.model.set_weights(weights)

# Instantiate the model
model_dfo = NeuralNetworkDFO(model, swarm_size=30, generations=100)

# Train the model with DFO
model_dfo.train(X_train, y_train)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {accuracy}")

# Make predictions
predictions = model.predict(X_test)
print(predictions)
