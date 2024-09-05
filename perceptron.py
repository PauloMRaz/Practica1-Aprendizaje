import random

class Perceptron:
    def __init__(self, num_inputs):
        self.weights = [random.uniform(-1, 1) for _ in range (num_inputs)]
        self.bias = random.uniform(-1,1)

    def predict(self, inputs):
        activation = self.bias
        for i in range(len(inputs)):
            activation += inputs[i] * self.weights[i]
        return 1 if activation >= 0 else 0
    
def train(self, inputs, target):
    output = self.predict(inputs)
    error = target - output
    self.bias += error
    for i in range(len(self.weights)):
        self.weights[i] += error * inputs[i]

def get_weights(self):
    return self.weights

def save_weights(self, filename):
    with open(filename, "w") as f:
        f.write(f"{self.bias}\n")
        for w in self.weights:
            f.write(f"{w}\n")

def load_weights(self, filename):
    with open(filename, "r") as f:
        self.bias = float(f.readline())