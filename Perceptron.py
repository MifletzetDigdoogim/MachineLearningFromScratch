import random
import math
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1 + math.e ** (-x))

class Perceptron:
    def __init__(self, w=random.uniform(1, -1)):
        print("Weight is " + str(w))
        self.weight = w
        self.LEARNING_RATE = 0.01

    def forward_propagate(self, input):
        self.net = input * self.weight
        predicted_output = sigmoid(self.net)
        return predicted_output

    def train(self, input, expected_output):
        # Calculate error
        predicted_output = self.forward_propagate(input)
        self.error = 0.5 * (expected_output - predicted_output) ** 2

        # Calculate gradient
        gradient = (predicted_output - expected_output) * (predicted_output)*(1 - predicted_output) * input
        # print("Gradient: " + str(gradient))

        # Calculate change in w
        change_in_w = ((self.error) / gradient) * self.LEARNING_RATE

        # Change w
        self.weight -= change_in_w
        # Keep weight in range of 1>=w>=-1
        if self.weight > 1:
            self.weight = 1
        if self.weight < -1:
            self.weight = -1



p = Perceptron()
errors = []
weights = []
for i in range(1000):
    num = random.uniform(1, -1)
    label = sigmoid(num / 2)
    p.train(num, label)
    errors.append(p.error)
    weights.append(p.weight)

plt.plot(errors)
plt.ylabel("Error")
plt.xlabel("Iteration")
plt.show("Error")

plt.plot(weights)
plt.ylabel("Weight")
plt.xlabel("Iteration")
plt.show("Weight")

plt.scatter(weights, errors)
plt.ylabel("Error")
plt.xlabel("Weight")
plt.show("Error(Weight)")

for i in range(2):
    num = random.uniform(1, -1)
    print(p.forward_propagate(num))
    label = sigmoid(num / 2)
    print("Expected: " + str(label))