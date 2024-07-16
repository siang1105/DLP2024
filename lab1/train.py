import numpy as np
import matplotlib.pyplot as plt
from data_generate import generate_linear, generate_XOR_easy

# Sigmoid function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# The derivative of sigmoid function
def derivative_sigmoid(x):
    return np.multiply(x, 1.0 - x)

def relu(x):
    return np.maximum(0, x)

def derivative_relu(x):
    return (x > 0).astype(float)

# mse loss function
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# initialize weights
def initialize_weights(input_dim, hidden_dim1, hidden_dim2, output_dim):
    W1 = np.random.randn(input_dim, hidden_dim1)
    b1 = np.zeros((1, hidden_dim1))
    W2 = np.random.randn(hidden_dim1, hidden_dim2)
    b2 = np.zeros((1, hidden_dim2))
    W3 = np.random.randn(hidden_dim2, output_dim)
    b3 = np.zeros((1, output_dim))
    return W1, b1, W2, b2, W3, b3

# forward propagation
def forward_propagation(X, W1, b1, W2, b2, W3, b3):
    z1 = np.dot(X, W1) + b1
    # a1 = sigmoid(z1)
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    # a2 = sigmoid(z2)
    a2 = relu(z2)
    z3 = np.dot(a2, W3) + b3
    # a3 = sigmoid(z3)
    a3 = sigmoid(z3)
    return z1, a1, z2, a2, z3, a3

# back propagation
def back_propagation(X, y, z1, a1, z2, a2, z3, a3, W1, W2, W3, b1, b2, b3, learning_rate):
    m = X.shape[0]
    dz3 = a3 - y
    dW3 = np.dot(a2.T, dz3) / m
    db3 = np.sum(dz3, axis=0, keepdims=True) / m

    dz2 = np.dot(dz3, W3.T) * derivative_relu(a2)
    dW2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    dz1 = np.dot(dz2, W2.T) * derivative_relu(a1)
    dW1 = np.dot(X.T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3

    return W1, b1, W2, b2, W3, b3

def train_neural_network(X, y, input_dim, hidden_dim1, hidden_dim2, output_dim, learning_rate, epochs):
    W1, b1, W2, b2, W3, b3 = initialize_weights(input_dim, hidden_dim1, hidden_dim2, output_dim)
    losses = []

    for epoch in range(epochs):
        z1, a1, z2, a2, z3, a3 = forward_propagation(X, W1, b1, W2, b2, W3, b3)
        loss = mse_loss(y, a3)
        losses.append(loss)
        W1, b1, W2, b2, W3, b3 = back_propagation(X, y, z1, a1, z2, a2, z3, a3, W1, W2, W3, b1, b2, b3, learning_rate)
        if epoch % 1000 == 0:
            print(f'Epoch {epoch} loss: {loss}')

    for i in range(len(y)):
        ground_truth = y[i][0]
        prediction = a3[i][0]
        print(f'Iter{i} | Ground truth: {ground_truth:.1f} | prediction: {prediction:.5f}')
    return W1, b1, W2, b2, W3, b3, a3, loss, losses


def show_result(x, y, pred_y, losses):
    plt.clf()
    plt.subplot(1,2,1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.show()
    plt.savefig('generate_linear.png')
    plt.close()

    plt.plot(losses)
    plt.title('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    plt.savefig('learning_curve_linear.png')
    plt.close()

# generate dataset
# X, y = generate_linear()
X, y = generate_XOR_easy()

# defined layer
input_dim = 2
hidden_dim1 = 4
hidden_dim2 = 4
output_dim = 1
learning_rate = 0.1
epochs = 50000

W1, b1, W2, b2, W3, b3, a3, loss, losses = train_neural_network(X, y, input_dim, hidden_dim1, hidden_dim2, output_dim, learning_rate, epochs)

pred_y = (a3 >= 0.5).astype(int)
accuracy = np.mean(pred_y == y) * 100.0
print(f'loss = {loss:.5f} accuracy = {accuracy:.2f}%')

# plot result
# show_result(X, y, pred_y, losses)
