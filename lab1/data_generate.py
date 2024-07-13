import matplotlib.pyplot as plt
import numpy as np

def generate_linear(n=100):
    np.random.seed(42)
    pts = np. random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels. append (0)
        else:
            labels. append (1)
    return np.array(inputs), np.array(labels).reshape(n, 1)


def generate_XOR_easy():
    np.random.seed(42)
    inputs = []
    labels = []

    for i in range (11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue

        inputs.append([0.1*i, 1-0.1*i])
        labels.append (1)
    return np.array(inputs), np.array(labels). reshape(21, 1)

# Generate datasets
x_linear, y_linear = generate_linear(n=100)
x_xor, y_xor = generate_XOR_easy()

# Plot Linear Data
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(x_linear[y_linear[:, 0] == 0][:, 0], x_linear[y_linear[:, 0] == 0][:, 1], color='red', label='Class 0')
plt.scatter(x_linear[y_linear[:, 0] == 1][:, 0], x_linear[y_linear[:, 0] == 1][:, 1], color='blue', label='Class 1')
plt.title('Linear Dataset')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()

# Plot XOR Data
plt.subplot(1, 2, 2)
plt.scatter(x_xor[y_xor[:, 0] == 0][:, 0], x_xor[y_xor[:, 0] == 0][:, 1], color='red', label='Class 0')
plt.scatter(x_xor[y_xor[:, 0] == 1][:, 0], x_xor[y_xor[:, 0] == 1][:, 1], color='blue', label='Class 1')
plt.title('XOR Dataset')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()

plt.savefig('datasets_plot.png')