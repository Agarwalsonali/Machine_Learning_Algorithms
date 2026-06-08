import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    return x**2 + x*y + y**2

def grad_x(x, y):   
    return 2*x + y

def grad_y(x, y):
    return x + 2*y


x = 3
y = 4

alpha = 0.1

iterations = 50

losses = []

for i in range(iterations):

    dx = grad_x(x, y)
    dy = grad_y(x, y)

   
    x = x - alpha * dx
    y = y - alpha * dy

   
    loss = f(x, y)
    losses.append(loss)

    print(f"Iteration {i+1}: x = {x:.4f}, y = {y:.4f}, loss = {loss:.4f}")


print("\nFinal Values")
print("x =", x)
print("y =", y)

plt.plot(range(1, iterations + 1), losses)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Gradient Descent Loss Curve")
plt.grid(True)
plt.show()