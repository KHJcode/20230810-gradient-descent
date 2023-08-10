import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2 - 3*x + 2

def f_prime(x):
    return 2*x - 3

x = np.linspace(-2, 4, 400)
y = f(x)
gradient = f_prime(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='f(x) = x^2 - 3x + 2', color='blue')
plt.plot(x, gradient, label="f'(x) = 2x - 3", color='red')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(1.5, color='green', linewidth=0.5, linestyle='--', label='Critical Point (x = 1.5)')
plt.fill_between(x, gradient, where=(gradient > 0), interpolate=True, color='lightcoral', alpha=0.3, label='Positive Gradient')
plt.fill_between(x, gradient, where=(gradient < 0), interpolate=True, color='lightgreen', alpha=0.3, label='Negative Gradient')
plt.xlabel('x')
plt.ylabel('f(x), f\'(x)')
plt.title('Function and Gradient Visualization')
plt.legend()
plt.grid()
plt.savefig('function_and_gradient.png')
plt.show()
