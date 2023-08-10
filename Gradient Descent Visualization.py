import numpy as np
import matplotlib.pyplot as plt

# 선형 회귀를 위한 데이터 생성
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 경사 하강법 관련 함수 정의
def compute_cost(theta, X, y):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
    return cost

def batch_gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    cost_history = np.zeros(num_iterations)
    for i in range(num_iterations):
        gradient = (1 / m) * X.T.dot(X.dot(theta) - y)
        theta = theta - learning_rate * gradient
        cost_history[i] = compute_cost(theta, X, y)
    return theta, cost_history

def stochastic_gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    cost_history = np.zeros(num_iterations)
    for i in range(num_iterations):
        for j in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradient = xi.T.dot(xi.dot(theta) - yi)
            theta = theta - learning_rate * gradient
            cost_history[i] = compute_cost(theta, X, y)
    return theta, cost_history

def mini_batch_gradient_descent(X, y, theta, learning_rate, num_iterations, batch_size):
    m = len(y)
    cost_history = np.zeros(num_iterations)
    for i in range(num_iterations):
        for j in range(0, m, batch_size):
            xi = X[j:j+batch_size]
            yi = y[j:j+batch_size]
            gradient = (1 / batch_size) * xi.T.dot(xi.dot(theta) - yi)
            theta = theta - learning_rate * gradient
            cost_history[i] = compute_cost(theta, X, y)
    return theta, cost_history

# 데이터 전처리
X_b = np.c_[np.ones((100, 1)), X]
theta_initial = np.random.randn(2, 1)

# 하이퍼파라미터 설정
learning_rate = 0.01
num_iterations = 100
batch_size = 16

# Batch Gradient Descent
theta_batch, cost_history_batch = batch_gradient_descent(X_b, y, theta_initial, learning_rate, num_iterations)

# Stochastic Gradient Descent
theta_stochastic, cost_history_stochastic = stochastic_gradient_descent(X_b, y, theta_initial, learning_rate, num_iterations)

# Mini-Batch Gradient Descent
theta_mini_batch, cost_history_mini_batch = mini_batch_gradient_descent(X_b, y, theta_initial, learning_rate, num_iterations, batch_size)

# 시각화
plt.figure(figsize=(15, 5))

# Batch Gradient Descent 시각화
plt.subplot(1, 3, 1)
plt.plot(cost_history_batch, label='Batch GD')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Batch Gradient Descent')
plt.legend()

# Stochastic Gradient Descent 시각화
plt.subplot(1, 3, 2)
plt.plot(cost_history_stochastic, label='Stochastic GD')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Stochastic Gradient Descent')
plt.legend()

# Mini-Batch Gradient Descent 시각화
plt.subplot(1, 3, 3)
plt.plot(cost_history_mini_batch, label='Mini-Batch GD')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Mini-Batch Gradient Descent')
plt.legend()

plt.savefig('gradient_descent.png')
plt.tight_layout()
plt.show()
