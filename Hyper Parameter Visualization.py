import numpy as np
import matplotlib.pyplot as plt

# 가상의 데이터 생성
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 학습 데이터에 절편 항을 추가
X_b = np.c_[np.ones((100, 1)), X]

def compute_cost(theta, X, y):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
    return cost

def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    cost_history = []

    for _ in range(num_iterations):
        gradient = (1 / m) * X.T.dot(X.dot(theta) - y)
        theta -= learning_rate * gradient
        cost = compute_cost(theta, X, y)
        cost_history.append(cost)

    return theta, cost_history

def plot_cost_history(ax, cost_history, learning_rate):
    ax.plot(cost_history, label=f'Learning Rate = {learning_rate}')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Cost History')

# 초기 하이퍼파라미터 설정
learning_rates = [0.01, 0.1, 0.5, 1.0]
num_iterations = 100

# 서브플롯을 사용하여 그래프 시각화
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.ravel()

for i, learning_rate in enumerate(learning_rates):
    theta = np.random.randn(2, 1)
    theta_final, cost_history = gradient_descent(X_b, y, theta, learning_rate, num_iterations)
    plot_cost_history(axs[i], cost_history, learning_rate)
    axs[i].legend()

plt.tight_layout()
plt.savefig('hyper_parameter.png')
plt.show()
