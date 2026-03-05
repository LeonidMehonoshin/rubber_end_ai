class Model:
    def __init__(self, np, input_dim, lr = 0.01):
        self.np = np
        fan_in = self.np.sqrt(2 / input_dim)
        self.W1 = self.np.random.randn(input_dim, 64) * fan_in
        self.b1 = 0
        self.W2 = self.np.random.randn(64, 1) * fan_in
        self.b2 = 0
        self.lr = lr

    def forward(self, X):
        self.h = self.np.tanh(X @ self.W1 + self.b1)
        return self.h @ self.W2 + self.b2

    def backward(self, X, y, y_pred):
        grad = 2 * (y_pred - y) / X.shape[0]
        grad_W2 = self.h.T @ grad
        grad_W1 = X.T @ (grad @ self.W2.T * (1 - self.np.tanh(X @ self.W1 + self.b1) ** 2))

        self.W2 -= self.lr * grad_W2
        self.W1 -= self.lr * grad_W1

    def train(self, X, y, epochs = 1000):
        for i in range(epochs):
            y_pred = self.forward(X)
            self.backward(X, y, y_pred)
            if i % 100 == 0:
                print(f'Эпоха {i}, MSE(погрешность): {self.np.mean((y_pred-y) ** 2):.3f}')

    def predict(self, X):
        return self.forward(X)
