class Model:
    def __init__(self, np, pickle, input_dim, lr = 0.01):
        self.np = np
        fan_in = self.np.sqrt(2 / input_dim)
        self.W1 = self.np.random.randn(input_dim, 64) * fan_in # Входной > скрытый слой
        self.b1 = 0
        self.W2 = self.np.random.randn(64, 1) * fan_in # Скрытый > выходной слой
        self.b2 = 0
        self.lr = lr
        self.pickle = pickle

    # input-hidden weights
    def forward(self, X):
        self.h = self.np.tanh(X @ self.W1 + self.b1)
        return self.h @ self.W2 + self.b2

    # hidden-output weights
    def backward(self, X, y, y_pred):
        grad = 2 * (y_pred - y) / X.shape[0]
        grad_W2 = self.h.T @ grad
        grad_W1 = X.T @ (grad @ self.W2.T * (1 - self.np.tanh(X @ self.W1 + self.b1) ** 2))

        self.W2 -= self.lr * grad_W2
        self.W1 -= self.lr * grad_W1

    # training
    def train(self, X, y, epochs = 1000):
        for i in range(epochs):
            y_pred = self.forward(X)
            self.backward(X, y, y_pred)
            if i % 100 == 0:
                print(f'Эпоха {i}, MSE(погрешность): {self.np.mean((y_pred-y) ** 2):.3f}')

    def save_weights(self, filename):
        with open(filename, 'wb') as file:
            self.pickle.dump({
                'W1': self.W1,
                'b1': self.b1,
                'W2': self.W2,
                'b2': self.b2
            }, file)

    # result
    def predict(self, X):
        return self.forward(X)
