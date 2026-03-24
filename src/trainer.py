class Trainer:
    def __init__(
        self, np,
        input_data, output_data,
        weights, biases,
        Backward, Forward,
        Hidden_layer,
        learning_rate,
        epochs
    ):
        self.__np = np
        self.__input_data = input_data
        self.__output_data = output_data
        self.__weights = weights
        self.__biases = biases
        self.__Backward = Backward
        self.__Forward = Forward
        self.__Hidden_layer = Hidden_layer
        self.__learning_rate = learning_rate
        self.__epochs = epochs
        self.__results = []

    def train(self):
        for epoch in range(self.__epochs):
            forward = self.__Forward(
                self.__np, self.__input_data,
                self.__weights, self.__biases,
                self.__Hidden_layer
            )

            current_predictions = forward.get()
            backward = self.__Backward(
                self.__np,
                self.__weights, self.__biases,
                self.__input_data, self.__output_data,
                self.__Hidden_layer, self.__learning_rate,
                current_predictions
            )

            self.__weights, self.__biases = backward.get()

            if epoch % 50 == 0:
                self.__results.append({
                    'epoch': epoch,
                    'mse': round(self.__np.mean((current_predictions - self.__output_data) ** 2), 2),
                    'weights': self.__weights,
                    'biases': self.__biases
                })
        return self.__results
