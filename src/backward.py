class Backward:
    def __init__(
        self, np,
        weights, biases,
        input_data, output_data,
        Hidden_layer, learning_rate,
        current_predictions
    ):
        self.__np = np
        self.__weights = weights
        self.__biases = biases
        self.__input_data = input_data
        self.__output_data = output_data
        self.__Hidden_layer = Hidden_layer
        self.__learning_rate = learning_rate
        self.__current_predictions = current_predictions

    def get(self):
        loss_gradient = 2 * (self.__current_predictions - self.__output_data) / self.__input_data.shape[0]
        hidden_layer = self.__Hidden_layer(self.__np, self.__input_data, self.__weights, self.__biases)
        dh_dz = (1.0 - hidden_layer.get() ** 2)

        gradients = {
            'weights': [
                self.__input_data.T @ (loss_gradient @ self.__weights[1].T * dh_dz),
                hidden_layer.get().T @ loss_gradient
            ],
            'biases': [
                (loss_gradient @ self.__weights[1].T * dh_dz).sum(axis=0),
                loss_gradient.sum(axis=0)
            ]
        }

        self.__weights[0] -= self.__learning_rate * gradients['weights'][0]
        self.__weights[1] -= self.__learning_rate * gradients['weights'][1]
        self.__biases[0] -= self.__learning_rate * gradients['biases'][0]
        self.__biases[1] -= self.__learning_rate * gradients['biases'][1]

        return self.__weights, self.__biases
