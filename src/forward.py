class Forward:
    def __init__(
        self, np,
        input_data,
        weights, biases,
        Hidden_layer
    ):
        self.__np = np
        self.__input_data = input_data
        self.__weights = weights
        self.__biases = biases
        self.__Hidden_layer = Hidden_layer

    def get(self):
        hidden_layer = self.__Hidden_layer(self.__np, self.__input_data, self.__weights, self.__biases)
        return hidden_layer.get() @ self.__weights[1] + self.__biases[1]
