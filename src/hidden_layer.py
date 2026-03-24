class Hidden_layer:
    def __init__(self, np, input_data, weights, biases):
        self.__np = np
        self.__input_data = input_data
        self.__weights = weights
        self.__biases = biases

    def get(self):
        return self.__np.tanh(self.__input_data @ self.__weights[0] + self.__biases[0])
