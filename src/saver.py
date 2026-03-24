class Saver:
    def __init__(
        self,
        pickle,
        weights,
        biases,
        path
    ):
        self.__pickle = pickle
        self.__weights = weights
        self.__biases = biases
        self.__path = path

    def save(self):
        with open(self.__path, 'wb') as file:
            self.__pickle.dump(
                {
                    'weights': self.__weights,
                    'biases': self.__biases,
                },
                file
            )
