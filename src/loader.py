class Loader:
    def __init__(self, pickle, path):
        self.__pickle = pickle
        self.__path = path

    def get(self):
        with open(self.__path, 'rb') as file:
            return self.__pickle.load(file)
