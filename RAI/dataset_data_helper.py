class DatasetDataHelper:
    def __init__(self, dataset):
        self.__dataset_columns = dataset.columns.tolist()
        self.__data = {
            'features': self.__dataset_columns[:-1],
            'target': self.__dataset_columns[-1]
        }

    def get(self, name):
        return self.__data[name]
