class DatasetDataHelper:
    def __init__(self, dataset, target_column=None):
        self.__dataset_columns = dataset.columns.tolist()

        target = target_column if target_column is not None else self.__dataset_columns[-1]

        if target not in self.__dataset_columns:
            raise ValueError(f"Target column '{target}' not found in dataset columns.")

        features = [col for col in self.__dataset_columns if col != target]

        self.__data = {
            'features': features,
            'target': target
        }

    def get(self, name):
        if name not in self.__data:
            raise KeyError(
                f"Invalid key '{name}'. Accessible data keys are: {list(self.__data.keys())}"
            )
        return self.__data[name]
