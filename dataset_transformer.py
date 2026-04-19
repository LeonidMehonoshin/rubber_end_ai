class DatasetTransformer:
    def __init__(self, DatasetDataHelper, dataset, pd, sklearn):
        self.__pd = pd
        self.__sklearn = sklearn
        self.__dataset = dataset
        self.__dataset_data_helper = DatasetDataHelper(self.__dataset)
        self.__data = {
            'features': self.__dataset_data_helper.get('features'),
            'encoders': {},
            'scaler': self.__sklearn.preprocessing.StandardScaler(),
        }
        self.__transform()

    def __transform(self):
        self.__data['transformed'] = self.__dataset[self.__data['features']].copy()
        del self.__dataset
        self.__data['medians'] = self.__data['transformed'].median(numeric_only = True)
        self.__data['transformed'] = self.__data['transformed'].fillna(self.__data['medians'])

        for feature in self.__data['features']:
            if not self.__pd.api.types.is_numeric_dtype(self.__data['transformed'][feature]):
                if feature not in self.__data['encoders']:
                    encoder = self.__sklearn.preprocessing.LabelEncoder()
                    self.__data['transformed'][feature] = encoder.fit_transform(
                        self.__data['transformed'][feature].astype(str)
                    )
                    self.__data['encoders'][feature] = encoder
                else:
                    self.__data['transformed'][feature] = self.__data['encoders'][feature].transform(
                        self.__data['transformed'][feature].astype(str)
                    )

        self.__data['transformed'] = self.__data['scaler'].fit_transform(self.__data['transformed']).astype('float32')

    def get(self, name):
        return self.__data[name]
