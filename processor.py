class Processor:
    def __init__(self, features, pd, np, LabelEncoder, StandardScaler):
        self.__features = features
        self.__pd = pd
        self.__np = np
        self.__encoders = {}
        self.__scaler = StandardScaler()
        self.__LabelEncoder = LabelEncoder
        self.__medians = None

    def __encode_categories(self, dataset):
        dataset_copy = dataset.copy()

        for feature in self.__features:
            if not self.__pd.api.types.is_numeric_dtype(dataset_copy[feature]):
                if feature not in self.__encoders:
                    label_encoder = self.__LabelEncoder()
                    dataset_copy[feature] = label_encoder.fit_transform(dataset_copy[feature].astype(str))
                    self.__encoders[feature] = label_encoder

                else: dataset_copy[feature] = self.__encoders[feature].transform(dataset_copy[feature].astype(str))

        return dataset_copy

    def fit_transform(self, dataset):
        self.__medians = dataset[self.__features].median(numeric_only = True)
        dataset_filled = dataset[self.__features].fillna(self.__medians)
        dataset_encoded = self.__encode_categories(dataset_filled)
        numeric_dataset = dataset_encoded.select_dtypes(include = [self.__np.number])
        return self.__scaler.fit_transform(numeric_dataset).astype('float32')

    def transform_row(self, input_dict):
        row = self.__pd.DataFrame([input_dict])

        for feature in self.__features:
            if feature not in row.columns:
                row[feature] = self.__medians.get(feature, 0)

        row = row[self.__features].fillna(self.__medians)

        for feature, label_encoder in self.__encoders.items():
            try: row[feature] = label_encoder.transform(row[feature].astype(str))
            except: row[feature] = 0

        return self.__scaler.transform(row).astype('float32')

    def get_states(self):
        return {
            'encoders': self.__encoders,
            'scaler': self.__scaler,
            'medians': self.__medians
        }

    def set_states(self, states):
        self.__encoders = states['encoders']
        self.__scaler = states['scaler']
        self.__medians = states['medians']
