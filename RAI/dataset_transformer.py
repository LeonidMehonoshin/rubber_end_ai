class DatasetTransformer:
    def __init__(self, dataset):
        import pandas as pd
        import sklearn
        from .dataset_data_helper import DatasetDataHelper
        dataset_data_helper = DatasetDataHelper(dataset)

        data = {
            'features': dataset_data_helper.get('features'),
            'encoders': {},
            'scaler': sklearn.preprocessing.StandardScaler()
        }

        transformed_df = dataset[data['features']].copy()

        data['medians'] = transformed_df.median(numeric_only=True)
        transformed_df = transformed_df.fillna(data['medians'])

        for feature in data['features']:
            if not pd.api.types.is_numeric_dtype(transformed_df[feature]):
                encoder = sklearn.preprocessing.LabelEncoder()
                transformed_df[feature] = encoder.fit_transform(
                    transformed_df[feature].astype(str)
                )
                data['encoders'][feature] = encoder

        data['transformed'] = data['scaler'].fit_transform(transformed_df).astype('float32')
        self.__data = data

    def get(self, name):
        return self.__data[name]
