import pandas as pd
import sklearn.preprocessing
from .dataset_data_helper import DatasetDataHelper

class DatasetTransformer:
    def __init__(self, dataset):
        dataset_data_helper = DatasetDataHelper(dataset)
        features = dataset_data_helper.get('features')

        data = {
            'features': features,
            'encoders': {},
            'scaler': sklearn.preprocessing.StandardScaler(),
            'numeric_features': [],
            'categorical_features': [],
            'medians': {}
        }

        transformed_df = dataset[features].copy()

        for feature in features:
            if pd.api.types.is_numeric_dtype(transformed_df[feature]):
                data['numeric_features'].append(feature)
                median_val = float(transformed_df[feature].median())
                data['medians'][feature] = median_val
                transformed_df[feature] = transformed_df[feature].fillna(median_val)
            else:
                data['categorical_features'].append(feature)
                most_frequent = transformed_df[feature].mode().values[0] if not transformed_df[feature].mode().empty else 'unknown'
                data['medians'][feature] = most_frequent
                transformed_df[feature] = transformed_df[feature].fillna(most_frequent)

                encoder = sklearn.preprocessing.LabelEncoder()
                transformed_df[feature] = encoder.fit_transform(transformed_df[feature].astype(str))
                data['encoders'][feature] = encoder

        if data['numeric_features']:
            transformed_df[data['numeric_features']] = data['scaler'].fit_transform(
                transformed_df[data['numeric_features']]
            )

        data['transformed'] = transformed_df.values.astype('float32')
        self.__data = data

    def get(self, name):
        if name not in self.__data:
            raise KeyError(f"Key '{name}' not found in transformer data.")
        return self.__data[name]
