import pandas as pd

class InputTransformer:
    def __init__(self, input_data, encoders, scaler, medians, features):
        input_transform = pd.DataFrame([input_data])

        for feature in features:
            if feature not in input_transform.columns:
                input_transform[feature] = medians.get(feature, 0)

        input_transform = input_transform[features].copy()
        numeric_features = [col for col in features if col not in encoders]

        for col in features:
            if input_transform[col].isnull().any():
                val = medians.get(col, 0)
                input_transform[col] = input_transform[col].fillna(val)

        for feature, encoder in encoders.items():
            try:
                input_transform[feature] = encoder.transform(input_transform[feature].astype(str))
            except Exception:
                input_transform[feature] = 0

        if numeric_features and hasattr(scaler, 'mean_') and scaler.mean_ is not None:
            for col in numeric_features:
                input_transform[col] = pd.to_numeric(input_transform[col], errors='coerce')
                if input_transform[col].isnull().any():
                    input_transform[col] = input_transform[col].fillna(medians.get(col, 0.0))

            try:
                input_transform[numeric_features] = scaler.transform(input_transform[numeric_features].astype('float32'))
            except Exception:
                for col in numeric_features:
                    input_transform[col] = 0.0

        self.__input_transform = input_transform.values.astype('float32')

    def get(self):
        return self.__input_transform
