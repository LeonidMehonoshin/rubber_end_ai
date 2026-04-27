class InputTransformer:
    def __init__(self, input_data, encoders, scaler, medians, features):
        import pandas as pd

        input_transform = pd.DataFrame([input_data])

        for feature in features:
            if feature not in input_transform.columns:
                input_transform[feature] = medians.get(feature, 0)

        input_transform = input_transform[features].fillna(medians)

        for feature, encoder in encoders.items():
            try:
                input_transform[feature] = encoder.transform(input_transform[feature].astype(str))
            except:
                input_transform[feature] = 0

        self.__input_transform = scaler.transform(input_transform).astype('float32')

    def get(self):
        return self.__input_transform
