class InputTransformer:
    def __init__(self, input_data, encoders, scaler, medians, features, pd):
        self.__pd = pd
        self.__input_data = input_data
        self.__medians = medians
        self.__encoders = encoders
        self.__scaler = scaler
        self.__features = features
        self.__input_transform = None
        self.__transform()

    def __transform(self):
        self.__input_transform = self.__pd.DataFrame([self.__input_data])

        for feature in self.__features:
            if feature not in self.__input_transform.columns:
                self.__input_transform[feature] = self.__medians.get(feature, 0)

        self.__input_transform = self.__input_transform[self.__features].fillna(self.__medians)

        for feature, encoder in self.__encoders.items():
            try:
                self.__input_transform[feature] = encoder.transform(self.__input_transform[feature].astype(str))
            except:
                self.__input_transform[feature] = 0

        self.__input_transform = self.__scaler.transform(self.__input_transform).astype('float32')

    def get(self):
        return self.__input_transform
