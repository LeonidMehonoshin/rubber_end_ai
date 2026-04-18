class Predictor:
    def __init__(self, checkpoint, processor, torch, model, np, pd, LabelEncoder, StandardScaler):
        self.__torch = torch
        self.__checkpoint = checkpoint
        self.__y_max = self.__checkpoint['y_max']
        self.__processor = processor
        self.__processor.set_states(self.__checkpoint['processor_state'])
        self.__model = model
        self.__model.load_state_dict(self.__checkpoint['model_state'])
        self.__model.eval()

    def get(self, user_input):
        with self.__torch.no_grad():
            processed_input = self.__torch.tensor(self.__processor.transform_row(user_input)).float()
            prediction = self.__model(processed_input)
            return max(0, int(prediction.item() * self.__y_max))
