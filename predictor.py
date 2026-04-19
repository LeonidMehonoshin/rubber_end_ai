class Predictor:
    def __init__(self, checkpoint, InputTransformer, model, torch, pd):
        self.__torch = torch
        self.__pd = pd
        self.__target_max = checkpoint['target_max']
        self.__model = model
        self.__model.load_state_dict(checkpoint['model_state'])
        self.__model.eval()
        self.__processor_state = checkpoint['processor_state']
        self.__InputTransformer = InputTransformer

    def predict(self, input_data):
        with self.__torch.no_grad():
            transformer = self.__InputTransformer(
                input_data,
                self.__processor_state['encoders'],
                self.__processor_state['scaler'],
                self.__processor_state['medians'],
                self.__processor_state['features'],
                self.__pd
            )

            processed_tensor = self.__torch.tensor(transformer.get()).float().to(next(self.__model.parameters()).device)
            output = self.__model(processed_tensor)
            result = output.item() * self.__target_max
            return max(0, int(result))
