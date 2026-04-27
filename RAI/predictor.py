class Predictor:
    def __init__(self, checkpoint, input_data):
        import torch
        from .input_transformer import InputTransformer
        from .model import Model
        model = Model(len(checkpoint['processor_state']['features']))
        model.load_state_dict(checkpoint['model_state'])
        model.eval()

        with torch.no_grad():
            transformer = InputTransformer(
                input_data,
                checkpoint['processor_state']['encoders'],
                checkpoint['processor_state']['scaler'],
                checkpoint['processor_state']['medians'],
                checkpoint['processor_state']['features']
            )

            processed_tensor = torch.tensor(transformer.get()).float().to(next(model.parameters()).device)
            output = model(processed_tensor)
            result = output.item() * checkpoint['target_max']
            self.__result = max(0, int(result))

    def get(self):
        return self.__result
