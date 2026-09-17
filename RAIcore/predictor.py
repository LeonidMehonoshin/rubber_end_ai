import torch
from .input_transformer import InputTransformer
from .model import Model

class Predictor:
    def __init__(self, checkpoint, input_data):
        features_list = checkpoint['processor_state']['features']

        model = Model(len(features_list))
        model.load_state_dict(checkpoint['model_state'])
        model.eval()

        with torch.no_grad():
            transformer = InputTransformer(
                input_data,
                checkpoint['processor_state']['encoders'],
                checkpoint['processor_state']['scaler'],
                checkpoint['processor_state']['medians'],
                features_list
            )

            raw_tensor = torch.tensor(transformer.get(), dtype=torch.float32)
            processed_tensor = raw_tensor.unsqueeze(0).to(next(model.parameters()).device)

            output = model(processed_tensor)
            raw_output_value = output.detach().cpu().item()

            target_scale = checkpoint.get('target_max', 1.0)
            result = raw_output_value * target_scale

            self.__result = max(0, int(round(result)))

    def get(self):
        return self.__result
