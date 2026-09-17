import torch

class Model(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.__layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),

            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),

            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        output = self.__layers(x)

        if output.dim() > 1:
            return output.squeeze(-1)
        return output
