class ModelPatcher:
    def __init__(self, input_size, nn):
        self.__input_size = input_size
        self.__nn = nn

    def get_model_class(self):
        nn = self.__nn
        input_size = self.__input_size

        class DynamicModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.__layers = nn.Sequential(
                    nn.Linear(input_size, 64),
                    nn.Tanh(),
                    nn.Linear(64, 32),
                    nn.Tanh(),
                    nn.Linear(32, 1)
                )

            def forward(self, x):
                return self.__layers(x)

        return DynamicModel
