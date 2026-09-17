import torch
from .dataset_transformer import DatasetTransformer
from .dataset_data_helper import DatasetDataHelper
from .model import Model

class Trainer:
    def __init__(self):
        self.__checkpoint = None
        self.__is_running = False

    def run(self, device, dataset, epochs, patience, learning_rate, log_callback=None):
        data_helper = DatasetDataHelper(dataset)
        features_list = data_helper.get('features')
        target_name = data_helper.get('target')

        model = Model(len(features_list)).to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        transformer = DatasetTransformer(dataset)

        target_max = float(dataset[target_name].max())
        if target_max == 0:
            target_max = 1.0

        target_np = (dataset[target_name].values / target_max).reshape(-1).astype('float32')
        input_np = transformer.get('transformed')

        best_loss = float('inf')
        counter = 0

        tensors = {
            'input': torch.tensor(input_np, dtype=torch.float32).to(device),
            'target': torch.tensor(target_np, dtype=torch.float32).to(device)
        }

        self.__is_running = True

        for epoch in range(epochs):
            if not self.__is_running:
                break

            model, counter, best_loss, logs = self.__train(
                model, optimizer, tensors, criterion, best_loss, counter, target_max, epoch
            )

            if log_callback:
                log_callback(logs)

            if counter >= patience:
                break

        self.__is_running = False
        self.__checkpoint = {
            'model_state': model.state_dict(),
            'processor_state': {
                'features': transformer.get('features'),
                'encoders': transformer.get('encoders'),
                'scaler': transformer.get('scaler'),
                'medians': transformer.get('medians')
            },
            'target_max': target_max
        }
        return 'Done'

    def __train(self, model, optimizer, tensors, criterion, best_loss, counter, target_max, epoch):
        model.train()
        optimizer.zero_grad()

        outputs = model(tensors['input'])
        loss = criterion(outputs, tensors['target'])

        loss.backward()
        optimizer.step()

        current_loss = float(loss.item())

        if torch.isnan(loss):
            self.stop()

        if current_loss < best_loss:
            best_loss = current_loss
            counter = 0
        else:
            counter += 1

        rmse_normalized = current_loss ** 0.5

        return model, counter, best_loss, {
            'epoch': epoch,
            'lr': float(optimizer.param_groups[0]['lr']),
            'MSE': float(rmse_normalized),
            'counter': counter
        }

    def stop(self):
        self.__is_running = False

    def get(self):
        return self.__checkpoint
