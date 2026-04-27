class Trainer:
    def __init__(self):
        self.__checkpoint = None
        self.__is_running = False

    def run(self, device, dataset, epochs, patience, learning_rate, log_callback = None):
        import torch
        from .dataset_transformer import DatasetTransformer
        from .dataset_data_helper import DatasetDataHelper
        from .model import Model

        model = Model(len(DatasetDataHelper(dataset).get('features'))).to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        transformer = DatasetTransformer(dataset)

        target_name = DatasetDataHelper(dataset).get('target')
        target_max = dataset[target_name].max()
        target_np = (dataset[target_name].values / target_max).reshape(-1, 1).astype('float32')
        input_np = transformer.get('transformed')

        best_loss = float('inf')
        counter = 0
        tensors = {
            'input': torch.tensor(input_np).to(device),
            'target': torch.tensor(target_np).to(device)
        }

        self.__is_running = True
        for epoch in range(epochs + 1):
            model, counter, best_loss, logs = self.__train(
                model, optimizer, tensors, criterion, best_loss, counter, target_max, torch
            )

            if not self.__is_running:
                if log_callback: log_callback(None)
                break

            if log_callback:
                log_callback(logs)

            if counter >= patience: break

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

    def __train(self, model, optimizer, tensors, criterion, best_loss, counter, target_max, torch):
        model.train()
        optimizer.zero_grad()
        outputs = model(tensors['input'])
        loss = criterion(outputs, tensors['target'])
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            counter = 0
        else:
            counter += 1

        mse = current_loss ** 0.5
        mse_km = mse * target_max
        return model, counter, best_loss, {
            'MSE_km': f'{mse_km:.2f} km',
            'MSE': f'{mse:.4%}',
            'counter': counter
        }

    def stop(self):
        self.__is_running = False

    def get(self):
        return self.__checkpoint
