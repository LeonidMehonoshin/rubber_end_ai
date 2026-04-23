class Trainer:
    def __init__(self,
        model, device,
        dataset, torch,
        DatasetTransformer, DatasetDataHelper,
        pd, sklearn, Color,
        epochs = 500, patience = 20, learning_rate = 0.001
    ):
        self.__Color = Color
        self.__torch = torch
        self.__device = device
        self.__model = model.to(self.__device)
        self.__dataset = dataset
        self.__transformer = DatasetTransformer(DatasetDataHelper, dataset, pd, sklearn)
        self.__target_name = DatasetDataHelper(dataset).get('target')
        self.__target_max = 0
        self.__criterion = self.__torch.nn.MSELoss()
        self.__learning_rate = learning_rate
        self.__optimizer = self.__torch.optim.Adam(self.__model.parameters(), lr = self.__learning_rate)
        self.__epochs = epochs
        self.__patience = patience
        self.go()
        self.__checkpoint = {
            'model_state': self.__model.state_dict(),
            'processor_state': {
                'features': self.__transformer.get('features'),
                'encoders': self.__transformer.get('encoders'),
                'scaler': self.__transformer.get('scaler'),
                'medians': self.__transformer.get('medians')
            },
            'target_max': self.__target_max
        }

    def go(self):
        self.__target_max = self.__dataset[self.__target_name].max()
        target_np = (self.__dataset[self.__target_name].values / self.__target_max).reshape(-1, 1).astype('float32')
        input_np = self.__transformer.get('transformed')

        tensors = {
            'input': self.__torch.tensor(input_np).to(self.__device),
            'target': self.__torch.tensor(target_np).to(self.__device)
        }

        best_loss = float('inf')
        counter = 0

        log_lines = []
        final_log = ''
        try:
            for epoch in range(self.__epochs + 1):
                self.__model.train()
                self.__optimizer.zero_grad()

                outputs = self.__model(tensors['input'])
                loss = self.__criterion(outputs, tensors['target'])

                loss.backward()
                self.__torch.nn.utils.clip_grad_norm_(self.__model.parameters(), max_norm = 1.0)
                self.__optimizer.step()

                current_loss = loss.item()
                if current_loss < best_loss:
                    best_loss = current_loss
                    counter = 0
                else:
                    counter += 1

                mse = current_loss ** 0.5
                mse_km = mse * self.__target_max

                log_lines = [
                    f'  Device: {self.__device}',
                    f'  Learning_rate: {self.__learning_rate}',
                    f'  Epoch: {epoch} / {self.__epochs}',
                    f'  AutoStopCounter: {counter} / {self.__patience}',
                    f'  MSE_km: {mse_km:0f} km',
                    f'  MSE: {mse:.10%}'
                ]

                output_logs = f'{self.__Color.bold}{self.__Color.purple}'
                for line in log_lines:
                    output_logs += line + self.__Color.clear_line + '\n'

                print(output_logs + self.__Color.end, end = '')
                print(self.__Color.up * len(log_lines), end = '', flush = True)

                if counter >= self.__patience:
                    final_log = f'{self.__Color.bold}{self.__Color.green}\n[STOP] Early stopping at epoch {epoch}.'
                    break

        except KeyboardInterrupt:
            final_log = f'{self.__Color.bold}{self.__Color.yellow}[WARNING] Training interrupted.'

        print('\n' * (len(log_lines) + 1))
        print(final_log + self.__Color.end)

    def get(self):
        return self.__checkpoint
