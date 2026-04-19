class Trainer:
    def __init__(self,
        model, device,
        dataset, torch,
        DatasetTransformer, DatasetDataHelper,
        pd, sklearn, Color,
        epochs = 500, patience = 20
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
        self.__optimizer = self.__torch.optim.Adam(self.__model.parameters(), lr = 0.0001)
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

        print(
            f'{self.__Color.bold}{self.__Color.yellow}'
            f'\rdevice: {self.__device}\n'
            f'epochs: {self.__epochs}\n'
            f'patience: {self.__patience}\n'
            f'{self.__Color.end}'
        )

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

                if counter >= self.__patience:
                    print(f'{self.__Color.bold}{self.__Color.green}\n[STOP] Early stopping at epoch {epoch}.{self.__Color.end}')
                    break

                mse = current_loss ** 0.5
                mse_km = mse * self.__target_max

                log_lines = [
                    f' | Epoch: {epoch}',
                    f' | AutoStopCounter: {counter}',
                    f' | MSE_km: {mse_km:0f} km',
                    f' | MSE: {mse:.10%}'
                ]

                max_log_len = max(len(line) for line in log_lines)

                output = (
                    f'{self.__Color.bold}{self.__Color.purple}'
                    f'{'\n'.join([f'{line.ljust(max_log_len)} | ' for line in log_lines])}'
                    f'{self.__Color.end}'
                )

                print(output)
                print(self.__Color.backspace * 4, end='')
            print('\n' * 4)

        except KeyboardInterrupt:
            print('\n' * 4)
            print(f'{self.__Color.bold}{self.__Color.yellow}\n[WARNING] Training interrupted. {self.__Color.end}', end='')

    def get(self):
        return self.__checkpoint
