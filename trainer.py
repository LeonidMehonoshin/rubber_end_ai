class Trainer:
    def __init__(self, model, processor, features_helper, torch, save_path):
        self.__torch = torch
        self.__device = self.__torch.device('cpu')
        self.__model = model
        self.__processor = processor
        self.__features_helper = features_helper
        self.__y_max = 0
        self.__criterion = self.__torch.nn.MSELoss()
        self.__optimizer = self.__torch.optim.Adam(self.__model.parameters(), lr = 0.0001)
        self.__save_path = save_path

    def fit(self, dataset, epochs = 500, patience = 20):
        target_name = self.__features_helper.get_target()
        self.__y_max = dataset[target_name].max()
        features_only = dataset.drop(columns = [target_name])

        train_np = {
            'x': self.__processor.fit_transform(features_only),
            'y': (dataset[target_name].values / self.__y_max).reshape(-1, 1).astype('float32')
        }

        tensors = {
            'x': self.__torch.tensor(train_np['x']).float(),
            'y': self.__torch.tensor(train_np['y']).float()
        }

        best_loss = float('inf')
        counter = 0

        print(f'Training started, epochs: {epochs}...')

        try:
            for epoch in range(epochs + 1):
                self.__model.train()
                self.__optimizer.zero_grad()

                outputs = self.__model(tensors['x'])
                loss = self.__criterion(outputs, tensors['y'])
                current_loss = loss.item()

                loss.backward()
                self.__torch.nn.utils.clip_grad_norm_(self.__model.parameters(), max_norm = 1.0)
                self.__optimizer.step()

                if current_loss < best_loss:
                    best_loss = current_loss
                    counter = 0
                else:
                    counter += 1

                if counter >= patience:
                    print(f'\n[STOP] Early stopping at epoch {epoch}. Loss stopped improving.')
                    break

                if epoch % 10 == 0:
                    error_km = self.__torch.sqrt(loss).item() * self.__y_max
                    print(f'Epoch {epoch}, RMSE: {error_km:.0f} km | Counter: {counter}/{patience}')

        except KeyboardInterrupt:
            print('[WARING] Training interrupted by user. ', end = '')

        print('Saving current state...')
        self.__save(self.__save_path)

    def __save(self, filename):
        checkpoint = {
            'model_state': self.__model.state_dict(),
            'processor_state': self.__processor.get_states(),
            'y_max': self.__y_max,
            'features_list': self.__features_helper.get_features()
        }

        self.__torch.save(checkpoint, filename)
        print(f'Model successfully saved to {filename}')
