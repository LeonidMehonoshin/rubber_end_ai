class Loader:
    @staticmethod
    def load_checkpoint(path, key_path, Fernet, io, torch, Color, device):
        try:
            with open(key_path, 'rb') as key_file:
                key = key_file.read().strip() # Добавь strip на всякий случай

            # Читаем именно ЧЕКПОИНТ (path), а не ключ (key_path)
            with open(path, 'rb') as f:
                encrypted_checkpoint = f.read()

            fernet = Fernet(key)
            decrypted_checkpoint = fernet.decrypt(encrypted_checkpoint)

            checkpoint = torch.load(io.BytesIO(decrypted_checkpoint), map_location=device, weights_only=False)
            return checkpoint # Возвращаем результат

        except FileNotFoundError:
            print(f'{Color.bold}{Color.red}[ERROR] Key or Checkpoint file not found!{Color.end}')
            return

        except Exception as e:
            print(f'{Color.bold}{Color.red}[ERROR] Decryption failed: {e}{Color.end}')
            return

    @staticmethod
    def load_input(path, yaml, Color):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                user_input = yaml.safe_load(f)

        except Exception as e:
            print(f'{Color.bold}{Color.red}[ERROR] Input error: {e}{Color.end}')
            return

        return user_input

    @staticmethod
    def load_config(path, yaml, Color):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

        except FileNotFoundError:
            print(f'{Color.bold}{Color.red}[ERROR] Config file not found at {path}!{Color.end}')
            return

        return config
