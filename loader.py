class Loader:
    def __init__(self, yaml, io, torch, Fernet, Color):
        self.__io = io
        self.__yaml = yaml
        self.__torch = torch
        self.__Color = Color
        self.__Fernet = Fernet

    def load_checkpoint(self, path, key_path, device):
        try:
            with open(key_path, 'rb') as key_file:
                key = key_file.read().strip()

            with open(path, 'rb') as f:
                encrypted_checkpoint = f.read()

            fernet = self.__Fernet(key)
            decrypted_checkpoint = fernet.decrypt(encrypted_checkpoint)

            checkpoint = self.__torch.load(self.__io.BytesIO(decrypted_checkpoint), map_location = device, weights_only = False)
            return checkpoint

        except FileNotFoundError:
            print(f'{self.__Color.bold}{self.__Color.red}[ERROR] Key or Checkpoint file not found!{self.__Color.end}')
            return

        except Exception as e:
            print(f'{self.__Color.bold}{self.__Color.red}[ERROR] Decryption failed: {e}{self.__Color.end}')
            return

    def load_input(self, path):
        try:
            with open(path, 'r', encoding = 'utf-8') as f:
                user_input = self.__yaml.safe_load(f)

        except Exception as e:
            print(f'{self.__Color.bold}{self.__Color.red}[ERROR] Input error: {e}{self.__Color.end}')
            return

        return user_input

    def load_config(self, path):
        try:
            with open(path, 'r', encoding = 'utf-8') as f:
                config = self.__yaml.safe_load(f)

        except FileNotFoundError:
            print(f'{self.__Color.bold}{self.__Color.red}[ERROR] Config file not found at {path}!{self.__Color.end}')
            return

        return config
