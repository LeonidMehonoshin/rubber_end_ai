class CheckpointSaver:
    def __init__(self, torch, checkpoint, filename, key_filename, Fernet, io, Color):
        self.__torch = torch
        self.__checkpoint = checkpoint
        self.__filename = filename
        self.__key_filename = key_filename
        self.__Fernet = Fernet
        self.__io = io
        self.__Color = Color
        self.save()

    def save(self):
        key = self.__Fernet.generate_key()
        with open(self.__key_filename, 'wb') as key_file: key_file.write(key)

        buffer = self.__io.BytesIO()
        self.__torch.save(self.__checkpoint, buffer)
        raw_data = buffer.getvalue()

        fernet = self.__Fernet(key)
        encrypted_data = fernet.encrypt(raw_data)

        with open(self.__filename, 'wb') as checkpoint_file:
            checkpoint_file.write(encrypted_data)

        print(
            f'{self.__Color.green}Checkpoint encrypted and saved to {self.__filename}\n'
            f'Key saved to {self.__key_filename}{self.__Color.end}'
        )
