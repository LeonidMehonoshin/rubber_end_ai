class CheckpointSaver:
    def __init__(self, torch, checkpoint, filename, Color):
        self.__torch = torch
        self.__checkpoint = checkpoint
        self.__filename = filename
        self.__Color = Color
        self.save()

    def save(self):
        self.__torch.save(self.__checkpoint, self.__filename)
        print(f'{self.__Color.green}Checkpoint successfully saved to {self.__filename}{self.__Color.end}')
