import torch

class Checkpoint:
    @staticmethod
    def load(path, device='cpu'):
        try:
            return torch.load(path, map_location=device, weights_only=True)
        except Exception:
            return torch.load(path, map_location=device, weights_only=False)

    @staticmethod
    def save(checkpoint, filename='checkpoint.pth'):
        if hasattr(checkpoint, "state_dict"):
            torch.save(checkpoint.state_dict(), filename)
        else:
            torch.save(checkpoint, filename)
