class Checkpoint:
    @staticmethod
    def load(path, key, device):
        import io, torch
        from cryptography.fernet import Fernet
        with open(path, 'rb') as file:
            encrypted_checkpoint = file.read()

        fernet = Fernet(key)
        decrypted_checkpoint = fernet.decrypt(encrypted_checkpoint)

        checkpoint = torch.load(io.BytesIO(decrypted_checkpoint), map_location = device, weights_only = False)
        return checkpoint

    def save(checkpoint, key, filename):
        import io, torch
        from cryptography.fernet import Fernet

        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        raw_data = buffer.getvalue()

        fernet = Fernet(key)
        encrypted_data = fernet.encrypt(raw_data)

        with open(filename, 'wb') as checkpoint_file:
            checkpoint_file.write(encrypted_data)
