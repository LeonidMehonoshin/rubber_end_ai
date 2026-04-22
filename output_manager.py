class OutputManager:
    @staticmethod
    def save(path, user_input, result, password, Fernet, Auth, os, hashlib, base64):
        fernet = Fernet(Auth.get_key(password, hashlib, base64))
        history = ''
        if os.path.exists(path):
            with open(path, 'rb') as file:
                try: history = fernet.decrypt(file.read()).decode()
                except: history = '[Error: Decryption failed]\n'

        history += f'Input: {user_input} | Result: {result} km\n'
        with open(path, 'wb') as file:
            file.write(fernet.encrypt(history.encode()))

    @staticmethod
    def decode(path, target, password, Fernet, Auth, Color, hashlib, base64):
        fernet = Fernet(Auth.get_key(password, hashlib, base64))
        with open(path, 'rb') as file:
            data = fernet.decrypt(file.read()).decode()
        with open(target, 'w', encoding='utf-8') as file:
            file.write(data)
        print(f'{Color.green}[SUCCESS] Decoded to {target}{Color.end}')
