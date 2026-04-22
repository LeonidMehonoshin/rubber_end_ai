class Auth:
    @staticmethod
    def get_key(password, hashlib, base64):
        key = hashlib.sha256(password.encode()).digest()
        return base64.urlsafe_b64encode(key)
