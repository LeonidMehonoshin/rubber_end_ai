class KeyGen:
    def __init__(self, password):
        import hashlib, base64
        self.__key = self.__generate(hashlib, base64, password)

    def __generate(self, hashlib, base64, password):
        hash = hashlib.sha256(password.encode()).digest()
        return base64.urlsafe_b64encode(hash)

    def get(self):
        return self.__key
