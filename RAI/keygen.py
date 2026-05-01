class KeyGen:
    def __init__(self, password):
        password_bytes = password.encode()
        import hashlib
        raw_key = hashlib.scrypt(
            password_bytes,
            salt = password_bytes,
            #cost factor, block size, arallelization factor, derived key length
            n = 16384, r = 8, p = 1, dklen = 32
        )
        import base64
        self.__key = base64.urlsafe_b64encode(raw_key)

    def get(self):
        return self.__key
