class UserManager:
    def __init__(self, users_root, base_root, hashlib, os, Color, shutil, username):
        self.__root = users_root
        self.__base = base_root
        self.__hashlib = hashlib
        self.__os = os
        self.__Color = Color
        self.__shutil = shutil
        self.__username = username

    def get_user_paths(self):
        user_id = self.__hashlib.sha256(self.__username.encode()).hexdigest()
        user_dir = self.__os.path.join(self.__root, user_id)
        default_dir = self.__os.path.join(self.__root, 'default')

        if not self.__os.path.exists(user_dir):
            self.__os.makedirs(user_dir)
            print(f'{self.__Color.green}[INFO] Created directory: {self.__Color.bold}{user_dir}{self.__Color.end}')

            for filename in ['config.yaml', 'input.yaml']:
                source = self.__os.path.join(default_dir, filename)
                destination = self.__os.path.join(user_dir, filename)
                if self.__os.path.exists(source):
                    self.__shutil.copy(source, destination)

            print(f'{self.__Color.yellow}[NOTICE] Profile initialized. Go to config.yaml for configuration.{self.__Color.end}')
            raise SystemExit

        return {'user_root': user_dir, 'config': self.__os.path.join(user_dir, 'config.yaml')}
