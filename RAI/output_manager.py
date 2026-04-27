class OutputManager:
    @staticmethod
    def save(path, user_input, result):
        import os
        history = ''
        if os.path.exists(path):
            with open(path, 'r', encoding = 'utf-8') as file:
                try:
                    history = file.read()
                except:
                    history = '[Error: Could not read file]\n'

        history += f'Input: {user_input} | Result: {result} km\n'
        with open(path, 'w', encoding = 'utf-8') as file:
            file.write(history)
