import os

class OutputManager:
    @staticmethod
    def save(path, user_input, result):
        log_entry = f"Input: {user_input} | Result: {result} km\n"
        with open(path, 'a', encoding='utf-8') as file:
            file.write(log_entry)
