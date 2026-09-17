import yaml
import pandas as pd

class Loader:
    @staticmethod
    def load_input(path):
        with open(path, 'r', encoding='utf-8') as f:
            user_input = yaml.safe_load(f)
        return user_input

    @staticmethod
    def load_dataset(path):
        try:
            return pd.read_csv(path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                return pd.read_csv(path, encoding='cp1251')
            except Exception:
                return pd.read_csv(path, encoding='utf-8', errors='replace')
