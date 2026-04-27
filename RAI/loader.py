class Loader:
    @staticmethod
    def load_input(path):
        import yaml
        with open(path, 'r', encoding='utf-8') as f:
            user_input = yaml.safe_load(f)
        return user_input

    @staticmethod
    def load_dataset(path):
        import pandas as pd
        return pd.read_csv(path)
