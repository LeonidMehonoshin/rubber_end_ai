def main():
    from color import Color
    print(f'{Color.bold}{Color.green}Init...\r{Color.end}', end='')

    import os, io, hashlib, shutil, base64
    import yaml, torch, sklearn
    import pandas as pd
    from cryptography.fernet import Fernet
    from model_patcher import ModelPatcher
    from checkpoint_saver import CheckpointSaver
    from dataset_data_helper import DatasetDataHelper
    from dataset_transformer import DatasetTransformer
    from input_transformer import InputTransformer
    from trainer import Trainer
    from predictor import Predictor
    from loader import Loader
    from auth import Auth
    from user_manager import UserManager
    from output_manager import OutputManager

    base_path = os.path.dirname(os.path.abspath(__file__))
    print(f'{Color.bold}    AUTH{Color.end}')

    while True:
        username = input('Login: ').strip()
        if username: break

    while True:
        password = input('Password: ').strip()
        if password: break

    user_manager = UserManager(os.path.join(base_path, 'users'), base_path, hashlib, os, Color, shutil, username)
    paths = user_manager.get_user_paths()

    loader = Loader(yaml, io, torch, Fernet, Color)
    config = loader.load_config(paths['config'])

    for key, value in config['paths'].items():
        paths[key] = os.path.join(paths['user_root'], value)

    torch.serialization.add_safe_globals([sklearn.preprocessing.LabelEncoder, sklearn.preprocessing.StandardScaler])
    mode = config.get('mode', 'default')
    print(f'{Color.yellow}[INFO] Mode: {mode}{Color.end}')

    if mode == 'decode-out':
        OutputManager.decode(
            paths['output'],
            paths['decoded_output'],
            password,
            Fernet,
            Auth,
            Color,
            hashlib,
            base64
        )

    elif mode == 'train':
        try:
            dataset = pd.read_csv(paths['dataset'])

        except FileNotFoundError:
            print(f'{Color.bold}{Color.red}[FAILED] Dataset file not found!{Color.end}')
            return

        features = DatasetDataHelper(dataset).get('features')
        model = ModelPatcher(len(features), torch.nn).get_model_class()()

        trainer = Trainer(
            model=model, device=config['device'], dataset=dataset, torch=torch,
            DatasetTransformer=DatasetTransformer, DatasetDataHelper=DatasetDataHelper,
            pd=pd, sklearn=sklearn, epochs=config['training']['epochs'],
            patience=config['training']['patience'], Color=Color,
            learning_rate=config['training']['learning_rate']
        )

        CheckpointSaver(torch, trainer.get(), paths['checkpoint'], paths['checkpoint_key'], Fernet, io, Color)

    elif mode == 'default':
        if not all(os.path.exists(paths[f]) for f in ['checkpoint', 'checkpoint_key']):
            print(f'{Color.red}[ERROR] Model not trained.{Color.end}')
            return

        checkpoint = loader.load_checkpoint(paths['checkpoint'], paths['checkpoint_key'], config['device'])
        if not checkpoint: return

        user_input = loader.load_input(paths['input'])
        features_count = len(checkpoint['processor_state']['features'])
        model = ModelPatcher(features_count, torch.nn).get_model_class()()

        predictor = Predictor(checkpoint, InputTransformer, model, torch, pd)
        res = predictor.predict(user_input)
        OutputManager.save(
            paths['output'],
            user_input,
            res,
            password,
            Fernet,
            Auth,
            os,
            hashlib,
            base64
        )

        print(f'{Color.cyan}{Color.bold}[RESULT]: {res} km{Color.end}')
        print(f'{Color.green}[INFO] Result saved to {paths['output']}{Color.end}')

if __name__ == '__main__': main()
