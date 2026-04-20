from color import Color
print(f'{Color.bold}{Color.green}Init...\r{Color.end}', end='')

import os
import yaml
import torch
import pandas as pd
import sklearn
import time
from cryptography.fernet import Fernet
import io

from model_patcher import ModelPatcher
from checkpoint_saver import CheckpointSaver
from dataset_data_helper import DatasetDataHelper
from dataset_transformer import DatasetTransformer
from input_transformer import InputTransformer
from trainer import Trainer
from predictor import Predictor

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_path, 'config.yaml')

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f'{Color.bold}{Color.red}[ERROR] Config file not found at {config_path}!{Color.end}')
        return

    paths = {
        'dataset': os.path.join(base_path, config['paths']['dataset']),
        'checkpoint': os.path.join(base_path, config['paths']['checkpoint']),
        'input': os.path.join(base_path, config['paths']['input']),
        'checkpoint_key': os.path.join(base_path, config['paths']['checkpoint_key'])
    }

    torch.serialization.add_safe_globals([sklearn.preprocessing.LabelEncoder, sklearn.preprocessing.StandardScaler])
    device = config['device']

    if config['training']['trainMode']:
        print(f'{Color.yellow}{Color.bold}TrainMode = True!{Color.end}')

        for repeat in range(5):
            for row in [
                f'{Color.bold}{Color.yellow}    starting!{Color.end}',
                f'{Color.bold}{Color.red}    STARTING!{Color.end}',
            ]:
                print(f'{row} {5-repeat}', end='\r')
                time.sleep(0.5)
        print('\r', '             ', '\r', '', end='')

        dataset = pd.read_csv(paths['dataset'])
        helper = DatasetDataHelper(dataset)
        features_list = helper.get('features')
        model_patcher = ModelPatcher(len(features_list), torch.nn)
        model = model_patcher.get_model_class()()
        learning_rate = config['training']['learning_rate']

        trainer = Trainer(
            model=model,
            device=device,
            dataset=dataset,
            torch=torch,
            DatasetTransformer=DatasetTransformer,
            DatasetDataHelper=DatasetDataHelper,
            pd=pd,
            sklearn=sklearn,
            epochs=config['training']['epochs'],
            patience=config['training']['patience'],
            Color=Color,
            learning_rate = learning_rate
        )

        CheckpointSaver(torch, trainer.get(), paths['checkpoint'], paths['checkpoint_key'], Fernet, io, Color)

    else:
        try:
            with open(paths['checkpoint_key'], 'rb') as key_file:
                key = key_file.read()

            with open(paths['checkpoint'], 'rb') as f:
                encrypted_checkpoint = f.read()

            fernet = Fernet(key)
            decrypted_checkpoint = fernet.decrypt(encrypted_checkpoint)

            checkpoint = torch.load(io.BytesIO(decrypted_checkpoint), map_location = device, weights_only = False)

        except FileNotFoundError:
            print(f'{Color.bold}{Color.red}[ERROR] Key or Checkpoint file not found!{Color.end}')
            return

        except Exception as e:
            print(f'{Color.bold}{Color.red}[ERROR] Decryption failed: {e}{Color.end}')
            return

        try:
            with open(paths['input'], 'r', encoding='utf-8') as f:
                user_input = yaml.safe_load(f)
        except Exception as e:
            print(f'{Color.bold}{Color.red}[ERROR] Input error: {e}{Color.end}')
            return

        features_from_ckpt = checkpoint['processor_state']['features']
        model_patcher = ModelPatcher(len(features_from_ckpt), torch.nn)
        model = model_patcher.get_model_class()()

        predictor = Predictor(
            checkpoint=checkpoint,
            InputTransformer=InputTransformer,
            model=model,
            torch=torch,
            pd=pd
        )

        result = predictor.predict(user_input)
        print(f'{Color.bold}{Color.cyan}[RESULT]: {result} (km){Color.end}')

if __name__ == '__main__': main()
