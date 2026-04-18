import sys, os
import yaml
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

from model_patcher import ModelPatcher
from processor import Processor
from features_helper import FeaturesHelper
from trainer import Trainer
from predictor import Predictor

def main():
    print('Starting...')
    base_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_path, 'config.yaml')

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

    except FileNotFoundError:
        print(f'[ERROR] Config file not found at {config_path}!')
        return

    paths = {
        'dataset': os.path.join(base_path, config['paths']['dataset']),
        'checkpoint': os.path.join(base_path, config['paths']['checkpoint']),
        'input': os.path.join(base_path, config['paths']['input'])
    }

    training_config = {
        'max_epochs': config['training']['epochs'],
        'patience': config['training']['patience'],
        'trainMode': config['training']['trainMode']
    }

    torch.serialization.add_safe_globals([LabelEncoder, StandardScaler, Processor])
    features_helper = FeaturesHelper()
    features_list = features_helper.get_features()
    device = torch.device('cpu')

    if training_config['trainMode']:
        dataset = pd.read_csv(paths['dataset'])
        processor = Processor(features_list, pd, np, LabelEncoder, StandardScaler)
        model_patcher = ModelPatcher(len(features_list), torch.nn)
        Model = model_patcher.get_model_class()
        model = Model()
        trainer = Trainer(model, processor, features_helper, torch, paths['checkpoint'])
        trainer.fit(dataset, training_config['max_epochs'], training_config['patience'])

    else:
        try:
            checkpoint = torch.load(
                paths['checkpoint'],
                map_location = device,
                weights_only = False
            )

        except FileNotFoundError:
            print(f'[ERROR] CheckPoint file not found at: {paths['checkpoint']}')
            return

        except Exception as e:
            print(f'[ERROR] Failed to load checkpoint: {e}')
            return

        try:
            with open(paths['input'], 'r', encoding='utf-8') as f:
                user_input = yaml.safe_load(f)

        except FileNotFoundError:
            print(f'[ERROR] Input file not found at {paths['input']}')
            return

        except yaml.YAMLError:
            print(f'[ERROR] Failed to decode YAML in {paths['input']}')
            return

        model_patcher = ModelPatcher(len(checkpoint['features_list']), torch.nn)
        Model = model_patcher.get_model_class()
        model = Model()
        processor = Processor(checkpoint['features_list'], pd, np, LabelEncoder, StandardScaler)
        predictor = Predictor(checkpoint, processor, torch, model, np, pd, LabelEncoder, StandardScaler)

        result = predictor.get(user_input)
        print(f'\n[RESULT]: _{result}_ (km)')

if __name__ == '__main__': main()
