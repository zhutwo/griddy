import itertools
import json
import os
from joblib import Parallel, delayed
from utils import load_mnist_trainval, load_mnist_test
from main import run

DEBUG = False
NJOBS = 8 # set to 0 for serial
JSON_FOLDER = 'griddy'

param_ranges = {
    'batch_size': [64,],
    'learning_rate': [0.1,],
    'reg': [0.001,],
    'epochs': [10,],
    'hidden_size': [128,],
    'type': ['TwoLayerNet']
}

# base config for reference
base_config = {
    'Train': {
        'batch_size': 64,
        'learning_rate': 0.1,
        'reg': 0.001,
        'epochs': 10,
        'debug': True
    },
    'Model': {
        'type': 'TwoLayerNet',
        'hidden_size': 128,
    }
}

def encode_name(params):
    # model_type = params['type']
    # if model_type == 'TwoLayerNet':
    #     model_type = 'TLN'
    #     model_type += f"-h{params['hidden_size']}"
    # else:
    #     model_type = 'SR'
    return f"b{params['batch_size']}-h{params['hidden_size']}-l{str(params['learning_rate']).replace('.', '')}-r{str(params['reg']).replace('.', '')}-e{params['epochs']}"

def run_grid_search_iteration(params, data):

    name = encode_name(params)

    print(f"START: {name}")
    params['debug'] = DEBUG

    train_loss_history, train_acc_history, valid_loss_history, valid_acc_history, best_acc, test_acc = run(params, data)
    
    out = {
        'name': name,
        'acc_train': best_acc,
        'acc_test': test_acc,
        'train_loss_history': train_loss_history,
        'train_acc_history': train_acc_history,
        'valid_loss_history': valid_loss_history,
        'valid_acc_history': valid_acc_history,
        'params': params
    }

    with open(f'{JSON_FOLDER}/{name}.json', 'w') as f:
        json.dump(out, f)

def grid_search():

    if not os.path.exists(JSON_FOLDER):
        os.makedirs(JSON_FOLDER)

    print("\"Hitting the griddy...\" -Ellie")

    keys, values = zip(*param_ranges.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    data = {}
    data['train'] = load_mnist_trainval()
    data['test'] = load_mnist_test()

    if NJOBS == 0:
        for params in permutations:
            run_grid_search_iteration(params, data)
    else:
        Parallel(n_jobs=NJOBS)(delayed(run_grid_search_iteration)(params, data) for params in permutations)

    print("DONE")

grid_search()