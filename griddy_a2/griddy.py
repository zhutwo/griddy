import itertools
import json
import os
import torch
import uuid
from griddy_solver import GriddySolver
from joblib import Parallel, delayed

DEVICE = "cuda"
NJOBS = 8 # set to 0 for serial
FOLDER = "griddy"

fixed_params = {
    'model': 'GriddyModel',
    'save_best': False, # override solver
    'print': False,
    'imbalance': 'balanced',
    'loss_type': 'CE',
    'device': DEVICE,
    'download': False,
    #'dtype': 'float16',
    'early_stop': True,
    'max_attempts': 20,
}

focal_params = {
    'optimizer': ["SGD"],
    'batch_size': [16, 32],
    'learning_rate': [0.03, 0.01],
    'momentum': [0.98],
    'reg': [0.00001, 0.000005],
    'steps': [[20, 30, 40, 60]],
    'warmup': [5],
    'epochs': [50],
    'beta': [0.9],
    'gamma': [1,],
}

sgd_params = {
    'optimizer': ["SGD"],
    'batch_size': [16],
    'learning_rate': [0.03, 0.02],
    'momentum': [0.98, 0.99],
    'reg': [0.00001,],
    'steps': [[32, 49, 60, 100]],
    'warmup': [5],
    'epochs': [100],
    "n_blocks": [3],
    "block_depth": [3],
    "pad": [1],
    "stride": [1],
    "k_conv": [3],
    "maxpool": [2],
    "dropout": [0.2, 0.3],
    "out_channels": [64]
}

param_ranges = sgd_params

def _str_digits(num):
    return "{:.4f}".format(num).replace('.', '')

def _model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def _griddy_iter(params):

    name = str(uuid.uuid1()).replace('-','')

    print(f"START: {name}\nPARAMS: {params}")

    solver = GriddySolver(**params)
    acc, model, stop_epoch, train_acc, val_acc, class_acc = solver.train()
    print(f"END: {name} - ACC: {_str_digits(acc)}")
    model.half()
    torch.save(
        model.state_dict(),
        f"./{FOLDER}/{_str_digits(acc)}_{name}.pth",
    )

    for key in fixed_params.keys():
        params.pop(key, None)

    out = {
        'name': name,
        'size': _model_size(model),
        'acc': round(acc,4),
        'class_acc': class_acc,
        'stop_epoch': stop_epoch,
        'train_acc_history': train_acc,
        'valid_acc_history': val_acc,
        'params': params
    }

    with open(f'{FOLDER}/{_str_digits(acc)}_{name}.json', 'w') as f:
        json.dump(out, f)

def hit_griddy():

    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)

    print("\"Hitting the griddy...\" -Ellie")

    keys, values = zip(*param_ranges.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    for params in permutations:
        params.update(fixed_params)

    if NJOBS == 0:
        for params in permutations:
            _griddy_iter(params)
    else:
        Parallel(n_jobs=NJOBS)(delayed(_griddy_iter)(params) for params in permutations)

    print("DONE")

hit_griddy()