import json
import os
import pandas as pd
from utils import plot_curves

JSON_FOLDER = 'griddy'
IMG_FOLDER = 'img'

def plot_from_json(json_folder=JSON_FOLDER, img_folder=IMG_FOLDER):
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    for filename in os.listdir(json_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(json_folder, filename)
            
            # Load the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)

            train_loss_history = data['train_loss_history']
            train_acc_history = data['train_acc_history']
            valid_loss_history = data['valid_loss_history']
            valid_acc_history = data['valid_acc_history']
            name = data['name']

            plot_curves(train_loss_history, train_acc_history, valid_loss_history, valid_acc_history, img_folder, name)

def json_to_csv(json_folder=JSON_FOLDER, csv_filename='acc_table.csv'):
    results = []
    
    for filename in os.listdir(json_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(json_folder, filename)
            
            with open(file_path, 'r') as file:
                data = json.load(file)

            entry = {
                'name': data['name'],
                'acc_test': data['acc_test'],
                'acc_train': data['acc_train']
            }
            results.append(entry)

    results_df = pd.DataFrame(results)

    results_df.to_csv(os.path.join(json_folder, csv_filename), index=False)

json_to_csv()
plot_from_json()