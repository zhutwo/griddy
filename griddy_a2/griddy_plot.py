import json
import os
import pandas as pd
import matplotlib.pyplot as plt

JSON_FOLDER = 'griddy'
IMG_FOLDER = 'img'

def plot_curves(train_acc_history, valid_acc_history, img_folder, filename, title):
    plt.figure()
    plt.plot(train_acc_history, label='Training Accuracy')
    plt.plot(valid_acc_history, label='Validation Accuracy')
    plt.title(f"{title}")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{img_folder}/{filename}.png')
    plt.close()

def plot_from_json(json_folder=JSON_FOLDER, img_folder=IMG_FOLDER):
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    for filename in os.listdir(json_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(json_folder, filename)
            
            # load the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)

            train_acc_history = data['train_acc_history']
            valid_acc_history = data['valid_acc_history']
            title = "{:.4f}".format(data['acc']).replace('.', '') + data['name']
            filename = title
            #title = f"LR:{data['params']['learning_rate']}"
            #title = f"Reg: {data['params']['reg']}"

            plot_curves(train_acc_history, valid_acc_history, img_folder, filename, title)

def json_to_csv(json_folder=JSON_FOLDER, csv_filename='acc_table.csv'):
    results = []
    
    for filename in os.listdir(json_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(json_folder, filename)
            
            with open(file_path, 'r') as file:
                data = json.load(file)

            entry = {
                'params': str(data['params']),
                'stop_epoch': data['stop_epoch'],
                'size': data['size'],
                'acc': data['acc'],
                'name': data['name']
            }
            results.append(entry)

    results_df = pd.DataFrame(results)

    results_df.to_csv(os.path.join(json_folder, csv_filename), index=False)

json_to_csv()
plot_from_json()