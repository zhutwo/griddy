# griddy

In order to comply with Georgia Tech policy, none of the original project starter code is included in this repo. Instead, I present instructions to patch the project code to allow interfacing with `griddy`. Though if you're thinking of actually doing this for Assignment 1, save yourself the trouble and don't.

### SETUP

`griddy.py` saves jsons into `JSON_FOLDER `

`griddy_plot.py` bulk plots them all using your `plot_curve()` function saving into `IMG_FOLDER `

`griddy_plot.py` also saves a table of accuracy results as csv into `JSON_FOLDER `


### PATCHING INSTRUCTIONS

Replace beginning of existing `run()` function inside `main.py` up to and including the data loading with this:

    def run(params, data):
        args = parser.parse_args()
    
        for key in params:
            setattr(args, key, params[key])
    
        train_data, train_label, val_data, val_label = data['train']
        test_data, test_label = data['test']
    
        ### REST OF CODE

Change return line to this:

    return train_loss_history, train_acc_history, valid_loss_history, valid_acc_history, best_acc, test_acc

Modify your `plot_curve()` function to take `img_folder`, `filename` and `title` args and adjust plot saving code accordingly.
