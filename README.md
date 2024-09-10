# griddy

griddy.py saves jsons into JSON_FOLDER 

griddy_plot.py bulk plots them all using your plot_curve() function saving into IMG_FOLDER 

griddy_plot.py also saves a table of accuracy results as csv into JSON_FOLDER 


### INSTRUCTIONS

replace beginning of existing run() function up to and including the data loading with this:

    def run(params, data):
        args = parser.parse_args()
    
        for key in params:
            setattr(args, key, params[key])
    
        train_data, train_label, val_data, val_label = data['train']
        test_data, test_label = data['test']
    
        ### REST OF CODE

change return line to this:

    return train_loss_history, train_acc_history, valid_loss_history, valid_acc_history, best_acc, test_acc

modify your plot_curve() function to take `img_folder` and `filename` args and adjust plot saving code accordingly
