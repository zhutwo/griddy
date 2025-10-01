import json
import yaml
import shutil
import os



id = "09362_6227c61b808911ef83aa8c8d28f6175f"

JSON_FOLDER = 'griddy'


def replace_initialization_code_in_file(params, python_file_path):
    # Define the default values for each parameter
    defaults = {
        "n_blocks": 4,
        "block_depth": 2,
        "pad": 1,
        "stride": 1,
        "k_conv": 3,
        "maxpool": 2,
        "dropout": 0.2,
        "out_channels": 16
    }

    # Generate the code strings, updating defaults with any values present in params
    code_lines = ['        # START PARAMS\n']
    for param, default in defaults.items():
        value = params.get(param, default)
        code_line = f'        self.{param} = params.pop("{param}", {value})\n'
        code_lines.append(code_line)
    code_lines.append('        # END PARAMS\n')

    # Read the entire file into memory
    with open(python_file_path, 'r') as file:
        lines = file.readlines()
    
    # Find the start and end index
    start_index = -1
    end_index = -1
    for i, line in enumerate(lines):
        if '# START PARAMS' in line:
            start_index = i
        elif '# END PARAMS' in line and start_index != -1:
            end_index = i
            break

    # Replace or insert the new code
    if start_index != -1 and end_index != -1:
        # Replace existing section
        lines = lines[:start_index] + code_lines + lines[end_index+1:]
    else:
        # Append at the end if no markers found
        lines.extend(code_lines)

    # Write the updated content back to the file
    with open(python_file_path, 'w') as file:
        file.writelines(lines)

def update_config_with_params():
    # Read the JSON file
    with open(f'{JSON_FOLDER}/{id}.json', 'r') as file:
        data = json.load(file)
    
    # Extract 'params' from the JSON data, ensuring 'steps' is formatted as a list if present
    params = data.get('params', {})
    replace_initialization_code_in_file(params, 'part2-pytorch/models/my_model.py')
    # Default settings to be appended
    defaults = {
        'network': {
            'model': 'MyModel'
        },
        'data': {
            'imbalance': 'regular',
            'save_best': True
        },
        'loss': {
            'loss_type': 'CE'
        }
    }
    
    # Prepare the YAML content with 'Train' as the top key
    yaml_data = {'Train': params}
    yaml_data.update(defaults)  # This appends the default settings under the 'Train' section
    
    # Write the updated parameters to the YAML file
    with open('part2-pytorch/configs/config_mymodel.yaml', 'w') as file:
        yaml.dump(yaml_data, file, sort_keys=False)

    source_file_path = os.path.join(JSON_FOLDER, f'{id}.pth')
    destination_file_path = os.path.join('part2-pytorch/checkpoints', 'mymodel.pth')
    shutil.copy(source_file_path, destination_file_path)

update_config_with_params()