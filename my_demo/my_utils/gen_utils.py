import re
import os 

def parse_label_file(file_path):
    # Initialize an empty dictionary to store label_id and label_name pairs
    label_dict = {}

    # Read the contents of the file
    with open(file_path, 'r') as file:
        contents = file.read()

    # Use regular expressions to extract label_id and label_name pairs
    matches = re.findall(r'label \{ name: "(.*?)" label_id: (\d+)', contents)

    # Iterate over the matches and populate the label dictionary
    for label_name, label_id in matches:
        label_dict[int(label_id)] = label_name

    return label_dict


def create_experiment_folder(base_dir, experiment_name):
    # Check if the base directory exists, if not create it
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Check if there are any folders in the base directory starting with the experiment name
    existing_folders = [folder for folder in os.listdir(base_dir) if folder.startswith(experiment_name)]
    
    # If there are no existing folders, create the first folder with the experiment name
    if not existing_folders:
        new_folder = os.path.join(base_dir, f"{experiment_name}_1")
    else:
        # Find the highest numbered experiment folder
        existing_folders.sort()
        last_experiment = existing_folders[-1]
        last_experiment_number = int(last_experiment.split('_')[-1])
        
        # Create a new folder with the next experiment number
        new_folder_number = last_experiment_number + 1
        new_folder = os.path.join(base_dir, f"{experiment_name}_{new_folder_number}")
    
    # Create the new experiment folder
    os.makedirs(new_folder)
    return new_folder


def find_largest_exp_folder(directory, tag='exp_'):
    # Get all directories in the given directory
    all_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    
    # Filter out directories that start with 'exp_'
    exp_dirs = [d for d in all_dirs if d.startswith(tag)]
    
    if not exp_dirs:
        return None  # No directories starting with 'exp_' found
    
    # Extract the numeric part of each directory name
    indices = [int(d.split('_')[1]) for d in exp_dirs]
    
    # Find the index of the largest 'i'
    max_index = max(indices)
    
    # Get the path of the directory with the largest 'i'
    largest_exp_dir = os.path.join(directory, f'{tag}{max_index}')
    
    return largest_exp_dir
