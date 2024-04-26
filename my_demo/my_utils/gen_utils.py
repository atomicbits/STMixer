import argparse
import re
import os 

from alphaction.config import cfg

def list_parser(value):
    """
    Parse a comma-separated list of strings.
    """
    return value.split(',')


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Your script description here.")

    parser.add_argument("--video_path", "-vp", type=str, required=True,
                        help="Path to video file")
    
    parser.add_argument("--actions", type=list_parser, default=['fall', 'fight', 'kick', 'push'],
                        help="A list of action labels separated by commas (default: fall,fight,kick,push)")
    
    parser.add_argument("-sf", "--starting_frame_index", type=int, default=50,
                        help="Starting frame index (default: 50)")
    parser.add_argument("-li", "--length_input", type=int, default=100,
                        help="Number of frames to be processed (default: 100)")
    parser.add_argument("-sh", "--slice_height", type=int, default=800,
                        help="Patch height (default: 800)")
    parser.add_argument("-sw", "--slice_width", type=int, default=1000,
                        help="Patch width (default: 1000)")
    parser.add_argument("-or", "--overlap_ratio", type=float, default=0.2,
                        help="Patch overlap ratio (default: 0.2)")

    
    parser.add_argument("-mn", "--model_name", type=str, default="VMAEv2",
                        help="Name of the model (default: VMAEv2)")
    parser.add_argument("-label", "--label_path", type=str, default="/work/my_demo/labels.txt",
                        help="Label of actions in ava dataset")

    parser.add_argument("-pt", "--person_threshold", type=float, default=0.3,
                        help="Confidence threshold on actor (default: 0.3)")

    parser.add_argument("-sr", "--sampling_rate", type=int, default=3,
                        help="Sampling rate (default: 3)")

    parser.add_argument("-tk", "--top_k", type=int, default=5,
                        help="Number of actions per person (default: 5)")
    
    parser.add_argument("--agnostic", action="store_true",
                        help="If present, set agnostic to True, otherwise False (default: False)")
    parser.add_argument("--nms_score", type=str, default="obj_score", 
                       help="score used for nms postprocessing, obj_score, action_score, joint")
    parser.add_argument("--nms_distance", type=str, default="IOS", 
                       help="distance used for nms postprocessing, IOS or IOU")
    parser.add_argument("--nms_thresh", type=float, default=0.6, 
                       help="nms threshold defualt 0.6")
    
    parser.add_argument("--add_patch_index", action="store_false",
                        help="If present, set patch_index to False, otherwise True (default: True)")
    

    

    return parser.parse_args()


def create_exp_dict(args):
    """
    Create a dictionary representing experiment parameters from parsed arguments.

    Parameters:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        dict: Dictionary representing experiment parameters.
    """
    exp_dict = {
        'model_name': args.model_name,
        'model_params': {
            'person_threshold': args.person_threshold,
            'sampling_rate': args.sampling_rate
        },
        'orig_post_processing': {
            'top_k': args.top_k
        },
        'aggregation': {
            'method': {"agnostic" if args.agnostic else "class_based"},
            'params': {"score": args.nms_score, 
                       "distance": args.nms_distance, 
                       "nms_thr": args.nms_thresh}
        },
        'video_path': args.video_path,
        'slicing_params': {
            'slice_height': args.slice_height,
            'slice_width': args.slice_width,
            'overlap_ratio': args.overlap_ratio
        },
        'video_params': {
            'st_frame_index': args.starting_frame_index,
            'length_input': args.length_input
        },
        'vis_params': {
            'actions': args.actions,
            'add_patch_index': args.add_patch_index
        }
    }
    
    return exp_dict


def parse_label_file(file_path):
    """
    Parse a label file and extract label_id and label_name pairs.

    Parameters:
        file_path (str): Path to the label file.

    Returns:
        dict: A dictionary containing label_id and label_name pairs.
    """
    
    # Initialize an empty dictionary to store label_id and label_name pairs
    label_dict = {}

    # Read the contents of the file
    with open(file_path, 'r') as file:
        contents = file.read()

    # Regular expressions to extract label_id and label_name pairs
    matches = re.findall(r'label \{ name: "(.*?)" label_id: (\d+)', contents)

    # Iterate over the matches and populate the label dictionary
    for label_name, label_id in matches:
        label_dict[int(label_id)] = label_name

    return label_dict


def create_experiment_folder(base_dir, experiment_name):
    
    """
    Create a new experiment folder in the specified base directory with a unique experiment number.

    Parameters:
        base_dir (str): The base directory where the experiment folder will be created.
        experiment_name (str): The name of the experiment.

    Returns:
        str: The path of the newly created experiment folder.

    Example:
        If the base directory 'experiments' contains folders 'experiment_1', 'experiment_2', and 'experiment_3',
        and you call create_experiment_folder('experiments', 'experiment'), it will create a new folder named
        'experiment_4' within the 'experiments' directory and return the path to it.
    """
    
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
        _ , last_experiment_number = find_largest_exp_folder(base_dir, tag=experiment_name+'_')
        
        # Create a new folder with the next experiment number
        new_folder_number = last_experiment_number + 1
        new_folder = os.path.join(base_dir, f"{experiment_name}_{new_folder_number}")
    
    # Create the new experiment folder
    os.makedirs(new_folder)
    return new_folder


def find_largest_exp_folder(directory, tag='exp_'):
    
    """
    Find the directory with the largest index suffix starting with a given tag in the specified directory.

    Parameters:
        directory (str): The directory where the search will be performed.
        tag (str, optional): The prefix tag used to filter directories (default is 'exp_').

    Returns:
        (str, int) or None: The path of the directory with the largest index suffix and its suffix if found, else None.


    Example:
        If the directory 'experiments' contains folders 'exp_1', 'exp_2', and 'exp_3',
        and you call find_largest_exp_folder('experiments'), it will return the path to the folder 'exp_3'.
    """
    
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
    
    return largest_exp_dir, max_index


def create_config(args):
    if args.model_name == 'VMAEv2':
        config_file = '/work/config_files/VMAEv2-ViTB-16x4.yaml'
    if args.model_name == 'VMAE':
        config_file = '../config_files/VMAE-ViTB-16x4.yaml'
    
    cfg.merge_from_file(config_file)

    # change model weight path
    if args.model_name == 'VMAEv2':
        cfg.merge_from_list(["MODEL.WEIGHT", "/work/checkpoints/VMAEv2_ViTB_16x4.pth"])
    if args.model_name == 'VMAE':
        cfg.merge_from_list(["MODEL.WEIGHT", "/work/checkpoints/VMAE_ViTB_16x4.pth"])

    # change output dir
    cfg.merge_from_list(["OUTPUT_DIR", "/work/output_dir/"])

    # change person threshold
    cfg.merge_from_list(["MODEL.STM.PERSON_THRESHOLD", args.person_threshold])

    # change sampling rate
    cfg.merge_from_list(["DATA.SAMPLING_RATE", args.sampling_rate])

    # change path for data_dir
    cfg.merge_from_list(["DATA.PATH_TO_DATA_DIR", "/work/ava"])

    # folder name of annotations
    cfg.merge_from_list(["AVA.ANNOTATION_DIR", "annotations/"])

    # file name of  frame_lists
    cfg.merge_from_list(["AVA.TRAIN_LISTS", ['sample.csv']])
    cfg.merge_from_list(["AVA.TEST_LISTS", ['sample.csv']])

    # file name of predicted_bboxes
    cfg.merge_from_list(["AVA.TRAIN_GT_BOX_LISTS", ['ava_sample_predicted_boxes.csv']])
    cfg.merge_from_list(["AVA.TEST_GT_BOX_LISTS", ['ava_sample_predicted_boxes.csv']])

    # file name of exlusions
    cfg.merge_from_list(["AVA.EXCLUSION_FILE", 'ava_sample_train_excluded_timestamps_v2.2.csv'])

    # number of batches in test scenario
    cfg.merge_from_list(["TEST.VIDEOS_PER_BATCH", 1])

    # number of workers
    cfg.merge_from_list(["DATALOADER.NUM_WORKERS", 1])

    return cfg

def config_show(cfg):
     # The shape of model input should be divisible into this. Otherwise, padding 0 to left and bottum. 
    print("cfg.DATALOADER.SIZE_DIVISIBILITY: ", cfg.DATALOADER.SIZE_DIVISIBILITY)
    
    # Sampling rate in constructing the clips.
    print("cfg.DATA.SAMPLING_RATE: ", cfg.DATA.SAMPLING_RATE)
    
    # Length of clip
    print("cfg.DATA.NUM_FRAMES: ", cfg.DATA.NUM_FRAMES)
    
    # Length of sequence frames from which a clip is constructed.
    seq_len = cfg.DATA.SAMPLING_RATE * cfg.DATA.NUM_FRAMES
    print("Length of clip: ", seq_len)
    
    print("cfg.MODEL.STM.ACTION_CLASSES: ", cfg.MODEL.STM.ACTION_CLASSES)
    print("cfg.MODEL.STM.MEM_ACTIVE: ", cfg.MODEL.STM.MEM_ACTIVE)
  

    # Augmentation params.
    print("Augmentation params: mean= {}, std={} and BGR={}".format(cfg.DATA.MEAN, cfg.DATA.STD, cfg.AVA.BGR))
    print("scale and flip params: min={}, max={} and flip={}".format(cfg.DATA.TEST_MIN_SCALES, cfg.DATA.TEST_MAX_SCALE, cfg.AVA.TEST_FORCE_FLIP))


    

if __name__ == "__main__":
    pass
