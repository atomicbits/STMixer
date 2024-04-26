import torch
import numpy as np


def clip_boxes_tensor(boxes, height, width):
    boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], min=0., max=width-1)
    boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], min=0., max=height-1)
    return boxes

def map_bbox_from_prep_to_crop(bboxes, original_shape, resized_shape):
    # Calculate scaling factors
    width_scale_factor = original_shape[1] / resized_shape[1]
    height_scale_factor = original_shape[0] / resized_shape[0]
    
    # Map bounding box coordinates to original frame size
    mapped_bboxes = torch.zeros_like(bboxes)
    mapped_bboxes[:, 0] = (bboxes[:, 0] * width_scale_factor).int()
    mapped_bboxes[:, 1] = (bboxes[:, 1] * height_scale_factor).int()
    mapped_bboxes[:, 2] = (bboxes[:, 2] * width_scale_factor).int()
    mapped_bboxes[:, 3] = (bboxes[:, 3] * height_scale_factor).int()
    
    return mapped_bboxes

def map_bbox_from_crop_to_orig(bboxes, crop_position):
    """
    Maps bounding box coordinates from cropped frame to original frame.
    
    Args:
        bboxes (torch.Tensor): Tensor of shape Nx4 containing bounding box coordinates in the cropped frame.
        crop_position (tuple): Tuple containing the (x, y) coordinates of the top-left corner of the cropped frame.
        
    Returns:
        torch.Tensor: Tensor of shape Nx4 containing bounding box coordinates in the original frame.
    """
    # Map bounding box coordinates to original frame size
    mapped_bboxes = torch.zeros_like(bboxes)
    mapped_bboxes[:, 0] = (bboxes[:, 0] + crop_position[0]).int()
    mapped_bboxes[:, 1] = (bboxes[:, 1] + crop_position[1]).int()
    mapped_bboxes[:, 2] = (bboxes[:, 2] + crop_position[0]).int()
    mapped_bboxes[:, 3] = (bboxes[:, 3] + crop_position[1]).int()
    
    return mapped_bboxes


def concatenate_results(results_dict_list, top_k, patch_index=True):
    """
    Concatenate results from a dictionary of numpy arrays or lists

    Args:
        results_dict_list (dict): A dictionary containing results where keys represent frame indices
            and values are either numpy arrays or lists of numpy arrays.
        top_k (int): The number of top action indices and scores to consider.

    Returns:
        dict: A dictionary containing concatenated numpy arrays of results, where keys represent frame indices.
    """
    length = 1 + 4 + 2 * top_k + int(patch_index) # obj, bbox, actions indices and scores, patch_index
    
    output_results_dict_np = {}
    
    for cur_frame, results_frame in results_dict_list.items():
        if isinstance(results_frame, np.ndarray):
            output_results_dict_np[cur_frame] = results_frame
        elif isinstance(results_frame, list):
            # Filter out empty lists from the result_list
            non_empty_arrays = [arr for arr in results_frame if len(arr)!=0]

            # Convert the list to a numpy array if it's not empty
            if non_empty_arrays:
                output_results_dict_np[cur_frame] = np.concatenate(non_empty_arrays, axis=0)
            else:
                output_results_dict_np[cur_frame] = np.empty((0, length))  #  Create an empty array if the list is empty
                
    return output_results_dict_np
