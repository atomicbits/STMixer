import numpy as np

def compute_iou(box1, box2, nms_distance='IOS'):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    if nms_distance == 'IOU':
        union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
            (box2[2] - box2[0]) * (box2[3] - box2[1]) - intersection
    elif nms_distance == 'IOS':
        min_area = min((box1[2] - box1[0]) * (box1[3] - box1[1]),
                       (box2[2] - box2[0]) * (box2[3] - box2[1]))
        union = min_area

    iou = intersection / union if union > 0 else 0
    return iou

def agnostic_nms_action_detection(detections, nms_score='obj_score', nms_distance='IOU', nms_thresh=0.5):
    """
    Apply Non-Maximum Suppression (NMS) on action detection results.

    Parameters:
        detections (numpy.ndarray): Array of shape Nx15 where N is the number of detected objects.
                                    Each row contains [obj_score, x1, y1, x2, y2, action_id1, action_id2, ...,
                                    action_id5, action_score1, action_score2, ..., action_score5].
        agnostic (bool): If True, NMS will be applied to all detections together.
                         If False, NMS will be applied separately for each class.
        type_score (str): Type of score to be used for sorting: 'obj_score', 'action_score', or 'joint'.
        nms_method (str): NMS method to be used: 'IOU' or 'IOS'.
        iou_threshold (float): Threshold for IOU or IOS.

    Returns:
        numpy.ndarray: Filtered detections after applying NMS.
    """

    # Sort detections based on score
    if nms_score == 'obj_score':
        score_index = 0
        sorted_indices = np.argsort(detections[:, score_index])[::-1]
    elif nms_score == 'action_score':
        score_index = slice(-5, None)  # Selecting action score columns
        sorted_indices = np.argsort(detections[:, score_index])[::-1]
    elif nms_score == 'joint':
        sorted_indices = np.argsort(detections[:, 0] * detections[:, 5])[::-1]

    detections = detections[sorted_indices]

    keep_indices = []
    while len(detections) > 0:
        keep_indices.append(detections[0])

        iou_scores = np.array([compute_iou(detections[0, 1:5], detections[i, 1:5], nms_distance) for i in range(1, len(detections))])
        mask = iou_scores <= nms_thresh
        detections = detections[1:][mask]

    return np.array(keep_indices)


def class_based_nms_action_detection(detections, nms_score='obj_score', nms_distance='IOU', nms_thresh=0.5):
    """
    Apply Non-Maximum Suppression (NMS) on action detection results based on their classes. Top action define the class.

    Parameters:
        detections (numpy.ndarray): Array of shape Nx15 where N is the number of detected objects.
                                    Each row contains [obj_score, x1, y1, x2, y2, action_id1, action_id2, ...,
                                    action_id5, action_score1, action_score2, ..., action_score5].
        nms_score (str): Type of score to be used for sorting: 'obj_score', 'action_score', or 'joint'.
        nms_distance (str): NMS method to be used: 'IOU' or 'IOS'.
        nms_thresh (float): Threshold for IOU or IOS.

    Returns:
        numpy.ndarray: Filtered detections after applying NMS.
    """

    # Sort detections based on score
     # Sort detections based on score
    if nms_score == 'obj_score':
        score_index = 0
        sorted_indices = np.argsort(detections[:, score_index])[::-1]
    elif nms_score == 'action_score':
        score_index = slice(-5, None)  # Selecting action score columns
        sorted_indices = np.argsort(detections[:, score_index])[::-1]
    elif nms_score == 'joint':
        sorted_indices = np.argsort(detections[:, 0] * detections[:, 5])[::-1]
    
    detections = detections[sorted_indices]
    
    # unique top class ids
    unique_action_ids = np.unique(detections[:, 5])
    
    
    keep_indices = []
    for action_id in unique_action_ids:
        action_mask = detections[:, 5] == action_id
        action_detections = detections[action_mask]

        while len(action_detections) > 0:
            keep_indices.append(action_detections[0])

            iou_scores = np.array([compute_iou(action_detections[0, 1:5], action_detections[i, 1:5], nms_distance) for i in range(1, len(action_detections))])

            mask = iou_scores <= nms_thresh
            action_detections = action_detections[1:][mask]

    return np.array(keep_indices)

def apply_nms_to_dict(detections_dict, agnostic=True, nms_score='obj_score', nms_distance='IOS', nms_thresh=0.5):
    """
    Apply NMS post-processing on a dictionary of action detections.

    Parameters:
        detections_dict (dict): Dictionary where keys are frame indices and values are arrays of action detections.
        agnostic (bool): If True, NMS will be applied to all detections together.
                         If False, NMS will be applied separately for each action ID.
        nms_score (str): Type of score to be used for sorting: 'obj_score', 'action_score', or 'joint'.
        nms_distance (str): NMS method to be used: 'IOU' or 'IOS'.
        nms_thresh (float): Threshold for IOU or IOS.

    Returns:
        dict: Dictionary with NMS-applied detections.
    """

    nms_applied_dict = {}
    for frame, detections in detections_dict.items():
        if agnostic:
            nms_applied_detections = agnostic_nms_action_detection(detections,
                                                                   nms_score=nms_score, 
                                                                   nms_distance=nms_distance, 
                                                                   nms_thresh=nms_thresh)
        else:
            nms_applied_detections = class_based_nms_action_detection(detections,
                                                                   nms_score=nms_score, 
                                                                   nms_distance=nms_distance, 
                                                                   nms_thresh=nms_thresh)


        nms_applied_dict[frame] = nms_applied_detections

    return nms_applied_dict