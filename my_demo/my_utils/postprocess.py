# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2021.

from typing import List

import torch
import numpy as np


def nms(
    predictions: torch.tensor,
    sort_metric: str = "scores",
    match_threshold: float = 0.5,
    match_metric: str = "IOU"
):
    """
    Apply non-maximum suppression to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        predictions: (tensor) The location preds for the image
            along with the class predscores, Shape: [num_boxes,5].
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap thresh for
            match metric.
    Returns:
        A list of filtered indexes, Shape: [ ,]
    """

    # we extract coordinates for every
    # prediction box present in P
    
    # we extract the confidence scores as well
    scores = predictions[:, 0]
    
    
    x1 = predictions[:, 1]
    y1 = predictions[:, 2]
    x2 = predictions[:, 3]
    y2 = predictions[:, 4]

    

    # calculate area of every block in P
    areas = (x2 - x1) * (y2 - y1)

    # sort the prediction boxes in P
    # according to their confidence scores
    if sort_metric == "scores":
        order = scores.argsort()
    elif sort_metric == "areas":
        order = areas.argsort()
    else:
        raise ValueError()

    # initialise an empty list for
    # filtered prediction boxes
    keep = []

    while len(order) > 0:
        # extract the index of the
        # prediction with highest score
        # we call this prediction S
        idx = order[-1]

        # push S in filtered predictions list
        keep.append(idx.tolist())

        # remove S from P
        order = order[:-1]

        # sanity check
        if len(order) == 0:
            break

        # select coordinates of BBoxes according to
        # the indices in order
        xx1 = torch.index_select(x1, dim=0, index=order)
        xx2 = torch.index_select(x2, dim=0, index=order)
        yy1 = torch.index_select(y1, dim=0, index=order)
        yy2 = torch.index_select(y2, dim=0, index=order)

        # find the coordinates of the intersection boxes
        xx1 = torch.max(xx1, x1[idx])
        yy1 = torch.max(yy1, y1[idx])
        xx2 = torch.min(xx2, x2[idx])
        yy2 = torch.min(yy2, y2[idx])

        # find height and width of the intersection boxes
        w = xx2 - xx1
        h = yy2 - yy1

        # take max with 0.0 to avoid negative w and h
        # due to non-overlapping boxes
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)

        # find the intersection area
        inter = w * h

        # find the areas of BBoxes according the indices in order
        rem_areas = torch.index_select(areas, dim=0, index=order)

        if match_metric == "IOU":
            # find the union of every prediction T in P
            # with the prediction S
            # Note that areas[idx] represents area of S
            union = (rem_areas - inter) + areas[idx]
            # find the IoU of every prediction in P with S
            match_metric_value = inter / union

        elif match_metric == "IOS":
            # find the smaller area of every prediction T in P
            # with the prediction S
            # Note that areas[idx] represents area of S
            smaller = torch.min(rem_areas, areas[idx])
            # find the IoU of every prediction in P with S
            match_metric_value = inter / smaller
        else:
            raise ValueError()

        # keep the boxes with IoU less than thresh_iou
        mask = match_metric_value < match_threshold
        order = order[mask]
    return keep


class PostprocessPredictions:
    """Utilities for calculating IOU/IOS based match for given ObjectPredictions"""

    def __init__(
        self,
        sort_metric: str = "scores",
        match_threshold: float = 0.5,
        match_metric: str = "IOU"
    ):
        self.sort_metric = sort_metric
        self.match_threshold = match_threshold
        self.match_metric = match_metric

    def __call__(self):
        raise NotImplementedError()


class NMSPostprocess(PostprocessPredictions):
    def __call__(
        self,
        action_predictions,
    ):
        if action_predictions.shape[0] == 0:
            return None
        
        action_predictions_torch = torch.from_numpy(action_predictions)
        keep = nms(
            action_predictions_torch, 
            sort_metric = self.sort_metric,
            match_threshold=self.match_threshold, 
            match_metric=self.match_metric
            )
        if len(keep) != 0:
            selected_action_predictions= action_predictions_torch[keep].numpy()
        else:
            return None
        

        return selected_action_predictions, keep


