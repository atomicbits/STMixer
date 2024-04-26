from alphaction.dataset.datasets import cv2_transform as cv2_transform
from alphaction.dataset.datasets.utils import pack_pathway_output


import numpy as np
import torch
import math


def ava_preprocessing_cv2(sample_cropped_clip, cfg):
    
    #height, width, _ = sample_cropped_clip[0].shape
    
    imgs, boxes = cv2_transform.random_short_side_scale_jitter(
        sample_cropped_clip,
        min_sizes=cfg.DATA.TEST_MIN_SCALES,
        max_size=cfg.DATA.TEST_MAX_SCALE,
        boxes=None,
    )
    ### imgs: (list, 144, (256, 320, 3))
    ### boxes: None
    
    # Convert image to CHW keeping BGR order.
    imgs = [cv2_transform.HWC2CHW(img) for img in imgs]
    ### imgs: (list, 144, (3, 256, 320))
    
    # Image [0, 255] -> [0, 1].
    imgs = [img / 255.0 for img in imgs]
    
    imgs = [np.ascontiguousarray(
        # img.reshape((3, self._crop_size, self._crop_size))
        img.reshape((3, imgs[0].shape[1], imgs[0].shape[2]))
    ).astype(np.float32) for img in imgs]
    ### imgs: list, 144, (3, 256, 320)
    
    imgs = [
        cv2_transform.color_normalization(
            img,
            np.array(cfg.DATA.MEAN, dtype=np.float32),
            np.array(cfg.DATA.STD, dtype=np.float32),
        )
        for img in imgs
    ]
    ### imgs: list,144, (3, 256, 320), array([-2.,2.4444447, -2.], dtype=float32)    
    
    imgs = np.concatenate([np.expand_dims(img, axis=1) for img in imgs], axis=1)
    ### imgs: numpy.ndarray, (3, 144, 256, 320)
    
    # Convert image format from BGR to RGB.
    if not cfg.AVA.BGR:
        imgs = imgs[::-1, ...]
    ### (3, 144, 256, 320)
    
    imgs = np.ascontiguousarray(imgs)
    
    imgs = torch.from_numpy(imgs)
    ### torch.Size([3, 144, 256, 320])
    
    return imgs




def clip_constructor(tensor, rate_sample, num_frames):
    """
    Sample frames from a given tensor with all possible combinations.

    Args:
        tensor (torch.Tensor): Input tensor of shape 3xNxHxW.
        rate_sample (int): Sampling rate.
        num_frames (int): Number of frames in each sampled tensor.

    Returns:
        Tuple: List of sampled tensors, each with shape 3xnum_framesxHxW,
               and list of corresponding center frame indices.
    """
    sampled_tensors = []
    sampled_center_frames = []

    # Calculate the maximum possible starting index
    max_start_index = tensor.size(1) - num_frames * rate_sample + 1
    
    # Iterate over all possible starting indices
    for start_index in range(max_start_index):
        # Extract frames using the specified sampling rate and number of frames
        
        sampled_center_frames.append(int(start_index + rate_sample * num_frames/2))
        
        sampled_frames = tensor[:, start_index:start_index+rate_sample*num_frames:rate_sample, :, :]
        sampled_tensors.append(sampled_frames)
        
    
    return sampled_tensors, sampled_center_frames

    

def batch_different_videos(videos, size_divisible=0):
    """
    :param videos: a list of video tensors
    :param size_divisible: output_size(width and height) should be divisble by this param
    :return: batched videos as a single tensor
    """
    assert isinstance(videos, (tuple, list))
    max_size = tuple(max(s) for s in zip(*[clip.shape for clip in videos]))

    if size_divisible > 0:
        stride = size_divisible
        max_size = list(max_size)
        max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
        max_size[3] = int(math.ceil(max_size[3] / stride) * stride)
        max_size = tuple(max_size)

    batch_shape = (len(videos),) + max_size
    batched_clips = videos[0].new(*batch_shape).zero_()
    for clip, pad_clip in zip(videos, batched_clips):
        pad_clip[:clip.shape[0], :clip.shape[1], :clip.shape[2], :clip.shape[3]].copy_(clip)

    return batched_clips


def prepare_collated_batches(prep_clips, center_frames, cfg):
    """
    Prepare collated batches for further processing.

    Args:
        prep_clips (list): List of prepared clips.
        center_frames (list): List of center frames.
        cfg: Configuration object.

    Returns:
        list: List of collated batches.
    """
    
    pathways = cfg.MODEL.BACKBONE.PATHWAYS

    list_collated_batches = []

    for idx, clip in enumerate(prep_clips):
        clip_pathways = pack_pathway_output(cfg, clip, pathways=pathways)

        if pathways == 1:
            slow, fast = clip_pathways[0], None
        else:
            slow, fast = clip_pathways[0], clip_pathways[1]

        h, w = slow.shape[-2:]
        whwh = torch.tensor([w, h, w, h], dtype=torch.float32)
        whwh = whwh.unsqueeze(0)

        metadata = ([0, center_frames[idx]], )
        boxes = (None,)
        label_arrs = (None,)

        if pathways == 1:
            slow_clip = batch_different_videos([slow], cfg.DATALOADER.SIZE_DIVISIBILITY)
            fast_clip = None
        else:
            slow_clip = batch_different_videos([slow], cfg.DATALOADER.SIZE_DIVISIBILITY)
            fast_clip = batch_different_videos([fast], cfg.DATALOADER.SIZE_DIVISIBILITY)

        clip_idx = (idx, )
        list_collated_batches.append([slow_clip, fast_clip, whwh, boxes, label_arrs, metadata, clip_idx])

    return list_collated_batches


def prepare_collated_batches_v2(prep_clips, cfg):
    """
    Prepare collated batches for further processing.

    Args:
        prep_clips (list): List of prepared clips.
        center_frames (list): List of center frames.
        cfg: Configuration object.

    Returns:
        list: List of collated batches.
    """
    
    pathways = cfg.MODEL.BACKBONE.PATHWAYS

    list_collated_batches = []

    for idx, clip in enumerate(prep_clips):
        clip_pathways = pack_pathway_output(cfg, clip, pathways=pathways)

        if pathways == 1:
            slow, fast = clip_pathways[0], None
        else:
            slow, fast = clip_pathways[0], clip_pathways[1]

        h, w = slow.shape[-2:]
        whwh = torch.tensor([w, h, w, h], dtype=torch.float32)
        whwh = whwh.unsqueeze(0)

        metadata = ([0,0], )
        boxes = (None,)
        label_arrs = (None,)

        if pathways == 1:
            slow_clip = batch_different_videos([slow], cfg.DATALOADER.SIZE_DIVISIBILITY)
            fast_clip = None
        else:
            slow_clip = batch_different_videos([slow], cfg.DATALOADER.SIZE_DIVISIBILITY)
            fast_clip = batch_different_videos([fast], cfg.DATALOADER.SIZE_DIVISIBILITY)

        clip_idx = (idx, )
        list_collated_batches.append([slow_clip, fast_clip, whwh, boxes, label_arrs, metadata, clip_idx])

    return list_collated_batches
