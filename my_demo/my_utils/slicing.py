import cv2

import numpy as np
from moviepy.editor import ImageSequenceClip
from typing import Dict, List, Optional, Sequence, Tuple, Union


def get_slice_bboxes(
    image_height: int,
    image_width: int,
    slice_height: Optional[int] = None,
    slice_width: Optional[int] = None,
    auto_slice_resolution: bool = True,
    overlap_ratio: Union[float, List[float]] = 0.1,
) -> List[List[int]]:
    """Slices `image_pil` in crops.
    Corner values of each slice will be generated using the `slice_height`,
    `slice_width`, `overlap_height_ratio` and `overlap_width_ratio` arguments.

    Args:
        image_height (int): Height of the original image.
        image_width (int): Width of the original image.
        slice_height (int, optional): Height of each slice. Default None.
        slice_width (int, optional): Width of each slice. Default None.
        overlap_ratio (float or List[float]): Fractional overlap in height and width of each slice. 
            (e.g. an overlap of 0.2 for a slice of size 100 yields an overlap of 20 pixels). Default 0.2.
        auto_slice_resolution (bool): if not set slice parameters such as slice_height and slice_width,
            it enables automatically calculate these params from image resolution and orientation.

    Returns:
        List[List[int]]: List of 4 corner coordinates for each N slices.
            [
                [slice_0_left, slice_0_top, slice_0_right, slice_0_bottom],
                ...
                [slice_N_left, slice_N_top, slice_N_right, slice_N_bottom]
            ]
    """
    slice_bboxes = []
    y_max = y_min = 0
    
    if not isinstance(overlap_ratio, list):
        overlap_ratio = [overlap_ratio, overlap_ratio]
    
    overlap_height_ratio, overlap_width_ratio = overlap_ratio

    if slice_height and slice_width:
        y_overlap = int(overlap_height_ratio * slice_height)
        x_overlap = int(overlap_width_ratio * slice_width)
    elif auto_slice_resolution:
        x_overlap, y_overlap, slice_width, slice_height = get_auto_slice_params(height=image_height, width=image_width)
    else:
        raise ValueError("Compute type is not auto and slice width and height are not provided.")

    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + slice_height
        while x_max < image_width:
            x_max = x_min + slice_width
            if y_max > image_height or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height, y_max)
                xmin = max(0, xmax - slice_width)
                ymin = max(0, ymax - slice_height)
                slice_bboxes.append([xmin, ymin, xmax, ymax])
            else:
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes


def generate_sliding_window_gif(frame, patch_coordinates, gif_filename):
    """
    Generate a GIF of sliding windows over a frame.

    Args:
        frame (numpy.ndarray): Input frame.
        patch_coordinates (list): list of patch box coordinates (xyxy).
        gif_filename (str): Filename for the generated GIF.
    """
    
    # Initialize list to store frames
    frames = []
    
    # Iterate over patch coordinates and generate frames with patches highlighted
    for patch_coord in patch_coordinates:
        x1, y1, x2, y2 = patch_coord
        
        # Copy frame to avoid modifying the original frame
        frame_copy = np.copy(frame)
        
        # Draw rectangle around patch
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 4)
        
        # Append frame to list
        frames.append(cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB))
    
    # Generate GIF using MoviePy
    clip = ImageSequenceClip(frames, fps=10)
    clip.write_gif(gif_filename, fps=10)

    
def calc_ratio_and_slice(orientation, slide=1, ratio=0.1):
    """
    According to image resolution calculation overlap params
    Args:
        orientation: image capture angle
        slide: sliding window
        ratio: buffer value

    Returns:
        overlap params
    """
    if orientation == "vertical":
        slice_row, slice_col, overlap_height_ratio, overlap_width_ratio = slide, slide * 2, ratio, ratio
    elif orientation == "horizontal":
        slice_row, slice_col, overlap_height_ratio, overlap_width_ratio = slide * 2, slide, ratio, ratio
    elif orientation == "square":
        slice_row, slice_col, overlap_height_ratio, overlap_width_ratio = slide, slide, ratio, ratio

    return slice_row, slice_col, overlap_height_ratio, overlap_width_ratio  # noqa


def calc_resolution_factor(resolution: int) -> int:
    """
    According to image resolution calculate power(2,n) and return the closest smaller `n`.
    Args:
        resolution: the width and height of the image multiplied. such as 1024x720 = 737280

    Returns:

    """
    expo = 0
    while np.power(2, expo) < resolution:
        expo += 1

    return expo - 1


def calc_aspect_ratio_orientation(width: int, height: int) -> str:
    """

    Args:
        width:
        height:

    Returns:
        image capture orientation
    """

    if width < height:
        return "vertical"
    elif width > height:
        return "horizontal"
    else:
        return "square"


def calc_slice_and_overlap_params(
    resolution: str, height: int, width: int, orientation: str
) -> Tuple[int, int, int, int]:
    """
    This function calculate according to image resolution slice and overlap params.
    Args:
        resolution: str
        height: int
        width: int
        orientation: str

    Returns:
        x_overlap, y_overlap, slice_width, slice_height
    """

    if resolution == "medium":
        split_row, split_col, overlap_height_ratio, overlap_width_ratio = calc_ratio_and_slice(
            orientation, slide=1, ratio=0.8
        )

    elif resolution == "high":
        split_row, split_col, overlap_height_ratio, overlap_width_ratio = calc_ratio_and_slice(
            orientation, slide=2, ratio=0.4
        )

    elif resolution == "ultra-high":
        split_row, split_col, overlap_height_ratio, overlap_width_ratio = calc_ratio_and_slice(
            orientation, slide=4, ratio=0.4
        )
    else:  # low condition
        split_col = 1
        split_row = 1
        overlap_width_ratio = 1
        overlap_height_ratio = 1

    slice_height = height // split_col
    slice_width = width // split_row

    x_overlap = int(slice_width * overlap_width_ratio)
    y_overlap = int(slice_height * overlap_height_ratio)

    return x_overlap, y_overlap, slice_width, slice_height


def get_resolution_selector(res: str, height: int, width: int) -> Tuple[int, int, int, int]:
    """

    Args:
        res: resolution of image such as low, medium
        height:
        width:

    Returns:
        trigger slicing params function and return overlap params
    """
    orientation = calc_aspect_ratio_orientation(width=width, height=height)
    x_overlap, y_overlap, slice_width, slice_height = calc_slice_and_overlap_params(
        resolution=res, height=height, width=width, orientation=orientation
    )

    return x_overlap, y_overlap, slice_width, slice_height


def get_auto_slice_params(height: int, width: int) -> Tuple[int, int, int, int]:
    """
    According to Image HxW calculate overlap sliding window and buffer params
    factor is the power value of 2 closest to the image resolution.
        factor <= 18: low resolution image such as 300x300, 640x640
        18 < factor <= 21: medium resolution image such as 1024x1024, 1336x960
        21 < factor <= 24: high resolution image such as 2048x2048, 2048x4096, 4096x4096
        factor > 24: ultra-high resolution image such as 6380x6380, 4096x8192
    Args:
        height:
        width:

    Returns:
        slicing overlap params x_overlap, y_overlap, slice_width, slice_height
    """
    resolution = height * width
    factor = calc_resolution_factor(resolution)
    if factor <= 18:
        return get_resolution_selector("low", height=height, width=width)
    elif 18 <= factor < 21:
        return get_resolution_selector("medium", height=height, width=width)
    elif 21 <= factor < 24:
        return get_resolution_selector("high", height=height, width=width)
    else:
        return get_resolution_selector("ultra-high", height=height, width=width)

    
if __name__ == '__main__':
    video_path = '../../input_dir/fight_prison_youtube.mp4'
    from video_processing import get_video_info
    video_info = get_video_info(video_path)
    frame = get_frame_from_video(video_path, frame_index=0)
    frame_shape = frame.shape[0:-1]
    patch_shape = [600, 800]
    overlap_ratio = 0.1
    patch_coordinates = get_slice_bboxes(frame_shape[0], 
                                         frame_shape[1], 
                                         patch_shape[0], 
                                         patch_shape[1], 
                                         False, 
                                         overlap_ratio)
    
    gif_filename = "sliding_windows.gif"  # Output GIF filename
    generate_sliding_window_gif(frame, patch_shape, overlap_ratio, gif_filename)
    