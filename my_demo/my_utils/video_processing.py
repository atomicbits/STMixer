import cv2

def get_video_info(video_path):
    """
    Get information about a video.

    Args:
        video_path (str): Path to the video file.

    Returns:
        dict: Dictionary containing video information.
    """
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        raise ValueError("Error opening video file: {}".format(video_path))

    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    video_capture.release()

    video_info = {
        "frame_count": frame_count,
        "frame_rate": frame_rate,
        "width": width,
        "height": height,
        "fps": fps
    }

    return video_info

def visualize_crop_on_frames(video_path, frame_index=0, length=1, crop=None):
    """
    Visualize a crops on one or multiple frames from a video.

    Args:
        video_path (str): Path to the video file.
        frame_index (int): Index of the frame to start visualization.
        length (int): Number of frames to visualize.
        crop (list): Bounding box coordinates in the form [x1, y1, x2, y2].

    Returns:
        list: List of visualized frames.
    """
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        raise ValueError("Error opening video file: {}".format(video_path))

    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    frames = []

    for _ in range(length):
        ret, frame = video_capture.read()

        if not ret:
            break

        if crop is not None:
            x1, y1, x2, y2 = crop
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)

        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    video_capture.release()

    return frames


def segment_crop_video(video_path, frame_index=0, length=-1, crop=None):
    """
    temporaly segment and spatially crop a video.

    Args:
        video_path (str): Path to the video file.
        frame_index (int): Index of the frame for segmentation.
        length (int): Number of frames for segmentation. -1 means all frames.
        crop (list): Bounding box coordinates in the form [x1, y1, x2, y2].

    Returns:
        list: List of visualized frames.
    """
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        raise ValueError("Error opening video file: {}".format(video_path))

    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    
    if crop is not None:
        crop_x1, crop_y1, crop_x2, crop_y2 = crop
    else:
        crop_x1, crop_y1, crop_x2, crop_y2 = 0, 0, -1, -1

    frames = []
    
    if length == -1:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        length = total_frames - frame_index
            

    for _ in range(length):
        ret, frame = video_capture.read()

        if not ret:
            break

        if crop is not None:
            frame = frame[crop_y1:crop_y2, crop_x1:crop_x2, :]

        frames.append(frame)

    video_capture.release()

    return frames


def get_frame_from_video(video_file, frame_index):
    """
    Get a frame with a specific index from a video.

    Args:
        video_file (str): Path to the video file.
        frame_index (int): Index of the frame to retrieve.

    Returns:
        numpy.ndarray: The frame with the specified index.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_file)

    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    # Read the frame
    ret, frame = cap.read()

    # Release the video capture object
    cap.release()

    # Check if the frame was successfully read
    if ret:
        return frame
    else:
        print(f"Error: Unable to read frame {frame_index} from the video.")
        return None

