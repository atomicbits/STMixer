import cv2
import numpy as np
from tqdm import tqdm
import os
from moviepy.editor import ImageSequenceClip
from my_utils.video_processing import get_frame_from_video


def setting_interesting_actions(list_actions):
    """
    Filter dictionaries based on a list of interesting actions.

    This function filters dictionaries containing information about interesting actions,
    such as labels, colors, and thresholds, based on a provided list of action labels.

    Parameters:
        list_actions (list): A list of action labels to filter the dictionaries.

    Returns:
        tuple: A tuple containing filtered dictionaries for interesting actions:
            - interesting_actions_indices (list): Indices of interesting actions.
            - interesting_actions_labels (dict): Labels of interesting actions.
            - action_colors (dict): Colors corresponding to interesting actions.
            - action_threshold (dict): Thresholds for interesting actions.

    """

    interesting_actions_labels = {5: 'fall', 64: 'fight', 71: 'kick', 76: 'push'}
    action_colors = {5: (0, 0, 210),  # Blue
                     64: (255, 0, 0),  # Red
                     71: (255, 165, 0),  # Purple
                     75: (128, 0, 128)}  # Orange
    action_thrshold = {5: 0.3, 64: 0.2, 71: 0.3, 76: 0.3}

    interesting_actions_indices = [key for key in interesting_actions_labels if interesting_actions_labels[key] in list_actions]
    interesting_actions_labels = {key: value for key, value in interesting_actions_labels.items() if value in list_actions}
    action_colors = {key: value for key, value in action_colors.items() if key in interesting_actions_indices}
    action_thrshold = {key: value for key, value in action_thrshold.items() if key in interesting_actions_indices}

    return interesting_actions_indices, interesting_actions_labels, action_colors, action_thrshold




def action_visualizer_frame_index(all_results_dict, 
                                  video_path, 
                                  label_dict, 
                                  output_directory,
                                  top_k=5,
                                  list_actions=['fall', 'fight', 'kick', 'push'],
                                  other_actions_color = (0, 255, 0), # green
                                  all_actions=False, 
                                  long_text_show=False,
                                  add_patch_index=False,
                                  mode='movie',
                                  fps=20):
    

    interesting_actions_indices, interesting_actions_labels, action_colors, action_thrshold = setting_interesting_actions(list_actions)

    if mode in ['gif', 'movie']:
        vis_frames_list = []
        if mode == 'gif':
            output_path = os.path.join(output_directory, 'gif.gif')
        else:
            output_path = os.path.join(output_directory, 'output_video.mp4')
    else:
        output_directory = os.path.join(output_directory, "frames")
    
    
    with tqdm(total=len(all_results_dict)) as pbar:
        
        for cur_frame, results_frame in all_results_dict.items():
            
            # checking results_frame type and converting it into np
            if isinstance(results_frame, list):
                results_frame_np = np.concatenate(results_frame, axis=0)
            elif isinstance(results_frame, np.ndarray):
                results_frame_np = results_frame
            else:
                raise ValueError("Input must be a list or a numpy array")
                
            # getting the frame
            frame = get_frame_from_video(video_path, int(cur_frame))
            vis_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
            
            # no detection on frame, we only visualize frame for video visualization
            if results_frame_np.shape[0] == 0:
                if mode in ['pic', 'gif']:
                    continue
                else:
                    vis_frames_list.append(vis_frame)
                    continue
            
            # extracing object_score, bboxes, top action indices and score, and patch_index (if exists)
            obj_scores_frame = results_frame_np[:, :1]
            bboxes_frame = results_frame_np[:, 1:5]
            top_indices_frame = results_frame_np[:, 5:5+top_k]
            top_values_frame = results_frame_np[:, 5+top_k:5+2*top_k]
            
            if results_frame_np.shape[-1] != 5+2*top_k:
                patch_indices_frame = results_frame_np[:, -1].astype(int)
            else:
                add_patch_index = False
                
            # id of interesting actor starting from 0
            id_actor = 0
            
            # if frame contains any interesting action
            interesting_frame = False
            
            # looping over each actor
            for object_score, bbox, top_action_indices, top_action_scores, patch_index in zip(obj_scores_frame, bboxes_frame,
                                                                                              top_indices_frame, top_values_frame, patch_indices_frame):
                x1, y1, x2, y2 = bbox.astype(int)
                
                # if actor is interesting
                interesting_actor = False
                for ind_act, act in enumerate(top_action_indices):
                    if act in interesting_actions_indices and top_action_scores[ind_act] > action_thrshold[int(act)]:
                        interesting_actor = True
                        interesting_frame = True
                        bbox_action_color = action_colors[int(act)]
                        main_interesting_act = int(act) # the main interesting action of actor
                        main_interesting_score = top_action_scores[ind_act]
                        break
                
                # visualization for interesting actor
                if interesting_actor:
                     
                    # plot the bbox of interesting actor using specific color for action
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), bbox_action_color, 3)
                    
                    # text on top of bbox of interesting actor
                    if add_patch_index:
                        id_text = '{}_{}_{}_{}'.format(id_actor, 
                                                       interesting_actions_labels[main_interesting_act], 
                                                       np.round(main_interesting_score, 2), 
                                                       patch_index)
                    else:
                        id_text = '{}_{}_{}'.format(id_actor, 
                                                    interesting_actions_labels[main_interesting_act], 
                                                    np.round(main_interesting_score, 2))
                    id_actor += 1
                    
                    # adding bbox text
                    cv2.putText(vis_frame, id_text, (x1+10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, bbox_action_color, 2)
            
                     # add text containing all actions of interesting actor op top of frame
                    if long_text_show:
                        long_text = '{}-{}:'.format(id_actor, np.round(object_score, 2)) # add ID of actor for frame visualization
                        # add act and their scores on long text
                        for act, score in zip(top_action_indices, top_action_scores):
                            long_text += '{}_{}-'.format(label_dict[act].replace('(', '').replace(')', '').split('/')[0], 
                                                 (np.round(score, 2)))
                        # add long text on frame
                        cv2.putText(vis_frame, long_text[:-1], (20, 100 + 20 * id_actor), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                bbox_action_color, 2)
                
                # visualization for other actors
                else:
                    if all_actions:
                        # plot bbox of other actors   
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), other_actions_color, 2)
                        # add patch index on top of bbox
                        if add_patch_index:
                            id_text = '{}'.format(patch_index)
                            cv2.putText(vis_frame, id_text, (x1+10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                        other_actions_color, 2)
       
            # add frame index on pic/gif mode if frame contains interesting action
            if interesting_frame and mode != 'video':
                cv2.putText(vis_frame, str(cur_frame), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
        
            if mode in ['gif', 'movie']:
                vis_frames_list.append(vis_frame)
            else:
                if True: # we can add other conditions on saving vis frames in `pic` mode
                    frame_path = os.path.join(output_directory, f"frame_{cur_frame}.jpg")
                    sucess = cv2.imwrite(frame_path, cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB))
        
        
    # Update tqdm progress bar
    pbar.update(1)
        
    if mode in ['gif', 'movie']:
        create_video_from_frames(vis_frames_list, output_path, fps=fps)
        return output_path
    else:
        return output_directory


def create_video_from_frames(frames, output_path, fps=20):
    """
    Create a video from a list of frames using moviepy.

    Parameters:
        frames (list): List of frames (each frame is a numpy array or an image file path).
        output_path (str): Path to save the output video file (including file extension, e.g., 'output.mp4').
        fps (int, optional): Frames per second for the output video (default is 30).

    Returns:
        None
    """
    # Create video clip from frames
    clip = ImageSequenceClip(frames, fps=fps)
    # Write video file
    clip.write_videofile(output_path)