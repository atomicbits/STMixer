import cv2
import numpy as np
from tqdm import tqdm
import os


from my_utils.video_processing import get_frame_from_video


def action_visualizer_frame_index(all_results_dict, 
                                  video_path, 
                                  label_dict, 
                                  output_directory,
                                  top_k=5,
                                  interesting_actions_indices = [5, 64, 71, 75],
                                  interesting_actions_labels = {5:'fall', 64:'fight', 71:'kick', 76:'push'},
                                  action_colors = {5: (0, 0, 255), # Blue
                                                   64 : (255, 0, 0), # Red
                                                   71: (255, 165, 0), # Purple
                                                   75: (128, 0, 128)}, # Orange
                                  other_actions_color = (0, 255, 0), # green
                                  all_actions=True, 
                                  add_patch_index=True,
                                  vis_patch_lines=False,
                                  mode='pic'):
    
    if mode in ['gif', 'movie']:
        vis_frames_list = []
    
    
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
            
            # no detection on frame
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
                patch_index = results_frame_np[:, -1]
            else:
                patch_index = None
                add_patch_index = False
                
            
            # id of interesting actor starting from 0
            id_actor = 0
            
            # if frame contains any interesting action
            interesting_frame = False
            
            # looping over each actor
            for object_score, bbox, top_action_indices, top_action_scores in zip(obj_scores_frame, bboxes_frame,
                                                                                 top_indices_frame, top_values_frame):
                x1, y1, x2, y2 = bbox.astype(int)
                
                # if actor is interesting
                interesting_actor = False
                for ind_act, act in enumerate(top_action_indices):
                    if act in interesting_actions_indices:
                        interesting_actor = True
                        interesting_frame = True
                        bbox_action_color = action_colors[int(act)]
                        main_interesting_act = act # the main interesting action of actor
                        break
                
                # visualization for interesting actor
                if interesting_actor:
                    long_text = '{}-{}:'.format(id_actor, np.round(object_score, 2)) # add ID of actor for frame visualization
                    # add act and their scores on long text
                    for act, score in zip(top_action_indices, top_action_scores):
                        long_text += '{}_{}-'.format(label_dict[act].replace('(', '').replace(')', '').split('/')[0], 
                                                 (np.round(score, 2)))
                    
                    # plot the bbox of interesting actor 
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), bbox_action_color, 2)
                    
                    # text on top of bbox of interesting actor
                    if add_patch_index:
                        id_text = '{}_{}'.format(id_actor, patch_index)
                    else:
                        id_text = '{}'.format(id_actor)
                    id_actor += 1
                    
                    
                    cv2.putText(vis_frame, id_text, (x1+10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, bbox_action_color, 2)
            
                     # add text containing all actions of interesting actor
                    cv2.putText(vis_frame, long_text[:-1], (20, 100 + 20 * id_actor), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                bbox_action_color, 2)
                
                # visualization for other actors
                else:
                    if all_actions:
                        # plot bbox of other actors   
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), other_actions_color, 2)
                        # add text
                        if add_patch_index:
                            id_text = '{}'.format(patch_index)
                            cv2.putText(vis_frame, id_text, (x1+10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                        other_actions_color, 2)
       
            if interesting_frame:
                cv2.putText(vis_frame, str(cur_frame), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)


            if vis_patch_lines:
                cv2.rectangle(vis_frame, (crop_box[0], crop_box[1]), (crop_box[2], crop_box[3]), (255, 255, 0), 1)
               
        
            if mode in ['gif', 'movie']:
                vis_frames_list.append(vis_frame)
            else:
                if interesting_frame:
                    frame_path = os.path.join(output_directory_frames, f"frame_{cur_frame}.jpg")
                    sucess = cv2.imwrite(frame_path, cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB))
        
        
        # Update tqdm progress bar
        pbar.update(1)
        
    if mode in ['gif', 'movie']:
        return vis_frames_list
    else:
        return sucess
