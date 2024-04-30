import os
import cv2
import numpy as np
import torch
import json_tricks as json_tricks
from tqdm import tqdm

from alphaction.config import cfg
from alphaction.modeling.detector import build_detection_model
from alphaction.utils.checkpoint import ActionCheckpointer

from my_utils.gen_utils import create_experiment_folder, parse_arguments, create_exp_dict, create_config, parse_label_file
from my_utils.video_processing import get_video_info, segment_crop_video
from my_utils.slicing import get_slice_bboxes
from my_utils.ava_preprocessing import ava_preprocessing_cv2, clip_constructor, prepare_collated_batches
from my_utils.ava_postprocessing import clip_boxes_tensor, map_bbox_from_prep_to_crop, map_bbox_from_crop_to_orig
from my_utils.ava_postprocessing import concatenate_results
from my_utils.postprocess import apply_nms_to_dict
from my_utils.visualization import action_visualizer_frame_index



def main(args):

    # EXPERIENCE DICT
    exp_dict = create_exp_dict(args)

    # INPUT 
    video_name = os.path.basename(args.video_path).split('.')[0]
    video_info = get_video_info(args.video_path)
    exp_dict['video_params'].update(video_info)
    frame_height = video_info['height']
    frame_width = video_info['width']
    

    # OUTPUT FOLDER
    output_directory = f'../output_dir/{video_name}/{args.model_name}/patch_batch/' 
    output_directory = create_experiment_folder(output_directory, 'exp')

    # MODEL CONFIG
    cfg = create_config(args)
    #config_show(cfg)

    # PATCH
    patches_coordinates = get_slice_bboxes(frame_height, 
                                           frame_width, 
                                           args.slice_height, 
                                           args.slice_width, 
                                           False, 
                                           args.overlap_ratio)
    
    # BUILD MODEL
    model = build_detection_model(cfg)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        #num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    else:
        device = torch.device("cpu")
    model.to(device)

    checkpointer = ActionCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(cfg.MODEL.WEIGHT)
    
    model.eval()
    
    # LABEL FILE
    label_dict = parse_label_file(args.label_path)


    temp_results_dict = {}

    # INFERENCE ON VIDEO
    for patch_index, patch_coordinates in tqdm(enumerate(patches_coordinates), desc='Processing video patches'):
    
        # cropping and segmenting input video based on patch_coordinates and temporal window  
        cropped_video = segment_crop_video(args.video_path, 
                                           frame_index=args.starting_frame_index, 
                                           length=args.length_input, 
                                           crop=patch_coordinates)
    
        crop_height, crop_width = cropped_video[0].shape[:2]
    
        # applying ava preprocessing on each cropped_video
        prep_video = ava_preprocessing_cv2(cropped_video, cfg)
        prep_height, prep_width = prep_video.shape[-2:]
    
        # constructing clips 
        prep_clips, center_frames = clip_constructor(prep_video, 
                                                     rate_sample=cfg.DATA.SAMPLING_RATE, 
                                                     num_frames=cfg.DATA.NUM_FRAMES)
        # constructing input batches
        list_collated_batches = prepare_collated_batches(prep_clips, center_frames, cfg)
    

        for batch, center_frame_index in tqdm(zip(list_collated_batches, center_frames), 
                                              desc='Processing frames', 
                                              total=len(list_collated_batches),
                                              leave=False):
        
            # updating frame index based on starting_frame_index
            cur_frame_index = center_frame_index + args.starting_frame_index
        
            # adding current frame index to result dict
            if cur_frame_index not in temp_results_dict:
                temp_results_dict[cur_frame_index] = []
        
            # passing batch to model
            with torch.no_grad():
                slow_video, fast_video, whwh, boxes, labels, metadata, idx = batch
                clips_height, clips_width = slow_video.shape[-2:]
                slow_video = slow_video.to(device)
                if fast_video is not None:
                    fast_video = fast_video.to(device)
                whwh = whwh.to(device)
    
                # INFERENCE
                action_score_list, box_list, objectness_score_list = model(slow_video, fast_video, whwh, boxes, labels)
                
        
            # Check if any detection happened
            if len(box_list) != 0:
           
                output_bbox = box_list[0]
                output_action = action_score_list[0]
                output_objectness = objectness_score_list[0]
        
                if output_bbox.shape[0] != 0:
            
                    # denormalizing bboxes w.r.t. clips shape
                    output_bbox_inp = output_bbox.clone()

                    output_bbox_inp[:, 0] = output_bbox[:, 0] * clips_width
                    output_bbox_inp[:, 1] = output_bbox[:, 1] * clips_height
                    output_bbox_inp[:, 2] = output_bbox[:, 2] * clips_width
                    output_bbox_inp[:, 3] = output_bbox[:, 3] * clips_height
    
                    # clipping bbonx coordinates with prep shape because clip shape is right/bottum padded version of prep shape.
                    output_bbox_prep = clip_boxes_tensor(output_bbox_inp, 
                                                         height=whwh[0,1], 
                                                         width = whwh[0,0])
    
                    # scaling bboxes from prep shape to crop shape
                    output_bbox_crop = map_bbox_from_prep_to_crop(output_bbox_prep, 
                                                                  (crop_height, crop_width), 
                                                                  (prep_height, prep_width))
    
                    # mapping from crop to original frame
                    output_bbox_frame = map_bbox_from_crop_to_orig(output_bbox_crop , patch_coordinates[:2])
                
                    # getting top_k action: scores and indices
                    top_values, top_indices = torch.topk(output_action, k=args.top_k, dim=1)
    
                    output_objectness_np = np.reshape(output_objectness.cpu().numpy(), (-1, 1))
                    output_bbox_frame_np = output_bbox_frame.cpu().numpy()
                
                    # shifting to ava dataset labeling
                    top_indices_np = top_indices.cpu().numpy() + 1
                    top_values_np = top_values.cpu().numpy()
                
                    if args.add_patch_index:
                        # adding patch index to result.
                        patch_index_np = np.full((output_objectness_np.shape[0], 1), patch_index)
                        agg_result = np.concatenate((output_objectness_np, 
                                                     output_bbox_frame_np, 
                                                    top_indices_np, 
                                                    top_values_np,
                                                    patch_index_np), axis=1)
                    else:
                        agg_result = np.concatenate((output_objectness_np, 
                                                     output_bbox_frame_np, 
                                                    top_indices_np, 
                                                    top_values_np), axis=1)
                
                
                    temp_results_dict[cur_frame_index].append(agg_result)
        
    
    
    # AGGREGATION (convert values to numpy array)
    all_results_dict = concatenate_results(temp_results_dict, top_k=args.top_k, patch_index=True)

    # POST PROCESSING
    post_all_results_dict = apply_nms_to_dict(all_results_dict)

    # VISUALIZATION
    _ = action_visualizer_frame_index(post_all_results_dict, 
                                                 args.video_path, 
                                                 label_dict, 
                                                 output_directory,
                                                 top_k=args.top_k,
                                                 list_actions = args.actions,
                                                 other_actions_color = (0, 255, 0), # green
                                                 all_actions=True,
                                                 long_text_show=False,
                                                 mode='movie',
                                                 fps = int(video_info['fps']))

    # SAVE RAW RESULTS
    exp_json_path = os.path.join(output_directory, 'exp.json')

    # Save the dictionary as a JSON file
    with open(exp_json_path, 'w') as f:
        json_tricks.dumps(exp_dict, f)
    
    result_json_path = os.path.join(output_directory, 'result.json')

    # Save raw results as a JSON file
    with open(result_json_path, 'w') as f:
        json_tricks.dumps(temp_results_dict, f, indent=4)
    

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
