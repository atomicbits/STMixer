
## SA-Inference of STMixer
### Step 1: Sequential patch inferencing
STMixer with slicing aided inference with sequential inferencing of patches in `my_demo/video_pb.py`. 
We have done this modification to `alphaction/modeling/stm_decoder` such that:
* In case of objectness (actor) score lower than threshold, no action is cosidered (without that, we will have a lot of ghsot actions!)
* In addition to box and action list,  objectness score will also be returned in inference and therefore, it can be used for post-processing (NMS post-processing for overlapping patches).

### RealTime processing of STMixer
We have created a duplicated version of `alphaction` in `rt_alphaction`. The idea is to be able to modify the modules in `alphacaction` in an isolated manner to avoid any conflic with previous step.
The code is [here](https://github.com/atomicbits/STMixer/blob/main/my_demo/realtime_patch_batch.ipynb).
