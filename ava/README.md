This `ava` folder is the `ava` in `/data/ava`. 

You have to add the following folders into it: 

 - annotations
 - frame_lists
 - frames

The scripts in this folder are not necessary for training, only for the preparation of the dataset. 

For training, read the top-level README.md file. You have to start the docker container and then run 
the training command. Check how to detach and re-attach from that container in the docker specs.

From the top-level STMixer folder (in docker mounted under `/work`):  

```shell
python -m torch.distributed.launch --nproc_per_node=8 train_net.py --config-file "config_files/my_VMAEv2-ViTB-16x4.yaml" --transfer --no-head --use-tfboard
```

