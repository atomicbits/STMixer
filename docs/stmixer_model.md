

# STMixer model

Model in trainins script is created [in](https://github.com/MCG-NJU/STMixer/blob/2c3cd1de2623fc47ad93cfda3a8fcb9713736a55/train_net.py#L243C5-L245C32) 

```python
model = train(cfg, 
              args.local_rank, # local rank of each process
              args.distributed, # True
              tblogger, # True
              args.transfer_weight, # True
              args.adjust_lr, # False
              args.skip_val, # False
              args.no_head # True
             )
```

and then [in](https://github.com/MCG-NJU/STMixer/blob/2c3cd1de2623fc47ad93cfda3a8fcb9713736a55/train_net.py#L33C5-L33C39)

```
model = build_detection_model(cfg)
```

wheere `buid_detection_model` is defined in [here](https://github.com/MCG-NJU/STMixer/blob/main/alphaction/modeling/detector/stm_detector.py#L124).

1. Construct backbone using `build_backbone(cfg)`.
2. Construct neck using `self._construct_space(cfg)`
3. Construct STM head using `build_stm_decoder(cfg)`.

In forward (with `self.backbone.num_pathways = 1`):

1. `slow_video` as a tensor of shape $(B, 3, T, H_{\text{prep}}, W_{\text{prep}})$ is put inside a list of length 1 and is fed into `self.backbone`:

   ```
   features = self.backbone([slow_video])
   ```

2. `features` are passed to the neck where `features` are 

   ```
   mapped_features = self.space_forward(features)
   ```

   and `mapped_features` is a list of ...

3. the output of model is obtained by passing `mapped_features`, `whwh`, `boxes` and `labels` to `stm_head`:

   * `whwh`: torch tensor of shape $(B,4)$.
   * `boxes`: list of np.array. Each item corresponds to one sample and is a np.array of shape $(M_i,4)$ containing `xyxyx` of bboox coordinates of actions (clipped to $(H_{\text{crop}},W_{\text{crop}})$).
   * `labels`: list of np.array. Each item corresponds to one sample and is a np.array of shape $(M_i,80)$ containing one-hot encoded action lists of sample $i$. 

