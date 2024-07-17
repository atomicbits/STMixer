## STMixer neck

In this part, we explain neck  i.e., ` _construct_space(self, cfg)`.

Input:

* a list of length 4 (repeated), each element is a tensor of shape $(B, \text{embed_dim}, n_t,h_t,w_t)$ where $\text{embed_dim}=768$.

Output:

* a list of length 4:
  * first element a tensor of shape: $(B, \text{hidden_dim}, n_t, 4h_t, 4w_t)$ ($\text{hidden_dim} = 256$).
  * Second element a tensor of shape: $(B, \text{hidden_dim}, n_t, 2h_t, 2w_t)$ 
  * first element a tensor of shape: $(B, \text{hidden_dim}, n_t, h_t, w_t)$ 
  * first element a tensor of shape: $(B, \text{hidden_dim}, n_t, h_t/2, w_t/2)$ .

Some params:

1. `out_channel = cfg.MODEL.STM.HIDDEN_DIM = 256 `from config params.
2. `in_channels = [cfg.ViT.EMBED_DIM]*4 = [768, 768, 768, 768]`.

### 1.1 Constucting Laterl Convolutions for ViT Backbone

An empty `nn.ModuleList` named `self.lateral_convs` is initialized to hold the constructed layers.

```python
self.lateral_convs = nn.ModuleList()
```

this container holds submodules in a list ([ref](https://github.com/FrancescoSaverioZuppichini/Pytorch-how-and-when-to-use-Module-Sequential-ModuleList-and-ModuleDict)), where each submodule corresponds to one scale.

The scales are `[4, 2, 1, 0.5]` and the method iterates over them, create one submodule for each scale. These submodules are appended in `self.lateral_convs`.

### 1.2 Lateral convolution for each scale

##### Scale 4

For scale 4, starting with input shape of $(N, 768, D, H, W)$:

- A first `ConvTranspose3d` layers for upsampling with output shape of ($N, 384, D, 2H, 2W$)
- Intermediate normalization and activation layers (no change in output shape)
- Another `ConvTranspose3d` layer for upsampling with output shape of $(N, 192, D, 4H, 4W)$.

- Then we have two  `Conv3d` layers to match the output channel dimnesion to `out_channel=256`.
- Final output shape is $(N, 256, D, 4H, 4W)$.

##### Scale 2:

For scale 2 with same input shape of  $(N, 768, D, H, W)$:

- One `ConvTranspose3d` layer for upsampling.
- Final `Conv3d` layers to match the output channel dimension.
- Final output shape is $(N, 256, D, 2H, 2W)$.

##### Scale 1:

For scale 1

- Direct final `Conv3d` layers without additional upsampling.
- Final output shape is $(N, 256, D, H, W)$.

##### Scale 0.5:

- One `MaxPool3d` layer for downsampling with output shape of $(N, 768, D, H/2, W/2)$
- Final `Conv3d` layers to match the output channel dimension.
- Final output shape is $(N,256,D,H/2,W/2)$.

In forward, each feature map is processed through a corresponding convolutional layer (scale). The output is a list of 4 elements.

