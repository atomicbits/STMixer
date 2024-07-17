# STMixer Backbone

in [here](https://github.com/MCG-NJU/STMixer/blob/2c3cd1de2623fc47ad93cfda3a8fcb9713736a55/alphaction/modeling/detector/stm_detector.py#L35)

```python
self.backbone = build_backbone(cfg)
```

then from [build_backbone.py](https://github.com/MCG-NJU/STMixer/blob/main/alphaction/modeling/backbone/backbone.py), for ViT-B, we have this [class](https://github.com/MCG-NJU/STMixer/blob/2c3cd1de2623fc47ad93cfda3a8fcb9713736a55/alphaction/modeling/backbone/video_model_builder.py#L538).

For rest of model, its input output shape is important and they are:

* list of length 1 containing `slow_video` as a tensor of shape $(B,3,T,H_{\text{prep}},W_{\text{prep}})$.
* a list of length 4 containing same tensor of shape $(B, \text{embed_dim}, n_t, h_t,w_t)$ where:
  -  `embed_dim`: dimension of each token (768).
  - $n_t = \lfloor T/{\text{tubelet_size}}\rfloor $.
  - $n_h=\lfloor H_{\text{prep}}/{\text{patch_size[0]}}\rfloor  $ (Default:  `patch_size[0]=16`).
  - $n_w=\lfloor W_{\text{prep}}/{\text{patch_size[1]}}\rfloor  $ (Default:  `patch_size[0]=16`).

### 1.2 ViT class

#### 1.2.1 `__init__`

set number of pathways:

```python
self.num_pathways = 1
```

and call `self._construct_netwrok(cfg)`

#### 1.2.2 `_consruct_network(self, cfg)`

The goal is to set model params and architecture.

##### cfg values

Cfg values are:

```python
tubelet_size = cfg.ViT.TUBELET_SIZE # 2 from default config
patch_size = cfg.ViT.PATCH_SIZE # 16 from default config
in_chans = cfg.ViT.IN_CHANS # 3 from default config
embed_dim = cfg.ViT.EMBED_DIM # 768 from VMAE config
pretrain_img_size = cfg.ViT.PRETRAIN_IMG_SIZE # 224 from default config
use_learnable_pos_emb = cfg.ViT.USE_LEARNABLE_POS_EMB # False from default
drop_rate = cfg.ViT.DROP_RATE # 0 from default config
attn_drop_rate = cfg.ViT.ATTN_DROP_RATE # 0 from degault config
drop_path_rate = cfg.ViT.DROP_PATH_RATE # 0.2 from VMAE
depth = cfg.ViT.DEPTH # 12 from VMAE config
num_heads = cfg.ViT.NUM_HEADS # 12 from default
mlp_ratio = cfg.ViT.MLP_RATIO # 4 from default
qkv_bias = cfg.ViT.QKV_BIAS # True from default
qk_scale = cfg.ViT.QK_SCALE # None from default
init_values = cfg.ViT.INIT_VALUES # 0 from default
use_checkpoint = cfg.ViT.USE_CHECKPOINT # True from VMAE config
```
##### norm layer

Norm Layer is definaed as `nn.LayerNorm(eps=1e-6)`.

##### `depth`, `tubelet` and `checkpoint`

we set some properties as

```python
self.depth = depth  # 12
self.tubelet_size = tubelet_size # 2

self.use_checkpoint = use_checkpoint # True
```

Checkpointing is a technique used during training to trade off between memory usage and computation time. It allows you to trade off the amount of memory needed to store intermediate activations during backpropagation for recomputation of those activations during the backward pass. 

##### patch embed

we create a patch embed based on config values

```python
self.patch_embed = PatchEmbed(
  img_size=pretrain_img_size, # cfg.ViT.PRETRAIN_IMG_SIZE = 224
  patch_size=patch_size, # cfg.ViT.PATCH_SIZE = 16 
  in_chans=in_chans, # cfg.ViT.IN_CHANS = 3
  embed_dim=embed_dim, # cfg.ViT.EMBED_DIM = 768
  tubelet_size=self.tubelet_size # config = 2
)
```

this is in fact a conv3D that maps the paches of pixels into vectors. 

##### number of patches `num_patches ` and grid size  `gird_size`

then we have number of patches 

```python
num_patches = self.patch_embed.num_patches
```

which will be equal to

```python
(cfg.ViT.PRETRAIN_IMG_SIZE[0]/ cfg.ViT.PATCH_SIZE[0]) * 
(cfg.ViT.PRETRAIN_IMG_SIZE[1]/ cfg.ViT.PATCH_SIZE[1]) * 
(cfg.DATA.NUM_FRAMES / cfg.ViT.TUBELET_SIZE) # (224/16)*(224/16)*(16/2) # 14 * 14 * 8
```

**NOTE 1** : number of patches is independent of shape of input $\mathbf{x}$ passed to the forward method and is a function of `cfg.ViT.PRETRAIN_IMG_SIZE`,i.e., the size of pretrain image of ViT (and `cfg.ViT.PATCH_SIZE` and `cfg.DATA.NUM_FRAMES` and `cfg.ViT.TUBELET_SIZE`), and **NOT of the shape of input**:
$$
\lfloor T/{\text{tubelet_size}}\rfloor \times \lfloor H_{\text{pretrain}}/{\text{patch_size[0]}}\rfloor   \times \lfloor W_{\text{pretrain}}/{\text{patch_size[1]}}\rfloor  \\
= 8 \times 14 \times 14 \text{      (default values)}
$$
**NOTE2**: $H_{\text{pretrain}}$ is the image size in pretraining the backone (224 defualt) while  $H_{\text{prep}}$ is the size used in training of STMixer. This mismatch is making an issue in positional embedding that will be solver later. 

**NOTE3**: we use this intialized number of patches for defining **positional embedding** dimension. Therefore, 

`grid_size` will be 

```python
self.grid_size = [pretrain_img_size // patch_size, pretrain_img_size // patch_size]  
# [cfg.ViT.PRETRAIN_IMG_SIZE[0]/ cfg.ViT.PATCH_SIZE[0]),
#  cfg.ViT.PRETRAIN_IMG_SIZE[1]/ cfg.ViT.PATCH_SIZE[1]] = [14,14]
```

##### positional embedding

if `use_learnable_pos_emb = cfg.ViT.USE_LEARNABLE_POS_EMB`, then we use learnable poistional embedding as:

```python
 self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
```

which is a tensor parameter fo shape (1, `num_patches`, `embed_dim`), i.e., for default values $(1, 1568, 768)$.

However, `cfg.ViT.USE_LEARNABLE_POS_EMB` is `False` by defualt, so:

```python
self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)
```

generates a sinusoidal position encoding tensor of shape (1, `num_patches`, `embed_dim`). This means that in ViT-B STMixer, we are using fixed positional embedding and it is not learnt during training. 

##### dropout 

```
self.pos_drop = nn.Dropout(p=drop_rate) # p=0
```

and we set the drop path rate of each block based on this linespace which generates `depth` number of `dpr`s between 0 and `cfg.ViT.DROP_PATH_RATE=0.2`.

```
dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
```

##### block

initializes a list of `Block` modules

```python
self.blocks = nn.ModuleList([
  Block(
    dim=embed_dim, # cfg.ViT.EMBED_DIM:768 from VMAE config
    num_heads=num_heads, # cfg.ViT.NUM_HEADS: 12 from default
    mlp_ratio=mlp_ratio, # cfg.ViT.MLP_RATIO: 4 from default
    qkv_bias=qkv_bias, # cfg.ViT.QKV_BIAS: True from default
    qk_scale=qk_scale, # cfg.ViT.QK_SCALE: None from default
    drop=drop_rate, # cfg.ViT.DROP_RATE: 0 from default config
    attn_drop=attn_drop_rate, # cfg.ViT.ATTN_DROP_RATE: 0 from degault config
    drop_path=dpr[i], # ith value between 0 and cfg.ViT.DROP_PATH_RATE: 0.2
    norm_layer=norm_layer, # partial(nn.LayerNorm, eps=1e-6)
    init_values=init_values, # cfg.ViT.INIT_VALUES: 0 from default
    use_checkpoint=use_checkpoint) # cfg.ViT.USE_CHECKPOINT: True from VMAE config
  for i in range(depth)]) # depth=12
```

#### 1.2.1 `forward(self, x_list)`

Takes as input a list but only the first item is used:

```
x = x_list[0]
```

Consider output of preprocessing as $\mathbf{x} \in \mathbb{R} ^{B \times  C \times  T \times  H_{\text{prep}} \times W_{\text{prep}}}$ where $B$ is the batch size, $T$ is number of frames in a video clip (16) and  $H_{\text{prep}}$ and $W_{\text{prep}}$ are height and width of pre-processed frames and the output

```
x = self.patch_embed(x)
```

with the output shape of$(B, \text{embed_dim}, n_{t}, h_{t}, w_{t})$ where:

-  `embed_dim`: dimension of each token (768).
- $n_t = \lfloor T/{\text{tubelet_size}}\rfloor $.
- $n_h=\lfloor H_{\text{prep}}/{\text{patch_size[0]}}\rfloor  $ (Default:  `patch_size[0]=16`).
- $n_w=\lfloor W_{\text{prep}}/{\text{patch_size[1]}}\rfloor  $ (Default:  `patch_size[0]=16`).

we then get the shape correspoinding to number of patches (tubelets) as

```python
ws = x.shape[2,:]
```

so
$$
\text{ws} = [n_t, h_t,w_t]
$$
there is a codingtion to be checked
$$
n_t \text{  should be even}
$$
then we reshaped embedded input as

```python
x = x.flatten(2).transpose(1, 2).contiguous()  # b,thw,768
B, _, C = x.shape
```

Therefore $\mathbf{x}$ becaome a tensor of dim 3 in form of $B \times N_{\text{patch}} \times \text{embed_dim}$, i.e, $(B,  n_t h_t w_t, \text{embed_dim})$. 

We check then the number of patches in $\mathbf{x}$ which is $n_t h_t w_t$ (a function of $H_{\text{prep}}, $$W_{\text{prep}}$) with number of patches defined in `init` part of `PatchEmbed` module. In other words, it checks if 
$$
\lfloor H_{\text{prep}}/{\text{patch_size[0]}}\rfloor \times  \lfloor W_{\text{prep}}/{\text{patch_size[0]}}\rfloor == \lfloor H_{\text{pretrain}}/{\text{patch_size[0]}}\rfloor \times  \lfloor W_{\text{pretrain}}/{\text{patch_size[0]}}\rfloor
$$
if there is a mismatch, i.e., mismatch between size of img used in pretraining of ViT and pre-processed image in STMixer training,  we interpolates positinal embeddings `pos_embed` using bicubic interpolation to resize it from the original grid size $[H_{\text{pretrain}}/{\text{patch_size[0]}}\rfloor, W_{\text{pretrain}}/{\text{patch_size[0]}}\rfloor]$ to the new grid size $[H_{\text{prep}}/{\text{patch_size[0]}}\rfloor, W_{\text{prep}}/{\text{patch_size[0]}}\rfloor]$.

```
pos_embed = interpolate_pos_embed_online(
                pos_embed, self.grid_size, [ws[1], ws[2]], 0).reshape(1, -1, C)
```

Now `pos_embed` is of shape $(1, n_t h_t w_t, \text{embed_dim})$, same as $\mathbf{x}$, ignoring $B$ , i.e., batch dimension. 

Then we add positonal tokens and pass the result through `nn.Dropout(p=drop_rate)`. Note that `drop_rate=0` in default.

```
x = x + pos_embed.type_as(x).to(x.device).clone().detach()
x = self.pos_drop(x)
```

Then we pass $\mathbf{x}$ through `depth` times of the block which produce a tensor of the same shape.

```python
x = self.pos_drop(x)
for i in range(self.depth):
  blk = self.blocks[i]
  if self.use_checkpoint:
    x = checkpoint.checkpoint(blk, x)
  else:
    x = blk(x)
x = self.norm(x)
# b,thw,768->b,768,t,h,w
x = x.reshape(B, ws[0], ws[1], ws[2], -1).permute(0, 4, 1, 2, 3).contiguous()
features = [x, x, x, x]
return features
```

the output is a list of repeated `x` where `x` is reshaped to $(B, \text{embed_dim}, n_t,h_,w_t)$.

### 1.3. ViT utils

#### 1.3.1. `PatchEmbed`

processing video frames, converting them into a sequence of embeddings that can be fed into a transformer.

Input: $(B, 3, T_{\text{clip}}, H_{\text{prep}}, W_{\text{prep}})$

Output: $(B, \text{embed_dim}, n_{t}, h_{t}, w_{t})$ where:

*  `embed_dim`: dimension of each token (768).
* $n_t = \lfloor T/{\text{tubelet_size}}\rfloor $.
* $n_h=\lfloor H_{\text{prep}}/{\text{patch_size[0]}}\rfloor  $ (Default:  `patch_size[0]=16`).
* $n_w=\lfloor W_{\text{prep}}/{\text{patch_size[1]}}\rfloor  $ (Default:  `patch_size[0]=16`).

##### Params

- `img_size`: The size of the input image (default is 224).
- `patch_size`: The size of each patch (default is 16), corresponding to $h$ and $w$ in ViViT paper.
- `in_chans`: The number of input channels (default is 3, for RGB images).
- `embed_dim`: The dimension of the embedding space (default is 768).
- `num_frames`: The number of frames in the input video (default is 16).
- `tubelet_size`: The size of the temporal tubelet (default is 2), corresponding to $t$ in ViViT paper.

in forward:

```python
self.proj = nn.Conv3d(in_channels=in_chans, 
                      out_channels=embed_dim,
                      kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
                      stride=(self.tubelet_size, patch_size[0], patch_size[1]))
```

For input in shape of $(B, C, T, H_{\text{prep}}, W_{\text{prep}})$ where

- `B` is the batch size
- `C` is the number of channels (3 for RGB images)
- `T` is the number of frames 
- `H` is the height of the image ($H_{\text{prep}}$)
- `W` is the width of the image ($W_{\text{prep}}$)

Regarding the discussion above, the tublet dimension  is given by `tublet_size * patch_size[0] * patch_size[1]`. 

The output of forward method of `PatchEmbed` will be of shape $(B, C^{\prime}, T^{\prime}, H^{\prime}, W^{\prime})$ where

* $C^\prime$: embed dimnesion `embed_dim` (set to 768)

* $T^\prime=\lfloor T/{\text{tubelet_size}}\rfloor  = 8$ (fot $T=16$ and `tubelet_size=2`).
* $H^\prime=\lfloor H_{\text{prep}}/{\text{patch_size[0]}}\rfloor  $ (Default:  `patch_size[0]=16`).
* $W^\prime=\lfloor W_{\text{prep}}/{\text{patch_size[1]}}\rfloor  $ (Default:  `patch_size[0]=16`).

#### 1.3.2 DropPath

[/backbone/vit](https://github.com/MCG-NJU/STMixer/blob/2c3cd1de2623fc47ad93cfda3a8fcb9713736a55/alphaction/modeling/backbone/vit_utils.py#L28)

```python
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

```

The `drop_path` function implements a regularization technique called Stochastic Depth (or Drop Path), which is used to randomly drop entire paths (layers or blocks) in a neural network during training. This technique is particularly useful in deep neural networks, such as residual networks (ResNets), to improve generalization and reduce overfitting.

- `training` (default `False`): A boolean flag indicating whether the model is in training mode. Path dropping only occurs during training.
- `scale_by_keep` (default `True`): A boolean flag indicating whether to scale the remaining paths by the keep probability to maintain the overall scale of the input tensor.

Notes:

* If `drop_prob` is 0 or the model is not in training mode (`training` is `False`), the function returns the input tensor `x` without any modification. This ensures that path dropping only happens during training and when a non-zero drop probability is specified.

* If `keep_prob` is greater than 0 and `scale_by_keep` is `True`, the `random_tensor` is scaled by dividing by `keep_prob`. This scaling step ensures that the expected sum of the input elements remains unchanged after some paths are dropped.

* The output shape will be the same as input shape.

* Example

  ```
  Input Tensor:
  tensor([[1., 2., 3.],
          [4., 5., 6.],
          [7., 8., 9.]])
  
  Output Tensor with drop_path:
  tensor([[ 2.,  4.,  6.],
          [ 8., 10., 12.],
          [ 0.,  0.,  0.]])
  ```

#### 1.3.4 Mlp

[/backbone/vit](https://github.com/MCG-NJU/STMixer/blob/main/alphaction/modeling/backbone/vit_utils.py#L43)

```python
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x
```

A `nn.Module` with follwowing forward:
$$
y=\text{Dropout}\Big(W_2 \text{act}(W_1x+b_1)+b_2,\text{drop_prob}\Big)
$$
where $x$ input is of shaoe $(*,H_{\text{in}})$, the output of first fc layer is defined as (*, $H_{\text{hidden}}$) where $H_{\text{hidden}} $ is defined as `hidden_features` (if not `None`) or `in_features` and the ouuptut of second fc layer is $\text{H}_{\text{output}}$ is `out_featurs` (if not `None`) or `in_features`. 

#### 1.3.5 Attention

[/backbone/vit](https://github.com/MCG-NJU/STMixer/blob/main/alphaction/modeling/backbone/vit_utils.py#L63)

```python
class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```

##### init params

* `dim`: the input dimension of the feature vectors.
* `num_heads`: The number of attention heads (default: 8).
* `qkv_bias`: If `True`, biases are added to the query, key, and value projections (default: `False`).
* `qk_scale`: A scaling factor for the dot products of the query and key vectors. If not provided, it defaults to $\frac{1}{\sqrt{\text{head_dim}}}$.
* `attn_drop`: Dropout probability applied to the attention scores (default: 0).
* `proj_drop`: Dropout probability applied to the output of the attention mechanism (default: 0).
* `attn_head_dim`: Optional dimension for each attention head. If provided, this overrides the default calculation of head dimensions.

##### Key Components:

1. **Number of Heads and Head Dimension**:
   - `num_heads` is the number of attention heads.
   - `head_dim` is the dimension of each attention head, calculated as dimnum_headsnum_headsdim unless `attn_head_dim` is specified.
   - `all_head_dim` is the total dimension for all heads, which is `head_dim * num_heads`.
2. **Scaling Factor**:
   - `self.scale` is the scaling factor for the dot products of the query and key vectors, which helps to stabilize the gradients during training.
3. **Linear Projections for Q, K, V**:
   - `self.qkv` is a linear layer that projects the input tensor into query, key, and value tensors. It outputs a tensor of size `3 * all_head_dim`.
4. **Optional Bias for Q and V**:
   - If `qkv_bias` is `True`, biases for the query and value projections are added (`self.q_bias` and `self.v_bias`).
5. **Dropout Layers**:
   - `self.attn_drop` is applied to the attention scores.
   - `self.proj_drop` is applied to the output of the final linear projection.
6. **Final Linear Projection**:
   - `self.proj` is a linear layer that projects the concatenated outputs of all attention heads back to the original dimension `dim`.

##### Forward Pass:

1. **Input Shape**:
   - The input tensor `x` has shape `(B, N, C)`, where `B` is the batch size, `N` is the sequence length, and `C` is the input dimension.
2. **Linear Projection for Q, K, V**:
   - `qkv` is computed by applying the linear layer `self.qkv` to `x`, optionally adding the biases `qkv_bias` if they are specified. The output is reshaped to `(B, N, 3, num_heads, head_dim)` and permuted to `(3, B, num_heads, N, head_dim)`.
3. **Splitting Q, K, V**:
   - The reshaped `qkv` tensor is split into separate query `q`, key `k`, and value `v` tensors, each with shape `(B, num_heads, N, head_dim)`.
4. **Scaling Q**:
   - The query tensor `q` is scaled by `self.scale`.
5. **Attention Scores**:
   - Attention scores are computed as the dot product of `q` and the transpose of `k`, resulting in a tensor of shape `(B, num_heads, N, N)`.
6. **Softmax and Dropout**:
   - The attention scores are normalized using softmax, and dropout is applied to the normalized scores.
7. **Weighted Sum of Values**:
   - The attention scores are used to compute a weighted sum of the value tensors `v`, resulting in a tensor of shape `(B, num_heads, N, head_dim)`.
8. **Concatenation and Final Projection**:
   - The output tensors from all heads are concatenated, reshaped to `(B, N, all_head_dim)`, and projected back to the original dimension `dim` using the linear layer `self.proj`. Dropout is then applied to the final output.

##### mathematical formulation

Given input tensor $\mathbf{x}$:

1. **Linear Projection**:

   $\mathbf{QKV}=\mathbf{W}_{\text{QKV}}\mathbf{x}+\mathbf{b}_{\text{QKV}}$

   where $\mathbf{W}_{\text{QKV}}$ is the weight matrix and $\mathbf{b}_{\text{QKV}}$ is the bias for the combined QKV projection (if `qkv_bias` is `True`).

2. **Reshape and Permute**:

   $\mathbf{QKV} \to \mathbf{Q}, \mathbf{K}, \mathbf{V}$ with shapes $(B, \text{num_heads}, N, \text{head_dim})$.

3. **Scale Query**:

   $\mathbf{Q} = \frac{\mathbf{Q}}{\sqrt{\text{head_dim}}}$

4. **Attention Scores**:

   $\text{Attn}=\text{softmax}\Big(\frac{\mathbf{Q}.\mathbf{K}^\top}{\sqrt{\text{head_dim}}}\Big)$

   where $.$ denotes the dot product.

5. **Dropout on Attention Scores**:

   $\text{Attn}=\text{Dropout}(\text{Attn})$

6. **Weighted Sum of Values**:

   $\text{Output}=\text{Attn}⋅\mathbf{V}$

7. **Concatenate and Final Projection**:

   $\text{Output} \to \text{Output}$ reshaped to $(B,N,\text{all_head_dim})$ and

   $\text{Final Output}=\mathbf{W}_{\text{proj}}\text{Output}+\textbf{b}_{\text{proj}}$

   where $\mathbf{W}_{\text{proj}}$ and $\textbf{b}_{\text{proj}}$ are the weights and biases of the final linear projection.

#### 1.3.6 Block

[/backbone/vit](https://github.com/MCG-NJU/STMixer/blob/main/alphaction/modeling/backbone/vit_utils.py#L109)

```python
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.use_checkpoint = use_checkpoint

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward_part1(self, x):
        if self.gamma_1 is None:
            return self.drop_path(self.attn(self.norm1(x)))
        else:
            return self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))

    def forward_part2(self, x):
        if self.gamma_1 is None:
            return self.drop_path(self.mlp(self.norm2(x)))
        else:
            return self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

    def forward(self, x):
        # if self.gamma_1 is None:
        #     x = x + self.drop_path(self.attn(self.norm1(x)))
        #     x = x + self.drop_path(self.mlp(self.norm2(x)))
        # else:
        #     x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        #     x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part1, x)
        else:
            x = x + self.forward_part1(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)
        return x

```

The `Block` class implements a Transformer block, which is a fundamental component of the Transformer architecture used in models like BERT and Vision Transformers (ViTs). This block consists of a multi-head self-attention mechanism, followed by a feedforward neural network (MLP), with normalization and residual connections.

##### Initialization Parameters

- `dim`: Dimension of the input and output feature vectors.
- `num_heads`: Number of attention heads.
- `mlp_ratio`: Ratio of the hidden layer size in the MLP relative to `dim` (default: 4).
- `qkv_bias`: If `True`, biases are added to the query, key, and value projections in the attention mechanism.
- `qk_scale`: Scaling factor for the dot products of the query and key vectors.
- `drop`: Dropout probability applied to the output of the MLP.
- `attn_drop`: Dropout probability applied to the attention scores.
- `drop_path`: Dropout probability for stochastic depth (also known as DropPath).
- `init_values`: Initial values for the scaling parameters `gamma_1` and `gamma_2`.
- `act_layer`: Activation function used in the MLP (default: GELU).
- `norm_layer`: Normalization layer used before the attention and MLP blocks (default: LayerNorm).
- `attn_head_dim`: Optional dimension for each attention head.
- `use_checkpoint`: If `True`, gradient checkpointing is used to save memory during training.

##### Components

1. **Normalization Layers**:
   - `self.norm1` and `self.norm2` are normalization layers applied before the attention and MLP blocks, respectively.
2. **Attention Mechanism**:
   - `self.attn` is an instance of the `Attention` class, implementing multi-head self-attention.
3. **Drop Path (Stochastic Depth)**:
   - `self.drop_path` is a dropout layer applied to the residual connections. If `drop_path` is `0`, it acts as an identity function.
4. **MLP Block**:
   - `self.mlp` is an instance of the `Mlp` class, implementing a feedforward neural network with one hidden layer.
5. **Scaling Parameters**:
   - `self.gamma_1` and `self.gamma_2` are scaling parameters applied to the outputs of the attention and MLP blocks, respectively. These are initialized to `init_values` if specified.
6. **Gradient Checkpointing**:
   - `self.use_checkpoint` enables gradient checkpointing for memory-efficient training.

##### Forward Pass

The forward pass through the block consists of the following steps:

1. **Normalization and Attention**:
   - The input tensor `x` is normalized using `self.norm1`.
   - The normalized tensor is passed through the attention mechanism.
   - If `self.gamma_1` is defined, the attention output is scaled by `self.gamma_1`.
   - The output of the attention mechanism is passed through `self.drop_path`.
   - This result is added to the input tensor `x` to form the first residual connection.
2. **Normalization and MLP**:
   - The result of the first residual connection is normalized using `self.norm2`.
   - The normalized tensor is passed through the MLP.
   - If `self.gamma_2` is defined, the MLP output is scaled by `self.gamma_2`.
   - The output of the MLP is passed through `self.drop_path`.
   - This result is added to the output of the first residual connection to form the second residual connection.
3. **Gradient Checkpointing**:
   - If `self.use_checkpoint` is `True`, gradient checkpointing is used for the attention and MLP parts to save memory during training.

##### Mathematical Formulation

Given an input tensor $\mathbf{x}$:

1. **Attention Part**:
   $$
   \mathbf{y}_1 = \text{norm}_1(\mathbf{x}) \\
   \mathbf{a} = \text{Attention}(\mathbf{y}_1) \\
   \mathbf{a} = \gamma_1 . \mathbf{a} \;\;\;(\text{if } \gamma \text{ is defined}) \\
   \mathbf{a} = \text{Attedrop_path}(\mathbf{a}) \\
   \mathbf{z}_1 = \mathbf{x} + \mathbf{a}
   $$
   
2. **MLP Part**:
   $$
   \mathbf{y}_2 = \text{norm}_2(\mathbf{z}_1) \\
   \mathbf{m} = \text{MLP}(\mathbf{y}_2) \\
   \mathbf{m} = \gamma_2 . \mathbf{m} \;\;\;(\text{if } \gamma_2 \text{ is defined}) \\
   \mathbf{m} = \text{drop_path}(\mathbf{m}) \\
   \mathbf{z}_2 = \mathbf{z} + \mathbf{m}
   $$
   

   The final output is $\mathbf{z}_2$.

#### 1.3.6 `pose_interpolation`

This function interpolates positional embeddings `pos_embed` from one grid size to another. Here's a breakdown of what it does:

1. Extracts any extra tokens from the beginning of the positional embeddings tensor `pos_embed` based on the specified `num_extra_tokens`.
2. Separates the remaining positional tokens from the extra tokens.
3. Reshapes the positional tokens into a 4D tensor, where the first dimension represents the batch size, the last dimension represents the embedding size, and the other two dimensions represent the original grid size `[H_0, W_0]`.
4. Permutes the dimensions of the positional tokens tensor to `[batch_size, embedding_size, original_height, original_width]`.
5. Interpolates the positional tokens tensor using bicubic interpolation to resize it from the original grid size `[H_0, W_0]` to the new grid size `[H_1, W_1]`.
6. Permutes the dimensions of the interpolated tensor back to `[batch_size, new_height, new_width, embedding_size]`.
7. Flattens the interpolated tensor along the spatial dimensions.
8. Concatenates the extra tokens (if any) with the interpolated positional tokens along the embedding dimension.
9. Returns the new positional embeddings tensor.

The output shape of the `new_pos_embed` tensor will be `(T, H_1*W_1, C)` where `T` is the number of tokens, `H_1` and `W_1` are the new grid dimensions, and `C` is the embedding size.

### Appendix 1. Gradient Checkpoint

References:

https://github.com/Mountchicken/Efficient-Deep-Learning/blob/main/Efficient_GPUtilization.md#23-gradient-checkpoint

https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9

https://huggingface.co/docs/transformers/perf_train_gpu_one#gradient-checkpointing

- Gradient Checkpoint is another useful trick to save your GPU memory which is a time for space technique
- **How neural networks use memory**: In order to understand how gradient checkpoint helps, we first need to understand a bit about how model memory allocation works. The total memory used by a neural network is basically the sum of two components.
  - The first component is the static memory used by the model
  - The second component is the dynamic memory taken up by the model's computational graph. Every forward pass through a neural network in train mode computes an activation for every neuron in the network; this value is then stored in the so-called computation graph. One value must be stored for every single training sample in the batch, so this adds up quickly. The total cost is determined by model size and batch size, and sets the limit on the maximum batch size that will fit into your GPU memory.
- **How Gradient Checkpoint Helps**: The network retains intermediate variables during the forward propagation in order to calculate the gradient during the backward. Gradient checkpoint works by not retaining intermediate variables during the forward propagation but recalculating them during the backward. This can save a lot of memory, but the consequence is that the training time will be longer.

```python
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
class Model(nn.Module):
    super(Model, self).__init__()
    def __init__(self):
        self.layer1 = nn.Conv2d(3,64,3,1,1)
        self.layer2 = nn.Conv2d(64,64,3,1,1)
        self.layer3 = nn.Conv2d(64,32,3,1,1)
    def forward(self,x):
        x = self.layer1(x)
        x = checkpoint(self.layer2, x)
        x = self.layer(x)
        return x
```

- **checkpoint** in forward takes a module and its arguments as input. Then the intermediate output by this module (self.layer2) will not be kept in the computation graph, and will be recalculated during the backpropagation instead.

![](https://github.com/Mountchicken/Efficient-Deep-Learning/raw/main/images/gradientckp.gif)

- **Experiments**: Here is an experiment from [Explore Gradient-Checkpointing in PyTorch](https://qywu.github.io/2019/05/22/explore-gradient-checkpointing.html). It's is an experiment to do classification with BERT, and gradient checkpoint is add to the Multi Head Self Attention module (MHSA) and GeLU in BERT. Using gradient checkpoint can save of lot of GPU memory without droping the performance.

  | Gradient checkpoint | Batch size | GPU Memory | Time for one epoch | Validation Accuracy after one epoch |
  | ------------------- | ---------- | ---------- | ------------------ | ----------------------------------- |
  | No                  | 24         | 10972MB    | 27min05s           | 0.7997                              |
  | Yes                 | 24         | 3944MB     | 36min50s           | 0.7997                              |
  | Yes                 | 132        | 10212MB    | 31min20s           | 0.7946                              |

- **Cautions**: Gradient checkpoint is very useful, but be careful where you use it

  - **Do not use on input layer**. The checkpoint detects whether the input tensor has a gradient or not, and thus performs the relevant operation. The input layer generally uses an image as input, which has no gradient, so it is useless to perform checkpoint on the input layer.
  - **No dropout, BN or so**: This is because checkpointing is incompatible with dropout (recall that effectively runs the sample through the model twice—dropout would arbitrarily drop different values in each pass, producing different outputs). Basically, any layer that exhibits non-idempotent behavior when rerun shouldn't be checkpointed (nn.BatchNorm is another example).

### Appendix 2. Embedding

A patch embedding layer **maps patches of pixels to vectors**. 

#### Theory

##### Embedding in ViT

From [ViViT paper](https://arxiv.org/pdf/2103.15691)

ViT (for images) extracts $N$ non-overlapping image patches $x_i \in \mathbb{R}^{H\times W}$, performs a linear projection and then rasterises them into 1D tokens $z_i \in \mathbb{R}^d$. The sqeunece of tokens input to the following transformer encoder is 
$$
\mathbf{z} = [z_{cls}, \mathbf{E}x_1, \cdots,\mathbf{E}x_N]+ \mathbf{p}
$$
where the projection by $\mathbf{E}$ is equivalent to a 2D convolution, $z_{cls}$ is an optional learned classification token prepended to this sequence, and its representation at the final layer of the encoder serves as the final representation used by the classification layer [16]. In addition, a learned positional embedding, $\mathbf{p} \in \mathbb{R}^{N \times d}$
is added to the tokens to retain positional information, as the subsequent self-attention operations in the transformer are permutation invariant.

#### Embedding video clips

We consider two simple methods for mapping a video $\mathbf{V} \in \mathbb{R}^{T \times H \times W \times C}$ to a sequence of tokens $\tilde{\mathbf{z}} \in \mathbb{R}^{n_t \times n_h \times  n_w \times d}$. We then add the positional embedding and reshapre into $\mathbb{R}^{N\times d}$ to obtain $\mathbf{z}$, the input to the trasnformer.

##### unifrom frame sampling

A straightforward method of tokenising the input video is to uniformly sample $n_t$ frames from the input video clip, embed each 2D frame independetly using the same method as ViT, and concatenate all these tokens together. Conceretely, if $n_h\cdot n_w$ non overlapping image patches are extracted from each frame, then a total of $n_t \cdot n_h \cdot n_w$ tokens will be forwarded trough the transformer encoder. Intuitively, this process may be seen as simply constructing a large 2D image to be tokenised following ViT. 

##### tubelent embedding

An alternate method, as shown in Fig. 3, is to extract non-overlapping, spatio-temporal “tubes” from the input volume, and to linearly project this to $\mathbb{R}^d$. This method is an extension of ViT’s embedding to 3D,
and corresponds to a 3D convolution. For a tubelet of dimension $t \times h\times w$, $n_t = \lfloor \frac{T}{t}\rfloor$, $n_h=\lfloor \frac{H}{h} \rfloor$ and $n_w=\lfloor \frac{W}{w}\rfloor$ tokens are extracted from the temporal, height, and width dimensions respectively. 

Smaller tubelet dimensions thus result in more tokens which increases the computation.
Intuitively, this method fuses spatio-temporal information during tokenisation, in contrast to “Uniform frame sampling” where temporal information from different frames is fused by the transformer.