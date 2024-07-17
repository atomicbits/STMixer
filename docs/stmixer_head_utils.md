# Introduction

![](/Users/hamed/Desktop/Screenshot 2024-05-31 at 11.27.09.png)

## 1. STMDecoder



### 1.2 ` AdaptiveMixing`

[code](https://github.com/MCG-NJU/STMixer/blob/main/alphaction/modeling/stm_decoder/util/adaptive_mixing_operator.py)

#### 1.2.1 Introduction

The `AdaptiveMixing` class is a  neural network module that dynamically generates and applies adaptive mixing parameters based on an input query. It performs **both channel-wise and spatial mixing**, followed by a projection to the query dimension. This module can be particularly useful in applications requiring dynamic and context-dependent transformations of input data, such as in attention mechanisms or adaptive filtering.
$$
\text{AdaptiveMixing}: \mathbf{x} \in \mathbb{R} ^{B \times N \times G \times P \times C} , \mathbf{q} \in \mathbb{R}^{B\times N \times \text{query_dim}}\to \mathbb{R}^{B\times N \times \text{query_dim}}
$$
The output shape of the `forward` method of the `AdaptiveMixing` class is `(B, N, query_dim)` where:

*  `B` and `N` are batch size and number of sequences. $G$, $P$ and $C$ are also number of groups, number of points and number of channels.
* `query_dim` is the dimension of the query input, defaulting to `in_dim` if not explicitly provided..

#### 1.2.1 Initialization

```python
def __init__(self, 
             in_dim, 
             in_points, 
             n_groups, 
             query_dim=None,
             out_dim=None, 
             out_points=None, 
             sampling_rate=None):
```

- **Parameters:**
  
  - `in_dim`: Input dimensionality.
  
  - `in_points`: Number of input points.
  
  - `n_groups`: Number of groups for group-wise operations.
  
  - `query_dim`: Dimensionality of the query input (optional, defaults to `in_dim`).
  
  - `out_dim`: Output dimensionality (optional, defaults to `in_dim`).
  
  - `out_points`: Number of output points (optional, defaults to `in_points`).
  
  - `sampling_rate`: Sampling rate for downsampling the input points (optional, defaults to 1 and not used in code). If used, then it will reduce number of input pounts
  
    ```
    self.in_points = in_points//sampling_rate
    ```
- **Attributes:**
  
  - Computes effective input and output dimensions based on the number of groups:
  
    ```python
    self.eff_in_dim = in_dim//n_groups
    self.eff_out_dim = out_dim//n_groups
    
    self.pad_bias_dim = 0
    self.pad_bias_points = 0
    
    self.eff_in_dim = self.eff_in_dim + self.pad_bias_dim
    
    self.in_points = self.in_points + self.pad_bias_points
    ```
  
  * `self.m_parameters`: Number of parameters for the mixing channel matrix is multipilcation of effective input and output dimensions
  
    ```
    self.m_parameters = self.eff_in_dim * self.eff_out_dim
    ```
  
  * `self.s_parameters`: Number of parameters for the spatial mixing is multiplication of input and output points:
  
    ```
     self.s_parameters = self.in_points * self.out_points
    ```
  
  * `self.total_parameters`: Total number of parameters for both mixing and spatial operations.
  
    ```
    self.total_parameters = self.m_parameters + self.s_parameters
    ```
  
  - Initializes a sequential model (`parameter_generator`) to generate mixing parameters from the query. This is a `nn.Sequential` composed of a linear layer of shape ($Q, G \times (\text{m_parameters}+\text{s_parameters})$). 
  - Initializes an output projection layer (`out_proj`) to transform the mixed output back to the query dimension. This is a linear layer of shape $(\text{eff_out_dim} \times \text{out_points} \times G, Q)$.
  - Applies ReLU activation (`act`) and initializes weights:
    - For `parameter_generator`: weights are initialized to zero.
    - For `out_ptoj`:
      - Weights are initialized using a uniform distribution $U(-\sqrt{k}, \sqrt{k})$, where $k = \frac{1}{\text{fan_in}}$ and  `fan_in` is the number of input units in the weight tensor.
      - Biases are initialized to zeros.

#### 1.2.2 Forward Pass (`forward` method)

- **Inputs:**

  - `x` as Input tensor of shape 
    
    ```
(B, N, G, P, C)
    ```

     where:
    
    - `B`: Batch size.
- `N`: Number of sequences.
    - `G`: Number of groups (should match `n_groups`).
- `P`: Number of points.
    - `C`: Number of channels.
    
  - `query`: Query tensor used to generate adaptive mixing parameters.
  
- **Steps:**

  1. **Parameter Generation:**
     - The `query` tensor is passed through the `parameter_generator` (a linear layer) to produce mixing parameters.
     - The parameters are reshaped and split into channel mixing parameters (`M`) and spatial mixing parameters (`S`).
  2. **Adaptive Channel Mixing:**
     - The input `x` is reshaped and multiplied with the channel mixing parameters (`M`).
     - Layer normalization and ReLU activation are applied.
  3. **Adaptive Spatial Mixing:**
     - The output from the channel mixing step is multiplied with the spatial mixing parameters (`S`).
     - Layer normalization and ReLU activation are applied.
  4. **Linear Transformation to Query Dimension:**
     - The mixed output is reshaped and passed through the `out_proj` layer to match the query dimension.
     - A residual connection is applied by adding the `query` tensor to the transformed output.

- **Output:**

  - The final output tensor, with the same shape as the `query`, is returned.

#### 1.2.3 Forward in math equations

##### Inputs

- $X \in \mathbb{R} ^{B \times N \times G \times P \times C}$ is the input tensor.
- $\mathbf{q} \in \mathbb{R}^{B \times N \times Q}$ is the query tensor.

##### Parameter Generation

1. **Generate Mixing Parameters:**

   $ \mathbf{p}=\text{parameter_generator}(\mathbf{q})$ where $\mathbf{p} \in \mathbb{R} ^{B \times N \times (G \times (\text{m_parameters}+\text{s_parameters}))}$

2. Reshape and Split Parameters:

   $\mathbf{p} \to  \mathbf{P} \in \mathbf{R}^{(B \times N) \times G \times (\text{m_parameters}+\text{s_parameters})}$

   and 

   $$\mathbf{P}=[\mathbf{M},\mathbf{S}]$$

    where:

   - $\mathbf{M} \in \mathbb{R}^{(B \times N) \times G \times \text{eff_in_dim} \times \text{eff_in_dim}}$
   - $\mathbf{S} \in \mathbb{R}^{(B \times N) \times G \times \text{out_points} \times \text{out_points}}$

##### Adaptive Channel Mixing

1. **Reshape Input**: $\mathbf{X} \to \mathbf{X}^\prime \in \mathbb{R}^{(B \times N) \times G \times P \times C}$.
2. **Channel Mixing**: $\mathbf{Y}=\mathbf{X}^\prime \cdot M$ (matrix multiplication) where $\mathbf{Y} \in \mathbb{R}^{(B \times N) \times G\times P \times \text{eff_in_dim}}$
3. **Layer Normalization and Activation:** $\mathbf{Y}=\text{ReLU}(\text{LayerNorm}(\mathbf{Y}))$.

##### Adaptive Spatial Mixing

1. **Spatial Mixing**:

   $$\mathbf{Z}= \mathbf{S} \cdot \mathbf{Y}$$ (implicit transpose and matrix multiplication) 

   where $\mathbf{Z} \in \mathbb{R}^{(B\times N)\times G \times \text{out_points} \times \text{eff_in_dim}}$.

2. **Layer Normalization and Activation:** 

   $$\mathbf{Z}=\text{ReLU}(\text{LayerNorm}(\mathbf{Z}))$$

##### Linear Transformation to Query Dimension

1. **Reshape**: 

   $$\mathbf{Z}→\mathbf{Z}^\prime \in \mathbb{R}^{B \times N\times (G \times \text{out_points} \times \text{eff_in_dim})}$$

2. **Linear Projection:** 

   $$\mathbf{O}=\text{out_proj}(\mathbf{Z}^\prime)$$ where $\mathbf{O} \in \mathbb{R}^{B \times N \times Q}$.

##### Output

1. **Residual Connection:** 

   $$\mathbf{O}_{\text{final}}=\mathbf{q}+\mathbf{O}$$

##### summary

The output shape of the `forward` method of the `AdaptiveMixing` class is `(B, N, query_dim)`, where `query_dim` is the dimension of the query input, defaulting to `in_dim` if not explicitly provided..

### 1.3 `make_sample_points`

It processes input tensors to generate sample points based on given offsets and region of interest (ROI) information (center coordinates, scale, and ratio parameters of RIOs).

The resulting tensor contains the sampled coordinates and levels for each query, batch, and group. 
$$
\text{make_sample_points}: \text{offset} \in \mathbb{R}^{B\times L \times (3.\text{num_groups} )}, \text{xyzr} \in \mathbb{R}^{B \times L \times 4} \to \mathbb{R}^{B \times L \times 1 \times \text{num_groups} \times 3}
$$
 where the last dimension consists of the `[x, y, level]` values where `B` and `L` are first two dimensions of input `offset` and `xyzr`.

##### Function Signature and Docstring

```python
def make_sample_points(offset, num_group, xyzr):
    '''
        offset_yx: [B, L, num_group*3], normalized by stride
        L: n_query
        num_group: in_points * n_heads
        xyzr : B, L, 4
        return: [B, L, 1, num_group, 3]
    '''
```

- `offset`: A tensor of shape `[B, L, num_group*3]` containing offset values normalized by stride:
  - `B`: batch size
  - `L`: number of queries.
- `num_group`: An integer representing the number of groups (in_points * n_heads).
- `xyzr`: A tensor of shape `[B, L, 4]` containing coordinates and dimensions information.

##### Reshaping the `offset` Tensor

```python
B, L, _ = offset.shape
offset = offset.view(B, L, 1, num_group, 3)
```

- Extract the dimensions `B` (batch size) and `L` (number of queries) from `offset`.
- Reshape `offset` to shape `[B, L, 1, num_group, 3]`.

##### Extracting and Calculating ROI Parameters

```python
roi_cc = xyzr[..., :2]
scale = 2.00 ** xyzr[..., 2:3]
ratio = 2.00 ** torch.cat([xyzr[..., 3:4] * -0.5, xyzr[..., 3:4] * 0.5], dim=-1)
roi_wh = scale * ratio
```

- `roi_cc`: Extracts the first two dimensions from `xyzr` (centroid coordinates).
- `scale`: Calculates the scale factor using the third dimension of `xyzr`.
- `ratio`: Computes the width-to-height ratio using the fourth dimension of `xyzr`, adjusted by `-0.5` and `0.5`.
- `roi_wh`: Computes the width and height by multiplying `scale` and `ratio`.

##### Preparing `roi_lvl` for Sample Points Calculation

```python
roi_lvl = xyzr[..., 2:3].view(B, L, 1)
roi_lvl = roi_lvl.view(B, L, 1, 1, 1)
```

- `roi_lvl`: Extracts and reshapes the third dimension of `xyzr` to shape `[B, L, 1, 1, 1]`.

##### Calculating Sample Points

```python
offset_yx = offset[..., :2] * roi_wh.view(B, L, 1, 1, 2)
sample_yx = roi_cc.contiguous().view(B, L, 1, 1, 2) + offset_yx
sample_lvl = roi_lvl + offset[..., 2:3]
```

- `offset_yx`: Adjusts the offset values by the width and height.
- `sample_yx`: Adds the adjusted offset to the centroid coordinates to get the sample points' coordinates.
- `sample_lvl`: Adjusts the level by adding the third component of the offset.

##### Returning the Final Sample Points

```python
return torch.cat([sample_yx, sample_lvl], dim=-1)
```

- Concatenates `sample_yx` and `sample_lvl` along the last dimension to form the final sample points tensor of shape `[B, L, 1, num_group, 3]`.

##### Summary

The function `make_sample_points` computes and returns sample points by:

1. Reshaping the input offset tensor.
2. Extracting and calculating region of interest (ROI) parameters from the `xyzr` tensor.
3. Calculating the sample points' coordinates and levels.
4. Concatenating these results to form a tensor representing the sample points.

#### math format

##### Inputs

- `offset`: $\mathbf{O} \in \mathbb{R}^{B \times L \times (\text{num_group} \cdot 3)}$
- `xyzr`: $X \in \mathbb{R}^{B \times L \times 4}$.

##### Reshape `offset`

Reshape $\mathbf{O}$ to $\mathbb{R}^{B \times L \times 1 \times \text{num_group}  \times 3}$:

$$\mathbf{O}_{\text{reshaped}} = \text{reshape}\big( \mathbf{O}, (B,L,1,\text{num_group}, 3)\big)$$

##### Extract ROI Parameters

1. **Centroid coordinates** $\mathbf{C}$:

   $$\mathbf{C} = \mathbf{X}_{[\cdots,:2]} \in \mathbb{R}^{B \times L \times 2}$$

2. **Scale** $\mathbf{S}$:

   $$\mathbf{S} = 2^{\mathbf{X}_{[\cdots,2:3]}} \in \mathbb{R}^{B \times L \times 1}$$.

3. **Ratio** $\mathbf{R}$:

   $$\mathbf{R} = 2 ^{[\mathbf{X}_{[\cdots,3:4]}\cdot -0.5, \mathbf{X}_{[\cdots,3:4]}\cdot 0.5]} \in \mathbb{R}^{B \times L \times 2}$$

4. **Width and Height $\mathbf{WH}$**:

   $$\mathbf{WH} = \mathbf{S}\cdot\mathbf{R}\in \mathbb{R}^{B\times L \times 2}$$

##### ROI Level

Reshape and extract the level component:

1. Extract level $\mathbf{L}_{\text{level}}$:

$$
\mathbf{L}_{\text{level}}=\mathbf{X}_{[\cdots,2:3]} \in \mathbb{R}^{B \times L \times 1}
$$

2. Reshape $\mathbf{L}_{\text{level}}$:

$$
\mathbf{L}_{\text{level}}= \text{reshape}(\mathbf{L}_{\text{level}}, (B,L,1,1,1))
$$


##### Calculate Offset and Sample Points

1. **Calculate Offset in YX**:
   $$
   \mathbf{O}_{\text{yx}}= \mathbf{O}_{\text{reshape}}[\cdots,:2].\text{reshape}\big(\mathbf{WH}, (B,L,1,1,2) \big)
   $$

2. Sample Points YX:

$$
\mathbf{S}_{\text{yx}}= \text{reshape}\big(\mathbf{C}, (B,L,1,1,2) \big) + \mathbf{O}_{\text{yx}}
$$

3. **Sample Levels**:

$$
\mathbf{S}_{\text{level}}= \mathbf{L}_{\text{level}} + \mathbf{O}_{\text{reshaped}}[\cdots,2:3]
$$

##### Concatenate and Return

Concatenate the sample points and levels:
$$
\mathbf{S}_{\text{final}} = \text{concatenate} \big([\mathbf{S}_{\text{yx}}, \mathbf{S}_{\text{level}}], \text{dim}=-1 \big) \in \mathbb{R}^{B \times L \times 1 \times \text{num_groups} \times 3}
$$

##### Summary

In summary, the function can be expressed with the following steps in mathematical form:

1. Reshape the `offset` tensor.
2. Extract the centroid coordinates, scale, ratio, width, height, and level from the `xyzr` tensor.
3. Calculate the adjusted offset and sample points in the YX plane.
4. Adjust the sample levels.
5. Concatenate the adjusted YX coordinates and levels to form the final sample points tensor.

Thus, the final result $\mathbf{S}_{\text{final}}$ contains the sample points and levels, with each point represented as $(y,x, \text{level})$.

### 1.4 `translate_to_linear_weight`

It takes a reference tensor `ref`, a total number of feature levels `num_total`, and a parameter `tau` to calculate a set of weights. These weights are computed based on the distance between the reference tensor and a linearly spaced grid of feature levels, which are then normalized using a softmax function.
$$
\text{translate_to_linear_weight}: \mathbb{R}^{n\times \text{n_query} \times (\text{in_points } \cdot \text{ n_heads}) } \to \mathbb{R}^{n\times \text{n_query} \times (\text{in_points } \cdot \text{ n_heads}) \times \text{num_total} } 
$$

##### Function Signature and Input

```python
def translate_to_linear_weight(ref: torch.Tensor, num_total, tau=2.0):
    # ref: [n, n_query, 1, in_points * n_heads]
    # num_total: feature levels (typically 4)
```

- `ref`: A tensor of shape `[n, n_query, 1, in_points * n_heads]`.
- `num_total`: An integer representing the number of feature levels (typically 4).
- `tau`: A parameter (default is 2.0) used for scaling the distances in the weight calculation.

##### Creating a Linearly Spaced Grid

```python
grid = torch.arange(num_total, device=ref.device, dtype=ref.dtype).view(
    *[len(ref.shape)*[1, ]+[-1, ]])
    # [1, 1, 1, 1, num_total]
```

- `torch.arange(num_total)`: Creates a tensor containing values `[0, 1, 2, ..., num_total-1]`.
- `view(*[len(ref.shape)*[1, ]+[-1, ]])`: Reshapes the grid tensor to `[1, 1, 1, 1, num_total]`, adding extra singleton dimensions to match the shape of `ref` for broadcasting.

##### Unsqueeze `ref` and Clone

```
ref = ref.unsqueeze(-1).clone()
    # [n, n_query, 1, in_points * n_heads, 1]
```

- `ref.unsqueeze(-1)`: Adds a new dimension at the end, changing the shape to `[n, n_query, 1, in_points * n_heads, 1]`.
- `.clone()`: Creates a copy of the tensor to avoid in-place operations affecting the original tensor.

##### Calculating the Squared Distance and Weights

```python
l2 = (ref - grid).pow(2.0).div(tau).abs().neg()
    # [n, n_query, 1, in_points * n_heads, num_total]
```

- `ref - grid`: Broadcasts and subtracts the grid values from `ref`, resulting in a tensor of shape `[n, n_query, 1, in_points * n_heads, num_total]`.
- `.pow(2.0)`: Squares the differences.
- `.div(tau)`: Divides by `tau`, scaling the squared distances.
- `.abs().neg()`: Takes the absolute value and negates it to prepare for the softmax operation.

##### Applying Softmax

```python
weight = torch.softmax(l2, dim=-1)
```

- `torch.softmax(l2, dim=-1)`: Applies the softmax function along the last dimension (num_total), converting the distances to a set of normalized weights.

##### Returning the Weights

```python
return weight
```

- Returns the calculated weights tensor of shape `[n, n_query, 1, in_points * n_heads, num_total]`.

##### Summary

The `translate_to_linear_weight` function performs the following operations:

1. Creates a linearly spaced grid of values from 0 to `num_total - 1`.
2. Expands `ref` tensor by adding an extra dimension.
3. Computes the squared distance between `ref` and the grid, scaled by `tau`.
4. Converts these distances into normalized weights using the softmax function.

This results in a tensor of weights that represent the softmax probabilities computed based on the distance between the reference tensor `ref` and the grid values. These weights can be used to linearly interpolate or assign importance based on the proximity to the grid values.

##### math

- R as the input reference tensor of shape $(n, n_{\text{query}}, 1, \text{in_points} \times n_{\text{heads}})$. 
- $\tau$ as the temperature parameter (default value 2.0).
- $\text{num_total}$as the number of feature levels.

1. **Create a Grid Tensor:**

   Define a grid tensor $\mathbf{G}$ of shape $(1, 1, 1, 1, \text{num_total})$:
   $$
   \mathbf{G} = [0,1,2,\cdots,\text{num_total-1}]
   $$

2. **Expand the Reference Tensor:**

   Expand the reference tensor $\mathbf{R}$ to include an additional dimension at the end:
   $$
   \mathbf{R'} = \mathbf{R} \text{ unsqueeze}(-1)
   $$
   This changes the shape of $\mathbf{R}$ to $(n, n_{\text{query}}, 1, \text{in_points} \times n_{\text{heads}}, 1)$.

3. **Compute the L2 Distance and Normalize:**

   Compute the L2 distance between the reference tensor $\mathbf{R}^\prime$ and the grid tensor $\mathbf{G}$, scale it by $\tau$, and negate:
   $$
   \mathbf{L2} = -\frac{(\mathbf{R'} - \mathbf{G})^2}{\tau}
   $$
   Here, $\mathbf{L2}$ will have the shape $n, n_{\text{query}}, 1, \text{in_points} \times n_{\text{heads}}, \text{num_total})$.

4. **Compute the Softmax Weights:**

   Apply the softmax function along the last dimension to get the weights:
   $$
   \text{weight}_{i, j, k, l, m} = \frac{\exp(\mathbf{L2}_{i, j, k, l, m})}{\sum_{m'=1}^{\text{num_total}} \exp(\mathbf{L2}_{i, j, k, l, m'})}
   $$
   This results in the final weight tensor:
   $$
   \mathbf{W} = \text{softmax}(\mathbf{L2}, \text{dim}=-1)
   $$
   The shape of $\mathbf{W}$ is $(n, n_{\text{query}}, 1, \text{in_points} \times n_{\text{heads}}, \text{num_total})$.

### 1.5 `MHAQ3D`

The function applies a form of multi-head attention in a 3D context (spatial + temporal) by:

- Sampling values from a 3D feature map (`value`) at specific points (`sample_points`).
- Optionally weighting these sampled values (`weight`).
- Organizing the output in a way that preserves the multi-head structure of the attention mechanism.

Inputs:

* **`sample_points`**:  $ \in \mathbb{R}^{B \times H_q \times  W_q \times  (\text{n_heads} \times \text{n_points}) \times  2}$ . A tensor specifying the points at which to sample the `value` tensor:
  * $B$: Batch size.
  * $H_q, W_q$: Spatial dimensions of the query. Note that $H_q$ is actually `num_query` and $W_q=1$.
  * $\text{n_heads}$: Number of attention heads.
  * $\text{n_points}$: Number of sampling points per head.
  * The last dimension $2$ represents the 2D coordinates for sampling.
* **`value`**: $\in \mathbb{R}^{B\times C_k \times T_k \times H_k \times W_k}$ A tensor representing the feature maps from which the values will be sampled:
  * $C_k$: Number of channels.
  * $T_k$: Temporal dimension.
  * $H_k, W_k$: Spatial dimensions.
* **`weight`** (optional): $\in \mathbb{R}^{B \times  H_q \times  W_q \times  (\text{n_heads} \times \text{n_points})}$ A tensor of weights for the sampled values.
* **`n_points`**: Number of points per head.

$$
\text{MHAQ3D}\to \mathbb{R}^{B \times  (C_k / \text{n_heads}) \times \text{n_heads}\times  T_k \times  \text{n_points} \times  H_q \times W_q}
$$

##### Function Signature and Input

```python
def MHAQ3D(sample_points: torch.Tensor, 
           value: torch.Tensor, 
           weight=None, 
           n_points=1):
    '''
    Args:
        sample_points: [n, n_query, 1, in_points * n_heads, 2]
        value: [n, c, t, h, w]
        weight: [n, n_query, 1, in_points * n_heads]
        n_points: in_points

    Returns:
        [n, c//n_heads, n_heads, t, in_points, n_query, 1]
    '''
```

- `sample_points`: Tensor of shape $[n, n_\text{query}, 1, \text{in_points} \times n_\text{heads}, 2]$, representing the coordinates of the sample points.
- `value`: Tensor of shape $[n,t,c,h,w]$ representing the input feature map.
- `weight`: Optional tensor of shape $[n, n_\text{query}, 1, n_\text{points} \times n_\text{heads}]$, representing the attention weights.
- `n_points`: Integer representing the number of points per head.

##### Extract and Adjust Shapes

```python
B, Hq, Wq, n_heads_points, _ = sample_points.shape
B, Ck, Tk, Hk, Wk = value.shape

n_heads = n_heads_points // n_points
```

- Extracts the dimensions from `sample_points` and `value`.
- Calculates the number of heads (`n_heads`) by dividing the total number of head points by `n_points`.

##### Reshape and Permute `sample_points`

```python
sample_points = sample_points.view(B, Hq, Wq, n_heads, n_points, 2) \
    .permute(0, 3, 1, 2, 4, 5).contiguous().flatten(0, 1)
# n*n_heads, n_query, 1, in_points, 2
```

- Reshapes `sample_points` to separate heads and points.
- Permutes dimensions to get the desired shape.
- Flattens the first two dimensions to prepare for repeating along the time dimension.
- The results of this step is: $[B, H_q, W_q, n_{\text{heads}} \cdot n_{\text{points}}, 2] \rightarrow [B.\text{n_heads}, H_q, W_q, n_{\text{points}}, 2]$.

##### Repeat and Adjust `sample_points`

```python
sample_points = sample_points.repeat(Tk, 1, 1, 1, 1)
# n*n_heads*Tk, n_query, 1, in_points, 2
sample_points = sample_points.flatten(2, 3)
# n*n_heads*Tk, n_query, in_points, 2
sample_points = sample_points * 2.0 - 1.0
```

- Repeats `sample_points` along the time dimension $T_k$ ($[B⋅\text{n_heads}⋅T_k,H_q,W_q,n_\text{points},2]$). 
- Flattens the third and fourth dimensions. $[B⋅\text{n_heads}⋅T_k,H_q \cdot W_q,n_\text{points},2]$
- Adjusts `sample_points` to a range suitable for `grid_sample` ([-1, 1]).

##### Reshape and Permute `value`

```python
value = value.view(B*n_heads, Ck//n_heads, Tk, Hk, Wk).permute(2, 0, 1, 3, 4).flatten(0, 1)
```

- Reshapes `value` to separate heads and channels: $[B, C_k, T_k, H_k, W_k] \rightarrow [B \cdot n_{\text{heads}}, \frac{C_k}{n_{\text{heads}}}, T_k, H_k, W_k$].
- Permutes dimensions to bring the time dimension to the front: $[B \cdot n_{\text{heads}}, \frac{C_k}{n_{\text{heads}}}, T_k, H_k, W_k] \rightarrow [T_k, B \cdot n_{\text{heads}}, \frac{C_k}{n_{\text{heads}}},  H_k, W_k] $.
- Flattens the first two dimensions to prepare for `grid_sample`:  $[T_k, B \cdot n_{\text{heads}}, \frac{C_k}{n_{\text{heads}}},  H_k, W_k] \rightarrow [T_k⋅B⋅\text{n_heads},\frac{C_k}{n_{\text{heads}}},H_k,W_k]$ 

##### Perform Grid Sampling

```python
out = F.grid_sample(
  value, 
  sample_points,
  mode='bilinear', padding_mode='zeros', align_corners=False,
)
# n*n_heads*Tk, Ck//n_heads, n_query, in_points
```

- Applies `grid_sample` to get the interpolated values using `value` values at `sample_points` locations..

- `grid_sample`, explained in [here](https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html),  performs bilinear interpolation.

- Resulting shape: $[B⋅n_\text{heads}⋅T_k,C_k/{\text{n_heads}},H_q,n_\text{points}]$.

  

##### Apply Weights (if provided)

```python
if weight is not None:
    weight = weight.view(B, Hq, Wq, n_heads, n_points) \
        .permute(0, 3, 1, 2, 4).flatten(0, 1).flatten(2, 3).unsqueeze(1).repeat(Tk, 1, 1, 1)
    # n*n_heads*Tk, 1, n_query, in_points
    out *= weight
```

- Reshapes and permutes `weight` to match the shape of `out`.
  - Reshape:$[B, H_q, W_q, n_{\text{heads}} \cdot n_{\text{points}}] \rightarrow [B, H_q, W_q, n_{\text{heads}}, n_{\text{points}}]$. 
  - Permute: $[B, H_q, W_q, n_{\text{heads}}, n_{\text{points}}] \rightarrow [B, n_{\text{heads}}, H_q, W_q, n_{\text{points}}$]. 
  - Flatten: $[B, n_{\text{heads}}, H_q, W_q, n_{\text{points}}] \rightarrow [B \cdot n_{\text{heads}}, H_q, W_q, n_{\text{points}}]$. 
  - Unsqueeze and repeat: $[B \cdot n_{\text{heads}}, H_q, W_q, n_{\text{points}}] \rightarrow [B \cdot n_{\text{heads}} \cdot T_k, 1, H_q, n_{\text{points}}]$.
- Multiplies `out` by `weight` to apply the attention weights.

##### Reshape and Permute Output

```python
return out.view(Tk, B, n_heads, Ck//n_heads, Hq, Wq, n_points).permute(1, 3, 2, 0, 6, 4, 5)
```

- Reshapes `out` to the desired final shape.
  - $[B \cdot n_{\text{heads}} \cdot T_k, \frac{C_k}{n_{\text{heads}}}, H_q, n_{\text{points}}] \rightarrow [T_k, B, n_{\text{heads}}, \frac{C_k}{n_{\text{heads}}}, H_q, W_q, n_{\text{points}}]$.
- Permutes dimensions to get the final output shape: $(B, Ck/n_{\text{heads}}, n_{\text{heads}}, T_k, \text{n_points}, H_q, W_q)$ where $H_q$ is actually `num_query` and $W_q=1$.

##### Summary

The `MHAQ3D` function performs a 3D multi-head attention operation. It transforms `sample_points` and `value` tensors, applies grid sampling to extract values at the sampled points, optionally applies attention weights, and reshapes the result to the desired output format. This process allows for attending to different parts of the input feature map across multiple heads and time steps.

### 1.6 `SAMPLE4D`

This function processes multi-scale 3D feature maps using sampled points and performs a multi-head attention operation over these scales. 

##### Function Signature and Input

```python
def SAMPLE4D(sample_points: torch.Tensor, 
             values: torch.Tensor, 
             featmap_strides, 
             n_points: int = 1, 
             num_levels: int = None,
             mapping_stride=3.0, 
             tau=2.0):
    B, Hq, Wq, n_heads_points, _ = sample_points.shape
    B, C, t, _, _ = values[0].shape
```

* `sample_points`: `[B, Hq, Wq, n_heads_points, 3]` representing the sampling points, where:
  - `B`: Batch size.
  - `Hq`: Height of the query.
  - `Wq`: Width of the query.
  - `n_heads_points`: Number of heads multiplied by the number of points.
  - The last dimension (3) represents the coordinates `[x, y, level]`.

* `values`: A list of tensors representing multi-scale feature maps at different levels, each of shape `[B, C, T, H, W]`.
  * `B`: Batch size.
  * `C`: Number of channels.
  * `t`: Temporal dimension.
  * `H`: Height of the feature map.
  * `W`: Width of the feature map.

* `featmap_strides`: A list of strides for each level in `values`.
* `n_points`: Number of points per head (default is 1).
* `num_levels`: Number of levels in `values` (default is `None`, which sets it to the length of `values`).
* `mapping_stride`: A stride value used in mapping sample points to feature levels (default is 3.0).
* `tau`: A temperature parameter for scaling (default is 2.0).

##### initial setup

```python
n_heads = n_heads_points // n_points

if num_levels is None:
    num_levels = len(values)
```

- `n_heads`: Number of heads calculated by dividing `n_heads_points` by `n_points`.
- Sets `num_levels` to the number of feature levels if not provided.

##### Extract Sample Points and Levels

```python
sample_points_xy = sample_points[..., 0:2]
sample_points_lvl = sample_points[..., 2].clone()
sample_points_lvl_mapped = sample_points_lvl - mapping_stride
sample_points_lvl_weight = translate_to_linear_weight(sample_points_lvl_mapped, num_levels, tau=tau)
sample_points_lvl_weight_list = sample_points_lvl_weight.unbind(-1)
```

- `sample_points_xy`: Extracts the 2D  `[x,y]` coordinates from `sample_points`.
- `sample_points_lvl`: Extracts the level information.
- `sample_points_lvl_mapped`: Adjusts the level information by subtracting `mapping_stride`.
- `sample_points_lvl_weight`: Calculates the weights for each level using `translate_to_linear_weight`.
- `sample_points_lvl_weight_list`: Unbinds the weights tensor into a list of tensors, one for each level.

##### Prepare Output Tensor

```python
out = sample_points.new_zeros(B, C // n_heads, n_heads, t, n_points, Hq, Wq)
```

- Initializes the output tensor `out` with zeros, of shape `[B, C // n_heads, n_heads, t, n_points, Hq, Wq]`.

##### Loop Over Feature Levels

```python
for i in range(num_levels):
    value = values[i]
    lvl_weights = sample_points_lvl_weight_list[i]
    stride = featmap_strides[i]

    mapping_size = value.new_tensor([value.size(4), value.size(3)]).view(1, 1, 1, 1, -1) * stride        
    normalized_xy = sample_points_xy / mapping_size

    out += MHAQ3D(normalized_xy, value, weight=lvl_weights, n_points=n_points)
```

- Loops over each feature level:
  - `value`: The feature map for the current level.
  - `lvl_weights`: Weights for the current level.
  - `stride`: The stride for the current level.
  - `mapping_size`: Calculates the mapping size by multiplying the spatial dimensions of `value` by `stride`.
  - `normalized_xy`: Normalizes `sample_points_xy` by `mapping_size` to map coordinates to the range [-1, 1] for `grid_sample`.
  - Adds the output of `MHAQ3D` to `out`. `MHAQ3D` applies multi-head attention to the normalized sample points and the feature map, weighted by `lvl_weights`.

##### Return Output

```python
return out, None
```

- Returns the output tensor `out` and `None` (placeholder for potential additional outputs).

##### Summary

The `SAMPLE4D` function performs the following steps:

1. **Setup**: Initializes variables and prepares the input data.
2. **Weight Calculation**: Computes level-specific weights for the sample points.
3. **Output Initialization**: Initializes an output tensor to accumulate results.
4. **Feature Map Processing**: Iterates through each feature level, normalizes sample points, and performs multi-head attention using `MHAQ3D`.
5. **Return**: Returns the accumulated output tensor and `None`.

The function essentially combines information from multiple scales of feature maps using attention mechanisms, allowing for more robust and multi-scale feature aggregation.

To express the `SAMPLE4D` function mathematically, let's denote the following:

- $\mathbf{S}$ as the input sample points tensor of shape$(B, H_q, W_q, n_\text{heads_points}, 3)$.
- $\mathbf{V}_i$ as the feature map tensor at level $i$ of shape $(B, C, T, H, W)$.
- $\mathbf{F}$ as the list of feature map tensors $\{\mathbf{V}_i\}$ for $i = 1, 2, \ldots, \text{num_levels}$.
- $\mathbf{f}_i$ as the stride for the $i$-th feature map.
- $\tau$ as the temperature parameter.
- $n_\text{npoints}$ as the number of points per head.
- $n_\text{nheads}$ as the number of heads, calculated as $n_\text{heads_points} / n_\text{points}$.
- mapping_stride $\text{mapping_stride}$ as the stride value for mapping levels.

### Step-by-Step Mathematical Formulation

1. **Extract Dimensions:**

   Given $\mathbf{S}$ of shape $(B, H_q, W_q, \text{n_heads_points}, 3)$, we can extract:

   - Sample points coordinates: $\mathbf{S}_{xy}$ of shape $(B, H_q, W_q, \text{n_heads_points}, 2)$. 
   - Sample points levels: $\mathbf{S}_{lvl}$ of shape $(B, H_q, W_q, \text{n_heads_points})$. 

2. **Map Sample Points Levels:**

   Map sample points levels by subtracting the mapping stride:
   $$
   \mathbf{S}_{lvl\_mapped} = \mathbf{S}_{lvl} - \text{mapping_stride}
   $$

3. **Compute Weights for Each Level:**

   Using the function `translate_to_linear_weight`, compute weights $\mathbf{W}$ for each level:
   $$
   \mathbf{W} = \text{softmax}\left(-\frac{(\mathbf{S}_{lvl\_mapped} - i)^2}{\tau}\right) \quad \forall i \in [0, \text{num_levels} - 1]
   $$
   This yields $\mathbf{W}$ of shape $(B, H_q, W_q, \text{n_heads_points}, \text{num_levels})$.

4. **Initialize Output Tensor:**

   Initialize the output tensor $\mathbf{O}$ with zeros:
   $$
   \mathbf{O} = \mathbf{0} \quad \text{of shape} \quad (B, \frac{C}{n_\text{heads}}, n_\text{heads}, T, n_\text{points}, H_q, W_q)
   $$

5. **Iterate Over Feature Levels:**

   For each feature level $i$:

   * Extract the $i$-th feature map $\mathbf{V}_i$ of shape $(B, C, T, H, W)$ and corresponding level weights $\mathbf{W}_i$:
     $$
     \mathbf{W}_i = \mathbf{W}[:, :, :, :, i]
     $$

   * Compute the mapping size $\mathbf{M}$ for normalization:
     $$
     \mathbf{M} = \left[\frac{W}{\mathbf{f}_i}, \frac{H}{\mathbf{f}_i}\right]
     $$

   * Normalize

   $$
   \mathbf{S}_{xy\_norm} = \frac{\mathbf{S}_{xy}}{\mathbf{M}}
   $$

   * Perform multi-head attention sampling using `MHAQ3D` and accumulate:
     $$
     \mathbf{O}+= \text{MHAQ3D}(\mathbf{S}_{xy\_norm}, \mathbf{V}_i, \mathbf{W}_i, \text{n_points})
     $$
     

##### Summary

The `SAMPLE4D` function can be summarized mathematically as:
$$
\mathbf{O} = \sum_{i=0}^{\text{num_levels}-1} \text{MHAQ3D}\left(\frac{\mathbf{S}_{xy}}{\left[\frac{W}{\mathbf{f}_i}, \frac{H}{\mathbf{f}_i}\right]}, \mathbf{V}_i, \text{softmax}\left(-\frac{(\mathbf{S}_{lvl} - \text{mapping_stride} - i)^2}{\tau}\right), \text{n_points}\right)
$$
where the multi-head attention function `MHAQ3D` handles the sampling and attention mechanism, and the output $\mathbf{O}$ is the accumulated result across all levels.



