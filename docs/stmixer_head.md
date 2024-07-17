# `stm-decoder`

#### 1.1.1 _generate queires

We create two embeddings where an embedding is essentially acts as a lookup table that maps each index  to a dense vector (embedding).

- `cfg.MODEL.STM.NUM_QUERIES = 100`: size of the dictionary of spatial and embeddings
- `cfg.MODEL.STM.HIDDEN_DIM = 256`:  the size of each embedding vector

query embeddings are used to generate proposals or attend to different parts of the input data. Here, `self.init_spatial_queries` will be used to focus on different spatial regions, while `self.init_temporal_queries will be used to focus on different time steps/frames in a sequence.

```python
def _generate_queries(self, cfg):
  self.num_queries = cfg.MODEL.STM.NUM_QUERIES # 100
  self.hidden_dim = cfg.MODEL.STM.HIDDEN_DIM # 256
  
  # Build Proposals
  self.init_spatial_queries = nn.Embedding(self.num_queries, self.hidden_dim)
  self.init_temporal_queries = nn.Embedding(self.num_queries, self.hidden_dim)
```

These embeddings are intended to be used as attention queries within a model, helping it to focus on specific parts of the input data during processing. The embeddings are initialized with random values and will be learned during the model's training process.

#### 1.1.2 `_decode_init_queries`

This function takes `whwh` as a tensor of shape $(B,4)$ cotaining $(W_{\text{crop}},H_{\text{crop}},W_{\text{crop}},H_{\text{crop}})$. 

Output:

* `xyzr`: a tensor of $(B,N_{\text{queries}},4)$ containing proposal `xy` center, `z` scale as $\log_2(\sqrt{HW})$ and `r` as the aspect ratio $\log_2(H/W)$.
* `init_spatial_queries`: a tensor of shape $(B, N_{\text{queries}}, \text{hidden_dim})$ containing normalized spatial query embeddings.
* `init_temporal_queries`: same as `init_spatial_queries` but for temporal query embeddings.

#### 1.1.3 `AdaptiveSamplingMixing`

the functions outputs two tensors:

* `spatial_queries`: `[B, n_query, query_dim]`.
* `temporal_queries`: `[B, n_query, query_dim]`.

#### 1.1.4 `ATMStage`

The `ATMStage` class produces multiple outputs with the following shapes:

1. **cls_score**:
   - Shape:`[N, n_query, num_classes_object + 1]`
   - Description: The classification scores for the human classifier.
   - Derived from: The output of `self.human_fc_cls` reshaped with `.view(N, n_query, -1)`.
2. **action_score**:
   - Shape: `[N, n_query, num_classes_action]`
   - Description: The classification scores for the action classifier.
   - Derived from: The output of `self.fc_action` reshaped with `.view(N, n_query, -1)`.
3. **xyzr_delta**:
   - Shape: `[N, n_query, 4]`
   - Description: The regression deltas for the bounding boxes.
   - Derived from: The output of `self.fc_reg` reshaped with `.view(N, n_query, -1)`.
4. **spatial_queries**:
   - Shape:`[N, n_query, query_dim]`
   - Description: The refined spatial queries.
   - Derived from: The output of `self.ffn_norm_s(self.ffn_s(spatial_queries))` reshaped with `.view(N, n_query, -1)`.
5. **temporal_queries**:
   - Shape: `[N, n_query, query_dim]`
   - Description: The refined temporal queries.
   - Derived from: The output of `self.ffn_norm_t(self.ffn_t(temporal_queries))` reshaped with `.view(N, n_query, -1)`.

Here's a breakdown of the main components:

- **cls_score**: This is generated from the spatial queries after passing through a series of fully connected layers and normalization layers defined in `self.human_cls_fcs`.
- **action_score**: This is generated from a concatenation of the spatial and temporal queries, which are then processed through a series of fully connected layers and normalization layers defined in `self.action_cls_fcs`.
- **xyzr_delta**: This is generated from the spatial queries after passing through a series of fully connected layers and normalization layers defined in `self.reg_fcs`.
- **spatial_queries and temporal_queries**: These are refined versions of the input queries after processing through the multihead attention layers (`self.attention_s` and `self.attention_t`), followed by feedforward networks (`self.ffn_s` and `self.ffn_t`) and layer normalization (`self.attention_norm_s`, `self.ffn_norm_s`, `self.attention_norm_t`, and `self.ffn_norm_t`).

### Appendix 1. _decode_init_queries

#### A.1.1 Function 1: `box_cxcywh_to_xyxy`

```
python
Copy code
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (x_c + 0.5 * h)]
    return torch.stack(b, dim=-1)
```

##### Purpose:

This function converts bounding box coordinates from the format `(center_x, center_y, width, height)` to the format `(x_min, y_min, x_max, y_max)`.

1. **Unbinding the tensor**:
   - `x_c, y_c, w, h = x.unbind(-1)` splits the tensor `x` along the last dimension into four separate tensors: `x_c`, `y_c`, `w`, and `h`.
   - These represent the center x-coordinate, center y-coordinate, width, and height of the bounding boxes, respectively.
2. **Calculating the corners**:
   - The corners of the bounding box are calculated using these formulas:
     - `x_min = x_c - 0.5 * w`
     - `y_min = y_c - 0.5 * h`
     - `x_max = x_c + 0.5 * w`
     - `y_max = y_c + 0.5 * h`
3. **Stacking the results**:
   - `b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]` creates a list of the corner coordinates.
   - `torch.stack(b, dim=-1)` stacks these tensors back into a single tensor along the last dimension.
4. **Return value**:
   - The function returns a tensor of shape `Bx4`, where each bounding box is represented as `(x_min, y_min, x_max, y_max)`.

#### A.1.2 Function 2: `_decode_init_queries`

```python
def _decode_init_queries(self, whwh):
    num_queries = self.num_queries
    proposals = torch.ones(num_queries, 4, dtype=torch.float, device=self.device, requires_grad=False)
    proposals[:, :2] = 0.5
    proposals = box_cxcywh_to_xyxy(proposals)

    batch_size = len(whwh)
    whwh = whwh[:, None, :] # B, 1, 4
    proposals = proposals[None] * whwh # B, N, 4

    xy = 0.5 * (proposals[..., 0:2] + proposals[..., 2:4])
    wh = proposals[..., 2:4] - proposals[..., 0:2]
    z = (wh).prod(-1, keepdim=True).sqrt().log2()
    r = (wh[..., 1:2] / wh[..., 0:1]).log2()
    xyzr = torch.cat([xy, z, r], dim=-1).detach()

    init_spatial_queries = self.init_spatial_queries.weight.clone()
    init_spatial_queries = init_spatial_queries[None].expand(batch_size, *init_spatial_queries.size())
    init_spatial_queries = torch.layer_norm(init_spatial_queries,
                                            normalized_shape=[init_spatial_queries.size(-1)])

    init_temporal_queries = self.init_temporal_queries.weight.clone()
    init_temporal_queries = init_temporal_queries[None].expand(batch_size, *init_temporal_queries.size())
    init_temporal_queries = torch.layer_norm(init_temporal_queries,
                                             normalized_shape=[init_temporal_queries.size(-1)])

    return xyzr, init_spatial_queries, init_temporal_queries
```

This function generates initial spatial and temporal queries for an object detection model, transforming the input bounding boxes and normalizing the query weights.

1. **Initialization**:
   - `num_queries = self.num_queries` gets the number of queries from the class.
   - `proposals` creates a tensor of shape `num_queries x 4` filled with ones and sets the first two columns to 0.5.
2. **Box transformation**:
   - `proposals = box_cxcywh_to_xyxy(proposals)` converts the proposals from `(center_x, center_y, width, height)` to `(x_min, y_min, x_max, y_max)` format.
3. **Batch processing**:
   - `batch_size = len(whwh)` determines the batch size.
   - `whwh = whwh[:, None, :]` reshapes the `whwh` tensor to `Bx1x4` for broadcasting.
   - `proposals = proposals[None] * whwh` scales the proposals by `whwh`, resulting in a tensor of shape `BxNx4`.
   - At this stage, `proposals` contains for each sample for each query: $[W_{crop},H_{crop},W_{crop},H_{crop}]$.
4. **Bounding box parameters**:
   - `xy` calculates the center `(x, y)` of each bounding box.
   - `wh` calculates the width and height of each bounding box.
   - `z` computes the log2 of the square root of the product of width and height.
   - `r` computes the log2 of the aspect ratio (height/width).
   - `xyzr` concatenates `xy`, `z`, and `r` to form a tensor of shape `BxNx4`.
5. **Query normalization**:
   - `init_spatial_queries` and `init_temporal_queries` are initialized from the model's weights and expanded to match the batch size.
   - Both queries are normalized using `torch.layer_norm`.
6. **Return values**:
   - The function returns `xyzr`, `init_spatial_queries`, and `init_temporal_queries`.

#####Summary:

- **box_cxcywh_to_xyxy**: Converts bounding box format from `(center_x, center_y, width, height)` to `(x_min, y_min, x_max, y_max)`.
- **_decode_init_queries**: Initializes spatial and temporal queries for an object detection model, transforming bounding boxes and normalizing the queries.

