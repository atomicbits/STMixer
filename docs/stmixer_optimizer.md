#stmixer_optimizer

## 1.1 make_optimizer(cfg, model)

First, we need to `params` (iterable of parameters to optimize or dicts defining parameter groups) for our optimizer. We do it by defining a dict  

For ViT backbone, we can have two options: 

```python
layer_decay = cfg.ViT.LAYER_DECAY < 1.0
if layer_decay:
  assigner = LayerDecayValueAssigner(
    list(cfg.ViT.LAYER_DECAY ** (cfg.ViT.DEPTH + 1 - i)
         for i in range(cfg.ViT.DEPTH + 2)))
  else:
    assigner = None
if assigner is not None:
  print("Assigned values = %s" % str(assigner.values))
```

* if `cfg.ViT.LAYER_DECAY >= 1.0`, then `assigner=None`. This means that we are assiging learning rates as fixed and not dependent on layer index. 
  *  Note that in VMAEv2_VITB config file, `cfg.ViT.LAYER_DECAY` value is set to 1 but for default config file this value is set to 0.75. 
* otherwise, learning rate scale of layers of backbone model follows the exponential function based on layer id and `cfg.ViT.LAYER_DECAY`.  Note that there is no difference in weight decay between these two scenarios. 

First we consider first case (no assiginer). We can see later that in this case, we have 4 groups (in theory but 2 groups in real): all have learning rate = 1 and weight decay either 1 or 1e-4. 

We get list of layers that should have `weight_decay` equal to 0 (`cfg.ViT.NO_WEIGHT_DECAY`).

```python
skip_weight_decay_list = set(cfg.ViT.NO_WEIGHT_DECAY) # ('pos_embed')
print("Skip weight decay list: ", skip_weight_decay_list)
```

and we get the weight decay value for layers with weight_decay:

```python
weight_decay = cfg.ViT.WEIGHT_DECAY # 1e-4
```

 in case of `layer_decay` in other words if `cfg.ViT.LAYER_DECAY < 1.0`, we pass this list to 
$$
[\alpha ^{(D+1−i)}∣i=0,1,2,…,D+1]
$$
 `LayerDecayValueAssigner` class to create `assiginer` object where $\alpha$ is `cfg.ViT.LAYER_DECAY`and $D$ is `cfg.ViT.DEPTH`. 

otherwise, we set `assigner` to `None`.

Then we get the list of dictionary containing params of backbone model:

```python
backbone_parameters = get_parameter_groups(model.backbone,
                                           weight_decay,# 1e-4 from config
                                           skip_weight_decay_list, # 'pos_embed' from 
                                           get_num_layer=assigner.get_layer_id
                                           if assigner is not None else None,
                                           get_layer_scale=assigner.get_scale
                                           if assigner is not None else None,
                                           lr_scale=1.0 # default value in function defition
                                          )
                                           
```

### 1.1.1 backbone_parameters

```
parameter_group_names = {}
parameter_group_vars = {}
```

* **parameter_group_names**: A dictionary to store parameter group names and their corresponding parameters.
* **parameter_group_vars**: A dictionary to store parameter group names and their actual parameter values (tensors i.e., `torch.nn.Parameter`).

Then we iterate over all named params in the model (backbone) and assigns either passed weight decay and no weight decay but learing rate scale equalt to 1 (either hardcoded or defualt value).

* skip params that do not require gradients (i.e., frozen parameters)
* **no_decay_encoder**: 
  * if the parameter is a bias, has only one dimension, or is in the skip list, and if its name starts with 'encoder.'
  * no weight decay and a scale of 1.0.
* **no_decay_others**:
  * if the parameter is a bias, has only one dimension, or is in the skip list.
  * no weight decay and a scale of `lr_scale` (default=1).
* **decay_encoder**:
  * if the parameter name starts with 'encoder.'
  * assigns passed weight decay (`1e-4` from default) and a scale of 1.0.
* **decay_others**:
  * assigns passed weight decay (`1e-4` from default)  and a scale of `lr_scale` ( default =1). 

If `assigner != None`, we will change only `lr_scale` of the layers in backone by:

* `get_num_layer` function is used to determine the layer ID for the parameter and modifies the group name to include the layer ID.

* it creates a new entry for the group.

* it uses the provided `get_layer_scale` to adjusts the learning rate scale (set initially as 1) for this param.

  ```python
   scale = get_layer_scale(layer_id) * scale
  ```

  which is equal to $\alpha^{D+1-i}$ (`scale=1`).

Then:

* Adds the actual parameter tensor of this layer in `params` value (list) in the value of the appropriate group of `parameter_group_vars`

  ```python
  parameter_group_vars[group_name]["params"].append(param)
  ```

* Adds the  parameter name of this layer in `params` value (list) in the value of the appropriate group of `parameter_group_names`

  ```
  parameter_group_names[group_name]["params"].append(name)
  ```

The output of `parameter_group_names` is:

```python
Param groups = {
  "decay_others": {
    "weight_decay": 0.0001,
    "params": [
      "patch_embed.proj.weight",
      "blocks.0.attn.qkv.weight",
      "blocks.0.attn.proj.weight",
      "blocks.0.mlp.fc1.weight",
      "blocks.0.mlp.fc2.weight",
      "blocks.1.attn.qkv.weight",
      "blocks.1.attn.proj.weight",
      "blocks.1.mlp.fc1.weight",
      "blocks.1.mlp.fc2.weight",
      "blocks.2.attn.qkv.weight",
      "blocks.2.attn.proj.weight",
      "blocks.2.mlp.fc1.weight",
      "blocks.2.mlp.fc2.weight",
      "blocks.3.attn.qkv.weight",
      "blocks.3.attn.proj.weight",
      "blocks.3.mlp.fc1.weight",
      "blocks.3.mlp.fc2.weight",
      "blocks.4.attn.qkv.weight",
      "blocks.4.attn.proj.weight",
      "blocks.4.mlp.fc1.weight",
      "blocks.4.mlp.fc2.weight",
      "blocks.5.attn.qkv.weight",
      "blocks.5.attn.proj.weight",
      "blocks.5.mlp.fc1.weight",
      "blocks.5.mlp.fc2.weight",
      "blocks.6.attn.qkv.weight",
      "blocks.6.attn.proj.weight",
      "blocks.6.mlp.fc1.weight",
      "blocks.6.mlp.fc2.weight",
      "blocks.7.attn.qkv.weight",
      "blocks.7.attn.proj.weight",
      "blocks.7.mlp.fc1.weight",
      "blocks.7.mlp.fc2.weight",
      "blocks.8.attn.qkv.weight",
      "blocks.8.attn.proj.weight",
      "blocks.8.mlp.fc1.weight",
      "blocks.8.mlp.fc2.weight",
      "blocks.9.attn.qkv.weight",
      "blocks.9.attn.proj.weight",
      "blocks.9.mlp.fc1.weight",
      "blocks.9.mlp.fc2.weight",
      "blocks.10.attn.qkv.weight",
      "blocks.10.attn.proj.weight",
      "blocks.10.mlp.fc1.weight",
      "blocks.10.mlp.fc2.weight",
      "blocks.11.attn.qkv.weight",
      "blocks.11.attn.proj.weight",
      "blocks.11.mlp.fc1.weight",
      "blocks.11.mlp.fc2.weight"
    ],
    "lr_scale": 1.0
  },
  "no_decay_others": {
    "weight_decay": 0.0,
    "params": [
      "patch_embed.proj.bias",
      "blocks.0.norm1.weight",
      "blocks.0.norm1.bias",
      "blocks.0.attn.q_bias",
      "blocks.0.attn.v_bias",
      "blocks.0.attn.proj.bias",
      "blocks.0.norm2.weight",
      "blocks.0.norm2.bias",
      "blocks.0.mlp.fc1.bias",
      "blocks.0.mlp.fc2.bias",
      "blocks.1.norm1.weight",
      "blocks.1.norm1.bias",
      "blocks.1.attn.q_bias",
      "blocks.1.attn.v_bias",
      "blocks.1.attn.proj.bias",
      "blocks.1.norm2.weight",
      "blocks.1.norm2.bias",
      "blocks.1.mlp.fc1.bias",
      "blocks.1.mlp.fc2.bias",
      "blocks.2.norm1.weight",
      "blocks.2.norm1.bias",
      "blocks.2.attn.q_bias",
      "blocks.2.attn.v_bias",
      "blocks.2.attn.proj.bias",
      "blocks.2.norm2.weight",
      "blocks.2.norm2.bias",
      "blocks.2.mlp.fc1.bias",
      "blocks.2.mlp.fc2.bias",
      "blocks.3.norm1.weight",
      "blocks.3.norm1.bias",
      "blocks.3.attn.q_bias",
      "blocks.3.attn.v_bias",
      "blocks.3.attn.proj.bias",
      "blocks.3.norm2.weight",
      "blocks.3.norm2.bias",
      "blocks.3.mlp.fc1.bias",
      "blocks.3.mlp.fc2.bias",
      "blocks.4.norm1.weight",
      "blocks.4.norm1.bias",
      "blocks.4.attn.q_bias",
      "blocks.4.attn.v_bias",
      "blocks.4.attn.proj.bias",
      "blocks.4.norm2.weight",
      "blocks.4.norm2.bias",
      "blocks.4.mlp.fc1.bias",
      "blocks.4.mlp.fc2.bias",
      "blocks.5.norm1.weight",
      "blocks.5.norm1.bias",
      "blocks.5.attn.q_bias",
      "blocks.5.attn.v_bias",
      "blocks.5.attn.proj.bias",
      "blocks.5.norm2.weight",
      "blocks.5.norm2.bias",
      "blocks.5.mlp.fc1.bias",
      "blocks.5.mlp.fc2.bias",
      "blocks.6.norm1.weight",
      "blocks.6.norm1.bias",
      "blocks.6.attn.q_bias",
      "blocks.6.attn.v_bias",
      "blocks.6.attn.proj.bias",
      "blocks.6.norm2.weight",
      "blocks.6.norm2.bias",
      "blocks.6.mlp.fc1.bias",
      "blocks.6.mlp.fc2.bias",
      "blocks.7.norm1.weight",
      "blocks.7.norm1.bias",
      "blocks.7.attn.q_bias",
      "blocks.7.attn.v_bias",
      "blocks.7.attn.proj.bias",
      "blocks.7.norm2.weight",
      "blocks.7.norm2.bias",
      "blocks.7.mlp.fc1.bias",
      "blocks.7.mlp.fc2.bias",
      "blocks.8.norm1.weight",
      "blocks.8.norm1.bias",
      "blocks.8.attn.q_bias",
      "blocks.8.attn.v_bias",
      "blocks.8.attn.proj.bias",
      "blocks.8.norm2.weight",
      "blocks.8.norm2.bias",
      "blocks.8.mlp.fc1.bias",
      "blocks.8.mlp.fc2.bias",
      "blocks.9.norm1.weight",
      "blocks.9.norm1.bias",
      "blocks.9.attn.q_bias",
      "blocks.9.attn.v_bias",
      "blocks.9.attn.proj.bias",
      "blocks.9.norm2.weight",
      "blocks.9.norm2.bias",
      "blocks.9.mlp.fc1.bias",
      "blocks.9.mlp.fc2.bias",
      "blocks.10.norm1.weight",
      "blocks.10.norm1.bias",
      "blocks.10.attn.q_bias",
      "blocks.10.attn.v_bias",
      "blocks.10.attn.proj.bias",
      "blocks.10.norm2.weight",
      "blocks.10.norm2.bias",
      "blocks.10.mlp.fc1.bias",
      "blocks.10.mlp.fc2.bias",
      "blocks.11.norm1.weight",
      "blocks.11.norm1.bias",
      "blocks.11.attn.q_bias",
      "blocks.11.attn.v_bias",
      "blocks.11.attn.proj.bias",
      "blocks.11.norm2.weight",
      "blocks.11.norm2.bias",
      "blocks.11.mlp.fc1.bias",
      "blocks.11.mlp.fc2.bias",
      "norm.weight",
      "norm.bias"
    ],
    "lr_scale": 1.0
  }
}
```

the output of this function is a **list of dictionaries** where each item is a dictionary containing parameters in the same group with following keys: 

- `weight_decay`: weight decay applying to all parameters of this group.
-  `params` : list of params (the actual parameter tensor (i.e., `torch.nn.Parameter`) ) sharing the same `weight_decay` and `lr_scale`.
-  `lr_scale`. 

#### 1.1.2 rest_parameters

For other layers, we define first this dictionaroy:

```python
rest_parameters = {'params':[]}
```

Then we add all actual tensor parameters to this list.

```python
for name, p in model.named_parameters():
  if "backbone" not in name:
    rest_parameters['params'].append(p)
```

We combine all parameters in a same list

```python
optim_params = backbone_parameters + [rest_parameters]
```

where each item of this list a dictionary:

* dictionaries corresponding to backbone have `params`, `weight_decay` and `lr_scale` keys.
* dictionaries corresponding to rest have only `params` key.

Then, we create optimizer based on `cfg.SOLVER.OPTIMIZING_METHOD`. For our scenario, ``cfg.SOLVER.OPTIMIZING_METHOD=adamW`

```python
optimizer = torch.optim.AdamW(
  optim_params,
  lr=cfg.SOLVER.BASE_LR, # 0.00001 from VMAE config
  betas=cfg.SOLVER.BETAS, # (0.9, 0.999)
  weight_decay=cfg.SOLVER.WEIGHT_DECAY, # 1e-4 from VMAE config
)
```

The information about this optimizer is [here](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW).

**Note**

For rest layers, `weight_decay` and `lr_scale` are defined in `torch.optim.AdamW` passed variables but for backbone layers, these parameters are set in `get_parameter_groups` function.

###1.2 make_lr_scheduler(cfg, optimizer)

The idea to to define a custom learning rate scheduler

#### 1.2.1 Learning rate scheduler

To do that, we need to create a new class that inherits from `torch.optim.lr_scheduler._LRScheduler` and overriding the `_get_lr` method, which returns a list of learning rates for each parameter group.

Here is a step-by-step guide to creating a custom learning rate scheduler:

1. **Import necessary modules**: Start by importing the necessary modules from PyTorch.
2. **Define the custom scheduler class**: Create a new class that inherits from `torch.optim.lr_scheduler._LRScheduler`.
3. **Initialize the scheduler**: Implement the `__init__` method to initialize your scheduler. This includes storing any parameters you might need to compute the learning rate.
4. **Compute the learning rate**: Override the `get_lr` method to define how the learning rate changes over time.
5. **Use the custom scheduler**: Finally, create an instance of your custom scheduler and use it in your training loop.

An example:

```python
import torch
from torch.optim.lr_scheduler import _LRScheduler

class CustomExponentialLR(_LRScheduler):
    def __init__(self, optimizer, gamma, last_epoch=-1):
        self.gamma = gamma
        super(CustomExponentialLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        # Get current learning rates for all parameter groups
        current_lrs = [base_lr * (self.gamma ** self.last_epoch) for base_lr in self.base_lrs]
        return current_lrs

# Usage example
model = torch.nn.Linear(10, 2)  # Example model
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Instantiate the custom learning rate scheduler
scheduler = CustomExponentialLR(optimizer, gamma=0.9)

# Example training loop
for epoch in range(10):
    # Training code goes here

    # Step the scheduler
    scheduler.step()
    
    # Print current learning rates
    print(f"Epoch {epoch}: {scheduler.get_lr()}")

```

* The `__init__` method initializes the scheduler with the optimizer, a decay factor (`gamma`), and the last epoch.

* The `get_lr` method computes the learning rates by multiplying the initial learning rate (`base_lr`) by `gamma` raised to the power of `last_epoch`.
* `last_epoch` is an attribute that keeps track of the number of epochs completed. It is used internally to determine the current point in training and to compute the updated learning rates based on the specific scheduling policy.
* In the training loop, the scheduler's `step` method is called at the end of each epoch to update the learning rates. The current learning rates are printed to verify the updates.
* Each time the `step()` method of the scheduler is called, `last_epoch` is incremented by 1. This typically happens at the end of each epoch during training.



#### 1.2.2 `WarmupMultiStepLR`

```python
scheduler = cfg.SOLVER.SCHEDULER #  "warmup_multi_step"
```

```python
iter_per_epoch = cfg.SOLVER.ITER_PER_EPOCH # 23048
```

```python
steps = tuple(step*iter_per_epoch for step in cfg.SOLVER.STEPS) # (5, 8)
```

```
warmup_iters = cfg.SOLVER.WARMUP_EPOCH*iter_per_epoch # 2 * 23048
```

Then:

```python
WarmupMultiStepLR(
  optimizer,
  steps,
  cfg.SOLVER.GAMMA, # 0.1
  warmup_factor=cfg.SOLVER.WARMUP_FACTOR, #0.1
  warmup_iters=warmup_iters if cfg.SOLVER.WARMUP_ON else 0, # True
  warmup_method=cfg.SOLVER.WARMUP_METHOD, # linear
)
```

#####Initialization (`__init__` Method)

- **Parameters:**
  - `optimizer`: The optimizer for which to adjust the learning rate.
  - `milestones`: A list of epoch indices at which to decrease the learning rate.
  - `gamma`: Multiplicative factor of learning rate decay.
  - `warmup_factor`: Factor by which to scale the learning rate during warmup.
  - `warmup_iters`: Number of iterations for the warmup phase.
  - `warmup_method`: Method of warmup, either "constant" or "linear".
  - `last_epoch`: The index of the last epoch.
- **Validations:**
  - Ensures `milestones` is sorted.
  - Ensures `warmup_method` is either "constant" or "linear".

#### `get_lr` Method

- **Functionality:**

  - Computes the learning rate for each parameter group based on the current epoch (`self.last_epoch`).

  - If in the warmup phase (

    ```
    self.last_epoch < self.warmup_iters
    ```

    ), applies the warmup factor:

    - **Constant Warmup**: Keeps the learning rate scaled by `warmup_factor`.
    - **Linear Warmup**: Linearly interpolates between `warmup_factor` and 1.

  - After warmup, adjusts the learning rate based on `milestones` and `gamma`.

- **Return Value:**

  - A list of learning rates for each parameter group.



#### get_num_layer_for_vit

This function determines the layer number for a Vision Transformer based on the parameter name and the maximum number of layers. It maps various types of parameters (e.g., `cls_token`, `patch_embed`, `blocks`) to corresponding layer indices.

* special tokens ("cls_token" and "mask_token") or positional embeddings that are usually considered to be at the initial layer (value `0`).
* Patch embedding layers are also typically considered part of the initial layer (value `0`).
* relative position bias parameters are associated with the last layer.
* For block layers, the function extracts the layer index from the variable name and returns it incremented by one.
* **Default**: Any other parameter names are considered part of the last layer.

This function helps in determining which layer a particular variable belongs to, which can then be used for various purposes such as applying specific training strategies, adjusting learning rates, or setting different hyperparameters for different layers.

#### `LayerDecayValueAssigner`

This class assigns decay values to different layers. It has two methods:

- `get_scale(layer_id)`: Returns the decay value for a given layer ID.
- `get_layer_id(var_name)`: Returns the layer ID for a given variable name using `get_num_layer_for_vit`.

### Parameter Grouping

#### `get_parameter_groups`

This function groups model parameters into different categories based on their names and attributes, such as whether they belong to the encoder or if they should have weight decay applied. It returns a list of parameter groups, each with specific learning rate scales and weight decay settings.

**Parameters**:

- `model`: The neural network model.
- `weight_decay`: The default weight decay value.
- `skip_list`: A list of parameter names to skip.
- `get_num_layer`: A function to determine the layer ID for a variable name.
- `get_layer_scale`: A function to get the scaling factor for a layer.
- `lr_scale`: The learning rate scale for parameters.

**Logic**:

- Iterates over model parameters.
- Classifies parameters into groups based on their names and characteristics:
  - **"no_decay_encoder"**: For encoder parameters that do not require decay (e.g., biases).
  - **"no_decay_others"**: For other parameters that do not require decay.
  - **"decay_encoder"**: For encoder parameters that do require decay.
  - **"decay_others"**: For other parameters that require decay.
- Optionally, assigns a scale to each parameter group, modifying it with `get_layer_scale` if available.
- Collects parameter names and their corresponding groups.



### Optimizer Construction

#### `make_optimizer`

This function constructs an optimizer based on the configuration (`cfg`) and the model. It supports different optimizers like SGD, Adam, and AdamW. The function:

1. Determines if the model is a Vision Transformer.
2. Sets up layer decay if applicable.
3. Groups parameters into those that should and should not have weight decay.
4. Constructs the optimizer using the specified method (SGD, Adam, AdamW) with the grouped parameters.



### Learning Rate Scheduler

#### `make_lr_scheduler`

This function creates a learning rate scheduler based on the configuration. It supports two types of schedulers:

1. `WarmupMultiStepLR`: Adjusts the learning rate at specific steps with an initial warmup period.
2. `HalfPeriodCosStepLR`: Uses a half-period cosine schedule for the learning rate with an initial warmup period.

### Putting It All Together

Here's how these components work together:

1. **Parameter Grouping**: `get_parameter_groups` is used to create parameter groups with specific learning rate scales and weight decay settings.
2. **Optimizer Creation**: `make_optimizer` uses these parameter groups to construct an optimizer.
3. **Scheduler Creation**: `make_lr_scheduler` creates a learning rate scheduler based on the optimizer and the specified configuration.

### 



`WarmupMultiStepLR` and `HalfPeriodCosStepLR`, both of which include a warmup phase to gradually increase the learning rate at the beginning of training. Here's a breakdown of each class and its functionality:

### `WarmupMultiStepLR` Class

- - 

### `HalfPeriodCosStepLR` Class

#### Initialization (`__init__` Method)

- **Parameters:**
  - `optimizer`: The optimizer for which to adjust the learning rate.
  - `warmup_factor`: Factor by which to scale the learning rate during warmup.
  - `warmup_iters`: Number of iterations for the warmup phase.
  - `max_iters`: Maximum number of iterations for the training.
  - `warmup_method`: Method of warmup, either "constant" or "linear".
  - `last_epoch`: The index of the last epoch.
- **Validations:**
  - Ensures `warmup_method` is either "constant" or "linear".

#### `get_lr` Method

- **Functionality:**

  - Computes the learning rate for each parameter group based on the current epoch (`self.last_epoch`).

  - If in the warmup phase (

    ```
    self.last_epoch < self.warmup_iters
    ```

    ), applies the warmup factor:

    - **Constant Warmup**: Keeps the learning rate scaled by `warmup_factor`.
    - **Linear Warmup**: Linearly interpolates between `warmup_factor` and 1.

  - After warmup, uses a half-period cosine schedule to adjust the learning rate.

- **Return Value:**

  - A list of learning rates for each parameter group.

### Summary

Both classes are subclasses of PyTorch's `_LRScheduler` and are designed to manage the learning rate schedule during training:

- **WarmupMultiStepLR**:
  - Applies a warmup phase followed by a step-wise decay at specified milestones.
  - Suitable for training where learning rate needs to drop at certain epochs.
- **HalfPeriodCosStepLR**:
  - Applies a warmup phase followed by a half-period cosine decay.
  - Suitable for smoother transitions in learning rates over time.

```
python -m torch.distributed.launch --nproc_per_node=1 train_net.py --config-file "config_files/my_VMAEv2-ViTB-16x4.yaml" --transfer --no-head --use-tfboard
```