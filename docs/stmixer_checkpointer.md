# stmixer_checkpointer

### Checkpointer Class

#### Initialization (`__init__` Method)

- **Parameters:**
  - `model`: The model whose state will be saved or loaded.
  - `optimizer`: The optimizer whose state will be saved or loaded.
  - `scheduler`: The learning rate scheduler whose state will be saved or loaded.
  - `save_dir`: Directory where checkpoints will be saved.
  - `save_to_disk`: Boolean indicating whether to save checkpoints to disk.
  - `logger`: Logger for logging messages.
- **Initialization:**
  - Sets the attributes for the class.
  - Initializes a logger if one is not provided.

#### Saving Checkpoints (`save` Method)

- **Parameters:**
  - `name`: Name of the checkpoint file.
  - `kwargs`: Additional data to save in the checkpoint.
- **Functionality:**
  - Checks if `save_dir` and `save_to_disk` are set.
  - Creates a dictionary `data` to store the model, optimizer, and scheduler states.
  - Saves the dictionary to a file using `torch.save`.
  - Updates the last checkpoint file using `tag_last_checkpoint`.

#### Loading Checkpoints (`load` Method)

- **Parameters:**
  - `f`: File path to the checkpoint.
  - `model_weight_only`: Boolean indicating whether to load only the model weights.
  - `adjust_scheduler`: Boolean indicating whether to adjust the scheduler.
  - `no_head`: Boolean indicating whether to skip loading the model head.
- **Functionality:**
  - Checks if a checkpoint file exists.
  - Loads the checkpoint file.
  - Loads the model state using `_load_model`.
  - Optionally loads the optimizer and scheduler states.
  - Resets iteration and person pool if `model_weight_only` is set.

#### Helper Methods

- **has_checkpoint**: Checks if a last checkpoint file exists.
- **get_checkpoint_file**: Reads the last checkpoint file and returns the filename.
- **tag_last_checkpoint**: Writes the last checkpoint filename to a file.
- **_load_file**: Loads the checkpoint file.
- **_load_model**: Loads the model state from the checkpoint.

### ActionCheckpointer Class

#### Initialization (`__init__` Method)

- Extends `Checkpointer` and adds a `cfg` (configuration) parameter.
- Calls the parent constructor and sets the `cfg` attribute.

#### Loading Files (`_load_file` Method)

- **Parameters:**
  - `f`: File path to the checkpoint.
- **Functionality:**
  - Checks if the file is a `.pkl` file and uses `load_c2_format` to load it.
  - Calls the parent `_load_file` method for other file types.
  - Adjusts the loaded checkpoint if the "model" key is not present.

### Summary

This code provides a comprehensive utility for managing checkpoints in a machine learning workflow, allowing you to save and load model states, optimizer states, and learning rate scheduler states. The `Checkpointer` class handles basic checkpointing tasks, while the `ActionCheckpointer` class extends this functionality to handle additional formats and configurations specific to the project.