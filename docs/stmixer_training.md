```shell
python -m torch.distributed.launch --nproc_per_node=8 train_net.py --config-file "config_files/config_file.yaml" --transfer --no-head --use-tfboard

```

## 1.  Training command line

### 1.1 `torch.distributed.launch`

`- m torch.distributed.launch --nproc_per_node=4 train_net.py ` :

- **-m**: This option allows you to run a module as a script. When you use `python -m`, Python treats the specified module as the main entry point of the program. In this case, `torch.distributed.launch` is the module being run.

- **`torch.distributed.launch`**: This is a PyTorch utility module designed to help launch distributed training jobs. It sets up the environment for distributed training and launches multiple processes.
- **`--nproc_per_node`**: This argument specifies the number of processes to run on each node. Since we are focusing on a single node with multiple GPUs, this effectively sets the number of GPUs to use on that node.
- **`--nnodes`**: The number of nodes to use for distributed training.
- **`--node_rank`**: The rank of the node (0-indexed).

This command will:

1. Launch the `torch.distributed.launch` module, which prepares the environment for distributed training.

   1. The `torch.distributed.launch` utility calculates the `WORLD_SIZE` as `nproc_per_node * nnodes` where `nproc_per_node` and `nnodes` are set in command line.

   2.  It then sets this value along with other necessary environment variables (`MASTER_ADDR`, `MASTER_PORT`, `RANK`, etc.) for each process.

   3. Note that without using `torch.distributed.launch` module, we have to set environment variables manually as:

      ```shell
      export MASTER_ADDR="127.0.0.1"  # or the IP address of the master node
      export MASTER_PORT=29500  # any open port number
      export WORLD_SIZE=8  # total number of processes
      export RANK=0  # rank of the current process
      ```

      and then `python train_net.py --config-file ...`

2. Create 4 processes, each corresponding to one GPU on your single-node system.

3. Each process will execute `train_net.py` with the appropriate environment variables set to enable distributed training.

When you run this command, the following happens:

1. **Environment Variables**: `torch.distributed.launch` sets up several environment variables for each process, including `WORLD_SIZE`, `RANK`, `LOCAL_RANK`, `MASTER_ADDR`, and `MASTER_PORT`.
   - `WORLD_SIZE`: Total number of processes (4 in this case).
   - `RANK`: Unique rank of each process, ranging from 0 to `WORLD_SIZE`-1.
   - `LOCAL_RANK`: Local rank of each process on the node, ranging from 0 to `nproc_per_node`-1 (0 to 3 here).
   - `MASTER_ADDR`: The address of the master node (typically set to `localhost` for single-node training).
   - `MASTER_PORT`: The port number for communication (usually a free port).
2. **Process Initialization**: Each process will read these environment variables and initialize its own distributed process group using `torch.distributed.init_process_group()`.
   - The backend (e.g., `nccl` for GPU communication) and initialization method (`env://`) will be used to set up the distributed training environment. Note that ``
3. **GPU Assignment**: Each process sets its device to the appropriate GPU using `torch.cuda.set_device(local_rank)`. This ensures that each process works with a different GPU.
4. **Distributed Training Execution**: The `your_training_script.py` script is executed by each process, allowing them to collaborate in training the model across multiple GPUs.

## `train_net.py`

```python
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1], rlimit[1]))
```

- The first line retrieves the current limits on the number of file descriptors.
- The second line sets the soft limit to the maximum value allowed by the hard limit, allowing the process to open as many file descriptors as the system allows.

This is useful when a process needs to handle a large number of open files simultaneously and ensures that it can do so up to the maximum limit permitted by the system.

## 2. args 

We note:

* `args.transfer = True`: transfier weight from a pretrained model.

* `args.no_head = True`: no load the head layer params from weight file
* `args.use_tfboard = True`
* `args.skip_test = False`: do not test the final model
* `args.skip_val = False`: do not validate during training
* `args.adjust_lr =False`: adjust learning rate scheduler from old checkpoint  

Note that:

```python
parser.add_argument("--local_rank", type=int, default=0)
```

is required. When you use the `torch.distributed.launch` utility to launch your training script in a distributed manner, it spawns multiple processes on each node. Each process is given a unique identifier, which is passed to the script as the `--local_rank` argument. This allows each process to know its own rank and differentiate itself from the other processes.

## 3. distributted training setup

```python
num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
args.distributed = num_gpus > 1
if args.distributed:
	torch.cuda.set_device(args.local_rank)
  torch.distributed.init_process_group(backend="nccl", init_method="env://")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Note that:

* `args.local_rank` is set by `torch.distributed.launch`.
* when `init_process_group` is called, PyTorch sets up the distributed environment for your training job. This involves initializing communication between different processes, setting up necessary configurations, and ensuring that the processes can synchronize and communicate with each other. 

* `torch.backends.cudnn.deterministic = True` ensures that the cuDNN library produces deterministic results. This means that given the same input and network, you will get the same output every time you run your model. 
  * Pros: reproducibility
  * Cons: May lead to a reduction in performance.
* `torch.backends.cudnn.benchmark = False` disables the cuDNN auto-tuner, which selects the best convolution algorithms for the hardware you are running on based on the input size.
  - Pros: ensure consistent timing across different runs, which is beneficial when reproducibility is a concern.
  - Cons: Disabling the auto-tuner may lead to suboptimal performance, especially if the input sizes vary, as the fastest algorithm may not be chosen.

```python
global_rank = get_rank()
```

which gives:

* `0` if `not dist.is_available()` or if `not dist.is_initialized()`
* gives  the rank of the current process in the provided `group`, default otherwise. Rank is a unique identifier assigned to each process within a distributed process group. They are always consecutive integers ranging from 0 to `world_size`.

## 4. Config

```python
# Merge config.
cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()
```

getting config params from:

1. initialized by [defualts.py](https://github.com/MCG-NJU/STMixer/blob/main/alphaction/config/defaults.py)
2. and updated by the passed `args.config_file` for example: [VMAEv2-ViTB-16x4.yaml](https://github.com/MCG-NJU/STMixer/blob/main/config_files/VMAEv2-ViTB-16x4.yaml).



## 5. Setting loggers

```python
logger = setup_logger("alphaction", output_dir, global_rank)
logger.info("Using {} GPUs".format(num_gpus))
logger.info(args)

logger.info("Collecting env info (might take some time)")
logger.info("\n" + get_pretty_env_info())

logger.info("Loaded configuration file {}".format(args.config_file))
with open(args.config_file, "r") as cf:
  config_str = "\n" + cf.read()
  logger.info(config_str)
logger.info("Running with config:\n{}".format(cfg))

tblogger = None
if args.tfboard:
  tblogger = setup_tblogger(output_dir, global_rank)
```

## 6. Random seed for each process

```python
set_seed(args.seed, global_rank, num_gpus)
```

The primary purpose of this function is to ensure reproducibility in a distributed training setup. Each process (rank) in the distributed system will get a unique, but reproducible, seed. By setting the seeds for Python, NumPy, and PyTorch, it ensures that random number generation in these libraries is consistent and reproducible across runs.

## 7. Calling traing method

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











## 