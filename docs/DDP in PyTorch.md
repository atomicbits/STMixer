#DDP in PyTorch

## 1. Single GPU

```python
import torch
from torch.utils.data import Dataset

class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.data[index]
```

```python
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datautils import MyTrainDataset


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int, 
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs():
    train_set = MyTrainDataset(2048)  # load your dataset
    model = torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
  # Takes the dataset and wraps the dataloader around it
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True
    )


def main(device, total_epochs, save_every, batch_size):
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, device, save_every)
    trainer.train(total_epochs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    device = 0  # shorthand for cuda:0
    main(device, args.total_epochs, args.save_every, args.batch_size)
```

## 2. MultiGPU DDP

Code: [here](https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py)

```python
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datautils import MyTrainDataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs():
    train_set = MyTrainDataset(2048)  # load your dataset
    model = torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)
```



### 2.1 Imports

A wrapper around Python's multiprocessing

```
import torch.multiprocessing as mp
```

The module that takes in our input data and distributes it across our GPUs

```
from torch.utils.data.distributed import DistributedSampler
```

The DDP wrapper

```
from torch.nn.parallel import DistributedDataParallel as DDP
```

Two functions to initialize and destroy our distributed process group

```
from torch.distributed import init_process_group, destroy_process_group
```

### 2.2 the main block

```python
world_size = torch.cuda.device_count() # Return the number of GPUs available.
```

The `mp.spawn` function in PyTorch's `torch.multiprocessing` module is used to launch multiple processes, each of which runs the specified target function (in this case, `main`). The processes are identified by a `rank` that ranges from `0` to `nprocs-1`.

```python
mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)
```

- `main` is the function to be called by each process.
- `args=(world_size, args.save_every, args.total_epochs, args.batch_size)` are the arguments passed to each `main` function call.
- `nprocs=world_size` indicates that `world_size` processes will be spawned.

For each of these `world_size` processes, `mp.spawn` will call `main` with the arguments provided. However, the `rank` argument is automatically supplied by `mp.spawn` and corresponds to the rank of the process (0, 1, 2, ..., `world_size-1`).

So, if `world_size` is 4, `mp.spawn` will spawn 4 processes, and `main` will be called with the following `rank` values in separate processes:

1. `main(0, world_size, args.save_every, args.total_epochs, args.batch_size)`
2. `main(1, world_size, args.save_every, args.total_epochs, args.batch_size)`
3. `main(2, world_size, args.save_every, args.total_epochs, args.batch_size)`
4. `main(3, world_size, args.save_every, args.total_epochs, args.batch_size)`

Each process receives a unique `rank` from 0 to `world_size-1`. The `rank` value helps differentiate between the different processes and can be used to perform tasks specific to each process, such as handling different parts of a dataset in distributed data parallel training.

#### 2.2.1 Difference between `mp.spwan` and `torch.multiprocessing`

##### `mp.spaws`

* A programmatic way to spawn multiple processes and typically used within a Pythn script.
* It spawns `nprocs` processes, each running the specified function (`main` in this case).
* It passes a unique `rank` (process ID) to each process.
* It is more flexible and easier to use within a Python script as it does not require command-line arguments.

```python
import torch.multiprocessing as mp

def main(rank, world_size, save_every, total_epochs, batch_size):
    # Setup and training code here
    pass

if __name__ == "__main__":
    world_size = 4
    mp.spawn(main, args=(world_size, save_every, total_epochs, batch_size), nprocs=world_size)

```

Pros

- **Flexibility:** Easy to integrate directly within a script.
- **Programmatic Control:** Allows more control over how processes are spawned and managed.

Cons

- **Less Familiar:** Less common for those used to command-line tools.
- **Manual Setup:** Requires explicit setup within the script.

#####`torch.distributed.launch`

Overview

- **Function:** A command-line utility to launch multiple processes.
- **Location:** Typically used from the command line.

How it works

- `torch.distributed.launch` is a utility script provided by PyTorch.
- It launches multiple processes as specified by `--nproc_per_node`.
- Each process runs the same Python script with arguments needed for distributed training.
- It passes environment variables to each process, which are used for distributed setup.

```
python -m torch.distributed.launch --nproc_per_node=4 my_script.py --arg1 --arg2
```

Pros

- **Ease of Use:** Simple to use for those familiar with command-line tools.
- **Standardized:** A well-known method for launching distributed training scripts.

Cons

- **Less Flexibility:** Limited to what can be specified via command-line arguments.
- **Environment Management:** Requires understanding of environment variables for distributed setup.

##### Key Differences

Setup

- `mp.spawn`: Set up within the Python script, calling a function directly.
- `torch.distributed.launch`: Set up from the command line, running a script with specified arguments.

Usage

- `mp.spawn`: More suitable for programmatically managing processes within a script.
- `torch.distributed.launch`: More suitable for quick and standardized launching of distributed training from the command line.

Environment Variables

- `mp.spawn`: Typically handles distributed setup manually within the function (e.g., `ddp_setup`).
- `torch.distributed.launch`: Automatically sets up necessary environment variables for distributed processes.

### 2.3 Constructing the process group

This process group consists of all of the processes that are running on our GPUs. Typically each GPU runs one process and setting up a group is necessary so that all the processes can discover and communicate with each other. 

The following function takes the `rank` and `world_size` and we set some environment variables:

1. `"MASTER_ADDR"`: IP address of the machine that running the rank 0 process. For one single machine setup, the rank 0 process is going to be on the same machine. 
2. `"MASTER_PORT"`: a free port on machine with rank 0.

Then, we call `init_process_group` which initilized the default distributed process group. 

```python
def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

```

#### 2.3.1 `init_process_group`

When this function is called, PyTorch sets up the distributed environment for your training job. This involves initializing communication between different processes, setting up necessary configurations, and ensuring that the processes can synchronize and communicate with each other. 

**NOTE**: this function must be called from each process invloved in the distributed training, not just process with `RANK=0`. So, this `ddp_setup` is part of the `main` function called in `mp.spawn`.

##### Steps in  `init_process_group`

* **Backend initialisation**: the backend determines the communication protocol and is crucial for performance and compatibility. For instance, `nccl` is optimized for GPU communication, while `gloo` is used for CPU and is also GPU-compatible.

* **Environment variables**: 

  * Reads necessary environment variables like `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, and `RANK`.
  * `MASTER_ADDR` and `MASTER_PORT` specify the address and port of the master node, which coordinates the processes.
  * `WORLD_SIZE` specifies the total number of processes involved.
  * `RANK` specifies the unique identifier for each process.

* **Network Communication Setup**:

  - Establishes communication channels between processes using the specified backend.

  - For `nccl`, this involves setting up high-performance GPU-to-GPU communication.

  - For `gloo`, this involves setting up communication over TCP/IP or other supported networks.

    - The process with `RANK=0` (the master process) listens on the specified `MASTER_ADDR` and `MASTER_PORT`.

      Other processes connect to the master process using these details.

* **Process Group Initialization**:

  - The global process group is established, enabling collective communication operations like broadcast, scatter, gather, and reduce.
  - Creates subgroups if needed, which can be useful for hierarchical communication patterns.

* **Barrier Synchronization**:

  - Ensures that all processes reach the same point before proceeding. This is typically done using a barrier, which blocks each process until all processes have reached the barrier.

### 2.4 Constructing the DDP model

we wrap the model with DDP:

* `device_ids`: a list consisting of the GPU ID that the model lives on

```python
- self.model = model.to(gpu_id)
+ self.model = DDP(model, device_ids=[gpu_id])
```

Note that `gpu_id` is the same as `rank` of process:

```python
trainer = Trainer(model, train_data, optimizer, rank, save_every)
```

and 

```python
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        ...
```

###2.5 Distributing input data

* [DistributedSampler](https://pytorch.org/docs/stable/data.html?highlight=distributedsampler#torch.utils.data.distributed.DistributedSampler) chunks the input data across all distributed processes. IT is a Sampler that restricts data loading to a subset of the dataset. It is useful in conjunction with [`torch.nn.parallel.DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel). In such a case, each process can pass a `DistributedSampler` instance as a [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) sampler, and load a subset of the original dataset that is exclusive to it.

* Each process will receive an input batch of 32 samples; the effective batch size is `32 * nprocs`, or 128 when using 4 GPUs.

  ```python
  train_data = torch.utils.data.DataLoader(
      dataset=train_dataset,
      batch_size=32,
  -   shuffle=True,
  +   shuffle=False,
  +   sampler=DistributedSampler(train_dataset),
  )
  
  ```

* Calling the `set_epoch()` method on the `DistributedSampler` at the beginning of each epoch is necessary to make shuffling work properly across multiple epochs. Otherwise, the same ordering will be used in each epoch

  ```python
  def _run_epoch(self, epoch):
      b_sz = len(next(iter(self.train_data))[0])
  +   self.train_data.sampler.set_epoch(epoch)
      for source, targets in self.train_data:
        ...
        self._run_batch(source, targets)
  ```

* Note that for shuffling:

  * We set `shuffle=False` in `torch.utils.data.DataLoader`

  * Before creating the dataloader iterator (`for source, targets in …`), we call `set_epoch`

    ```python
    sampler = DistributedSampler(dataset) if is_distributed else None
    loader = DataLoader(dataset, shuffle=(sampler is None),
                        sampler=sampler)
    for epoch in range(start_epoch, n_epochs):
      if is_distributed:
        sampler.set_epoch(epoch)
      train(loader)
    ```

### 2.6 Saving model checkpoints

We only need to save model checkpoints from one process. Without this condition, each process would save its copy of the identical mode. Read more on saving and loading models with DDP [here](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#save-and-load-checkpoints)

```python
- ckp = self.model.state_dict()
+ ckp = self.model.module.state_dict()
...
...
- if epoch % self.save_every == 0:
+ if self.gpu_id == 0 and epoch % self.save_every == 0:
  self._save_checkpoint(epoch)
```

Note the difference between `model.state_dict()` and `model.module.state_dict()`:

* use `self.model.state_dict()` for single-device training and
* use  `self.model.module.state_dict()` when using `DataParallel` or `DistributedDataParallel` to ensure you are accessing the state dictionary of the underlying model rather than the parallel wrapper.

### 2.7 Running the distributed training job

- Include new arguments `rank` (replacing `device`) and `world_size`.
- `rank` is auto-allocated by DDP when calling [mp.spawn](https://pytorch.org/docs/stable/multiprocessing.html#spawning-subprocesses).
- `world_size` is the number of processes across the training job. For GPU training, this corresponds to the number of GPUs in use, and each process works on a dedicated GPU.

```python
- def main(device, total_epochs, save_every):
+ def main(rank, world_size, total_epochs, save_every):
+  ddp_setup(rank, world_size)
   dataset, model, optimizer = load_train_objs()
   train_data = prepare_dataloader(dataset, batch_size=32)
-  trainer = Trainer(model, train_data, optimizer, device, save_every)
+  trainer = Trainer(model, train_data, optimizer, rank, save_every)
   trainer.train(total_epochs)
+  destroy_process_group()

if __name__ == "__main__":
   import sys
   total_epochs = int(sys.argv[1])
   save_every = int(sys.argv[2])
-  device = 0      # shorthand for cuda:0
-  main(device, total_epochs, save_every)
+  world_size = torch.cuda.device_count()
+  mp.spawn(main, args=(world_size, total_epochs, save_every,), nprocs=world_size)
```