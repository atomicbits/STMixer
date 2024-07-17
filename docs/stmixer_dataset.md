# Dataset and Dataloader

## 1. Introduction

Dataloader in created in [/train_net.py](https://github.com/MCG-NJU/STMixer/blob/2c3cd1de2623fc47ad93cfda3a8fcb9713736a55/train_net.py#L68C5-L74C1) as

```python
data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments['iteration'],
    )
```

and passed to `do_train` function defined in [/engine/trainer.py](https://github.com/MCG-NJU/STMixer/blob/main/alphaction/engine/trainer.py). 

* `start_iter` can be set if we want to start iteration count, useful for resuming training from a checkpoint. Note that ` arguments["iteration"] = 0`, so, in general, we start from scratch.
* `is_distributed` is flag saying if we have a distributed environment or not.

this `data_loader` is used in training as

```
for iteration, (slow_video, fast_video, whwh, boxes, labels, metadata, idx) in enumerate(data_loader, start_iter):
```

so the main question is what are shapes, types and meaning of these `slow_video, fast_video, whwh, boxes, labels, metadata, idx` data using in training. 

We will see that (for ViTb-STMixer):

- `slow_clips`: a tensor of shape $(B, 3, T, H_{\text{prep}}, W_{\text{prep}}$ ). 
- `fast_clips` : None
- `whwh`: a tensor of shape $(B, 4)$ each row contains $(W_{\text{crop}}, H_{\text{crop}}, W_{\text{crop}}, H_{\text{crop}})$.
- `boxes`: a list of length equal to $B$ where each item is a np.array containing bbox coordinates of action (clipped to $W_{\text{crop}}, H_{\text{crop}}$) of shape $(M_i,4)$ where $M_i$ is number of actions in $i$th sample.
- `label_arrs`: a list of length equal to $B$ where each item is a np.array of shape $(M_i,80)$ where $M_i$ is number of actions in $i$th sample.
- `metadata`: a list of length equal to $B$ where each item is a list containing `video_index` and `sec` of that sample (`video_index`: index of the video from which this sample is comming and `sec`: the time index of that sample in the video).
- `clip_ids`: a list of length equal to `B` where each item is index of the corresponding sample (its index in all possible training clips).

## 2. Dataloader creation

The details are given here.

### 2.1 `make_data_loader`

[ref: /alphaction/dataset/build.py](https://github.com/MCG-NJU/STMixer/blob/main/alphaction/dataset/build.py#L65)

```
make_data_loader(cfg, is_train=True, is_distributed=False, start_iter=0)
```

params:

* `start_iter`: 

steps:

* `cfg.SOLVER.VIDEOS_PER_BATCH`: 8 in `VMAEv2-ViTB-16x4` and must be divisible by the number of gpus `num_gpus`
* `videos_per_gpu`: number of videos per gpu
* if`is_train`:
  * `shuffle` and `drop_last` = `True`.
  * `num_iters=cfg.SOLVER.MAX_EPOCH*cfg.SOLVER.ITER_PER_EPOCH = 23048 * 12 ` in `VMAEv2-ViTB-16x4`.
  * `split=train`
* else:
  * `shuffle=True` in case of `is_dsitributed` else `False`  
  *  `drop_last` = `False`.
  * `num_iters=cfg.SOLVER.MAX_EPOCH*cfg.SOLVER.ITER_PER_EPOCH = 23048 * 12 ` in `VMAEv2-ViTB-16x4`.
  * `split=test`
* `DATALOADER.ASPECT_RATIO_GROUPING = False`  as defined in [`alphaction/config/defaults.py`](https://github.com/MCG-NJU/STMixer/blob/2c3cd1de2623fc47ad93cfda3a8fcb9713736a55/alphaction/config/defaults.py#L152) 
  * If True, each batch should contain only images for which the aspect ratio is compatible. This groups portrait images together, and landscape images are not batched with portrait images.
* `aspect_grouping = []`.
* `datasets = build_dataset(cfg, split=split)` where `split` is set based on `is_train`.
  * this `datasets` is a list with one element which is an object from class `Ava`.
* For each `dataset` in `datasets`, we create `sampler` and `dataloader`.

### 2.2 `build_dataset(cfg, split)` 

[`alphaction/dataset/build.py`](https://github.com/MCG-NJU/STMixer/blob/2c3cd1de2623fc47ad93cfda3a8fcb9713736a55/alphaction/dataset/build.py#L9)

```python
def build_dataset(cfg, split):
    if cfg.DATA.DATASETS[0] == 'ava_kinetics':
        dataset = D.AvaKinetics(cfg, split)
    else:
        dataset = D.Ava(cfg, split)

    return [dataset]
```

Note that the output is a list.

### 2.3 `class Ava(torch.utils.data.Dataset)`

[/alphaction/dataset/datasets/ava_dataset](https://github.com/MCG-NJU/STMixer/blob/main/alphaction/dataset/datasets/ava_dataset.py)

#### 2.3.1 init params

* `self.split=split` which is `train` if  `is_train=True` else is `test` else.
* `self._sample_rate = cfg.DATA.SAMPLING_RATE = 4` from `VMAEv2-ViTB-16x4.yaml`
* `self._video_length = cfg.DATA.NUM_FRAMES = 16` form `VMAEv2-ViTB-16x4.yaml`.
* `self._seq_len = self._video_length * self._sample_rate = 64.` It defines the length of sequence from which an input clipt of length `self._video_length` is constructed by sampling `self._sample_rate`.
* `self._num_classes = cfg.MODEL.STM.ACTION_CLASSES = 80`  form `VMAEv2-ViTB-16x4.yaml`.
* `self._data_mean = cfg.DATA.MEAN = [0.45, 0.45, 0.45]` from `/config/defaults.py`: the mean value of the video raw pixels across the R G B channels.
* `self._data_mean = cfg.DATA.MEAN = [0.225, 0.225, 0.225]` from `/config/defaults.py`: the std value of the video raw pixels across the R G B channels.
* `self._use_bgr = cfg.AVA.BGR = False` from ``/config/defaults.py`.
*  in case of **training**:
  * `self._jitter_min_scale = cfg.DATA.TRAIN_MIN_SCALES = [256, 320]` from `VMAEv2-ViTB-16x4.yaml`.
  *  `self._jitter_max_scale = cfg.DATA.TRAIN_MAX_SCALE = 1333 ` from same config file.
  *  `self.random_horizontal_flip = cfg.DATA.RANDOM_FLIP = True` from same config file: horizontal flip augmentation.
  * `self._use_color_augmentation = cfg.AVA.TRAIN_USE_COLOR_AUGMENTATION = False`: no color augmentation method.
  * `self._pca_jitter_only = cfg.AVA.TRAIN_PCA_JITTER_ONLY = True`: True means that we only use PCA jitter augmentation when using color augmentation method. False emans that combine it with color jitter method!
  * `self._pca_eigval = cfg.AVA.TRAIN_PCA_EIGVAL= [0.225, 0.224, 0.229]  `.
  *  `self._pca_eigvec = cfg.AVA.TRAIN_PCA_EIGVEC = [
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ]`          
* in case of **testing**:
  * `self._jitter_min_scale = cfg.DATA.TEST_MIN_SCALES = [256]` from `VMAEv2-ViTB-16x4.yaml`.
  *  `self._jitter_max_scale = cfg.DATA.TEST_MAX_SCALE = 1333 ` from same config file.
  * `self._test_force_flip = cfg.AVA.TEST_FORCE_FLIP = False` from `/alphaction/config/defaults.py`

#### 2.3.2 `self._load_data(cfg)` 

load frame paths and annotations from files. 

The output of this function:

* `self._image_paths`: list of lists. Each item (list) corresponds to one video and contains the paths of images for this video. `self_image_paths[video_idx][frame_index]` where `video_idx` is the index of the video and `frame_index` is the frame index (frame index computed after first 15 mins). Note that the images are extracted from videos after removing the first 15 mins. So, the names of these images contain `frame_index`. 

* `self._video_idx_to_name` list which stroes the correponding files.

* `self._keyframe_indices` (list): a list of indices of the keyframes (**NOTE**: This is the list from which training data indices are sampled in dataloader):

  - Each item corresponds to one keyframe and is a tuple containing: 

    - `video_idx`: index of video

    - `sec_idx`: the index of that keyframe in all keyframes of this video (starting from 0).

    - `sec`: time index corresponding to this keyframe (should be in `range(902, 1799)`)

    - `center_idx`: frame index of this keyframe after removing the first 15 minutes and computed based on `sec` and FPS as:

      ```python
      def sec_to_frame(sec):
        """
        Convert time index (in second) to frame index.
        0: 900
        30: 901
        """
        return (sec - 900) * FPS
      ```

* `self._keyframe_boxes_and_labels` (list[list[list]]): a list of list which maps from `video_idx` and `sec_idx to` a list of boxes and corresponding labels.
  * The outer index is `video_idx`.
  *  The first inner index is `sec`.
  * The content of most inner list is [`box_coord`, `box_labels`] where `box_coord` is the coordinates of box and `box_labels` are the corresponding labels for the box.

* `self._num_boxes_used`: (int): total number of used boxes.

#### 2.3.3 `__def__(getitem__)(self, idx)`

Generating training samples for dataloader from `self._keyframe_indices` for a given index of possible trainining clips.

It takes `idx` as an index of one possible training clip, and For ViTb, the returned list `slow, fast, whwh, boxes, label_arrs, metadata, idx` is

* `slow`: a torch tensor of shape $(3,N,H_{\text{crop}},W_{\text{crop}})$ with format of `torch.float32`. $N$ is number of frames in each clip and is equal to `cfg.DATA.NUM_FRAMES `, first dimension is in `RGB` format, and $H_{\text{crop}}$ and $W_{\text{crop}}$ are height and width of cropped frames.
* `fast: None`
* `whwh`: a torch tensor of length 4 with format of `torch.float32` and containing $[W_{\text{crop}}, H_{\text{crop}},W_{\text{crop}}, H_{\text{crop}}]$.
* `boxes`: a np.array of shape $M \times 4$ containing `xyxy` coordinates of bboxes of groundtruth actions. Note that bbox coordinates are mapped and clipped to be within cropped frame. 
* `label_arrs`: a np.array of one-hot encoded labels of shape $M\times 80$.
* `metadata=[video_index,sec]` a list containing the video index `video_idx` and the time index `sec`.
* `idx`: index of training clip.

##### 2.3.3.1 Creating the clip in terms of its frame indices

For a sample index `idx` (corresponding to a keyframe from all keyframes), we get its video index `video_idx`, its relative index in that video `sec_idx`, its time index `sec` and its relative frame index (after first 15 mins) `center_idx` :

```
video_idx, sec_idx, sec, center_idx = self._keyframe_indices[idx]
```

we then construct a list of relative frame indices corresponding to the clip corresponding to that keyframe

```python
 seq = utils.get_sequence(
            center_idx,
            self._seq_len // 2,
            self._sample_rate,
            num_frames=len(self._image_paths[video_idx]),
        )

```

```python
def get_sequence(center_idx, half_len, sample_rate, num_frames):
    """
    Sample frames among the corresponding clip.
    Args:
        center_idx (int): center frame idx for current clip
        half_len (int): half of the clip length
        sample_rate (int): sampling rate for sampling frames inside of the clip
        num_frames (int): number of expected sampled frames
    Returns:
        seq (list): list of indexes of sampled frames in this clip.
    """
    seq = list(range(center_idx - half_len, center_idx + half_len, sample_rate))

    for seq_idx in range(len(seq)):
        if seq[seq_idx] < 0:
            seq[seq_idx] = 0
        elif seq[seq_idx] >= num_frames:
            seq[seq_idx] = num_frames - 1
    return seq
```

note that `seq` is list of frame indices (after removing the first 15 mins) which composes the clip corresponding to the keyframe. 

##### 2.3.3.2 getting groundtruth information for constructed clip

then we get groundtruth information about this clip (this keyframe) from `self._keyframe_boxes_and_labels`

```python
clip_label_list = self._keyframe_boxes_and_labels[video_idx][sec_idx]
assert len(clip_label_list) > 0
# Get boxes and labels for current clip.
boxes = []
labels = []
for box_labels in clip_label_list:
  boxes.append(box_labels[0])
  labels.append(box_labels[1])
boxes = np.array(boxes)
# Score is not used.
boxes = boxes[:, :4].copy()
# ori_boxes = boxes.copy()
```

now `boxes` is np.array of shape $M\times4$ and `labels` is a list of length $M$, and $M$ is number of assigned actions.

##### 2.3.3.3 loading the images corresponding to the clip

we then load the images corresponding to this clip using `seq`

```python
# Load images of current clip.
image_paths = [self._image_paths[video_idx][frame] for frame in seq]
imgs = utils.retry_load_images(
  image_paths, backend='cv2')
```

where `imgs` is the list where each item is of shape $(H_{\text{orig}},W_{\text{orig}},3)$.

##### 2.3.3.4 Pre-processing the clip: `_images_and_boxes_preprocessing_cv2(self, imgs, boxes)`

First, we scale back groundtruth boxes from [0, 1] to frame resolution of original data (np.array) 

```python
height, width, _ = imgs[0].shape

boxes[:, [0, 2]] *= width
boxes[:, [1, 3]] *= height
boxes = cv2_transform.clip_boxes_to_image(boxes, height, width)

# `transform.py` is list of np.array. However, for AVA, we only have
# one np.array.
```

and then create a list of length one including this np.array

```python
boxes = [boxes]
```

###### 2.3.3.4.1 `cv2_transform.random_short_side_scale_jitter`

Applying resizing 

```python
imgs, boxes = cv2_transform.random_short_side_scale_jitter(
                imgs,
                min_sizes=self._jitter_min_scale,
                max_size=self._jitter_max_scale,
                boxes=boxes,
            )
```

`min_sizes=[256,320]` and `max_size=1333`. This function scales the shorter side of the images to a random size between `min_size` and `max_size`, maintaining the aspect ratio

* selects a `min_scale` randomly between `256` and `320`.
* if $\min(H_{\text{orig}},W_{\text{orig}}) = $ `min_scale` and $\max(H_{\text{orig}},W_{\text{orig}}) \leq $ `max_size=1333`:
  * return the original `imgs` and `boxes`
* otherwise, it sets the shorter side of images to `min_scale` and scales the larger side while maninting the aspect ratio.
* if $\min(H_{\text{orig}},W_{\text{orig}}) = W_{\text{orig}}$, i.e., the width is the shorter side, then sets $W_{\text{prop}}=$`min_scale` and $H_{\text{prop}}=\lfloor \frac{H_{\text{orig}}}{W_{\text{orig}}} \times W_{\text{prop} } + 0.5 \rfloor $. 
* if $\min(H_{\text{orig}},W_{\text{orig}}) = H_{\text{orig}}$, i.e., the height is the shorter side, then sets $H_{\text{prop}}=$`min_scale` and $W_{\text{prop}}=\lfloor \frac{W_{\text{orig}}}{H_{\text{orig}}} \times H_{\text{prop} } + 0.5 \rfloor $. 
* BBox coordinates are scaled by multiplying them with $\text{min_scale}/\min(H_{\text{orig}},W_{\text{orig}})$.

The output is:

* `imgs`: list of np arrays, each witgh shape $(H_{\text{prep}}, W_{\text{prep}}, 3)$ where $\min(H_{\text{prep}}, W_{\text{prep}}) \sim [256, 320]$ and $\frac{H_{\text{prep}}}{W_{\text{prep}}}=\frac{H_{\text{orig}}}{W_{\text{orig}}}$.
* `bboxes` : list of np.arrays each with shape of $M\times4$ where $M$ is number of ground truth actions in each frame of the clip.

###### 2.3.3.4.2 `cv2_transform.horizontal_flip_list`

Flipping the `imgs` and `boxes` horizontally:

```python
imgs, boxes = cv2_transform.horizontal_flip_list(
                    0.5, imgs, order="HWC", boxes=boxes
                )
```

function performs a horizontal flip on a list of images and optionally flips the corresponding bounding boxes as well. 

in case of flipping:

* Horizontally flip the bboxes .
* Horizontally flip the images and keeps the order as "HWC" with same shape as input $(H_{\text{prep}}, W_{\text{prep}}, 3)$

```python
def flip_boxes(boxes, im_width):
    """
    Horizontally flip the boxes.
    Args:
        boxes (array): box to flip.
        im_width (int): width of the image.
    Returns:
        boxes_flipped (array): flipped box.
    """

    boxes_flipped = boxes.copy()
    boxes_flipped[:, 0::4] = im_width - boxes[:, 2::4] - 1
    boxes_flipped[:, 2::4] = im_width - boxes[:, 0::4] - 1
    return boxes_flipped
```

###### 2.3.3.4.3  Converting `imgs` to CHW order keeping BGR order

then convert images to CHW keeping BGR order

```python
# Convert image to CHW keeping BGR order.
imgs = [cv2_transform.HWC2CHW(img) for img in imgs]
```

###### 2.3.3.4.4  Normalizing `imgs` values to [0,1]

normalizes the pixel values of each image from the range [0, 255] to the range [0, 1].

```
imgs = [img / 255.0 for img in imgs]
```

###### 2.3.3.4.5 Reshaping 

then it ensures that the images are stored in contiguous memory blocks, reshapes them to consistent dimensions, and converts the data type to `float32`.

```python
imgs = [
  np.ascontiguousarray(
    # img.reshape((3, self._crop_size, self._crop_size))
    img.reshape((3, imgs[0].shape[1], imgs[0].shape[2]))).astype(np.float32)
  for img in imgs
]
```

Now `imgs` is a list of numpy arrays each elment is of shape $(3,H_{\text{crop}}, W_{\text{crop}})$ where cropping size is determined in ``cv2_transform.random_short_side_scale_jitter`.

###### 2.3.3.4.6 `color_normalization`

in training of STMixer for ViTb backbone, `self._use_color_augmentation=False`. So, we only do normal color normalization given mean and std

```python
imgs = [
            cv2_transform.color_normalization(
                img,
                np.array(self._data_mean, dtype=np.float32),
                np.array(self._data_std, dtype=np.float32),
            )
            for img in imgs
        ]
```

###### 2.3.3.4.7 Concatenting `imgs` , making it in RGB format

```python
# Concat list of images to single ndarray.
imgs = np.concatenate(
  [np.expand_dims(img, axis=1) for img in imgs], axis=1
)
if not self._use_bgr:
	# Convert image format from BGR to RGB.
  imgs = imgs[::-1, ...]

imgs = np.ascontiguousarray(imgs)
imgs = torch.from_numpy(imgs)
```

The `imgs` is now a torch tensor of shape $(3, N, H_{\text{crop}}, W_{\text{crop}})$ where $N$ is the clip length and each frame is in `RGB` format.

###### 2.3.3.4.8 Clipping the groundtruth bboxes to crop sizes

Noting that `boxes` now is a list of length 1 containign a np.array of shape Mx4, 

```python
 boxes = cv2_transform.clip_boxes_to_image(
            boxes[0], imgs[0].shape[1], imgs[0].shape[2]
        )
```

generates a numpy array of shape $M\times 4$. 

##### 2.3.3.5 Creating a one-hot encoded NumPy array `label_arrs`

NumPy array of shape $M \times 80$, with one-hot encoded labels for each box where $M$ is number of actions in this clip.  Each entry in `label_arrs` will have a 1 at the position corresponding to the class label (minus 1, for zero-based indexing) if that class label is present for the corresponding box.

##### 2.3.3.6 Creating `slow` and `fast`

Since for ViTb, `cfg.MODEL.BACKBONE.PATHWAYS = 1 ` and `cfg.DATA.REVERSE_INPUT_CHANNEL = False` 

```
pathways = self.cfg.MODEL.BACKBONE.PATHWAYS
imgs = utils.pack_pathway_output(self.cfg, imgs, pathways=pathways)
```

is equaivalent to `imgs = list[imgs]`.

Now

```python
if pathways == 1:
	slow, fast = imgs[0], None
else:
	slow, fast = imgs[0], imgs[1]
```

creates:

* `slow`: a tensor of shape $(3,N,H,W)$ where first dimension is in RGB format and `N` is number of frames in each training clip.
* `fast : None`

##### 2.3.3.7 Other information

creating a tensor of shape 4 where `w` is width of cropped frame and `h` is the height of cropped frame of format `torch.float32`

```
h, w = slow.shape[-2:]
whwh = torch.tensor([w, h, w, h], dtype=torch.float32)
```

and a list containing the video index `video_idx` and the time index `sec` of that clip:

```python
 metadata = [video_idx, sec]
```

### 2.4 Sampler

 If `distributed = True`, then 

```
sampler = samplers.DistributedSampler(dataset, shuffle=shuffle)
```

if `distributed = False` and `shuffle = True`: 

we have a [random sampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.RandomSampler)

```
sampler = torch.utils.data.sampler.RandomSampler(dataset)
```

else, we have a [Sequential Sampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.SequentialSampler)

```
sampler = torch.utils.data.sampler.SequentialSampler(dataset)
```

#### 2.4.1 `distributed`

The `DistributedSampler` class is used to split the dataset across multiple processes in a distributed training setup. Each process (or replica) gets a unique subset of the dataset, and the subsets can be shuffled based on the epoch for training.

##### 2.4.1.1 init

```python
class DistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
```

* **dataset**: The dataset from which to sample.

* **num_replicas**: The number of processes participating in distributed training. If not provided, it defaults to the world size (total number of processes) of the current distributed environment.

* **rank**: The rank of the current process within the total number of replicas. If not provided, it defaults to the rank of the current process in the distributed environment.

* **shuffle**: Whether to shuffle the dataset indices before sampling. Defaults to `True`.

Then these parameters are obtained:

* If `num_replicas` or `rank` are not provided, the class fetches them from the current distributed environment.

* `self.num_samples`: The number of samples each process will handle. This is calculated by dividing the total dataset size by the number of replicas.

* `self.total_size`: The total number of samples needed to ensure each process gets an equal number of samples, which might include some repeated samples to make the dataset evenly divisible.

##### 2.4.1.2 Iterator

* If `shuffle` is `True`, the dataset indices are shuffled deterministically using the current epoch as the seed.

* Extra samples are added to the indices list to make the total number of indices equal to `self.total_size`.

* The indices are then divided among the different processes by calculating the `offset` and selecting a subset of the indices for the current process based on its `rank`:

  ```python
  offset = self.num_samples * self.rank
  indices = indices[offset : offset + self.num_samples]
  ```

* The resulting list of indices for the current process is returned as an iterator.

##### 2.4.1.3 Length

`__len__`: returns the number of samples assigned to the current process.

##### 2.4.1.4 Set epoch

```python
def set_epoch(self, epoch):
    self.epoch = epoch
```

Allows setting the epoch, which is used to seed the random number generator for shuffling the dataset. This ensures that the data is shuffled differently at each epoch. 

Note that in distributed mode, calling the `set_epoch()` method at the beginning of each epoch **before** creating the `DataLoader` iterator is necessary to make shuffling work properly across multiple epochs. Otherwise, the same ordering will be always used. Example:

```python
sampler = DistributedSampler(dataset) if is_distributed else None
loader = DataLoader(dataset, shuffle=(sampler is None),
                    sampler=sampler)
for epoch in range(start_epoch, n_epochs):
  if is_distributed:
    sampler.set_epoch(epoch)
	train(loader)
```

### 2.5 batch sampler

```python
batch_sampler = make_batch_data_sampler(
  dataset, sampler, aspect_grouping, videos_per_gpu, num_iters, start_iter, drop_last
)
```

We will cover only case of `aspect_grouping = False` as it is in config params.

#### 2.5.1 `BatchSampler`

`torch.utils.data.BatchSampler` is a class provided by PyTorch that helps in sampling batches of data. It wraps another sampler to yield a mini-batch of indices, which can be used to index into the dataset. This is useful when you want to generate batches of data indices to be used by a DataLoader.

```python
batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, videos_per_batch, drop_last=drop_last)
```

*  It takes another sampler (like `torch.utils.data.RandomSampler` or `torch.utils.data.SequentialSampler`) and generates batches of indices based on that.

* **Batch Size**: You can specify the size of each batch. Note that it is infact equal to `videos_per_gpu` set in ``make_data_loader` function.

* **Drop Last**: You can specify whether to drop the last batch if it's smaller than the specified batch size.

##### What Happens Internally

- **Iteration**:
  - When you iterate over the `DataLoader`, it uses the `BatchSampler` to yield batches of indices.
  - For each batch of indices, the DataLoader fetches the corresponding data points from the dataset.
- **BatchSampler**:
  - Combines indices from the wrapped sampler into batches of the specified size.
  - If `drop_last` is `True`, the last batch is dropped if it has fewer than the specified `batch_size`.

#### 2.5.2 `IterationBasedBatchSampler`

In training `num_iters` is not `None` and is equal to `cfg.SOLVER.MAX_EPOCH*cfg.SOLVER.ITER_PER_EPOCH`. So, we have another wrapper over `BatchSampler`.

The class extends the functionality of a `BatchSampler` to support iterating over batches for a specified number of iterations. This can be useful in scenarios where **you want to train a model for a fixed number of iterations rather than simply iterating over the dataset a fixed number of times (epochs)**.

##### 2.5.2.1 init

* `batch_sampler`: An instance of a `BatchSampler` (or its subclass) which provides batches of data.

* `num_iterations`: The total number of iterations (batches) to sample.

* `start_iter`: The starting iteration count, useful for resuming training from a checkpoint.

##### 2.5.2.2 Itetation

`__iter__` makes the class iterable.

* It maintains an `iteration` counter starting from `self.start_iter`.

* If the underlying sampler has a `set_epoch` method (like `DistributedSampler`), it is called to set the epoch, ensuring different processes see different splits of the dataset.

* The outer `while` loop continues until the specified number of iterations (`num_iterations`) is reached.

* The inner `for` loop iterates over batches provided by the `batch_sampler`.

* Batches are yielded one by one until the iteration count exceeds `num_iterations`.
* **Iteration Control**: The class keeps track of the number of iterations and ensures that it stops once the specified number of iterations is reached.

##### 2.5.2.3 Length

`__len__`: returns the total number of iterations, which is `num_iterations`.

### 2.6 Collator

```python
collator = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
```

`DATALOADER.SIZE_DIVISIBILITY = 32` in default config file.

#### 2.6.1 `batch_different_videos`

It takes a list of tensors `[video_clip1, video_clip2, ...]` from different batches and creates a tensor of shape $(B,3,T,H_{\text{prep}},W_{\text{prep}})$ where:

* `B`: batch size

* $T$: temporal length of a clip

* $H_{\text{prep}}$ and $W_\text{prep}$ are height and width of preprocessed frames obtained as
  $$
  H_{\text{prep}} = \lceil \frac{{H_{\text{crop}}}}{32} \rceil \times 32
  $$
  

  to make them divisible to `cfg.DATALOADER.SIZE_DIVISIBILITY`. Note that $H_{\text{crop}}$ and $W_{\text{crop}}$ are height and width of pre-processed frames in `_images_and_boxes_preprocessing_cv2(self, imgs, boxes)` and controlled by `cfg.DATA.TRAIN_MIN_SCALES`.

* The frames are padded in right and bottom from shape of $(H_{\text{crop}}, W_{\text{crop}})$ to shape of $(H_{\text{prep}}, W_{\text{prep}})$.

* The output is constructed in a single tensor.

#### 2.6.2 `BatchCollator`

At each time, it outputs following data to the training pipleline:

* `slow_clips`: a tensor of shape $(B, 3, T, H_{\text{prep}}, W_{\text{prep}}$ ). 
* `fast_clips` : None
* `whwh`: a tensor of shape $(B, 4)$ each row contains $(W_{\text{crop}}, H_{\text{crop}}, W_{\text{crop}}, H_{\text{crop}})$.
* `boxes`: a list of length equal to $B$ where each item is a np.array containing bbox coordinates of action (clipped to $W_{\text{crop}}, H_{\text{crop}}$) of shape $(M_i,4)$ where $M_i$ is number of actions in $i$th sample.
* `label_arrs`: a list of length equal to $B$ where each item is a np.array of shape $(M_i,80)$ where $M_i$ is number of actions in $i$th sample.
* `metadata`: a list of length equal to $B$ where each item is a list containing `video_index` and `sec` of that sample (`video_index`: index of the video from which this sample is comming and `sec`: the time index of that sample in the video).
* `clip_ids`: a list of length equal to `B` where each item is index of the corresponding sample (its index in all possible training clips).

