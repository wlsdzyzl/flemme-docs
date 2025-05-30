==============
Data Loading
==============

To train a model, we have to load the datasets firstly. 
In Flemme, Data loader can also be specified through configuration file as follows. 

.. code-block:: yaml
  :linenos:

  # loader config
  loader:
    ## dataset config
    dataset: 
      name: ImgSegDataset
      data_dir: images
      label_dir: masks
      data_suffix: jpg
      ### you can specify data path here. 
      # data_path: path/to/single_data_set
    ## you can specify a list of data paths
    ## we would read all the datasets and concatenate them into one dataset
    ## note: at least one of data_path and data_path_list need to be specified.
    data_path_list: 
      - path/to/datasets/CVC-ClinicDB/fold1/
      - path/to/datasets/CVC-ClinicDB/fold2/
      - path/to/datasets/CVC-ClinicDB/fold3/
    ## augmentations or transforms
    data_transforms:
      - name: Resize
        size: [320, 256]
      - name: ToTensor
    ### other parameters related to dataloader
    ### refer to https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader 
    batch_size: 16
    num_workers: 8
    shuffle: true

We use ``torch.Dataloader`` as our data loader. 
Parameters like ``batch_size`` and ``num_workers`` are directly from ``torch.Dataloader``. 
Such parameters not recognized by Flemme will be passed to the torch's loader initialization. 
In specific, Flemme will parse these parameters: ``dataset``, ``data_path_list``, ``data_suffix_list``, ``label_suffix_list``, ``data_transforms`` and ``label_transforms``.

Datasets
=========

We provide several dataset classes for easy and flexible data loading. 

Image Datasets
---------------

ImgDataset
^^^^^^^^^^^
``ImgDataset`` defines a dataset contains only images without any label or condition information.

.. code-block:: python
    :linenos:

    ImgDataset.__init__(self, 
                ## data_path, can be specified in dataset[data_path] or loader[data_path_list] 
                data_path, 
                ## dimension of image, can be 2 or 3
                dim = 2, 
                ## data transforms, specified in loader[data_transforms]
                data_transform = None, 
                ## sub dir in data_path, we only read data in data_dir
                data_dir = '', 
                ## data_suffix, we only read data with specified data_suffix
                ## can be specified in dataset[data_suffix] or loader[data_suffix_list] (if different dataset has different suffix) 
                data_suffix = '.png', 
                **kwargs)

For the above parameters, ``data_path`` can be specified in dataset config if there is only one dataset. 
It can also be specified through ``data_path_list`` in loader config. ``data_transform`` need to be specified in loader block.
Other parameters should be specified in dataset config.

ImgClsDataset
^^^^^^^^^^^^^^

``ImgClsDataset`` defines a classification dataset contains images and the corresponding class labels.

.. code-block:: python
    :linenos:

    ImgClsDataset.__init__(self, 
                data_path, 
                dim = 2,
                data_transform = None, 
                label_transform = None,
                mode = 'train', 
                data_suffix = '.png',
                ## pre-shuffle the samples, because the samples are load by categories.
                ## you can also do the shuffle in data-loader 
                pre_shuffle = True,
                ## map class names to class labels
                ## also, samples should be store in the subdirs of data_path whose names are the corresponding class names
                cls_label = {},
                **kwargs)


ImgSegDataset
^^^^^^^^^^^^^^

``ImgSegDataset`` defines a dataset contains images and segmentation maps. Each image corresponds to one target with the same shape.

.. code-block:: python
    :linenos:
    
    ImgSegDataset.__init__(self, 
                    ## data_path, can be specified in dataset[data_path] or loader[data_path_list]
                    data_path, 
                    ## dimension of image, can be 2 or 3
                    dim = 2, 
                    ## data transforms, specified in loader[data_transforms]
                    data_transform = None,
                    ## label transforms, specified in loader[label_transforms]
                    ## if loader[label_transforms] was not specified, 
                    ## we would perform necessary transforms based on data transforms.
                    ## necessary transforms indicate those transforms should be performed on data and label simultaneously, such as resize and crop
                    label_transform = None,  
                    ## sub dir for image data in data_path, we only read image in data_dir
                    data_dir = 'raw', 
                    ## sub dir for image label in data_path, we only read label in label_dir
                    label_dir = 'label', 
                    ## data_suffix, we only read data with specified data_suffix
                    ## can be specified in dataset[data_suffix] or loader[data_suffix_list]
                    data_suffix='.png', 
                    ## label_suffix, we only read label with specified label_suffix
                    ## can be specified in dataset[label_suffix] or loader[label_suffix_list]
                    ## if not specified, use the same setting as data_suffix
                    label_suffix = None, 
                    ## crop the non-zero region by image or label while keeping some margin
                    ## crop_nonzero should be a dict like: {'crop_by': label, 'margin':10}
                    crop_nonzero = None, 
                    **kwargs)

MultiModalityImgSegDataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``MultiModalityImgSegDataset`` defines a dataset contains multiple modalities (such as `BraTS <http://braintumorsegmentation.org/>`_), 
which means it may contains images from different imaging devices and labels of different organs, tissues or tumors.
It has a same initialization function with ImgSegDataset, but some of the parameters can be list. 
``MultiModalityImgSegDataset`` will load images and labels from all the listed sub directories and combine them to a dataset.

.. code-block:: python
    :linenos:
    
    MultiModalityImgSegDataset.__init__(self, 
                    data_path, 
                    dim = 2, 
                    data_transform = None,
                    label_transform = None,  
                    ## the following dir and suffix can be a list
                    data_dir = 'raw', 
                    label_dir = 'label', 
                    data_suffix='.png', 
                    label_suffix = None, 
                    crop_nonzero = None,
                    ## how to combine data of different modalities, can be mean, sum or cat
                    data_combine = 'mean',
                    ## how to combine label of different modalities, can be mean, sum or cat
                    ## if not specified, use the same combining method as data_combine
                    label_combine = None,
                    **kwargs)

The following configuration define a loader for BraTS21 dataset:

.. code-block:: yaml
  :linenos:

  loader:
    dataset: 
      name: MultiModalityImgSegDataset
      dim: 3
      data_dir: [flair, t1, t1ce, t2]
      data_suffix: [flair.nii.gz, t1.nii.gz, t1ce.nii.gz, t2.nii.gz]
      label_dir: seg
      label_suffix: seg.nii.gz
      data_combine: cat
      crop_nonzero:
        margin: [2,2,2]
        crop_by: raw
    data_path_list: 
      - /work/guoqingzhang/datasets/biomed_3d_dataset/BraTS2021/fold1
      - /work/guoqingzhang/datasets/biomed_3d_dataset/BraTS2021/fold2
    batch_size: 4
    num_workers: 8
    shuffle: false
    data_transforms:
      - name: Resize
        size: [120, 192, 120]
      - name: ToTensor
    label_transforms:
      - name: Resize
        size: [120, 192, 120]
      - name: Relabel
        map: 
          - [4, 3]
      - name: ToOneHot
        num_classes: 4
        ignore_background: False
      - name: ToTensor

Point Cloud Datasets
---------------------

PcdDataset
^^^^^^^^^^^^^^

``PcdDataset`` defines a dataset contains only point clouds without any label or condition information.

.. code-block:: python
    :linenos:

    PcdDataset.__init__(self, 
                data_path, 
                data_transform = None, 
                data_dir = '', 
                data_suffix = '.ply', 
                **kwargs)

PcdClsDataset
^^^^^^^^^^^^^^

``PcdClsDataset`` defines a classification dataset contains point clouds and the corresponding class labels.

.. code-block:: python
    :linenos:

    PcdClsDataset.__init__(self, 
                data_path, 
                data_transform = None, 
                label_transform = None,
                mode = 'train', 
                data_suffix = '.ply',
                pre_shuffle = True,
                ## a dict that maps class names to class labels.
                ## we pre-define the dicts of shapenet and medshapes,
                ## therefore, cls_label can also be 'shapenet' or 'medshapes'. 
                cls_label = {},
                **kwargs)


PcdSegDataset
^^^^^^^^^^^^^^

``PcdSegDataset`` defines a dataset contains images and segmentation labels.

.. code-block:: python
    :linenos:
    
    PcdSegDataset.__init__(self, 
                data_path, 
                data_transform = None, 
                label_transform = None, 
                mode = 'train', 
                data_dir = 'pcd', 
                label_dir = 'label', 
                data_suffix = '.ply', 
                label_suffix='.seg', 
                **kwargs):

PcdReconWithClassLabelDataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``PcdReconWithClassLabelDataset`` defines a completion dataset contains partial or noisy point clouds, reconstruction targets, and corresponding category labels.

.. code-block:: python
    :linenos:
    
    PcdReconWithClassLabelDataset.__init__(self, data_path, 
                 data_transform = None, 
                 ## transform for class labels
                 label_transform = None, 
                 target_transform = None,
                 mode = 'train', 
                 data_dir = 'partial', 
                 target_dir = 'target', 
                 data_suffix = '.ply', 
                 target_suffix='.ply', 
                 cls_label = {},
                 pre_shuffle = True,
                 **kwargs):

The following block defines a loader for MedPointS completion dataset:

.. code-block:: yaml
    :linenos:

    loader:
      dataset: 
        name: PcdReconWithClassLabelDataset
        data_dir: partial
        target_dir: target
        data_suffix: .ply
        target_suffix: .ply
        cls_label: MedPointS
      data_path_list: 
        - /media/wlsdzyzl/DATA/datasets/pcd/MedPointS/completion/fold1
        - /media/wlsdzyzl/DATA/datasets/pcd/MedPointS/completion/fold2
        - /media/wlsdzyzl/DATA/datasets/pcd/MedPointS/completion/fold3
      batch_size: 64
      num_workers: 8
      shuffle: true
      data_transforms:
        - name: Normalize
        - name: FixedPoints
          num: 2048
        - name: ToTensor
          dtype: float
      target_transforms:
        - name: Normalize
        - name: FixedPoints
          num: 2048
        - name: ToTensor
          dtype: float
      label_transforms:
        - name: ToOneHot
          num_classes: 47
          ignore_background: true
        - name: ToTensor
          dtype: float

Data Augmentations
===================

2D Image Data Augmentation
---------------------------

For 2D image, we adopt the following common transforms from ``torchvision.transforms``:

.. code-block:: console

  ToTensor
  RandomHorizontalFlip
  RandomVerticalFlip
  Normalize
  RandomRotation
  GaussianBlur
  CenterCrop
  RandomCrop

The required parameters can refer to `torchvision.transforms <https://pytorch.org/vision/0.9/transforms.html>`_.

Beside of these, we also implement the following transforms (some of them are wrapped to have the same parameters as their 3D counterparts):

.. code-block:: python
  :linenos:

  # resize image to certain shape
  ## size should be a list
  ## mode should be one of ['nearest', 'bilinear', 'bicubic']
  Resize.__init__(self, size, mode = 'nearest')
  # to one hot label
  ## number of classes
  ## if ignore_background is true, the one-hot encoding of background (zero values) will be zero vectors
  ToOneHot.__init__(self, num_classes = None, ignore_background = False, **kwargs)
  # to binary mask
  ## all values larger than threshold will be set as 1, others will be set as 0.
  ToBinaryMask.__init__(self, threshold=0)
  # to gray image
  ## some version of torch vision doesn't contain this transforms
  ## out_channel is the number of channel of output image
  GrayScale.__init__(self, out_channel = 1)
  # inverse color: white to black and black to white
  InverseColor.__init__(self)
  # Relabellabels into a consecutive numbers: [10, 10, 0, 6, 6] -> [2, 2, 0, 1, 1].
  ## a optional map can be provided. Map should be a list with a shape of (n, 2).
  ## A map like [[4, 3], [8, 4]] will relabel [4, 8] to [3, 4].
  Relabel.__init__(self, map = [], **kwargs)
  # perform elastic deformation on image
  ## parameters can refer to elastic deformation. Default values are good choices.
  ## for label transform, set spline_order = 1
  ElasticDeform.__init__(self, spline_order = 3, 
                alpha=2000, 
                sigma=50, 
                execution_probability=0.1)

For 2D image, most of the transforms need to be called after ``ToTensor`` because they should be performed on tensor.

3D Image Data Augmentation
---------------------------

We adopt and implement some common augmentations for 3D images.

.. code-block:: python
  :linenos:

  # Randomly flips the image.
  ## axis_prob define the flip probability for each axis
  RandomFlip.__init__(self, axis_prob=0.5)
  # Rotate an array by 90 degrees around z-axis
  RandomRotate90.__init__(self)
  # Rotate an array by a random degrees from taken from (-angle_spectrum, angle_spectrum) interval
  ## Rotation axis is picked at random from the list of provided axes.
  ## mode should be one of ['reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'] from scipy
  RandomRotate.__init__(self, angle_spectrum=30, axes=None, mode='reflect', order=0)
  # Adjust contrast by scaling each voxel to `mean + alpha * (v - mean)`.
  RandomContrast.__init__(self, alpha=(0.5, 1.5), mean=0.0, execution_probability=0.1)
  # elastic deformation, similar to the 2D counterpart
  ## if apply_3d is false, elastic deformation will be performed on each 2D slices
  ElasticDeform._init__(self, spline_order = 3, alpha=2000, sigma=50, 
            execution_probability=0.1, apply_3d=True)
  # crop width and height (x, y) to fixed shape
  ## centered: always crop center region
  CropToFixed.__init__(self, size=(256, 256), centered=False)
  # normalize image with mean and std.
  ## if mean and std are not specified, they will be computed based on the image
  ## if channelwise is true and image has multiple channels, 
  ## image will be normalized in a channelwise manner.
  Normalize.__init__(self, mean=None, std=None, channelwise=False)
  # apply simple min-max normalization
  ## if min_value and max_value are not specified, they will be computed based on the image
  ## if norm01 is true, image will be normalized to [0, 1]. 
  ## Otherwise, it will be normalized to [-1, 1]
  MinMaxNormalize.__init__(self, min_value=None, max_value=None, 
    norm01=True, channelwise=False)
  # resize the volume, similar to the 2D counterpart
  Resize.__init__(self, size, mode = 'nearest')
  # Converts a given input numpy.ndarray into torch.Tensor.
  ## expand_dims (bool): if True, adds a channel dimension to the input data
  ## dtype (np.dtype): the desired output data type
  ToTensor.__init__(self, expand_dims=True, dtype=np.float32)
  # relabel, similar to the 2D counterpart
  Relabel.__init__(self, map = [], **kwargs)
  # to one hot label, similar to the 2D counterpart
  ToOneHot.__init__(self, num_classes = None, ignore_background = False)
  # perform Gaussian blur with a certain probability.
  GaussianBlur.__init__(self, sigma=[.1, 2.], execution_probability=0.5)
  # perform binary closing for several iterations
  RemoveSmallGap.__init__(self, iterations)
  # perform binary opening for several iterations
  RemoveThinConnection.__init__(self, iterations)
  # to binary mask, similar to the 2D counterpart
  ToBinaryMask.__init__(self, threshold=0)

Point Cloud Augmentation
-------------------------

We adopt and implement some common augmentations for point clouds.

.. code-block:: python
  :linenos:

  # Numpy to tensor
  ToTensor.__init__(self, dtype = 'float')
  # Centers and normalizes node positions to the interval :math:`(-1, 1)`.
  # method should be one of ['minmax', 'mean']
  Normalize.__init__(self, method = 'minmax')
  # Samples a fixed number of points and features from a point cloud.
  FixedPoints.__init__(self, num, replace=True)
  # Transforms node positions with a square transformation matrix computed offline.
  LinearTransformation.__init__(self, matrix)
  # Rotates node positions around a specific axis by a randomly sampled factor within a given interval.
  #  Args:
  #    degrees (tuple or float): rotation degree
  #    axis (int, optional): The rotation axis. (default: :obj:`0`)
  Rotate.__init__(self, degree, axis=0)
  # Rotates node positions around a specific axis by a randomly sampled factor within a given interval.
  #    Args:
  #        degrees (tuple or float): Rotation interval from which the rotation
  #            angle is sampled. If :obj:`degrees` is a number instead of a
  #            tuple, the interval is given by :math:`[-\mathrm{degrees},
  #            \mathrm{degrees}]`.
  #        axis (int, optional): The rotation axis. (default: :obj:`0`)
  RandomRotate.__init__(self, degrees, axis=0)
  # Add gaussian noise to points
  AddNoise.__init__(self, std=0.01)
  # Add gaussian noise with random std to points
  AddRandomNoise.__init__(self, std_range=[0, 0.10])
  # Scales node positions by a randomly sampled factor s within given interval, *e.g.*, resulting in the transformation matrix
  RandomScale.__init__(self, scales)
  # Translates node positions by randomly sampled translation values within a given interval. 
  # In contrast to other random transformations, translation is applied separately at each position.
  #    Args:
  #        translate (sequence or float or int): Maximum translation in each
  #            dimension, defining the range
  #            :math:`(-\mathrm{translate}, +\mathrm{translate})` to sample from.
  #            If :obj:`translate` is a number instead of a sequence, the same
  #            range is used for each dimension.
  RandomTranslate.__init__(self, translate)
  # Shuffle order of points in point cloud
  ShufflePoints.__init__(self)
  # reorder points by a specified axis
  ReorderByAxis.__init__(self, axis=0)
  # reorder points by Hilbert curve
  ReorderByHilbert.__init__(self, bins = 16, radius = 1.0, origin = (0,0,0))
  # To one hot label, background value should be 0
  ToOneHot.__init__(self, num_classes = None, ignore_background = False, **kwargs)