train_unet_sem.yaml
===================

.. code-block:: yaml
  :linenos:

  mode: train
  # contents to define a model
  model:
  ## model architecture, SeM indicates Segmentation Model
    name: SeM
    ## loss function of architecture. For SeM, you need to specify the segmentation loss.
    segmentation_losses: 
      - name: Dice  
        weight: 1.0
      - name: BCEL
        weight: 1.0
    ## define encoder and decoder
    encoder:
      ### encoder name
      name: UNet
      ### input image size (after augmented)
      image_size: [320, 256]
      ### number of input image channels
      in_channel: 3
      ### number of output channels, for segmentation, it should be the number of categories
      out_channel: 1
      ### number of channel for image patches
      patch_channel: 32
      ### size of image patch, for 2D image, 'patch size = 2' indicates a 2*2 image patch.
      patch_size: 2
      ### number of channel for each layer in down-sample layers. 
      ### The length of list is the number of down-sample layers
      down_channels: [64, 128, 256]
      ### number of channel for each layer in middle layers. 
      ### The length of list is the number of middle layers
      middle_channels: [512, 512]
      ### building block
      building_block: conv
      ### normalization
      normalization: batch
  # data loader for training
  loader:
    ## define a dataset
    dataset: 
      name: ImgSegDataset
      data_dir: images
      label_dir: masks
      data_suffix: jpg
    ## merge listed datasets into one 
    data_path_list: 
      - path/to/datasets/CVC-ClinicDB/fold1/
      - path/to/datasets/CVC-ClinicDB/fold2/
      - path/to/datasets/CVC-ClinicDB/fold3/
    ### other parameters related to dataloader
    ### refer to https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader 
    batch_size: 16
    num_workers: 8
    shuffle: true
    ## augmentations performed on each sample from the dataset
    data_transforms:
      - name: Resize
        size: [320, 256]
      - name: ToTensor
  # data loader for validation
  val_loader:
    dataset: 
      name: ImgSegDataset
      data_dir: images
      label_dir: masks
      data_suffix: jpg
      data_path: path/to/datasets/CVC-ClinicDB/fold4/
    batch_size: 8
    num_workers: 8
    shuffle: false
    data_transforms:
      - name: Resize
        size: [320, 256]
      - name: ToTensor
  # define a optimizer
  optimizer:
    name: Adam
    lr: 0.0003
    weight_decay: 0.00000001
  # define a learning rate scheduler
  lr_scheduler: 
    name: LinearLR
    start_factor: 1.0
    end_factor: 0.01
  # evaluation metrics
  evaluation_metrics:
    seg:
      - name: Dice
      - name: ACC
      - name: mIoU
  
  score_metric:
    name: Dice
    higher_is_better: true

  # max training epochs
  max_epoch: 500
  # in warm-up epoch, learning rate will be fixed as the initial value
  warmup_epoch: 2
  # write intermediate results to tensorboard for visualization
  write_after_iters: 5
  # save checkpoint
  save_after_epochs: 2
  # directory for checkpoints
  check_point_dir: path/to/checkpoint/CVC-ClinicDB/UNet
