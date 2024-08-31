test_unet_sem.yaml
===============
.. code-block:: yaml
  :linenos:

  mode: test
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
  # data loader for test
  loader:
    ## define a dataset
    dataset: 
      name: ImgSegDataset
      data_dir: images
      label_dir: masks
      data_suffix: jpg
    ## merge listed datasets into one 
    data_path_list: 
      - path/to/datasets/CVC-ClinicDB/fold5/
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
  # evaluate test dataset
  evaluation_metrics:
    seg:
      - name: Dice
      - name: ACC
      - name: mIoU
  model_path: path/to/checkpoint/CVC-ClinicDB/UNet/ckp_last.pth
  # directory for saving results
  seg_dir: path/to/results/seg/CVC-ClinicDB/UNet
  # save transformed target
  save_target: true
  # save transformed input
  save_input: true
  # save colorized segmentation results
  save_colorized: true
