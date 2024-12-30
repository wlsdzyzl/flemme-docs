train_unet_ddpm.yaml
======================

.. code-block:: yaml
  :linenos:

  # train ae or vae
  mode: test
  rand_seed: 2024
  model:
    ## DDIM or DDPM. 
    name: DDPM
    num_steps: 1000
    sample_num_steps: 100
    beta_schedule: consine
    eps_model:
      # encoder config
      time_channel: 128
      encoder:
        name: UNet
        image_size: [28, 28]
        in_channel: 1
        patch_size: 1
        down_channels: [32, 64]
        middle_channels: [128, 128]
        building_block: res
        activation: silu
        num_blocks: 2
        normalization: group
        num_norm_groups: 16
        dropout: 0.1
  model_path: path/to/checkpoint/MNIST/DDPM/ckp_last.pth
  eval_gen:
    ## generated results will be saved in `gen_dir`
    gen_dir: path/to/results/gen/mnist/ddpm
    ## define a normal sampler
    sampler:
      clipped: true
      clip_range: [-1.0, 1.0]
      num_sample_steps: 1000
    ## number of generated results
    random_sample_num: 100
