===================
Model Architectures
===================
Flemme supports various architectures for segmentation, reconstruction and generation tasks, but they are all built from a base encoder-decoder architecture.

Base Architectures
==================

Base Architectures contains a encoder and decoder. The number of input channels of a base model is determined by the encoder, and the number of output channels is determined by the decoder. However, it contains several optional components for context embedding.

There are two main differences between condition embedding :math:`c` and time-step embedding :math:`t`. 

- :math:`c` can be encoded for both encoder and decoder
- :math:`c` will be combined with the input of encoder and decoder, while :math:`t` is the input of encoder and decoder. Actually, :math:`t` is the input of each building block in encoder and decoder.

We formulate context embedding as following equations to help you understand them:

.. math::
    \begin{equation}
    \label{equ:forward}
    \begin{split}
    &z = \mathcal{E}\left(x + \mathcal{C}_e (c), \mathcal{T}(t)\right),\\
    &\mathcal{M}(x, c, t) = \mathcal{D}\left( z + \mathcal{C}_d (c), \mathcal{T}(t )\right),
    \end{split}
    \end{equation}

We recommend to use :math:`t` to encode time-step for diffusion models and :math:`c` for common conditions. Context embeddings can also be defined in the configuration files.


Condition embedding
-------------------

Defining a condition embedding in model config looks like:

.. code-block:: yaml
   :linenos:

    condition_embedding:
      # how to combine condition embedding and input. Can be one of ['cat', 'add']
      combine_condition: cat
      # merge time-step embedding and context embedding, default: false
      # if you set this as true, 
      # make sure context embedding is a embedding vector and can be merged.
      merge_timestep_and_context: false
      # condition embedding for encoder
      # usually, you need to specity the output channel
      # if you use a supported encoder for encoding, 
      # the output channel will be determined by the parameters
      encoder:
        name: Identity
        out_channel: 1
      # contition embedding for decoder
      decoder:
        ## define a cnn encoder
        name: CNN
        image_size: [320, 256]
        in_channel: 3
        patch_channel: 32
        patch_size: 2
        down_channels: [64, 128, 256]
        middle_channels: [512, 512]
        fc_channels: [128]
        building_block: conv
        normalization: batch

In the above, for condition embedding of encoder, we simply concat the input and condition. For condition embedding of decoder, we use an CNN encoder to encode the image.


Time-step embedding
-------------------
For time-step, we use sinusoidal positional embedding. You just need to specify the feature channel of time-step in model configuration.

.. code-block:: yaml
   :linenos:

    model:
      name: Base
      # time step embedding
      time_channel: 128
      # condition embedding
      condition_embedding:
        combine_condition: cat
        merge_timestep_and_context: false
        encoder:
          name: Identity
          out_channel: 1
      ## there is no condition embedding for decoder
      # encoder configuration
      encoder:
        name: UNet
        image_size: [320, 320]
        in_channel: 1
        out_channel: 1
        proj_channel: 32
        proj_scaling: 2
        down_channels: [64, 128, 256]
        middle_channels: [512, 512]
        building_block: double
        activation: relu
        abs_pos_embedding: false
        normalization: group
        num_group: 16
        num_block: 1
Segmentation
============

Reconstruction
==============

Generation
==========