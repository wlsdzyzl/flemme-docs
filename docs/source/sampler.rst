========
Sampler
========

For generative models, we provide a ``NormalSampler`` to sample random points in latent space from a standard normal distribution for generation.
A sampler is bound to the actual generative model. To create a sampler, you have following parameters that need to be specified:

.. code-block:: python

  :linenos:
    ## model that this sampler bound with.
    NormalSampler.__init__(self, model, 
                      ## rand seed
                      rand_seed = None,
                      ## Sample steps for diffusion models.
                      ### Note that for a diffusion model, a small number of sample steps makes sampling faster, but it can lead to unstable results.
                      ### Note that for DDIM, this parameter will be ignored.   
                      ### By default, the model will perform a full sampling.
                      num_sample_steps = -1,
                      ## `clipped` and `clip_range` are also for diffusion models.
                      ## Clipping the noisy image to the right range might make the sampling process more stable. 
                      ## The clip range should be the numerical range of input elements.
                      clipped = None, 
                      clip_range = None, 
                      **kwargs):

You can create a sampler in training process to vasualize the generated results like follows:

.. code-block:: yaml
  :linenos:

  sampler:
    num_sample_steps: 100
    rand_seed: 2024
    clipped: true
    clip_range: [-1.0, 1.0]

The full configuration file can refer to `train_unet_ddim.yaml <train_unet_ddim.html>`_

Sampler can also be created in test process to evaluate and save the generated results as shown in `test_unet_ddpm.yaml <test_unet_ddpm.html>`_.

.. code-block:: yaml
  :linenos:

  eval_gen:
    gen_dir: path/to/results/gen/mnist/ddpm
    sampler:
      clipped: true
      clip_range: [-1.0, 1.0]
      num_sample_steps: 1000
    random_sample_num: 100

Sampler can be created for all supported generative architectures, including ``VAE``, ``DDPM``, ``DDIM``, ``LDPM``, ``LDIM``, ``SDPM``, and ``SDIM``.