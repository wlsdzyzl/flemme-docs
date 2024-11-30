Play with Flemme
==================

After installing flemme, you can test it with some small datasets to get an intuition about its capabilities.

Toy Example for Diffusion model
-------------------------------
Configuration file: `resources/toy_ddpm.yaml <https://github.com/wlsdzyzl/flemme/tree/main/resources/toy_ddpm.yaml>`_.

.. code-block:: console

  train_flemme --config resources/toy_ddpm.yaml

.. image:: _static/ddpm_toy.png

MINST
------ 

Configuration files are in  `resources/img/mnist <https://github.com/wlsdzyzl/flemme/tree/main/resources/img/mnist>`_.

AutoEncoder & Variational AutoEncoder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: _static/ae_mnist.png

Denoising Diffusion Probabilistic Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: _static/ddpm_mnist.png

CIFA10 
-------

Configuration files are in `resources/img/cifar10 <https://github.com/wlsdzyzl/flemme/tree/main/resources/img/cifar10>`_.

AutoEncoder & Variational AutoEncoder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: _static/ae_cifar10.png

Denoising Diffusion Probabilistic Model (conditional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: _static/cddpm_cifar10.png