==============================
Losses and Evaluation Metrics
==============================

Flemme supports a series of loss functions for training and evaluation metrics for testing.

Losses
=======

Loss are defined in model construction like follows:

.. code-block:: yaml
  :linenos:
  
  model:
  ## model architecture, SeM indicates Segmentation Model
    name: SeM
    # loss function of architecture. For SeM, you need to specify the segmentation loss.
    segmentation_losses: 
      - name: Dice  
        weight: 1.0
      - name: BCEL
        weight: 1.0
    # loss reduction, can be one of [mean, sum, null]
    loss_reduction: mean
    encoder:
      ## define a encoder you want ...

Although several losses in Flemme are based on torch, it's worth noting that all losses are computed per-sample in Flemme. This is a little different compared to torch's original implementation. 
For example, given a batch of images :math:`\mathbf x \in \mathbb R ^{N \times C \times W \times H}` and we compute the MSE loss without any reduction,
torch will compute loss per pixel and return a loss with shape :math:`N \times C \times W \times H` while flemme would return a loss with shape :math:`N \times 1`.
We want to compute the loss in a same manner as other losses such as Dice that are usually computed over a whole image.

The remainder of this section introduces the supported loss functions.

Segmentation
-------------
``Dice`` refers to Dice Loss, which is defined as 1 minus Dice score:

.. math::
  \mathcal L _\text{Dice}  = 1 - \frac{2 \cdot X \bigcap Y}{ \vert X\vert + \vert Y \vert}

You can specify some parameters based on the initialization of ``DiceLoss``:

.. code-block:: python
  :linenos:
  
  ## Normalization should be set as sigmoid and softmax for binary and multi-class segmentation, respectively.
  ## For point cloud segmentation, the channel_dim should be -1.
  DiceLoss.__init__(self, reduction = 'mean', normalization = 'sigmoid', channel_dim = 1)

``BCE`` and ``BCEL`` refers to ``BCELoss`` and ``BCEwithLogitsLoss`` in torch (binart cross-entropy loss), respectively. It's usually used for binary segmentation.


``CE`` refers to cross-entropy loss (``CrossEntropyLoss`` in torch) which are used for multi-class segmentation.

Loss implemented based on torch follows the following initialization:

.. code-block:: python
  :linenos:
  
  ## torch_loss is automatically determined by the loss name.
  TorchLoss.__init__(self, torch_loss = nn.MSELoss, reduction = 'mean', channel_dim = 1)

You only need to specify the reduction and channel_dim (1 for image and -1 for point cloud).


Reconstruction
---------------
``MSE`` refers to the mean squared error, based on torch's ``MSELoss``.

``L1`` refers to the mean absolute error, based on torch's ``L1Loss``.

``KL`` refers to KL divergence for comparing two Gaussian distributions:

.. code-block:: python
  :linenos:

  KLLoss.__init__(self, reduction = 'mean')

Given two Gaussian distribution :math:`N(\mu_1, \sigma_1^2)` and :math:`N(\mu_2, \sigma_2^2)`, the KL loss can be computed by:

.. math::

  \mathcal L_\text{KL} = \log \frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 -\mu_2)^2}{2 \sigma_2^2} - \frac{1}{2}

``EMD`` refers to Earth Mover's distance for comparing two point clouds, also known as **Wasserstein distance**. It calculates the minimal cost of transfering distribution :math:`A` to distribution :math:`B` through optimal transport. 
To measure the distance between two point clouds, the point clouds are considered as uniform distributions. In Flemme, ``EMD`` is approximated through auction algorithm. You can specify the reduction and number of iterations when defining the loss.

.. code-block:: python
  :linenos:

  EMDLoss.__init__(self, reduction = 'mean', eps = 1e-8, iters = 500)

``CD`` or ``Chamfer`` refers to Chamfer's distance for comparing two point clouds that is defined by the following equation: 

.. math::
  \mathcal L_\text{Chamfer} = \frac{1}{2 \cdot \vert P_1 \vert}\sum_{x \in P_1} \min_{y \in P_2}\Vert x - y\Vert^2 + \frac{1}{2 \cdot \vert P_2 \vert}\sum_{x \in P_2} \min_{y \in P_1}\Vert x - y\Vert^2 

You only need to specify the reduction for Chamfer loss:

.. code-block:: python
  :linenos:
  
  ChamferLoss.__init__(self, reduction = 'mean')

Evaluation Metrics
===================
The most significant difference between loss functions and evaluation metrics is that loss functions are computed over tensors, and evaluation are computed over numpy array.
Therefore, some of the metrics may not be differentiable. Flemme support a series of metrics for the evaluation of segmentation and reconstruction quality. 

The following block in configuration files defines the evaluation_metrics:

.. code-block:: yaml
  :linenos:

  # evaluate test dataset
  evaluation_metrics:
    seg:
      - name: Dice
      - name: ACC
      - name: mIoU

Specially, you can specify a metric in the ``evaluation_metrics`` as ``score_metric`` through following block, to select the model with best evaluation score (saved as ``ckp_best_score.pth``).

.. code-block:: yaml
  :linenos:

  score_metric:
    name: Dice
    higher_is_better: true

The remainder of this section introduces the supported metrics for evaluation.

Segmentation
-------------
``Dice`` refers to Dice score. The higher score indicates a better segmentation result.

.. code-block:: python
  :linenos:

  # channel dim = None means there is no channel dimension, because one-hot label is transferred into common label map.
  Dice.__init__(self, channel_dim = None)

``ACC`` refers to accuracy. The higher score indicates a better segmentation result.

.. code-block:: python
  :linenos:

  ACC.__init__(self, channel_dim = None)


``mIoU`` refers to the mean intersection of units. The higher score indicates a better segmentation result.

.. code-block:: python
  :linenos:

  mIoU.__init__(self, channel_dim = None)

``HD`` refers to hausdorff distance.  The lower value indicates a better segmentation result. 

.. code-block:: python
  :linenos:

  # method can be standard or modified
  HD.__init__(self, channel_dim = None, method='standard')

``SegARI`` refers to adjusted rand index (ARI) for **binary** segmentation. ARI is used to evaluate cluster results. 
We transfer label map through `scipy.ndimage.label` to get the connected components and compute ARI based on the transferred maps.
The ``SegARI`` is more sensitive for broken connections and wrong topologys.

.. code-block:: python
  :linenos:
  
  # if segmentation is boundary or has string-like shapes, we inverse the binary label map.
  # dim should be 2 and 3 for 2D and 3D images, respectively.
  SegARI.__init__(self, boundary = True, dim = 2, channel_dim=None)

Reconstruction
---------------
``MSE`` refers to the mean squared error, you don't need to specify any other parameters. The lower value indicates a better reconstruction result. 

``SSIM`` refers to structural similarity. The high value indicates a better reconstruction result.

.. code-block:: python
  :linenos:
  
  # data_range is set to the difference between the maximum and minimum values ​​supported by the image.
  SSIM.__init__(self, data_range = None, channel_dim = 0)

``PSNR`` refers to peak signal-to-noise ratio. The high value indicates a better reconstruction result.

.. code-block:: python
  :linenos:
  
  PSNR.__init__(self, data_range = None)

``EMD`` refers to the Earth Mover's distace for **point cloud**. The lower value indicates a better reconstruction result.

.. code-block:: python
  :linenos:
  
  EMD.__init__(self)

``CD`` or ``Chamfer`` refers to Chambfer distance for **point cloud**. The lower value indicates a better reconstruction result.

.. code-block:: python
  :linenos:

  # metric = l2 or l1, for l2 and l1 norm distance
  # direction should be one of ['bi', 'x_to_y' or 'y_to_x'].
  CD.__init__(self, metric = 'l2', direction = 'bi')

