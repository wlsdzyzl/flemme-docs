Global Configuration
====================
In Flemme, you can modity some global parameters in global configuration ``flemme/config.py`` to get rid of the requirements for some certain components and change the colors for visualization.

.. code-block:: python

    import logging
    module_config = {
        ### ViT and SwinTransformr, disable to get rid of [einops]
        'transformer': True,
        ### VMamba, disable to get rid of [mamba-ssm (cuda version >= 11.6)]
        'mamba': True,
        ### disable point cloud related encoders and algorithm to get rid of [plyfile, POT]
        'point-cloud': False,
        ### logger level: one of [logging.INFO, logging.DEBUG]
        'logger_level': logging.INFO,
        ### color_map for colorization of segmentation: [Scannet, Custom] or one of matplotlib's colormaps
        'color_map': 'Custom'
    }