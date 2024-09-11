=======
Scripts
=======
We provide several `scripts <https://github.com/wlsdzyzl/flemme/tree/main/scripts>`_ for batch processing of datasets and performance evaluation. 

Common Datasets
================

A common dataset should have a ``dataset_path`` that contains all sub directories. Files in different sub directories may have different suffix. 
These parameters may need to be specified in the following scripts.

Extract Files
-------------
Code refers to `extract_files <https://github.com/wlsdzyzl/flemme/blob/main/scripts/extract_files.py>`_. 
This script copies or moves files from a source directory to target directory based on a template directories.
Files from source directories should have the same names with the template directory.

.. code-block:: console

    python extract_files.py --source_dir <source_dir> --template_dir <template_dir=.> --output_dir <output_dir=> --suffix <suffix=\'\'> --method <method=copy>

Crop Image
-----------
Code refers to `crop_by <https://github.com/wlsdzyzl/flemme/blob/main/scripts/crop_by.py>`_. 
This script uses bounding box to crop images. 

If you run the command with ``--separately`` option, we compute a the boundingbox for each sample based on the label or raw data (specified by ``crop_by``). The boundingbox is computed for non-background region.

If you run the command without ``--separately`` option, you can choose to specify a ``boundingbox``, or the boundingbox will be computed based on all samples. Note that all samples should have the same shape in this situation. 

.. code-block:: console
    
    python crop_by.py -p <dataset_path> --sub_dirs <sub_dirs=.> --suffix <suffix=\'\'> --crop_by <crop_by=None> -o <output_dir=.> -m <margin = 20,20,20>  -b <background = 0.0> --boundingbox <boundingbox = None>

Print Informations about Dataset
--------------------------------
Code refers to `get_info_from_label <https://github.com/wlsdzyzl/flemme/blob/main/scripts/get_info_from_label.py>`_. 
This script prints the max, min, average size, average label counts of samples.

.. code-block:: console

    python get_info_from_label.py -d <label_dir> --suffix <suffix=.nii.gz>

Example results can refer to `dataset3d_info.log <https://github.com/wlsdzyzl/flemme/blob/main/scripts/dataset3d_info.log>`_.

Transfer h5 to images (png)
---------------------------
Code refers to `h5_to_png <https://github.com/wlsdzyzl/flemme/blob/main/scripts/h5_to_png.py>`_. 
H5 is a dict-like file, which can store multiple keys and its corresponding contents. 
To transfer H5 file to png, you need to specify the path and sub directories of h5 dataset, 
the keys corresponding to 2D/3D images, suffix of h5 files, and output directory. 
Extracted image will be stored in the sub directories of the output directory.

.. code-block:: console

    python h5_to_png.py -p <dataset_path> -o <output_dir=.> --sub_dirs <sub_dirs=\'.\'> --suffix <suffix=.h5> --keys <keys=\'\'> '

Randomly Split a Dataset
-------------------------
Code refers to `random_split_k_fold <https://github.com/wlsdzyzl/flemme/blob/main/scripts/random_split_k_fold.py>`_. 
This script randomly split a dataset into ``k`` folds. Each fold has a same file structure with original datasets.

.. code-block:: console

    python random_split_k_fold.py -p <dataset_path> --sub_dirs <sub_dirs=.> --suffix <suffix=.> -o <output_dir=.> -k <kfold=5> --method <method=copy>

FastMRI Dataset
================
We use FastMRI to evaluate the performance of reconstruction and generation. 
Specially, we construct noisy image by masking out 90% k-space content and reconstructing through zero-filled algorithm. 
Code refers to `fastmri_masked_zero_fill <https://github.com/wlsdzyzl/flemme/blob/main/scripts/fastmri/fastmri_masked_zero_fill.py>`_.

.. code-block:: console

    python fastmri_masked_zero_fill --data_path <data_path> --output_path <output_path> --challenge <singlecoil or multicoil>

Then we can use ``h5_to_png`` script to transfer the h5 file into png images.

Performance Evaluation
=======================
We also provide a script for performance evaluation. Code refers to `test_time_and_space <https://github.com/wlsdzyzl/flemme/blob/main/scripts/unittest/test_time_and_space.py>`_.
In specific, we construct models based on the ``model_config.yaml``, and create random tensors to stimulate the forward and backward processing. 
Time and space usage are recorded and printed after the test. To run this script, we don't need to specify any parameter in the command. 

.. code-block:: console

    python test_time_and_space.py

An example of evaluation results refers to `eval_time_and_space.log <https://github.com/wlsdzyzl/flemme/blob/main/scripts/unittest/eval_time_and_space.log>`_.