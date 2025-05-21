==================
histopreprocessing
==================


.. image:: https://img.shields.io/pypi/v/histopreprocessing.svg
        :target: https://pypi.python.org/pypi/histopreprocessing

.. image:: https://img.shields.io/travis/voreille/histopreprocessing.svg
        :target: https://travis-ci.com/voreille/histopreprocessing

.. image:: https://readthedocs.org/projects/histopreprocessing/badge/?version=latest
        :target: https://histopreprocessing.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




Preprocessing CLIs for WSIs


* Free software: MIT license
* Documentation: https://histopreprocessing.readthedocs.io.

TODO
--------
- [ ] Make the different tiling methods DRYer
- [ ] Implement unit tests especally for the basic functions, hope basic tiling still works since I Changed the map_masks_to_wsi function
- [ ] Don't bother with renaming slides IDs


Features
--------

* TODO



Installation
------------

To install **histopreprocessing** locally, follow these steps:

1. Clone the repository:

   ::

       git clone https://github.com/voreille/histopreprocessing.git

2. Change to the project directory:

   ::

       cd histopreprocessing

3. Install the package in editable mode:

   ::

       pip install -e .

.. warning::

   It is recommended to use a Python environment manager such as Conda or Virtualenv to avoid conflicts with your system packages.


After installation, you can run the CLI commands using the **histopreprocessing** command followed by the task name. For example, see the **Usage** section for detailed instructions on commands like:

   ::

       histopreprocessing run-histoqc
       histopreprocessing rename-masks
       histopreprocessing tile-wsi
       ...

Usages
--------
Command Usages
--------------

Below are usage examples for each command as configured in the launch.json:

1. **run_histoqc**

   This command executes HistoQC on raw WSIs. It processes images found in the raw WSI directory and outputs mask files to the specified output directory.
   
   **Example:**
   
   ::
   
       histopreprocessing run-histoqc \
         --raw-wsi-dir data/tcga_test \
         --output-dir data/masks_test \
         --num-workers 4

2. **rename_masks**

   This command renames mask directories based on the provided WSI identifier mapping style.
   
   **Example:**
   
   ::
   
       histopreprocessing rename-masks \
         --masks-dir data/masks_test \
         --wsi-id-mapping-style TCGA

3. **tile_wsi**

   This command generates tiles from WSIs using the HistoQC mask outputs. You can adjust tile size, coverage threshold, and the number of worker processes.
   
   **Example:**
   
   ::
   
       histopreprocessing tile-wsi \
         --raw-wsi-dir data/tcga_test \
         --masks-dir data/masks_test \
         --output-dir data/tile_test \
         --tile-size 224 \
         --threshold 0.5 \
         --num-workers-tiles 4 \
         --num-workers-wsi 4 

4. **write_metadata**

   This command generates metadata for the tiles and writes the output to a JSON file.
   
   **Example:**
   
   ::
   
       histopreprocessing write-tiles-metadata \
         --tiles-dir data/tile_test \
         --output-json data/tile_test/metadata.json

5. **superpixel_segmentation**

   This command performs superpixel segmentation on the WSIs using the corresponding mask images and saves the results to the specified output directory.
   
   **Example:**
   
   ::
   
       histopreprocessing superpixel-segmentation \
         --raw-wsi-dir data/tcga_test \
         --masks-dir data/masks_test \
         --output-dir data/superpixel_test \
         --num-workers 12

6. **tile_wsi_from_superpixel_no_overlap**

   This command generates non-overlapping tiles from WSIs using outputs from a superpixel segmentation. It requires directories for raw WSIs and superpixel results, and an output directory for the tiles.
   
   **Example:**
   
   ::
   
       histopreprocessing tile-wsi-from-superpixel-no-overlap \
         --raw-wsi-dir data/tcga_test \
         --superpixel-dir data/superpixel_test \
         --output-dir data/superpixel_tiling_no_test \
         --num-workers-tiles 4 \
         --num-workers-wsi 4 

7. **tile_wsi_from_superpixel_random_overlap**

   This command generates tiles using a random overlap method based on superpixel segmentation outputs. It allows you to process WSIs with an element of randomness in tile extraction.
   
   **Example:**
   
   ::
   
       histopreprocessing tile-wsi-from-superpixel-random-overlap \
         --raw-wsi-dir data/tcga_test \
         --superpixel-dir data/superpixel_test \
         --output-dir data/superpixel_tiling_ro_test \
         --num-workers-tiles 4 \
         --num-workers-wsi 4 

8. **create_superpixel_tile_mapping**

   This command creates a mapping between superpixels and their corresponding tiles, saving the results as a JSON file.
   
   **Example:**
   
   ::
   
       histopreprocessing create-superpixel-tile-mapping \
         --tiles-dir data/superpixel_tiling_ro_test \
         --output-json data/superpixel_ro_mapping.json \
         --num-workers 12


TASKS
--------
- [ ] Check HistoQC with a mapping config/wsi, check the error.log and such
- [ ] remove the need fotr the raw_wsi_path.csv

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
