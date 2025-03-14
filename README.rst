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


Features
--------

* TODO


Usages
--------
The command should be ran in the following order:

1. Run HistoQC as:
histopreprocessing run-histoqc data/tcga_test/ --output-dir data/masks_test -c path/to/config.ini --num-workers 12 

That command run HistoQC in a docker container, config.ini is the config used by HistoQC see their docs 
by default config_light.ini is used

2. Rename folder containings masks:


3. run tiling as:
histopreprocessing tile-wsi --masks-dir data/masks_test/ --output-dir data/tiles_test --num-workers 12

TASKS
--------
- [ ] remove the need fotr the raw_wsi_path.csv
- [ ] change of output folder is handled in the histoqc_task, since it 
- [ ] Refactor logging

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
