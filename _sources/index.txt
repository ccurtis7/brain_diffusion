.. brain_diffusion documentation master file, created by
   sphinx-quickstart on Wed Mar 14 08:29:02 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to brain_diffusion
===========================================

brain_diffusion is a python library designed to analyze and visualize trajectory
data from the ImageJ plugin `MOSAIC <http://mosaic.mpi-cbg.de/?q=downloads/imageJ>`_.

Usage
-----

.. code-block:: python

  import brain_diffusion.msd as msd

  DIR = '../data/'
  file = 'sample_data'

  # Prepare trajectory file
  total, frames, x, y, xs, ys = msd.MSD_iteration(DIR, file)

  # Calculate mean squared displacements and their geometric averages
  geoM2xy, gSEM, SM1x, SM1y, SM2xy = msd.vectorized_MMSD_calcs(frames, total, xs, ys)


Motivation
----------

Mean squared displacement (MSD) calculations are a standard calculation performed by
scientists and researchers who use multi-particle tracking techniques.
Brain_diffusion is a centralized analysis and visualization package to calculate
MSDs and diffusion coefficients from trajectory data as calculated using the MOSIAC
ImageJ plugin developed by in the MOSAIC Group at the Max Planck Institute of
Molecular Cell Biology and Genetics in Dresden.  This calculation package is the
primary tool for MSD calculations of nanoparticles in the brain in the `Nance
research group <https://www.nancelab.com/>`_ at the University of Washington.

Installation and getting started
--------------------------------

To install brain_diffusion and analyzing your data, visit :ref:`getting-started-label`

Documentation
-------------

For full documentation of brain_diffusion, see :ref:`doc-label`.

Bugs and issues
---------------

If you are having issues, please let us know by `opening a new issue
<https://github.com/ccurtis7/brain_diffusion/issues>`_.
Please tag your issues with the "bug" or "question" label.

License
-------

The project is licensed under the `BSD 2-Clause "Simplified" License
<https://github.com/ccurtis7/brain_diffusion/blob/master/LICENSE>`_.

Acknowledgements
----------------
Brain_diffusion development is supported by ....

.. toctree::
   :hidden:

   getting started <getting_started>
   documentation <documentation>
   examples <https://github.com/ccurtis7/brain_diffusion/tree/master/brain_diffusion/notebooks>
   code <https://github.com/ccurtis7/brain_diffusion>
   bugs <https://github.com/ccurtis7/brain_diffusion/issues>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
