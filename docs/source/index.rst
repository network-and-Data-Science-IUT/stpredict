.. stpredict documentation master file, created by
   sphinx-quickstart on Tue Aug  3 03:45:58 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |br| raw:: html

   <br />


home
====

stpredict is designed to apply forecasting methods on spatio-temporal data to predict the values of a target variable in the future time points based on the historical values of the data features. The main stages of the modeling process are implemented in this package including: 

- Data preprocessing
- Data partitioning
- Feature selection
- Model selection
- Model evaluation
- Prediction

Developers
~~~~~~~~~~

Complex Network and Data Analysis Lab (cndalab)

**Authors**

Arash Marioriyad, Maryam Meghdadi, Mahdi Naderi, Arezoo Haratian

**Supervisors**

Dr. Zeinab Maleki, Dr. Pouria Ramazi


**Acknowledgements**

We would thanks Nasrin Rafiei for contribution in the development of the package.

.. toctree::
   home
   installation
   :hidden:
   :maxdepth: 2
   :caption: Getting started

.. toctree::
   preprocess
   predict
   stpredict
   :hidden:
   :maxdepth: 3
   :caption: The API

.. toctree::
   data_structure
   user-defined-model
   :hidden:
   :maxdepth: 2
   :caption: Base structures

.. toctree::
   example
   :hidden:
   :maxdepth: 2
   :caption: Example
