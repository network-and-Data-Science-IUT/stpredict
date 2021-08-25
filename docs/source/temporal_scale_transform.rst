temporal_scale_transform
========================

**Description**

Change the data temporal scale to the desired temporal scale specified by the user and create a data set that contains covariate values for the units of the higher-level temporal scale. To obtain the values of each covariate for each unit of the bigger scale, the values of this covariate are averaged over all the units of the smaller scale which belong to that bigger scale unit, and if the user prefers to augment the data, the moving average method will be used to obtain data with a bigger temporal scale, but almost the same volume as input data.

**Usage**

.. py:function:: preprocess.temporal_scale_transform(data, column_identifier = None, temporal_scale_level = 2, augmentation = False, verbose = 0)

**Parameters**

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: temporal_scale_transform_in.csv


**Returns** 

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: temporal_scale_transform_out.csv

**Example** 

.. code-block:: python

   import pandas as pd
   from stpredict.preprocess import temporal_scale_transform

   df = pd.read_csv('USA COVID-19 spatial data.csv')


   transformed_df = temporal_scale_transform(data = df, temporal_scale_level = 3)

