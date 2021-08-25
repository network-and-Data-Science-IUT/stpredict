spatial_scale_transform
=======================

**Description**

Change the data spatial scale to the desired spatial scale specified by the user. To obtain a data frame that contains covariate values for the units of the higher level spatial scale,  each covariate is aggregated over all the units of smaller level scale which belong to a unit of bigger level scale (desired spatial scale).The aggregation of each covariate is performed based on a specified aggregation mode (mean or sum) for that covariate.

**Usage**

.. py:function:: preprocess.spatial_scale_transform(data, data_type, spatial_scale_table = None, spatial_scale_level = 2, aggregation_mode = 'mean', column_identifier = None, verbose = 0)

**Parameters**

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: spatial_scale_transform_in.csv


**Returns** 

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: spatial_scale_transform_out.csv

**Example** 

.. code-block:: python

   import pandas as pd
   from stpredict.preprocess import spatial_scale_transform

   df = pd.read_csv('USA COVID-19 spatial data.csv')
   scales_df = pd.read_csv('spatial scales data.csv')

   transformed_df = spatial_scale_transform(data = df, data_type = 'spatial', 
                                            spatial_scale_table = scales_df, 
                                            spatial_scale_level = 3)
