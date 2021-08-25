plot_data
=========

**Description**

Plot the temporal covariates evolution in the user specified time interval.

**Usage**

.. py:function:: plot_data(data, temporal_covariate = 'default', temporal_range = None, spatial_id = None, column_identifier = None, spatial_scale = 1, temporal_scale = 1,  spatial_scale_table = None,month_format_print = False, saving_plot_path = None)

**Parameters**

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: plot_data_in.csv

**Example** 

.. code-block:: python

   import pandas as pd
   from stpredict.preprocess import plot_data()

   df = pd.read_csv('USA COVID-19 temporal data.csv')

   plot_data(data = df, temporal_covariate = ['temperature'])

