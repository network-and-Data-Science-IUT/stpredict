data_preprocess
===============

**Description**

Transform data to the user defined format and prepare it for modeling.
The preprocessing procedure has several steps. First the imputation of missing values will be performed; and in the second step the temporal and spatial scales of data are transformed to the user's desired scale for prediction. Then in the third step, the target variable will be modified based on the user specified mode, and the last step is to reform the data to the historical format containing the historical values of input data covariates and values of the target variable at the forecast horizon. In addition, if the user prefers to output data frame(s) include the values of some covariates in the future temporal units, the name of these covariates could be specified using the futuristic_covariates argument.

**Usage**

.. py:function:: preprocess.data_preprocess(data, forecast_horizon, history_length = 1, column_identifier = None, spatial_scale_table = None, spatial_scale_level = 1, temporal_scale_level = 1, target_mode = 'normal', imputation = True, aggregation_mode = 'mean', augmentation = False, futuristic_covariates = None, future_data_table = None, save_address = None, verbose = 0)


**Parameters**

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: data_preprocess_in.csv

.. Note:: We assume that there is no gap in the time sequences of input *data*.

.. Note:: The gap in the sequence of temporal id levels is not allowed. More clearly if input *data* contains columns 'temporal id level 1','temporal id level 2', … , 'temporal id level x' , 'temporal id level x+2', the column 'temporal id level x+2' is not considered and will be removed from the data.

**Returns** 

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: data_preprocess_out.csv


**Example** 

.. code-block:: python

   import pandas as pd
   from stpredict.preprocess import data_preprocess

   df1 = pd.read_csv('USA COVID-19 temporal data.csv')
   df2 = pd.read_csv('USA COVID-19 spatial data.csv')

   historical_data = data_preprocess(data = {'temporal_data':df1,'spatial_data':df2},
                                     forecast_horizon = 2, history_length = 2,
                                     futuristic_covariates = {'Social distancing policy':[1,2]})

.. table::
.. _target tab 2:
.. table:: covariates history length in output historical data frames
   :align: center
   
   +-------------------------------------+----------------+----------------+----------------+------------------+
   |                                     |**Covariate 1** |**Covariate 2** |**Covariate 3** |**Covariate 4**   |
   |                                     +----------------+----------------+----------------+------------------+
   |                                     |**Covariate maxsimum history length**                                |
   |                                     +----------------+----------------+----------------+------------------+
   |                                     |3               |1               |5               |4                 |
   +-------------------------------------+----------------+----------------+----------------+------------------+
   |**Historical data frame number**     |**covariate history length in historical data frame**                |
   +-------------------------------------+----------------+----------------+----------------+------------------+
   |1                                    |1               |1               |1               |1                 |
   +-------------------------------------+----------------+----------------+----------------+------------------+
   |2                                    |2               |1               |2               |2                 |
   +-------------------------------------+----------------+----------------+----------------+------------------+
   |3                                    |3               |1               |3               |3                 |
   +-------------------------------------+----------------+----------------+----------------+------------------+
   |4                                    |3               |1               |4               |4                 |
   +-------------------------------------+----------------+----------------+----------------+------------------+
   |5                                    |3               |1               |5               |4                 |
   +-------------------------------------+----------------+----------------+----------------+------------------+
