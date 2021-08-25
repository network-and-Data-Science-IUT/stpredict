make_historical_data
====================

**Description**

Transforming input data to the historical format and extract features. This function prepares the reformed data frame including features and target variable for modeling. The set of features consists of spatial covariates, temporal covariates at current temporal unit (t) and historical values of these covariates at h-1 previous temporal units (t-1 , t-2 , … , t-h+1). The target of the output data frame is the values of the target variable at the temporal unit t+r, where h and r denote the user specified history length and forecast horizon. In addition, if the user prefers to output data frame(s) include the values of some covariates in the future temporal units, the name of these covariates could be specified using the futuristic_covariates argument.

**Usage**

.. py:function:: preprocess.make_historical_data(data, forecast_horizon, history_length = 1, column_identifier = None, futuristic_covariates = None, future_data_table = None, step = 1, verbose = 0)

**Parameters**

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: make_historical_data_in.csv

.. note:: We assume that there is no gap in the time sequences of input *data*.

**Returns** 

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: make_historical_data_out.csv

**Example** 

.. code-block:: python

   import pandas as pd
   from stpredict.preprocess import make_historical_data()

   df1 = pd.read_csv('USA COVID-19 temporal data.csv')
   df2 = pd.read_csv('USA COVID-19 spatial data.csv')


   historical_data_frame = make_historical_data(data = {'temporal_data':df1,'spatial_data':df2},
                                                forecast_horizon = 4, 
                                                history_length = {('temperature','precipitation'):2})


