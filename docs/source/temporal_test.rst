temporal_test
=============

**Description**

| Test for the correlation of the target variable in time.
| Options are:

   * Augmented dickey-fuller (ADF) test
   * Auto-correlation function (ACF) plot
   * Fit the autoregressive model with user-specified lags and report the coefficients


**Usage**

.. py:function:: temporal_test(data, spatial_id = None, test_type='ADF', lags = 1, column_identifier = None, saving_plot_path = './')

**Parameters**

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: temporal_test_in.csv

**Returns** 

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: temporal_test_out.csv

.. Note:: The implementation of the statsmodels package is used for all tests.

**Example** 

.. code-block:: python

   from stpredict.preprocess import temporal_test
   from stpredict import load_earthquake_data

   data = load_earthquake_data()

   column_identifier={'temporal id level 1':'month ID', 'spatial id level 1':'sub-region ID',
                      'target':'occurrence'}

   temporal_test(data=data, spatial_id = [1], test_type='autoreg', lags = 3, 
                 column_identifier = column_identifier, saving_plot_path = './')

