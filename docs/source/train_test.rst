train_test
==========

**Description**

training and testing process on the preprocessed data based on best configurations (best model, featureset and history)

**Usage**

.. py:function:: predict.train_test(data, instance_testing_size, forecast_horizon, feature_or_covariate_set, history_length, model='knn', base_models=None, model_type='regression', model_parameters=None, feature_scaler='logarithmic', target_scaler='logarithmic', labels=None, performance_measures=['MAPE'], performance_mode='normal', performance_report=True, save_predictions=True, verbose=0)


**Parameters**

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: train_test_in.csv

.. Note:: In the current version, 'AIC' and 'BIC' can only be calculated for the 'glm' model and 'classification' model_type.

**Returns** 

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: train_test_out.csv

**Example** 

.. code-block:: python

   import pandas as pd
   from stpredict.predict import train_test
   
   df = pd.read_csv('./historical_data h=1.csv')
   trained_model = train_test(data = df, instance_testing_size = 0.2, forecast_horizon = 4,
                              feature_or_covariate_set = ['temperature', 'population', 
                                                            'social distancing policy'])

