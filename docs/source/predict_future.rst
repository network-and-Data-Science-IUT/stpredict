predict_future
==============

**Description**

training the model on the training set and predict the target variable values in the future.

**Usage**

.. py:function:: predict.predict_future(data, future_data, forecast_horizon, feature_or_covariate_set, model = 'knn', base_models = None, model_type = 'regression', model_parameters = None, feature_scaler = None, target_scaler = None, labels = None, scenario  = 'current', save_predictions = True, verbose = 0)


**Parameters**

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: predict_future_in.csv

**Returns** 

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: predict_future_out.csv

**Example** 

.. code-block:: python

   import pandas as pd
   from stpredict.predict import predict_future
   
   df = pd.read_csv('./historical_data h=1.csv')
   data = df.iloc[:-120,:]
   future_data= df.iloc[-120:,:]
   trained_model = predict_future(data = data, future_data = future_data , forecast_horizon = 4,
                   feature_or_covariate_set = ['temperature', 'population', 'social distancing policy'],
                   model = 'knn', model_type = 'regression')

