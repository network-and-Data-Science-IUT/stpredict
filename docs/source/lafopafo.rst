lafopafo
========

**Description**

Implementation of all the modeling and forecasting stages including feature selection, model selection, evaluation, and forecasting of future time units, using Last Fold Partitioning Forecaster (LaFoPaFo). The last fold partitioning forecaster uses the last 'fold' or time unit in the data to evaluate the accuracy of the models, and thus simulate the real situation of forecasting where the models are trained only using the data available up until now, and perform the prediction for future time points. As a result the training and forecasting process is done separately for each future time point and using the data available prior to this time point, and hence the model, history length, and feature set selected for each of these points are different. Similarly, the Learner evaluation step is performed for each test point separately, taking that point as the test set and all the time units before it as the training set. It should be noted that for realistic evaluation of the models, a number of time units before the test point (i.e. forecst horizon - 1 units) that represent the distance between the predictive time point and the target variable time point are removed from the training data.

**Usage**

.. py:function:: predict.lafopafo(data, forecast_horizon=1, feature_sets={'covariate': 'mRMR'}, forced_covariates=[], models=['knn'], mixed_models=[], model_type='regression', instance_testing_size=0.2, fold_total_number=5, feature_scaler=None, target_scaler=None, performance_benchmark='MAPE', performance_measures=['MAPE'], performance_mode='normal', scenario='current', validation_performance_report=True, testing_performance_report=True, save_predictions=True, save_ranked_features = True, plot_predictions=False, verbose=0)

**Parameters**

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: lafopafo_in.csv

.. Note:: In the current version, 'AIC' and 'BIC' can only be calculated for the 'glm' model and 'classification' model_type.

**Returns** 

.. csv-table::   
   :header-rows: 1
   :widths: 1 , 3, 15
   :file: lafopafo_out.csv

**Example** 

.. code-block:: python

   from stpredict.predict import lafopafo

   lafopafo(data = ['./historical_data h=1.csv', './historical_data h=2.csv',
                    './historical_data h=3.csv'], forecast_horizon = 4)
