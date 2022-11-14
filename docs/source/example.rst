COVID data example
==================

Here is an example of using the stpredict package to predict the weekly average number of COVID-19 deaths.

The first step is to load the data.


.. code-block:: python

   from stpredict import load_covid_data
   covid_data = load_covid_data()

The columns of the data are:

- Temporal id level 1: ``date``
- Temporal id level 2: ``epidemic_week`` (the date of the last day of the epidemic week) 
- Spatial id level 1: ``country``
- Target: ``covid_19_deaths``
- Temporal covariates:

   - ``covid_19_confirmed_cases``: Daily number of COVID 19 confirmed cases
   - ``precipitation``: Average daily precipitation of the country regions
   - ``temperature``: Average daily temperature of the country regions
   - ``percent_fully_vaccinated_people``: Percent of residents who are fully vaccinated (two doses)

   | 
   | The rest of covariates represent the percent change in mobility trends in different palces compared to the pre-COVID-19 period

   - ``retail_and_recreation_mobility_percent_change``
   - ``transit_stations_mobility_percent_change``
   - ``workplaces_mobility_percent_change``
   - ``grocery_and_pharmacy_mobility_percent_change`` 
   - ``residential_mobility_percent_change``
   - ``parks_mobility_percent_change``
   

We first need to transform the temporal scale of data from daily to weekly:

.. code-block:: python

   from stpredict.preprocess import temporal_scale_transform

   column_identifier={'temporal id level 1':'date', 'temporal id level 2':'epidemic_week',
                   'spatial id level 1':'country', 'target':'covid_19_deaths',
                   'temporal covariates':['covid_19_deaths', 'covid_19_confirmed_cases', 
                                          'precipitation', 'temperature', 
                                          'retail_and_recreation_mobility_percent_change',
                                          'grocery_and_pharmacy_mobility_percent_change', 
                                          'parks_mobility_percent_change', 
                                          'transit_stations_mobility_percent_change',
                                          'workplaces_mobility_percent_change', 
                                          'residential_mobility_percent_change', 
                                          'percent_fully_vaccinated_people']}
   
   weekly_data = temporal_scale_transform(data = covid_data, column_identifier = column_identifier,
                                           temporal_scale_level = 2)


| Now we can create the historical data which includes the historical values of the covariates for the past weeks. The number of past weeks is specified using the ``history_length`` argument.
| We want to predict the weekly average number of deaths for 4 weeks in the future, so the ``forecast_horizon`` is set to 4.
| The mobility changes in the coming weeks are predictable considering the social distance policies. Therefore, the values of these covariates in the next 3 weeks can be used to predict the number of deaths in the 4th week.  So we specify these covariates as ``futuristic_covariates``. 

.. code-block:: python

   from stpredict.preprocess import make_historical_data

   # update column identifier for weekly data
   column_identifier['temporal id level 1'] = 'epidemic_week'
   del column_identifier['temporal id level 2']
   
   # consider the mobility covariate values in weeks 1-3
   futuristic_covariates = {'retail_and_recreation_mobility_percent_change':[1,3],
                         'grocery_and_pharmacy_mobility_percent_change':[1,3], 
                         'parks_mobility_percent_change':[1,3], 
                         'transit_stations_mobility_percent_change':[1,3], 
                         'workplaces_mobility_percent_change':[1,3], 
                         'residential_mobility_percent_change':[1,3], 
                         'percent_fully_vaccinated_people':[1,3]}

   historical_data = make_historical_data(data = weekly_data, forecast_horizon = 4, history_length = 3, 
                                          column_identifier = column_identifier, 
                                          futuristic_covariates = futuristic_covariates)

Features in the ``historical_data`` include covariate values for up to 3 weeks in the past. But to find the optimal history length we need to get a list of data frames with different history lengths.

.. code-block:: python

   preprocessed_data_list = []

   for history_length in range(1,4):

     historical_data = make_historical_data(data = weekly_data, forecast_horizon = 4, 
                                            history_length = history_length, 
                                            column_identifier = column_identifier, 
                                            futuristic_covariates = futuristic_covariates)

     historical_data = historical_data.rename(columns = {'Target':'Target (normal)'})
     historical_data['Normal target'] = historical_data['Target (normal)']
  
     preprocessed_data_list.append(historical_data)

| Note that we can use different modes for the target variable (i.e. cumulative, moving average, ...). To handle such cases in the prediction process, there is a need for unchanged values of the target to be included in the preprocessed data frames as a column (``Normal target`` column). Here we used normal mode and so specified it in the name of the target variable column (i.e. ``Target (Normal)``)).

| Now we can use ``preprocessed_data_list`` in the ``train_validate`` function to find the best model, history length, and feature sets based on the prediction performance on the validation set.
| To search over the sets of features, first the ranking is performed on the set of all covariates or all features (covariates and their historical values). The method of ranking and items to be ranked are specified using the ``feature_sets`` argument. 
| Here we used the 'mRMR' ranking method on covariates.
| The best feature set is selected from the n feature sets that include the features of the first 1, 2, ..., and n covariates of the ranked list.

.. code-block:: python

   from stpredict.predict import train_validate

   best_model, best_model_parameters, best_history_length, best_feature_or_covariate_set,\
   best_model_base_models, best_trained_model = train_validate(data = preprocessed_data_list, 
                                            feature_sets = {'covariate':'mRMR'}, 
                                            instance_validation_size = 0.2, 
                                            instance_testing_size = 0.2, forecast_horizon=4,
                                            models=['knn', 'glm', 'gbm'], mixed_models=['gbm'], 
                                            performance_benchmark='MAPE')

| To find the test performance the ``train_test`` function is called.

.. code-block:: python

   from stpredict.predict import train_test

   train_test(data = preprocessed_data_list[best_history_length-1], instance_testing_size = 0.2,
              forecast_horizon = 4, feature_or_covariate_set = best_feature_or_covariate_set, 
              history_length = best_history_length, model = best_model, 
              base_models = best_model_base_models, model_type='regression', 
              model_parameters = best_model_parameters, performance_measures=['MAPE'])


| The report of the model performance and the predictions will be saved as in '.csv' format in the subdirectories 'performance/testing process', and 'predictions/testing process' in the same directory as the code is running.
| To predict the future number of weekly deaths resulting from the preventive policies with different levels of stringency, we set the ``scenario`` argument to min, max, or mean in the ``predict_future`` function and predict the number of deaths.

.. code-block:: python

   from stpredict.predict import predict_future

   data = preprocessed_data_list[best_history_length-1]
   forecast_horizon = 4

   train_data = data.iloc[:-forecast_horizon]
   future_data = data.iloc[-forecast_horizon:]

   predict_future(data = train_data, future_data = future_data, forecast_horizon = 4, 
                  feature_or_covariate_set = best_feature_or_covariate_set,
                  model = best_model, base_models = best_model_base_models, model_type = 'regression', 
                  model_parameters = best_model_parameters, scenario='min')

| It will produce the predictions CSV file in the subdirectory 'prediction/future prediction'.

| The whole process can also be done using ``preprocess_data`` and ``predict`` functions.


.. code-block:: python

   from stpredict import load_covid_data
   from stpredict.preprocess import preprocess_data
   from stpredict.predict import predict

   covid_data = load_covid_data()

   column_identifier={'temporal id level 1':'date', 'temporal id level 2':'epidemic_week', 
                   'spatial id level 1':'country', 'target':'covid_19_deaths',
                   'temporal covariates':['covid_19_deaths', 'covid_19_confirmed_cases', 
                   'precipitation', 'temperature', 'retail_and_recreation_mobility_percent_change',
                   'grocery_and_pharmacy_mobility_percent_change', 'parks_mobility_percent_change',
                   'transit_stations_mobility_percent_change','workplaces_mobility_percent_change',
                   'residential_mobility_percent_change', 'percent_fully_vaccinated_people']}

   futuristic_covariates = {'retail_and_recreation_mobility_percent_change':[1,3],
                         'grocery_and_pharmacy_mobility_percent_change':[1,3], 
                         'parks_mobility_percent_change':[1,3], 
                         'transit_stations_mobility_percent_change':[1,3], 
                         'workplaces_mobility_percent_change':[1,3], 
                         'residential_mobility_percent_change':[1,3], 
                         'percent_fully_vaccinated_people':[1,3]}

   history_length = {covar : 3 for covar in column_identifier['temporal covariates']}

   preprocessed_data_list = preprocess_data(data = covid_data, forecast_horizon = 4, 
                                         history_length = history_length, column_identifier = 
                                         column_identifier, temporal_scale_level = 2,
                                         futuristic_covariates = futuristic_covariates)

   predict(data = preprocessed_data_list, forecast_horizon = 4,  feature_sets = {'covariate':'mRMR'},
           models=['knn', 'glm', 'gbm'], mixed_models=['gbm'], model_type = 'regression', 
           test_type = 'whole-as-one', instance_testing_size = 0.2, instance_validation_size = 0.2, 
           performance_benchmark = 'MAPE', scenario = 'current')


| It will produce the CSV files of the performance and prediction reports.
| To create an independent model for each test instance and train it using the most recent data before the test point time unit, we should only set the ``test_type`` argument to one-by-one.

| Another more precise code to perform the whole process is to use the ``stpredict`` function.

.. code-block:: python

   from stpredict import stpredict

   stpredict(data = covid_data, forecast_horizon = 4, history_length = history_length, 
             column_identifier = column_identifier, feature_sets = {'covariate': 'mRMR'}, 
             models = ['knn', 'glm', 'gbm'], mixed_models = ['gbm'],
             test_type = 'whole-as-one', performance_benchmark = 'MAPE', 
             instance_testing_size = 0.2, instance_validation_size = 0.2,
             futuristic_covariates = futuristic_covariates,
             scenario = 'min', temporal_scale_level = 2)