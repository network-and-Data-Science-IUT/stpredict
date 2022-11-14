import pytest
import json
import pandas as pd
import numpy as np
from stpredict import load_covid_data
from stpredict.preprocess import preprocess_data
from stpredict.predict import split_data, predict, performance, select_features, predict_future, train_evaluate

covid_data = load_covid_data()
earthquake_data = pd.read_csv('transformed_data.csv')

covid_data = covid_data[['date', 'epidemic_week', 'country', 'covid_19_deaths', 'covid_19_confirmed_cases', 
                         'retail_and_recreation_mobility_percent_change', 'transit_stations_mobility_percent_change',
                         'workplaces_mobility_percent_change', 'percent_fully_vaccinated_people']]

covid_column_identifier={'temporal id level 1':'date', 'temporal id level 2':'epidemic_week', 'spatial id level 1':'country', 'target':'covid_19_deaths',
                         'temporal covariates':['covid_19_deaths', 'covid_19_confirmed_cases', 'retail_and_recreation_mobility_percent_change',
                                                'transit_stations_mobility_percent_change', 'workplaces_mobility_percent_change', 'percent_fully_vaccinated_people']}

earthquake_column_identifier={'temporal id level 1':'month ID', 'spatial id level 1':'sub-region ID', 'target':'occurrence',
                              'temporal covariates':['occurrence']}

covid_futuristic_covariates = {'retail_and_recreation_mobility_percent_change':[1,3],'transit_stations_mobility_percent_change':[1,3]}

forecast_horizon = 4

covid_preprocessed_data_list = preprocess_data(data = covid_data, forecast_horizon = forecast_horizon, history_length = {key:3 for key in covid_column_identifier['temporal covariates']}, 
                                               column_identifier = covid_column_identifier, temporal_scale_level = 2, futuristic_covariates=covid_futuristic_covariates)
earthquake_preprocessed_data_list = preprocess_data(data = earthquake_data, forecast_horizon = forecast_horizon, history_length = {key:3 for key in earthquake_column_identifier['temporal covariates']}, 
                                               column_identifier = earthquake_column_identifier)
covid_historical_data = covid_preprocessed_data_list[2]
earthquake_historical_data = earthquake_preprocessed_data_list[2]

@pytest.mark.parametrize("data,splitting_type,instance_testing_size,instance_validation_size,instance_random_partitioning,fold_total_number,fold_number,forecast_horizon", 
                         [(covid_historical_data, 'instance', 0.2, 0.2, False, None, None, forecast_horizon), (covid_historical_data, 'fold', 0.2, None, False, 3, 2, forecast_horizon)])

def test_split_data(data, splitting_type, instance_testing_size, instance_validation_size, instance_random_partitioning, fold_total_number, fold_number, forecast_horizon):

  number_of_spatial_units = 1 # only USA country
  training_data, validation_data, testing_data, gap_data = split_data(data, splitting_type, instance_testing_size, instance_validation_size, instance_random_partitioning,
                                                                      fold_total_number, fold_number, forecast_horizon)

  if splitting_type == 'instance':
    assert len(training_data) + len(validation_data) + len(testing_data) + len(gap_data) == len(data)
    assert training_data['temporal id'].max() < validation_data['temporal id'].min()
    assert validation_data['temporal id'].max() < gap_data['temporal id'].min()
    assert gap_data['temporal id'].max() < testing_data['temporal id'].min()
    assert len(gap_data) == (forecast_horizon-1)*number_of_spatial_units

  if splitting_type == 'fold':
    assert len(training_data) + len(validation_data) == len(data)
    assert set(training_data['temporal id'].unique()).intersection(set(validation_data['temporal id'].unique())) == set()


@pytest.mark.parametrize("data,forecast_horizon,feature_sets,models", 
                         [(covid_preprocessed_data_list,forecast_horizon,{'covariate':'mRMR'},['knn', 'glm'])])

def test_predict_regression(data, forecast_horizon,  feature_sets, models):
  predict(data = data, forecast_horizon = forecast_horizon,  feature_sets = feature_sets, models = models, model_type = 'regression', test_type = 'whole-as-one', splitting_type = 'training-validation',  instance_testing_size = 0.2, instance_validation_size = 0.3, 
        save_ranked_features = False, performance_benchmark = 'MAPE', scenario = 'current',verbose = 1)
  
  validation_performance_df = pd.read_csv(f'./performance/validation process/validation performance report forecast horizon = {forecast_horizon}.csv')
  test_performance_df = pd.read_csv(f'./performance/test process/test performance report forecast horizon = {forecast_horizon}.csv')
  training_prediction_df = pd.read_csv(f'./prediction/validation process/training prediction forecast horizon = {forecast_horizon}.csv')
  validation_prediction_df = pd.read_csv(f'./prediction/validation process/validation prediction forecast horizon = {forecast_horizon}.csv')
  test_prediction_df = pd.read_csv(f'./prediction/test process/test prediction forecast horizon = {forecast_horizon}.csv')

  prediction_data = training_prediction_df.append(validation_prediction_df).append(test_prediction_df)

  best_measure = validation_performance_df['MAPE'].min()
  val_best_conf = validation_performance_df.loc[validation_performance_df['MAPE'] == best_measure].iloc[-1]

  assert val_best_conf['model name'] == test_performance_df.loc[0,'model name']
  assert val_best_conf['history length'] == test_performance_df.loc[0,'history length']
  assert val_best_conf['feature or covariate set'].replace(' ','').split(',')  == test_performance_df.loc[0,'feature or covariate set'].replace(' ','').split(',')

  assert all(np.array(prediction_data['prediction']<np.inf)) and all(np.array(prediction_data['prediction']>-np.inf))


@pytest.mark.parametrize("data,forecast_horizon,feature_sets,models", 
                         [(earthquake_preprocessed_data_list,forecast_horizon,{'covariate':'mRMR'},['knn', 'glm'])])

def test_predict_classification(data, forecast_horizon,  feature_sets, models):

  predict(data = data, forecast_horizon = forecast_horizon,  feature_sets = feature_sets, models = models, model_type = 'classification', test_type = 'whole-as-one', splitting_type = 'training-validation',  instance_testing_size = 0.2, instance_validation_size = 0.3, 
        save_ranked_features = False, performance_benchmark = 'AUC', performance_measures = ['AUC'], verbose = 1)
  
  validation_performance_df = pd.read_csv(f'./performance/validation process/validation performance report forecast horizon = {forecast_horizon}.csv')
  test_performance_df = pd.read_csv(f'./performance/test process/test performance report forecast horizon = {forecast_horizon}.csv')
  training_prediction_df = pd.read_csv(f'./prediction/validation process/training prediction forecast horizon = {forecast_horizon}.csv')
  validation_prediction_df = pd.read_csv(f'./prediction/validation process/validation prediction forecast horizon = {forecast_horizon}.csv')
  test_prediction_df = pd.read_csv(f'./prediction/test process/test prediction forecast horizon = {forecast_horizon}.csv')

  prediction_data = training_prediction_df.append(validation_prediction_df).append(test_prediction_df)

  best_measure = validation_performance_df['AUC'].max()
  val_best_conf = validation_performance_df.loc[validation_performance_df['AUC'] == best_measure].iloc[-1]

  assert val_best_conf['model name'] == test_performance_df.loc[0,'model name']
  assert val_best_conf['history length'] == test_performance_df.loc[0,'history length']

  assert all(np.array(prediction_data['class 1']<=1)) and all(np.array(prediction_data['class 1']>=0))
  assert all(np.array(prediction_data['class 0']<=1)) and all(np.array(prediction_data['class 0']>=0))


@pytest.mark.parametrize("model_type,performance_measures", 
                         [('regression', ['MAE', 'MAPE', 'MASE', 'MSE', 'R2_score']), ('classification', ['AUC','AUPR', 'likelihood', 'AIC', 'BIC'])])

def test_performance(model_type, performance_measures):

  if model_type == 'regression':
    y_true = [69, 92, 46, 92, 51, 21, 77, 53, 92, 72,  8, 76, 93, 39,  2, 75, 41,
        83, 41,  9, 94, 50, 78, 47, 36, 77, 23, 30, 21, 39, 12, 71,  6, 28,
        72, 95,  9, 53, 24, 72, 73, 56, 75,  2, 50, 25, 43, 73, 24, 66]

    y_pred = [ 76,  46,  14,  80, 100,  80,   0,  26,  54,  78,  52,  48,  83,
            25,   8,  98,  54,  11,   7,  44,  32,  46,  58,  54,  97,  24,
            80,  95,  36,  59,  88,  37,  64,  63,  64,  33,  51,  56,  81,
            43,  44,  92,  35,  61,  57,  61,  75,  61,  10,  88]

    perf = performance(true_values = y_true, predicted_values = y_pred, performance_measures=performance_measures, 
                      trivial_values=[i-1 for i in y_true], model_type=model_type)
    
    assert perf == [33.74, 1.8524001256823672, 33.74, 1586.18, -1.034645072972102]

  if model_type == 'classification':
    y_true = [0,1]*25

    y_pred = [[0.93263736, 0.06736264],[0.56407151, 0.43592849],
          [0.7707692 , 0.2292308 ],[0.83559927, 0.16440073],[0.87024468, 0.12975532],[0.65763411, 0.34236589],
          [0.5782574 , 0.4217426 ],[0.03655077, 0.96344923],[0.36596645, 0.63403355],[0.93517214, 0.06482786],
          [0.32456117, 0.67543883],[0.63835799, 0.36164201],[0.77271435, 0.22728565],[0.11741494, 0.88258506],
          [0.78932233, 0.21067767],[0.34540279, 0.65459721],[0.1642774 , 0.8357226 ],[0.72291163, 0.27708837],
          [0.53669793, 0.46330207],[0.80546977, 0.19453023],[0.08820708, 0.91179292],[0.4392431 , 0.5607569 ],
          [0.88090769, 0.11909231],[0.05965694, 0.94034306],[0.92397463, 0.07602537],[0.44474403, 0.55525597],
          [0.47080826, 0.52919174],[0.45161573, 0.54838427],[0.44191513, 0.55808487],[0.60416874, 0.39583126],
          [0.3537856 , 0.6462144 ],[0.42357949, 0.57642051],[0.82494634, 0.17505366],[0.95564808, 0.04435192],
          [0.51281535, 0.48718465],[0.3882395 , 0.6117605 ],[0.0970323 , 0.9029677 ],[0.64115425, 0.35884575],
          [0.04134373, 0.95865627],[0.86015911, 0.13984089],[0.64005379, 0.35994621],[0.19012802, 0.80987198],
          [0.42876724, 0.57123276],[0.9095136 , 0.0904864 ],[0.90095268, 0.09904732],[0.55911689, 0.44088311],
          [0.96025618, 0.03974382],[0.60540504, 0.39459496],[0.16549305, 0.83450695],[0.95634686, 0.04365314]]

    perf = performance(true_values = y_true, predicted_values = y_pred, performance_measures=performance_measures, model_type=model_type, labels=[0,1])
    assert perf == [0.4832,0.5036420450329683,0.9859123662067099,0.0794364946482684,5.883847737841566]

@pytest.mark.parametrize("data,ordered_covariates_or_features,history_length,item_type",[(covid_historical_data, ['covid_19_confirmed_cases t', 'percent_fully_vaccinated_people t'], 3, 'covariate'),
                                                                                         (covid_historical_data, ['covid_19_confirmed_cases t', 'percent_fully_vaccinated_people t-1'], 3, 'feature')])
def test_select_features(data, ordered_covariates_or_features, history_length, item_type):
  selected_data = select_features(data.rename(columns={'Target (normal)':'Target'}), ordered_covariates_or_features)

  for covar in ordered_covariates_or_features:
    features = list(filter(lambda x: covar in x, selected_data.columns))
    if item_type == 'covariate':
      assert len(features) == history_length
    else:
      assert len(features) == 1

@pytest.mark.parametrize("data,forecast_horizon,feature_or_covariate_set,model,model_type",[(covid_historical_data, forecast_horizon, 
                                                                                            ['covid_19_confirmed_cases t', 'percent_fully_vaccinated_people t'],'knn','regression')])
def test_predict_future(data, forecast_horizon, feature_or_covariate_set, model, model_type):
  predict_future(data.dropna(), data[pd.isna(data['Target (normal)'])], forecast_horizon = forecast_horizon, feature_or_covariate_set = feature_or_covariate_set,
                 model=model, base_models=None, model_type='regression')
  
  future_predictions = pd.read_csv(f'./prediction/future prediction/future prediction forecast horizon = {forecast_horizon}.csv')

  assert len(future_predictions) == forecast_horizon
  assert all(np.array(future_predictions['prediction']<np.inf)) and all(np.array(future_predictions['prediction']>-np.inf))

@pytest.mark.parametrize("data,model,model_parameters,model_type",[(covid_historical_data, 'nn', 
                                                                  {'hidden_layers_structure':[(2,None),(4,'relu'),(8,'relu')]},'regression'),(earthquake_historical_data, 'nn', 
                                                                  {'hidden_layers_structure':[(3,None),(5,'exponential'),(4,'relu')]},'classification')])
def test_train_evaluate_NN(data, model, model_parameters, model_type):

  if model_type == 'regression':
    train_predictions, validation_predictions, trained_model = train_evaluate(training_data = data.rename(columns={'Target (normal)':'Target'})[:90], 
                                                                              validation_data = data.rename(columns={'Target (normal)':'Target'})[90:].dropna(), 
                                                                              model = model, model_parameters = model_parameters, model_type = model_type)

    model_dict = json.loads(trained_model.to_json())

    assert [model_dict['config']['layers'][i]['config']['units'] for i in range(1,5)] == [2, 4, 8, 1]
    assert [model_dict['config']['layers'][i]['config']['activation'] for i in range(1,5)] == ['linear', 'relu', 'relu', 'exponential']

  else:
    train_predictions, validation_predictions, trained_model = train_evaluate(training_data = data.rename(columns={'Target (normal)':'Target'})[:90], 
                                                                              validation_data = data.rename(columns={'Target (normal)':'Target'})[90:].dropna(), 
                                                                              model = model, model_parameters = model_parameters, model_type = model_type)

    model_dict = json.loads(trained_model.to_json())

    assert [model_dict['config']['layers'][i]['config']['units'] for i in range(1,5)] == [3, 5, 4, 2]
    assert [model_dict['config']['layers'][i]['config']['activation'] for i in range(1,5)] == ['linear', 'exponential', 'relu', 'softmax']
    
@pytest.mark.parametrize("data,model,model_type,base_models",[(covid_historical_data,'glm','regression',['knn','gbm']),(covid_historical_data,'glm','regression',['knn','gbm','glm'])])

def test_train_evaluate_mixed_model(data, model, model_type, base_models):
  train_predictions, validation_predictions, trained_model = train_evaluate(training_data = data.rename(columns={'Target (normal)':'Target'})[:90], 
                                                                              validation_data = data.rename(columns={'Target (normal)':'Target'})[90:].dropna(), 
                                                                              model = model, model_type = model_type, base_models = base_models)
  # check if model inputs are the predictions of base models
  number_of_features = len(trained_model.coef_)
  assert number_of_features == len(base_models)
  
