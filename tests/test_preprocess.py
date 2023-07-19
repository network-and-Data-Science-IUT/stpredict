import pytest
import numpy as np
from stpredict import load_covid_data, load_earthquake_data
from stpredict.preprocess import impute, temporal_scale_transform, target_modification, make_neighbouring_data, make_historical_data
from stpredict.predict import split_data

covid_data = load_covid_data()
covid_column_identifier={'temporal id level 1':'date', 'temporal id level 2':'epidemic_week', 'spatial id level 1':'country', 'target':'covid_19_deaths',
                   'temporal covariates':['covid_19_deaths', 'covid_19_confirmed_cases', 'precipitation', 'temperature', 'retail_and_recreation_mobility_percent_change',
                                          'grocery_and_pharmacy_mobility_percent_change', 'parks_mobility_percent_change', 'transit_stations_mobility_percent_change',
                                          'workplaces_mobility_percent_change', 'residential_mobility_percent_change', 'percent_fully_vaccinated_people']}
covid_futuristic_covariates = {'retail_and_recreation_mobility_percent_change':[1,3],'transit_stations_mobility_percent_change':[1,3]}


earthquake_data = load_earthquake_data()
earthquake_column_identifier = {'temporal id level 1':'month ID', 'spatial id level 1':'sub-region ID', 'target':'occurrence', 'temporal covariates':['occurrence']}


historical_data = make_historical_data(data = covid_data, forecast_horizon = 4, history_length = 3, column_identifier = covid_column_identifier, futuristic_covariates = covid_futuristic_covariates)



@pytest.mark.parametrize("data,column_identifier", [(covid_data, covid_column_identifier)])
def test_imputation(data, column_identifier):
  imputed_data = impute(data=data,column_identifier=column_identifier)
  assert len(imputed_data.dropna()) == len(imputed_data)

@pytest.mark.parametrize("data,column_identifier,temporal_scale_level", [(covid_data, covid_column_identifier,2)])
def test_temporal_scale_transform(data, column_identifier, temporal_scale_level):
  weekly_data = temporal_scale_transform(data, column_identifier, temporal_scale_level)
  assert len(weekly_data) == 122 # total number of epidemic weeks

@pytest.mark.parametrize("data,target_mode,column_identifier", [(covid_data,'cumulative', covid_column_identifier), (covid_data,'moving average', covid_column_identifier), 
                                                                (covid_data,'differential', covid_column_identifier)])
def test_target_modification(data,target_mode,column_identifier):
  modified_data = target_modification(data = data,target_mode = target_mode,column_identifier = column_identifier)
  target_name = column_identifier['target']
  if target_mode == 'cumulative':
    assert modified_data['target'].iloc[-1] == data[target_name].sum()
  if target_mode == 'moving average':
    assert modified_data['target'].iloc[-1] == data[target_name].iloc[-7:].mean()
  if target_mode == 'differential':
    assert modified_data['target'].iloc[-1] == data[target_name].iloc[-1] - data[target_name].iloc[-2]

@pytest.mark.parametrize("data,forecast_horizon,history_length,column_identifier,futuristic_covariates", [(covid_data,4,3,covid_column_identifier,covid_futuristic_covariates)])
def test_make_historical_data(data, forecast_horizon, history_length, column_identifier, futuristic_covariates):
  target_name = column_identifier['target']
  historical_data = make_historical_data(data, forecast_horizon, history_length, column_identifier, futuristic_covariates)
  columns = ['temporal id', 'spatial id', 'Target']
  for covar in column_identifier['temporal covariates']:
    for t in range(history_length):
      if t == 0:
        columns.append(covar+' t')
      else:
        columns.append(covar+' t-'+str(t))
  for key, value in futuristic_covariates.items():
    for t in range(value[0], value[1]+1):
      columns.append(key+' t+'+str(t))
  assert set(columns) - set(historical_data.columns) == set(historical_data.columns) - set(columns) == set()
  target_values = list(historical_data['Target'].dropna()) # values of the target at t + forecast horizon
  feature_values = list(historical_data[target_name+' t'][-len(target_values):]) # values of the target at t
  assert target_values == feature_values


@pytest.mark.parametrize("data,column_identifier", [(earthquake_data, earthquake_column_identifier)])
def test_make_neighbouring_data(data, column_identifier):
  neighbouring_data = make_neighbouring_data(data = data, column_identifier = column_identifier, number_of_layers = 2, 
                                             neighbouring_matrix = np.array([[0,1,0,0,0,1,0,0,0],
                                                                              [1,0,0,1,0,1,0,0,0],
                                                                              [0,0,0,1,0,0,1,0,0],
                                                                              [0,1,1,0,1,0,0,1,0],
                                                                              [0,0,0,1,0,0,1,0,0],
                                                                              [1,1,0,0,0,0,0,0,1],
                                                                              [0,0,1,0,1,0,0,1,0],
                                                                              [0,0,0,1,0,0,1,0,0],
                                                                              [0,0,0,0,0,1,0,0,0]]))
  
  layer1_extracted_covariate_values = list(neighbouring_data.loc[neighbouring_data['sub-region ID']==1,'occurrence_l1'])
  layer1_neighbouring_units_average = list((np.array(neighbouring_data.loc[neighbouring_data['sub-region ID']==2,'occurrence'])+np.array(
                                            neighbouring_data.loc[neighbouring_data['sub-region ID']==6,'occurrence']))/2)
  layer2_extracted_covariate_values = list(neighbouring_data.loc[neighbouring_data['sub-region ID']==1,'occurrence_l2'])
  layer2_neighbouring_units_average = list((np.array(neighbouring_data.loc[neighbouring_data['sub-region ID']==4,'occurrence'])+np.array(
                                            neighbouring_data.loc[neighbouring_data['sub-region ID']==9,'occurrence']))/2)

  assert layer1_extracted_covariate_values == layer1_neighbouring_units_average
  assert layer2_extracted_covariate_values == layer2_neighbouring_units_average


@pytest.mark.parametrize("data,splitting_type,instance_testing_size,instance_validation_size,instance_random_partitioning,fold_total_number,fold_number", 
                         [(historical_data, 'instance', 0.2, 0.2, False, None, None), (historical_data, 'fold', 0.2, None, False, 3, 2)])

def test_split_data(data, splitting_type, instance_testing_size, instance_validation_size, instance_random_partitioning, fold_total_number, fold_number):

  number_of_spatial_units = 1 # only USA country
  forecast_horizon = 4
  training_data, validation_data, testing_data, gap_data = split_data(data, splitting_type, instance_testing_size, instance_validation_size, instance_random_partitioning, fold_total_number, fold_number, forecast_horizon)

  if splitting_type == 'instance':
    assert len(training_data) + len(validation_data) + len(testing_data) + len(gap_data) == len(data)
    assert training_data['temporal id'].max() < validation_data['temporal id'].min()
    if len(gap_data)>0:
        assert validation_data['temporal id'].max() < gap_data['temporal id'].min()
        assert gap_data['temporal id'].max() < testing_data['temporal id'].min()
    assert len(gap_data) == (forecast_horizon-1)*number_of_spatial_units
        
  if splitting_type == 'fold':
    assert len(training_data) + len(validation_data) == len(data)
    assert set(training_data['temporal id'].unique()).intersection(set(validation_data['temporal id'].unique())) == set()
