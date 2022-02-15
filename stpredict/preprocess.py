import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import pandas as pd
    import numpy as np
    import sys
    import datetime
    from sklearn.impute import KNNImputer
    import matplotlib.pyplot as plt
    import calendar


###################### renaming the columns to formal format

def rename_columns(data,column_identifier, mode = 'formalize'):
        
    if type(column_identifier) == dict:
        if any([key not in list(column_identifier.keys()) for key in ['target','temporal covariates']]):
            raise ValueError("The target and temporal covariates are not specified in column_identifier.")
        if mode == 'formalize':
            for key, value in column_identifier.items():
                if key not in ['temporal covariates','spatial covariates']:
                    if (key == 'target') and (value in column_identifier['temporal covariates']) and (value in data.columns):
                        if 'target' not in data.columns:
                            data.loc[:,('target')] = list(data[value])
                    else:
                        data.rename(columns = {value:key}, inplace = True)
        elif mode == 'deformalize':
            for key, value in column_identifier.items():
                if key not in ['temporal covariates','spatial covariates']:
                    if (key == 'target') and (value in column_identifier['temporal covariates']) and (value in data.columns):
                        data = data.drop(['target'],axis = 1)
                    else:
                        data.rename(columns = {key:value}, inplace = True)
            
    elif column_identifier is not None:
        raise TypeError("The column_identifier must be of type dict")

    return data

###################### check validity of input data

def check_validity(data, input_name = 'temporal_data', data_type = 'temporal', column_identifier = None): 
# data argument could accept temporal_data, full_data, spatial_data, spatial_scale_table or future_data_table
# the type will be determined based on data_type argument and the input_name argument will
# be used to produce more clear warning errors

    if 'spatial id level 1' not in data.columns:
        if data_type != 'spatial_scales':
            raise Exception("The input {0} has no spatial id column, and the name of this column is not specified in the column_identifier.\nmissing column: 'spatial id level 1'".format(input_name))
        else:
            raise Exception("The input {0} has no spatial id column.\nmissing column: 'spatial id level 1'".format(input_name))
    
    else:
        # check for null values
        for i in range(1,200):
            column_name = 'spatial id level ' + str(i)
            if column_name in data.columns:
                if data[column_name].isnull().values.any():
                    raise ValueError('spatial id must have value for all instances but spatial id level '+str(i)+' in {0} includes NULL values'.format(input_name))
            else: break

    if data_type in ['temporal','full']:
        
        if 'temporal id' in data.columns: # integrated temporal id format
            temporal_identifier_column_name = 'temporal id'
            if data[temporal_identifier_column_name].isnull().values.any():
                    raise ValueError('temporal id must have value for all instances but temporal id column in {0} includes NULL values'.format(input_name))

        elif 'temporal id level 1' in data.columns: # non-integrated temporal id format
            data = add_dummy_integrated_temporal_id(data.copy())
            temporal_identifier_column_name = 'dummy temporal id'
        else:
            raise Exception("The input {0} has no temporal id column, and the name of this column is not specified in the column_identifier.\nmissing column: 'temporal id' or 'temporal id level 1'".format(input_name))
            
        data = data.drop_duplicates(subset = ['spatial id level 1',temporal_identifier_column_name]).copy()

        if len(data) != len(data['spatial id level 1'].unique())* len(data[temporal_identifier_column_name].unique()):
            raise ValueError("The input {0} has different number of temporal units recorded for each spatial unit".format(input_name))
        if 'dummy temporal id' in data.columns:
            data.drop(['dummy temporal id'], axis = 1, inplace = True)
            
    if column_identifier is None:
        numerical_columns = list(filter(lambda x: x.startswith(('temporal covariate','spatial covariate','target')),data.columns))
    elif all([key in list(column_identifier.keys()) for key in ['target','spatial covariates','temporal covariates']]):
        numerical_columns = list(filter(lambda x: x in(['target']+column_identifier['spatial covariates']+column_identifier['temporal covariates']),data.columns))
    elif all([key in list(column_identifier.keys()) for key in ['target','temporal covariates']]):
        numerical_columns = list(filter(lambda x: x in(['target']+column_identifier['temporal covariates']),data.columns))
    else:
        raise ValueError("The target and temporal covariates are not specified in column_identifier.")
        
    for covar in numerical_columns:
        try:
            data[covar].astype(float)
        except (ValueError, TypeError):
            raise ValueError("The covariates and target variable must include only numerical values. But non-numerical values are recorded for the {0}".format(covar))
    return
            
############################ rename the columns of final historical data

def recorrect_hist_data(data, augmentation, granularity, target_mode, target_granularity):
    
    # name of target mode
    target_column_name = 'Target'
    if augmentation == True:
        target_column_name = 'Target (augmented with {0} units)'.format(granularity)
        data = data.rename(columns={'Target':target_column_name})

    if target_mode in ['differential', 'cumulative', 'normal']:
        data = data.rename(columns={target_column_name:'{0} ({1})'.format(target_column_name, target_mode)})
        target_column_name = '{0} ({1})'.format(target_column_name, target_mode)

    if target_mode == 'moving average':
        data = data.rename(columns={target_column_name:'{0} (moving average on {1} units)'.format(target_column_name, target_granularity)})
        target_column_name = '{0} (moving average on {1} units)'.format(target_column_name, target_granularity)

    data = data.rename(columns={target_column_name:target_column_name.replace(') (',' - ')})
    
    return data

############################ adding or separating the values of futuristic covariates (future_data_table) to the data

def current_future(data, future_data_table, futuristic_covariates, column_identifier , mode):
    '''
    mode = 'split' or 'add'
    '''
    if futuristic_covariates == None:
        return data, None, []
    
    futuristic_covariate_list = list(futuristic_covariates.keys())

    data = rename_columns(data.copy(), column_identifier)
    check_validity(data.copy(), input_name = 'data', data_type = 'temporal', column_identifier = column_identifier)
    
    if column_identifier is None:
        spatial_covariates = list(filter(lambda x:x.startswith('spatial covariate'), data.columns))
        temporal_covariates = list(filter(lambda x:x.startswith('temporal covariate'), data.columns))
    else:
        if 'spatial covariates' in column_identifier.keys():
            spatial_covariates = [item for item in list(column_identifier['spatial covariates']) if item in data.columns]
        else: spatial_covariates = []
        temporal_covariates = [item for item in list(column_identifier['temporal covariates']) if item in data.columns]
        
    id_columns = list(filter(lambda x: x.startswith(('spatial id','temporal id')),data.columns))
    spatial_id_columns = list(filter(lambda x: x.startswith(('spatial id')),data.columns))
    temporal_id_columns = list(filter(lambda x: x.startswith(('temporal id')),data.columns))
    target_columns = ['target']
    if 'Normal target' in data.columns:
        target_columns = target_columns + ['Normal target']
    
    spatial_data = None
    temporal_data = data[id_columns + temporal_covariates + target_columns]
    if len(spatial_covariates) > 0:
        # if input 'data' is full data, data on spatial covariates is saved to add to the data at the end of the process
        if len(list(set(spatial_covariates) - set(data.columns))) == 0 :
            spatial_data = data[spatial_id_columns + spatial_covariates].drop_duplicates(subset = spatial_id_columns)
    
    
    if future_data_table is not None:
        future_data_table = rename_columns(future_data_table.copy(), column_identifier)
        check_validity(future_data_table.copy(), input_name = 'future_data_table',
                       data_type = 'temporal', column_identifier = column_identifier)
        
        for col in id_columns:
            if col not in future_data_table.columns :
                raise ValueError("The temporal and spatial id columns must be identical in the future_data_table and the input data.")
        
        extra_columns = (set(future_data_table.columns)-set(id_columns)-set(futuristic_covariate_list))
        if len(extra_columns) > 0: print("\nWarning: some of the columns in the future_data_table are not in the futuristic_covariates and will be ignored.")
        unspecified_columns = (set(futuristic_covariate_list)-(set(future_data_table.columns)-set(id_columns)))
        if len(unspecified_columns) > 0:raise Exception("Some of the futuristic covariates in the futuristic_covariates dict are not included in the future_data_table.")
        
    if 'temporal id' in temporal_data.columns:
        temporal_identifier_column_name = 'temporal id'
        non_futuristic_covariates = list(set(temporal_data.columns) - set(futuristic_covariate_list + spatial_id_columns + ['temporal id']))


    else: # non-integrated temporal id format
        for level in range(1,200):
            if 'temporal id level ' + str(level) in temporal_data.columns:
                smallest_temporal_level = level
                break
        temporal_data = add_dummy_integrated_temporal_id(temporal_data.copy(), start_level = smallest_temporal_level)
        if future_data_table is not None:
            future_data_table = add_dummy_integrated_temporal_id(future_data_table.copy(), start_level = smallest_temporal_level)
        temporal_identifier_column_name = 'dummy temporal id'
        non_futuristic_covariates = list(set(temporal_data.columns) - set(futuristic_covariate_list + spatial_id_columns + ['dummy temporal id'] + temporal_id_columns))


    if mode == 'split':
        
        temporal_data = temporal_data.drop_duplicates(subset = ['spatial id level 1', temporal_identifier_column_name]).copy()
        temporal_data = temporal_data.sort_values(by = [temporal_identifier_column_name,'spatial id level 1'])

        temporal_data['...'] = list(range(len(temporal_data))) # add an unique id to the data frame

        current_data = temporal_data.dropna(subset = non_futuristic_covariates, how='all')
        current_data_index = list(current_data['...'])
        future_data = temporal_data[~(temporal_data['...'].isin(current_data_index))]
        temp = future_data.groupby([temporal_identifier_column_name]).count()
        
        # remove dates which is not compeletly missed the non_futuristic_covariates (are not related to the future data)
        total_number_of_spatial_ids = temp['spatial id level 1'].max()
        incomplete_dates = temp[temp['spatial id level 1'] < total_number_of_spatial_ids].index

        future_data = future_data[~(future_data[temporal_identifier_column_name].isin(incomplete_dates))]
        current_data = temporal_data[~(temporal_data['...'].isin(future_data['...']))]
        
        futuristic_temporal_units = list(future_data[temporal_identifier_column_name].unique())
        future_data_table_columns = id_columns + futuristic_covariate_list
        future_data_table = future_data[future_data_table_columns]

        if 'dummy temporal id' in current_data.columns:
            current_data = current_data.drop(['dummy temporal id','...'], axis = 1)
        else:
            current_data = current_data.drop(['...'], axis = 1)
        
        if len(future_data_table) == 0 :
            future_data_table = None
        
        if spatial_data is not None:
            current_data = pd.merge(current_data, spatial_data, on = 'spatial id level 1', how = 'left')
        
        return current_data, future_data_table, futuristic_temporal_units
        
    if mode == 'add':
        for col in non_futuristic_covariates:
            future_data_table.loc[:,(col)] = np.NaN
        future_data_table = future_data_table[list(temporal_data.columns)]
        
        future_data_table = future_data_table[future_data_table['spatial id level 1'].isin(temporal_data['spatial id level 1'].unique())]
        
        future_data_table = future_data_table[future_data_table[temporal_identifier_column_name]>max(temporal_data[temporal_identifier_column_name])]
        
        temporal_data = temporal_data.append(future_data_table)
        
        temporal_data.sort_values(by = [temporal_identifier_column_name, 'spatial id level 1'])
        if 'dummy temporal id' in temporal_data.columns:
            temporal_data = temporal_data.drop(['dummy temporal id'], axis = 1)
            
        futuristic_temporal_units = list(future_data_table[temporal_identifier_column_name].unique())
        
        if spatial_data is not None:
            data = pd.merge(temporal_data, spatial_data, on = 'spatial id level 1', how = 'left')
        else: data = temporal_data
            
        return data, None, futuristic_temporal_units

        
############################ filling up the future temporal units null values with inf to recognize them later
    
def fill_future_nulls(data, futuristic_temporal_units):
        if 'temporal id' in data.columns:
            temporal_identifier_column_name = 'temporal id'
        else:
            for level in range(1,200):
                if 'temporal id level ' + str(level) in data.columns:
                    smallest_temporal_level = level
                    break
            data = add_dummy_integrated_temporal_id(data.copy(), start_level = smallest_temporal_level)
            temporal_identifier_column_name = 'dummy temporal id'
        current_data = data[~(data[temporal_identifier_column_name].isin(futuristic_temporal_units))]
        future_data = data[data[temporal_identifier_column_name].isin(futuristic_temporal_units)]
        future_data = future_data.fillna(np.inf)
        data = current_data.append(future_data)
        return data
        
############################ adding an integrated teporal id to the data for non integrated temporal id format


# Integrating the temporal id columns with the scale levels greater than 'start_level',
# to obtain a single temporal id which is unique for each unit of the temporal scale level 'start_level'
def add_dummy_integrated_temporal_id(temporal_data, start_level = 1):
    
    temporal_data_columns = temporal_data.columns
    
    valid_temporal_id_levels = ['temporal id level ' + str(start_level)] # sequense of temporal levels without gap
    greatest_temporal_level = start_level

    # determine the greatest temporal scale level and valid temporal id levels
    for i in range(start_level+1,200):
        
        if 'temporal id level '+str(i) in temporal_data_columns:
            greatest_temporal_level += 1
            valid_temporal_id_levels.append('temporal id level '+str(i))
            
            # check for null values
            if temporal_data['temporal id level '+str(i)].isnull().values.any():
                raise ValueError('temporal id must have value for all instances but temporal id level '+str(i)+' includes NULL values')
        else:
            break
            
    # remove invalid temporal id levels from data
    all_temporal_id_levels = list(filter(lambda x: x.startswith('temporal id level '), temporal_data.columns))
    invalid_temporal_id_levels = list(set(all_temporal_id_levels)-set(valid_temporal_id_levels))
    if len(invalid_temporal_id_levels) > 0:
        temporal_data.drop(invalid_temporal_id_levels, axis = 1, inplace = True)
        if start_level>1:
            smaller_temporal_id_levels = ['temporal id level '+str(i) for i in range(1,start_level)]
            invalid_temporal_id_levels = list (set(invalid_temporal_id_levels)-set(smaller_temporal_id_levels))
            if len(invalid_temporal_id_levels) > 0:
                print('\nWarning: There is a gap in the sequence of temporal scale levels recorded in the data.\nIds for temporal scale levels greater than {0} are ignored.\n'.format(greatest_temporal_level))
    
    # construct the integrated (sortable) temporal id
    temporal_data.loc[:,('dummy temporal id')] = temporal_data['temporal id level ' + str(greatest_temporal_level)]

    if greatest_temporal_level > 1:
        for level in range(start_level,greatest_temporal_level)[::-1]:
            temporal_data.loc[:,('dummy temporal id')] = temporal_data['dummy temporal id'].astype(str) + '/' + temporal_data['temporal id level ' + str(level)].astype(str)
    
    return temporal_data


############################ adding secondary spatial scale id's to data

def add_spatial_ids(data,spatial_scale_table):
    
    
    # to use the spatial scale mapping in the spatial_scale_table
    # first the existing secondary spatial scale levels are removed from the data
    extra_spatial_ids = list(filter(lambda x: x.startswith('spatial id level '), data.columns))
    extra_spatial_ids.remove('spatial id level 1')
    data.drop(extra_spatial_ids, axis = 1, inplace = True)

    # check spatial_scale_table to have information of all the units in the spatial scale level 1
    intersection = list(set(data['spatial id level 1'].unique()) & set(spatial_scale_table['spatial id level 1'].unique()))
    if len(intersection) < len(data['spatial id level 1'].unique()):
        raise ValueError('The ids of some units in the spatial scale level 1 are missed in the spatial_scales_table.')

    spatial_scale_levels = ['spatial id level 1']
    for i in range(2,200):
        column_name = 'spatial id level ' + str(i)
        if column_name in spatial_scale_table.columns:
            spatial_scale_levels.append(column_name)
        else: break

    data = pd.merge(data,spatial_scale_table[spatial_scale_levels], how = 'left', on= 'spatial id level 1')

    return data


############################ transform temporal ids in the data frame to the time stamp format

def create_time_stamp(data, time_format, required_suffix):
    try:
        data.loc[:,('temporal id')] = data['temporal id'].astype(str) + required_suffix
        data.loc[:,('temporal id')] = data['temporal id'].apply(lambda x:datetime.datetime.strptime(x,time_format))
    except ValueError:
        raise ValueError("temporal id values doesn't match any supported integrated format for temporal id.\n")
    return data

############################ find the scale of temporal ids and transform to the time stamp format

def check_integrated_temporal_id(temporal_data):
        
    list_of_supported_formats_string_length = [4,7,10,13,16,19]
    temporal_data = temporal_data.sort_values(by = ['spatial id level 1','temporal id']).copy()
    temporal_id_instance = str(temporal_data['temporal id'].iloc[0])
    
    if len(temporal_id_instance) not in list_of_supported_formats_string_length:
        raise ValueError("temporal id values doesn't match any supported integrated format for temporal id.\n")
    
    # find the scale
    if len(temporal_id_instance) == 4:
        scale = 'year'
        temporal_data = create_time_stamp(temporal_data.copy(), '%Y/%m/%d', '/01/01')
            
    elif len(temporal_id_instance) == 7:
        scale = 'month'
        temporal_data = create_time_stamp(temporal_data.copy(), '%Y/%m/%d', '/01')
        
    elif len(temporal_id_instance) == 10:
        
        temporal_data = create_time_stamp(temporal_data.copy(), '%Y/%m/%d', '')
        
        first_temporal_id_instance = temporal_data['temporal id'].iloc[0]
        second_temporal_id_instance = temporal_data['temporal id'].iloc[1]
        
        delta = second_temporal_id_instance - first_temporal_id_instance
        if delta.days == 1:
            scale = 'day'
        elif delta.days == 7:
            scale = 'week'
        else:
            raise ValueError("temporal ids with format YYYY/MM/DD must be daily or weekly, but two consecutive id's of input data have a difference of {0} days.\n".format(delta.days))
    
    elif len(temporal_id_instance) == 13:
        scale = 'hour'
        temporal_data = create_time_stamp(temporal_data.copy(), '%Y/%m/%d %H:%M:%S', ':00:00')

    elif len(temporal_id_instance) == 16:
        scale = 'min'
        temporal_data = create_time_stamp(temporal_data.copy(), '%Y/%m/%d %H:%M:%S', ':00')
            
    elif len(temporal_id_instance) == 19:
        scale = 'sec'
        temporal_data = create_time_stamp(temporal_data.copy(), '%Y/%m/%d %H:%M:%S', '')
    
    return temporal_data, scale
            
############################ find the number of smaller scale units in each bigger scale unit

def find_granularity(data, temporal_scale_level):
    
    if (type(temporal_scale_level) != int) and (temporal_scale_level is not None):
        raise TypeError("The temporal_scale_level must be of type int.\n")
    
    
    ###################################### integrated temporal id format ################################
    
    if 'temporal id' in data.columns:
    
        scale_level_dict = {'sec':1, 'min':2, 'hour':3, 'day':4, 'week':5, 'month':6, 'year':7}
        scale_format = {'sec':'%Y/%m/%d %H:%M:%S', 'min':'%Y/%m/%d %H:%M', 'hour':'%Y/%m/%d %H', 'day':'%Y/%m/%d', 'week':'%Y/%m/%d', 'month':'%Y/%m', 'year':'%Y'}
        scale_second_units_number = {'sec':1, 'min':60, 'hour':3600, 'day':24*3600, 'week':7*24*3600, 'month':30*24*3600, 'year':365*24*3600}

        data = data.drop_duplicates(subset = ['spatial id level 1','temporal id']).copy()

        # determining input data temporal scale and transforming temporal id's to timestamp
        data , scale = check_integrated_temporal_id(data.copy())
        
        # in moving average target mode the temporal_scale_level is the next level of current level and must be detected
        # based on current level
        if temporal_scale_level is None:
            temporal_scale_level = 2
            desired_temporal_scale_level = scale_level_dict[scale] + temporal_scale_level - 1
            if desired_temporal_scale_level > 7:
                raise exception("The temporal scale level of the data is {0}. Thus the 'moving average' target_mode couldn't be applied cause the temporal scale bigger than {0} is ambiguous.\n".format(scale))


        # determine the numerical desired temporal scale level based on the input data
        # temporal scale and user specified temporal_scale_level
        desired_temporal_scale_level = scale_level_dict[scale] + temporal_scale_level - 1
        if desired_temporal_scale_level > 7:
            raise Exception('The first temporal scale level recorded in the data is {0}. So the temporal scale level {1} is out of the supported range of temporal scale levels: second, minute, hour, day, week, month, year'.format(scale,temporal_scale_level))

        # get the nominal form of desired temporal scale level
        desired_temporal_scale = list(scale_level_dict.keys())[list(scale_level_dict.values()).index(desired_temporal_scale_level)]
        
        # get number of smaller temporal units in each bigger temporal unit
        granularity = scale_second_units_number[desired_temporal_scale]//scale_second_units_number[scale]
        
    ###################################### non_integrated temporal id format ################################
        
    if 'temporal id level 1' in data.columns: 
        
        # if function is called for getting granularity of moving average target mode, the temporal_scale_level
        # is the next level of current level and must be detected based on current level
        if temporal_scale_level is None:
            for level in range(1,200):
                if 'temporal id level ' + str(level) in data.columns:
                    smallest_temporal_level = level
                    break
            temporal_scale_level = smallest_temporal_level + 1
            desired_scale_column_name = 'temporal id level ' + str(temporal_scale_level)
            if desired_scale_column_name not in data.columns:
                raise ValueError("The next bigger temporal scale after data current temporal scale (temporal scale level {0}) isn't included in the data or is located after a gap in the temporal scale levels sequence and is therefore ignored.\n".format(smallest_temporal_level))
            
        else:
            smallest_temporal_level = 1
            desired_scale_column_name = 'temporal id level ' + str(temporal_scale_level)
            if desired_scale_column_name not in data.columns:
                raise ValueError("temporal_scale_level {0} is not in the time scale levels recorded in the data or is located after a gap in the temporal scale levels sequence and is therefore ignored. ".format(temporal_scale_level))

            
        # next 2 line removes the duplicate data samples having same spatial and temporal id's.
        data = add_dummy_integrated_temporal_id(data.copy(), start_level = smallest_temporal_level)
        data = data.drop_duplicates(subset = ['spatial id level 1','dummy temporal id']).copy()

        
        # get number of smaller temporal units in each bigger temporal unit
        granularity = data.groupby(['spatial id level 1',desired_scale_column_name]).count()['dummy temporal id'].max()


    return granularity

############################ preprocessing the input data and split it to temporal and spatial data

def prepare_data(data, column_identifier):
    
    # initialize
    spatial_data = None
    temporal_data = None
    
    if type(data) != dict :

        if type(data) == str :
            try:
                data = pd.read_csv(data)
            except FileNotFoundError:
                raise FileNotFoundError("File '{0}' does not exist.\n".format(data))
        elif type(data) != pd.DataFrame :
            raise TypeError("The input data must be of type DataFrame, string, or a dict containing temporal and spatial DataFrames or addresses.\n")
        data = rename_columns(data.copy(), column_identifier)

        check_validity(data.copy(), input_name = 'data', data_type = 'full',
                       column_identifier = column_identifier)

        ##### split data to temporal and spatial data

        spatial_columns = []
        temporal_columns = []

        # if type of each column is specified in column_identifier
        if column_identifier is not None:
            temporal_columns = list(filter(lambda x: x.startswith(('temporal id', 'spatial id','target')), data.columns)) + [item for item in column_identifier['temporal covariates'] if item in data.columns]
            temporal_data = data[temporal_columns]
            # if data includes spatial covariates
            if 'spatial covariates' in column_identifier.keys():
                spatial_columns = list(filter(lambda x: x.startswith('spatial id'), data.columns)) + [item for item in column_identifier['spatial covariates'] if item in data.columns]
                spatial_data = data[spatial_columns].drop_duplicates(subset = ['spatial id level 1']).copy()

        # if type of columns are clear based on column name
        else:
            temporal_columns = list(filter(lambda x: x.startswith(('temporal id', 'spatial id','temporal covariate','target')), data.columns))
            temporal_data = data[temporal_columns]
            # if data includes spatial covariates
            if len(list(filter(lambda x: x.startswith('spatial covariate'), data.columns))) > 0 :
                spatial_columns = list(filter(lambda x: x.startswith('spatial id','spatial covariate'), data.columns))
                spatial_data = data[spatial_columns].drop_duplicates(subset = ['spatial id level 1']).copy()


        data_extra_columns = list(set(data.columns)-set(spatial_columns+temporal_columns))

        if len(data_extra_columns) > 0:
            print("\nWarning: Input data column names must match one of the formats:\n{'temporal id', 'temporal id level x', 'spatial id', 'spatial id level x', 'temporal covariate x', 'spatial covariate x', 'target'},")
            print("or be specified in the column_identifier, but the names of some of the columns do not match any of the supported formats and are not mentioned in the column_identifier:\n{0}\nThese columns will be ignored.\n".format(data_extra_columns))


    elif type(data) == dict:
        if 'temporal_data' in data.keys():

            if type(data['temporal_data']) == str:
                try:
                    temporal_data = pd.read_csv(data['temporal_data'])
                except FileNotFoundError:
                    raise FileNotFoundError("File '{0}' does not exist.\n".format(data['temporal_data']))

            elif type(data['temporal_data']) == pd.DataFrame:
                temporal_data = data['temporal_data']
            else:
                raise valueError("The value of the 'temporal id' key in the data dictionary must be a DataFrame or the address of temporal data.\n")
            temporal_data = rename_columns(temporal_data.copy(), column_identifier)
            check_validity(temporal_data.copy(), input_name = 'temporal_data', 
                           data_type = 'temporal', column_identifier = column_identifier)

            if column_identifier is not None:
                if 'temporal covariates' in column_identifier.keys():
                    temporal_data_extra_columns = list(set(filter(lambda x: not x.startswith(('temporal id','spatial id','target')), temporal_data.columns)) - set(column_identifier['temporal covariates']))
            else:
                temporal_data_extra_columns = list(filter(lambda x: not x.startswith('temporal id','spatial id','temporal covariate','target'), temporal_data.columns))

            if len(temporal_data_extra_columns) > 0:
                print("\nWarning: Input temporal_data column names must match one of the formats:\n{'temporal id', 'temporal id level x', 'spatial id', 'spatial id level x', 'temporal covariate x', 'target'}")
                print("or be specified in the column_identifier,but the names of some of the columns do not match any of the supported formats and are not mentioned in the column_identifier:\n{0}\nThese columns will be ignored.\n".format(temporal_data_extra_columns))
            temporal_data.drop(temporal_data_extra_columns, axis = 1, inplace = True)

        else:
            raise Exception("The data on temporal covariates and target variable must be passed to the function using data argument and as a DataFrame, Data address or value of 'temporal_data' key in the dictionary of data. But none is passed.\n")


        if ('spatial_data' in data.keys()) and (data['spatial_data'] is not None):

            if type(data['spatial_data']) == str:
                try:
                    spatial_data = pd.read_csv(data['spatial_data'])
                except FileNotFoundError:
                    raise FileNotFoundError("File '{0}' does not exist.\n".format(data['spatial_data']))

            elif type(data['spatial_data']) == pd.DataFrame:
                spatial_data = data['spatial_data']
            else:
                raise ValueError("The value of the 'spatial id' key in the data dictionary must be a DataFrame or the address of spatial data.\n")

            spatial_data = rename_columns(spatial_data.copy(), column_identifier)
            check_validity(spatial_data.copy(), input_name = 'spatial_data',
                           data_type = 'spatial', column_identifier = column_identifier)
            spatial_data = spatial_data.drop_duplicates(subset = ['spatial id level 1']).copy()

            if column_identifier is not None:
                if 'spatial covariates' in column_identifier.keys():
                    spatial_data_extra_columns = list(set(filter(lambda x: not x.startswith('spatial id'), spatial_data.columns)) - set(column_identifier['spatial covariates']))
            else:
                spatial_data_extra_columns = list(filter(lambda x: not x.startswith(('spatial id','spatial covariate')), spatial_data.columns))

            if len(spatial_data_extra_columns) > 0:
                print("\nWarning: Input spatial_data column names must match one of the formats:\n{'spatial id', 'spatial id level x', 'spatial covariate x'}")
                print("or be specified in the column_identifier,but the names of some of the columns do not match any of the supported formats and are not mentioned in the column_identifier:\n{0}\nThis columns will be ignored.\n".format(spatial_data_extra_columns))
            spatial_data.drop(spatial_data_extra_columns, axis = 1, inplace = True)
            
    return temporal_data, spatial_data


#############################################################################################
############################  imputing the tepmoral data ####################################
#############################################################################################

def impute(data, column_identifier = None, verbose = 0):
    
    if type(data) == str:
        data = pd.read_csv(data)
    
    data = rename_columns(data.copy(), column_identifier)
        
    check_validity(data.copy(), input_name = 'data', data_type = 'temporal',
                   column_identifier = column_identifier)
    
    if 'temporal id' in data.columns:
        temporal_identifier_column_name = 'temporal id'
        
    elif 'temporal id level 1' in data.columns: # non-integrated temporal id format
        # first integrated format for temporal id must constructed
        data = add_dummy_integrated_temporal_id(data.copy())
        temporal_identifier_column_name = 'dummy temporal id'

    data = data.drop_duplicates(subset = ['spatial id level 1', temporal_identifier_column_name]).copy()
        
    covariate_names = list(filter(lambda x: not x.startswith(('temporal id', 'spatial id', 'dummy temporal id')), data.columns))
    
    if 'target' in data.columns:
        if data['target'].isnull().values.any():
            if verbose > 0: print("\nWarning: The target variable includes missing values. This values will be imputed.\n")
            
    spatial_units_observed_value_count = data.groupby('spatial id level 1').count()
    temporal_units_observed_value_count = data.groupby(temporal_identifier_column_name).count()
    
    
    
    for covar in covariate_names:
        spatial_units_with_all_nulls=spatial_units_observed_value_count[spatial_units_observed_value_count[covar]==0].index
        temporal_units_with_all_nulls=temporal_units_observed_value_count[temporal_units_observed_value_count[covar]==0].index
        
        # check if covariate has no value for an temporal unit
        if len(temporal_units_with_all_nulls) > 0:
            raise ValueError("The input data has no value for {0}, in some temporal units.\nThe covariates must have value for at least one spatial unit in each temporal units recorded in the data".format(covar))
        
        data = data.drop_duplicates(subset = ['spatial id level 1',temporal_identifier_column_name]).copy()
        
        # create data frame with each row representing a spatial unit and each column representing a temporal unit 
        temp = data.pivot(index='spatial id level 1', columns=temporal_identifier_column_name, values=covar)
        
        # impute missing values using KNN imputation
        X = np.array(temp)
        imputer = KNNImputer(n_neighbors=5)
        imp=imputer.fit_transform(X)
        imp=pd.DataFrame(imp, index = temp.index, columns = temp.columns)
        
        # reshape imputed values of covariate into one column
        imp = pd.melt(imp.reset_index(), id_vars='spatial id level 1', value_vars=list(imp.columns),
                     var_name=temporal_identifier_column_name, value_name=covar)
        
        # add the imputed values back to the data set
        data.drop([covar], axis = 1, inplace = True)
        data = pd.merge(data, imp, how = 'left')
        
        # remove the imputed values for the spatial units with no observed values
        data.loc[data['spatial id level 1'].isin(spatial_units_with_all_nulls),covar]=np.NaN
        if len(spatial_units_with_all_nulls)>0:
            if verbose == 2:
                print('\nWarning: Following spatial units has no recorded values for {0} and therefore will be removed from the data:\n{1}\n'.format(covar,list(spatial_units_with_all_nulls)))
            elif verbose == 1:
                print('\nWarning: The number of {0} spatial units has no recorded values for {1} and therefore will be removed from the data\n.'.format(len(list(spatial_units_with_all_nulls)),covar))
    
    # remove temporarily added column
    if 'dummy temporal id' in data.columns: data.drop(['dummy temporal id'], axis = 1, inplace = True)
    
    # produce warning for removed spatial units
    number_of_spatial_units = len(data['spatial id level 1'].unique())
    imputed_data = data.dropna()
    number_of_removed_spatial_units = number_of_spatial_units - len(imputed_data['spatial id level 1'].unique())
    
    if number_of_removed_spatial_units == number_of_spatial_units:
        raise Exception("All the spatial units have no value for at least one temporal covariate and are removed from the data. Therefore, no spatial unit remains to make a prediction.\n")
    elif number_of_removed_spatial_units > 0:
        if verbose == 0:
            print('\nWarning: The number of {0} spatial units has no value for at least one temporal covariate and are removed from the data.\n'.format(number_of_removed_spatial_units))
    
    imputed_data = rename_columns(imputed_data.copy(), column_identifier, 'deformalize')
    
    return imputed_data

#################################################################################################
############################  transforming the spatial scale ####################################
#################################################################################################

def spatial_scale_transform(data, data_type, spatial_scale_table = None, spatial_scale_level = 2, aggregation_mode = 'mean', column_identifier = None, verbose = 0):
    
    # initializing list of covariates with sum or mean aggregation modes
    mean_covariates = []
    sum_covariates = []
    
    desired_scale_column_name = 'spatial id level ' + str(spatial_scale_level)
    base_columns = [desired_scale_column_name]
    
    if type(data) == str:
        data = pd.read_csv(data)
    data = rename_columns(data.copy(), column_identifier)
    
    check_validity(data.copy(), input_name = 'data', data_type = data_type, column_identifier = column_identifier)
    if spatial_scale_table is not None:
        if type(spatial_scale_table) == str:
            spatial_scale_table = pd.read_csv(spatial_scale_table)
        spatial_scale_table = rename_columns(spatial_scale_table.copy(), column_identifier)
        check_validity(spatial_scale_table.copy(), input_name = 'spatial_scale_table',
                       data_type = 'spatial_scales', column_identifier = column_identifier)
        # drop duplicate is needed ????
        data = add_spatial_ids(data.copy(),spatial_scale_table)

    if desired_scale_column_name not in data.columns:
        raise ValueError("spatial_scale_level {0} isn't in the spatial scale levels included in the data.\n".format(spatial_scale_level))


    if data_type == 'temporal':

        if 'temporal id' in data.columns: # integrated temporal id format
            temporal_identifier_column_name = 'temporal id'

        elif 'temporal id level 1' in data.columns: # non-integrated temporal id format
            # first integrated format for temporal id must constructed
            data = add_dummy_integrated_temporal_id(data.copy())
            temporal_identifier_column_name = 'dummy temporal id'

            non_integrated_temporal_levels = list(filter(lambda x: x.startswith('temporal id level '), data.columns))
            temporal_levels_data_frame = data[non_integrated_temporal_levels + ['dummy temporal id']]
            data.drop(non_integrated_temporal_levels, axis = 1, inplace = True)

        base_columns.append(temporal_identifier_column_name)
        data = data.drop_duplicates(subset = ['spatial id level 1',temporal_identifier_column_name]).copy()

    else:
        data = data.drop_duplicates(subset = ['spatial id level 1']).copy()


    # removing spatial id columns except for desired spatial scale ids
    extra_spatial_ids = list(filter(lambda x: x.startswith('spatial id '), data.columns))
    extra_spatial_ids.remove(desired_scale_column_name)
    data.drop(extra_spatial_ids, axis = 1, inplace = True)

    covariate_names = list(filter(lambda x: not x.startswith(('temporal id','spatial id','dummy temporal id')), data.columns))


    if aggregation_mode == 'mean':
        mean_covariates = covariate_names.copy()
    elif aggregation_mode == 'sum':
        sum_covariates = covariate_names.copy()
    elif type(aggregation_mode) == dict:
        extra_covariates_listed = list(set(aggregation_mode.keys())-set(covariate_names))
        if len(extra_covariates_listed) > 0:
            aggregation_mode = {covar:operator for covar,operator in aggregation_mode.items() if covar in covariate_names}
            if verbose == 1:
                print("\nWarning : Some of the covariates specified in aggregation_mode are not exist in the data.\n")
            if verbose == 2:
                print("\nWarning : Some of the covariates specified in aggregation_mode are not exist in the data:\n{0}\n".format(extra_covariates_listed))
        mean_covariates = [covar for covar,operator in aggregation_mode.items() if operator == 'mean']
        sum_covariates = [covar for covar,operator in aggregation_mode.items() if operator == 'sum']
    else:
        raise ValueError("aggregation_mode must be 'sum' or 'mean' or a dictionary with covariates name as the keys and 'sum' or 'mean' as the values")

    unspecified_covariates = list(set(covariate_names)-set(mean_covariates + sum_covariates))
    if len(unspecified_covariates)>0:
        if verbose < 2:
            print("\nWarning : The aggregation_mode is not specified for some of the covariates.\nThe mean operator will be used to aggregate these covariates' values.\n")
        if verbose == 2:
            print("\nWarning : The aggregation_mode is not specified for some of the covariates:\n{0}\nThe mean operator will be used to aggregate these covariates' values.\n".format(unspecified_covariates))
        mean_covariates = mean_covariates + unspecified_covariates

    if len(mean_covariates)>0:
        mean_covariates+=base_columns
        mean_data = data.copy()[mean_covariates]
        mean_data = mean_data.groupby(base_columns).mean()
        mean_data = mean_data.reset_index()

    if len(sum_covariates)>0:
        sum_covariates+=base_columns
        sum_data = data.copy()[sum_covariates]
        sum_data = sum_data.groupby(base_columns).sum(min_count=1)
        sum_data = sum_data.reset_index()

    if len(mean_covariates)>0 and len(sum_covariates)>0:
        data = pd.merge(mean_data,sum_data,on=base_columns)
    elif len(mean_covariates)>0:
        data = mean_data
    elif len(sum_covariates)>0:
        data = sum_data
    
    if 'dummy temporal id' in data.columns:
        data = pd.merge(temporal_levels_data_frame.drop_duplicates(), data, on = ['dummy temporal id']).drop_duplicates().copy()
        data.drop(['dummy temporal id'], axis=1, inplace = True)
        ordered_columns = [desired_scale_column_name]+list(temporal_levels_data_frame.columns.drop(['dummy temporal id']))+covariate_names
    elif 'temporal id' in data.columns:
        ordered_columns = [desired_scale_column_name,'temporal id']+covariate_names
    else:
        ordered_columns = [desired_scale_column_name]+covariate_names
        
    data = data.copy()[ordered_columns]
    
    data = rename_columns(data.copy(), column_identifier, 'deformalize')
                  
    return data

##############################################################################################
############################  transforming temporal scale ####################################
##############################################################################################

def temporal_scale_transform(data, column_identifier = None, temporal_scale_level = 2, augmentation = False, verbose = 0):
    
    if type(data) == str:
        data = pd.read_csv(data)
    data = rename_columns(data.copy(), column_identifier)
    check_validity(data.copy(), input_name = 'data', data_type = 'temporal', column_identifier = column_identifier)
    
    if type(temporal_scale_level) != int:
        raise TypeError("The temporal_scale_level must be of type int.\n")
        
    if temporal_scale_level == 1:
        if verbose > 0: print ('The temporal_scale_level = 1 is interpreted to no temporal transformation')
        return data
    
    # Next 4 lines save the secondary spatial scale levels information in a data frame to be added to the transformed data
    secondary_spatial_levels = list(filter(lambda x: x.startswith('spatial id level '), data.columns))
    spatial_levels_data_frame = data[secondary_spatial_levels].drop_duplicates().copy()
    secondary_spatial_levels.remove('spatial id level 1')
    data.drop(secondary_spatial_levels, axis = 1, inplace = True)
    
    ###################################### integrated temporal id format ################################
    
    if 'temporal id' in data.columns:
    
        scale_level_dict = {'sec':1, 'min':2, 'hour':3, 'day':4, 'week':5, 'month':6, 'year':7}
        scale_format = {'sec':'%Y/%m/%d %H:%M:%S', 'min':'%Y/%m/%d %H:%M', 'hour':'%Y/%m/%d %H', 'day':'%Y/%m/%d', 'week':'%Y/%m/%d', 'month':'%Y/%m', 'year':'%Y'}
        scale_second_units_number = {'sec':1, 'min':60, 'hour':3600, 'day':24*3600, 'week':7*24*3600, 'month':30*24*3600, 'year':365*24*3600}

        data = data.drop_duplicates(subset = ['spatial id level 1','temporal id']).copy()

        # determining input data temporal scale and transforming temporal id's to timestamp
        data , scale = check_integrated_temporal_id(data.copy())

        # determine the numerical desired temporal scale level based on the input data
        # temporal scale and user specified temporal_scale_level
        desired_temporal_scale_level = scale_level_dict[scale] + temporal_scale_level - 1
        if desired_temporal_scale_level > 7:
            raise Exception('The first temporal scale level recorded in the data is {0}. So the temporal scale level {1} is out of the supported range of temporal scale levels: second, minute, hour, day, week, month, year'.format(scale,temporal_scale_level))

        # get the nominal form of desired temporal scale level
        desired_temporal_scale = list(scale_level_dict.keys())[list(scale_level_dict.values()).index(desired_temporal_scale_level)]
        if verbose > 0 : print("\nTransformation of data to the temporal scale of the {0} is running.\n".format(desired_temporal_scale))
            
        # get number of smaller temporal units in each bigger temporal unit
        granularity = scale_second_units_number[desired_temporal_scale]//scale_second_units_number[scale]
        
        number_of_temporal_units=len(data['temporal id'].unique())
        number_of_spatial_units=len(data['spatial id level 1'].unique())
        
        if augmentation == False:

            # For the temporal scale of the week, each date must be replaced with the date of the first day
            # of the corresponding week
            if desired_temporal_scale == 'week' :
                data.loc[:,('temporal id')] = data['temporal id'].apply(lambda x : x - datetime.timedelta(days=x.weekday()))

            data.loc[:,('temporal id')] = data['temporal id'].apply(lambda x : datetime.datetime.strftime(x,scale_format[desired_temporal_scale]))
            
            # remove the data of the smaller scale temporal units in beggining and
            # ending of data which does'nt make up a complete bigger tempral scale unit
            max_temporal_id = max(data['temporal id'])
            min_temporal_id = min(data['temporal id'])
            if len(data[data['temporal id'] == max_temporal_id])<number_of_spatial_units*granularity:
                data = data[data['temporal id'] != max_temporal_id]
            if len(data[data['temporal id'] == min_temporal_id])<number_of_spatial_units*granularity:
                data = data[data['temporal id'] != min_temporal_id]
            
            base_columns = ['temporal id', 'spatial id level 1']
            data = data.groupby(base_columns).mean()
            data = data.reset_index()

        if augmentation == True:

            data = data.sort_values(by = ['temporal id' , 'spatial id level 1']).copy()
            data.reset_index(drop = True, inplace = True)

            transformed_data = pd.DataFrame(columns = data.columns)

            while number_of_temporal_units >= granularity:
                bigger_scale_unit_data = data.copy().tail(number_of_spatial_units*granularity) 
                current_temporal_id = data.copy().tail(1)['temporal id'].values[0]

                # average of covariates and target values on last bigger temporal scale unit for all spatial scale units
                bigger_scale_unit_data_average = bigger_scale_unit_data.groupby(['spatial id level 1']).mean().reset_index()
                bigger_scale_unit_data_average.loc[:,('temporal id')] = current_temporal_id

                transformed_data = transformed_data.append(bigger_scale_unit_data_average)
                # remove last smaller temporal scale unit for all spatial units from base data
                data = data.iloc[:-(number_of_spatial_units),:]
                number_of_temporal_units = number_of_temporal_units - 1

            data = transformed_data.copy()
            data.loc[:,('temporal id')] = data['temporal id'].apply(lambda x : datetime.datetime.strftime(x,scale_format[scale]))

        data = data.sort_values(by=['spatial id level 1','temporal id']).copy()
    
    ###################################### non_integrated temporal id format ################################
        
    if 'temporal id level 1' in data.columns: 
    
        # next 2 line removes the duplicate data samples having same spatial and temporal id's.
        data = add_dummy_integrated_temporal_id(data.copy(), start_level = 1)
        data = data.drop_duplicates(subset = ['spatial id level 1','dummy temporal id']).copy()

        desired_scale_column_name = 'temporal id level ' + str(temporal_scale_level)
        if desired_scale_column_name not in data.columns:
            raise ValueError("temporal_scale_level {0} is not in the temporal scale levels recorded in the data or is located after a gap in the temporal scale levels sequence and is therefore ignored. ".format(temporal_scale_level))
        elif verbose > 0 :
            if column_identifier is not None:
                print("\nTransformation of data to the temporal scale of the {0} is running.\n".format(column_identifier[desired_scale_column_name]))
            else:
                print("\nTransformation of data to the temporal scale of the {0} is running.\n".format(desired_scale_column_name))
        
        if augmentation == False :

            # remove dummy temporal id which is based on level 1 to add dummy temporal id based on 'temporal_scale_level'
            data.drop(['dummy temporal id'], axis = 1, inplace = True)
            data = add_dummy_integrated_temporal_id(data.copy(), start_level = temporal_scale_level)

            secondary_temporal_levels = list(filter(lambda x: x.startswith('temporal id level '), data.columns))
            temporal_levels_data_frame = data[secondary_temporal_levels + ['dummy temporal id']]
            data.drop(secondary_temporal_levels, axis = 1, inplace = True)

            base_columns = ['dummy temporal id', 'spatial id level 1']

            data = data.groupby(base_columns).mean()
            data = data.reset_index()
            data = pd.merge(temporal_levels_data_frame.drop_duplicates(), data, on = 'dummy temporal id', how = 'right').drop_duplicates().copy()
            # sorting columns
            data_columns = list(data.columns).copy()
            data_columns.remove('dummy temporal id')
            data = data[['dummy temporal id']+data_columns]


        elif augmentation == True :

            # get number of smaller temporal units in each bigger temporal unit
            granularity = data.groupby(['spatial id level 1',desired_scale_column_name]).count()['dummy temporal id'].max()

            secondary_temporal_levels = list(filter(lambda x: x.startswith('temporal id level '), data.columns))
            temporal_levels_data_frame = data[secondary_temporal_levels + ['dummy temporal id']]
            data.drop(secondary_temporal_levels, axis = 1, inplace = True)

            data = data.sort_values(by = ['dummy temporal id' , 'spatial id level 1']).copy()
            data.reset_index(drop = True, inplace = True)
            number_of_temporal_units=len(data['dummy temporal id'].unique())
            number_of_spatial_units=len(data['spatial id level 1'].unique())

            transformed_data = pd.DataFrame(columns = data.columns)

            while number_of_temporal_units >= granularity:
                bigger_scale_unit_data = data.copy().tail(number_of_spatial_units*granularity) 
                current_temporal_id = data.copy().tail(1)['dummy temporal id'].values[0]

                # average of covariates and target values on last bigger temporal scale unit for all spatial scale units
                bigger_scale_unit_data_average = bigger_scale_unit_data.groupby(['spatial id level 1']).mean().reset_index()
                bigger_scale_unit_data_average.loc[:,('dummy temporal id')] = current_temporal_id
                

                transformed_data = transformed_data.append(bigger_scale_unit_data_average)
                data = data.iloc[:-(number_of_spatial_units),:]# remove last smaller temporal scale unit for all spatial units from base data
                number_of_temporal_units = number_of_temporal_units - 1

            data = transformed_data.copy()

            data = pd.merge(temporal_levels_data_frame.drop_duplicates(), data, on = 'dummy temporal id', how = 'right').drop_duplicates().copy()

        data = data.sort_values(by=['spatial id level 1','dummy temporal id']).copy()
        data.drop(['dummy temporal id'] ,axis = 1 , inplace = True)

        
    # add secondary spatial scale levels back to data
    data = pd.merge(spatial_levels_data_frame.drop_duplicates(), data, on = 'spatial id level 1', how = 'right').drop_duplicates().copy()
    
    # sorting columns
    data_columns = list(data.columns).copy()
    data_columns.remove('spatial id level 1')
    data = data[['spatial id level 1']+data_columns]
    
    if column_identifier is not None:
        if 'temporal id' in data.columns: # integrated temporal_id format
            data = rename_columns(data, {key:value for key,value in column_identifier.items() if key != 'temporal id'}, 'deformalize')
        else: # non_integrated temporal_id format
            data = rename_columns(data, column_identifier, 'deformalize')
            
    if len(data)<1:
        raise Exception("The number of recorded units in the data with the specified temporal scale level is less than one.")
    
    return data

######################################################################################
############################  target modification ####################################
######################################################################################

# modifying target to the cumulative or differential or moving average mode

def target_modification(data, target_mode, column_identifier = None, verbose = 0):
    
    if type(data) == str:
        data = pd.read_csv(data)
    data = rename_columns(data.copy(), column_identifier)
    
    check_validity(data.copy(), input_name = 'data', data_type = 'temporal', column_identifier = column_identifier)
    
    if 'target' not in data.columns:
        raise ValueError("There is no column named 'target' in the input data, and the corresponding column is not specified in the column_identifier.\n")
    try:
        data.loc[:,('target')] = data['target'].astype(float)
    except ValueError:
        raise ValueError("The target column includes non-numerical values.\n")
        
    if (data['target'].isnull().values.any()) and (target_mode in ['cumulative', 'differential', 'moving average']):
        print("\nWarning: The target variable column includes Null values and therefore the resulting values of applying {0} target_mode is not valid.\n".format(target_mode))
        
    if 'temporal id' in data.columns: # integrated temporal id format
        temporal_identifier_column_name = 'temporal id'
    else: # Non_integrated temporal id format
        for level in range(1,200):
            if 'temporal id level ' + str(level) in data.columns:
                smallest_temporal_level = level
                break
        data = add_dummy_integrated_temporal_id(data.copy(), start_level = smallest_temporal_level)
        temporal_identifier_column_name = 'dummy temporal id'
        

    data = data.drop_duplicates(subset = [temporal_identifier_column_name,'spatial id level 1']).copy()
    data = data.sort_values(by=[temporal_identifier_column_name,'spatial id level 1']).copy()
    
    normal_target_df = data[['spatial id level 1', temporal_identifier_column_name, 'target']].rename(columns = {'target':'Normal target'})
    
    ######################################################################### cumulative
    
    if target_mode == 'cumulative':
        temporal_ids = data[temporal_identifier_column_name].unique()
        for i in range(len(temporal_ids)-1): 
            data.loc[data[temporal_identifier_column_name]==temporal_ids[i+1],'target']=\
            list(np.array(data.loc[data[temporal_identifier_column_name]==temporal_ids[i+1],'target'])+\
                  np.array(data.loc[data[temporal_identifier_column_name]==temporal_ids[i],'target']))
        
                
    ######################################################################### differential

    elif target_mode == 'differential': # make target differential
        reverse_temporal_ids = data[temporal_identifier_column_name].unique()[::-1]
        for i in range(len(reverse_temporal_ids)):
            temprl_unit = reverse_temporal_ids[i]
            past_temprl_unit = reverse_temporal_ids[i+1]
            data.loc[data[temporal_identifier_column_name] == temprl_unit,'target']=\
            list(np.array(data.loc[data[temporal_identifier_column_name]==temprl_unit,'target'])-\
                 np.array(data.loc[data[temporal_identifier_column_name]==past_temprl_unit,'target']))
            if i == len(reverse_temporal_ids)-2:
                break
            
    ######################################################################### moving average
            
    elif target_mode == 'moving average':
        
        ############### integrated temporal id format
        if temporal_identifier_column_name == 'temporal id': 
            
            scale_level_dict = {'sec':1, 'min':2, 'hour':3, 'day':4, 'week':5, 'month':6, 'year':7}
            scale_format = {'sec':'%Y/%m/%d %H:%M:%S', 'min':'%Y/%m/%d %H:%M', 'hour':'%Y/%m/%d %H', 'day':'%Y/%m/%d', 'week':'%Y/%m/%d', 'month':'%Y/%m', 'year':'%Y'}
            scale_second_units_number = {'sec':1, 'min':60, 'hour':3600, 'day':24*3600, 'week':7*24*3600, 'month':30*24*3600, 'year':365*24*3600}

            # determining input data temporal scale and transforming temporal id's to timestamp
            data , scale = check_integrated_temporal_id(data.copy())

            # determine the numerical level of the next temporal scale after input data
            # temporal scale for applying moving average on target values

            desired_temporal_scale_level = scale_level_dict[scale] + 1
            if desired_temporal_scale_level > 7:
                raise Exception("The temporal scale level of the data is {0}. Thus the 'moving average' target_mode couldn't be applied cause the temporal scale bigger than {0} is ambiguous.\n".format(scale))

             # get the nominal form of desired temporal scale level
            desired_temporal_scale = list(scale_level_dict.keys())[list(scale_level_dict.values()).index(desired_temporal_scale_level)]
            
            if verbose > 0:
                print("\nThe temporal scale level of the data is {0}, and using 'moving average' target_mode, the target value for each {0} is the average of values in the previous {1} of that {0}.\n".format(scale,desired_temporal_scale))

            # get number of smaller temporal units in each bigger temporal unit
            granularity = scale_second_units_number[desired_temporal_scale]//scale_second_units_number[scale]


            data = data.sort_values(by = ['temporal id' , 'spatial id level 1']).copy()
            data.reset_index(drop = True, inplace = True)
            number_of_temporal_units=len(data['temporal id'].unique())
            number_of_spatial_units=len(data['spatial id level 1'].unique())

            transformed_data = pd.DataFrame(columns = data.columns)

            while number_of_temporal_units >= granularity:

                bigger_scale_unit_data = data.copy().tail(number_of_spatial_units*granularity) 
                smaller_scale_unit_data = data.copy().tail(number_of_spatial_units)

                # average of target values on last bigger temporal scale unit for all spatial scale units
                bigger_scale_unit_data_average = bigger_scale_unit_data.groupby(['spatial id level 1']).mean().reset_index()
                # add averaged target values of last bigger temporal scale unit to the smaller temporal scale unit
                smaller_scale_unit_data.loc[:,('target')] = bigger_scale_unit_data_average.loc[:,('target')].tolist()

                transformed_data = transformed_data.append(smaller_scale_unit_data)
                # remove last smaller temporal scale unit for all spatial units from base data
                data = data.iloc[:-(number_of_spatial_units),:]
                number_of_temporal_units = number_of_temporal_units - 1

            data = transformed_data.copy()
            data.loc[:,('temporal id')] = data['temporal id'].apply(lambda x : datetime.datetime.strftime(x,scale_format[scale]))
            data = data.sort_values(by=['spatial id level 1','temporal id']).copy()
        
        ####################### non_integrated temporal id format
        
        elif temporal_identifier_column_name == 'dummy temporal id': 
            
            desired_scale_column_name = 'temporal id level ' + str(smallest_temporal_level+1)
            if desired_scale_column_name not in data.columns:
                raise Exception("The next bigger temporal scale after data current temporal scale (temporal scale level {0}) isn't included in the data or is located after a gap in the temporal scale levels sequence and is therefore ignored.\n".format(smallest_temporal_level))
                
            # get number of smaller temporal units in each bigger temporal unit
            granularity = data.groupby(['spatial id level 1',desired_scale_column_name]).count()['dummy temporal id'].max()

            data = data.sort_values(by = ['dummy temporal id' , 'spatial id level 1']).copy()
            data.reset_index(drop = True, inplace = True)
            number_of_temporal_units=len(data['dummy temporal id'].unique())
            number_of_spatial_units=len(data['spatial id level 1'].unique())

            transformed_data = pd.DataFrame(columns = data.columns)

            while number_of_temporal_units >= granularity:

                bigger_scale_unit_data = data.copy().tail(number_of_spatial_units*granularity) 
                smaller_scale_unit_data = data.copy().tail(number_of_spatial_units)

                # average of target values on last bigger temporal scale unit for all spatial scale units
                bigger_scale_unit_data_average = bigger_scale_unit_data.groupby(['spatial id level 1']).mean().reset_index()
                # add averaged target values of last bigger temporal scale unit to the smaller temporal scale unit
                smaller_scale_unit_data.loc[:,('target')] = bigger_scale_unit_data_average.loc[:,('target')].tolist()

                transformed_data = transformed_data.append(smaller_scale_unit_data)
                # remove last smaller temporal scale unit for all spatial units from base data
                data = data.iloc[:-(number_of_spatial_units),:]
                number_of_temporal_units = number_of_temporal_units - 1

            data = transformed_data.sort_values(by=['spatial id level 1','dummy temporal id']).copy()
            
    elif target_mode == 'normal':
        data = data
    else:
        raise ValueError("The specified target_mode is not recognized. The supported target_modes are:\n{'normal', 'cumulative', 'differential', 'moving average'}")
    
    data = pd.merge(data,
            normal_target_df[normal_target_df[temporal_identifier_column_name].isin(data[temporal_identifier_column_name].unique())],
            on = ['spatial id level 1', temporal_identifier_column_name], how = 'inner')
    
    if 'dummy temporal id' in data.columns:
        data.drop(['dummy temporal id'], axis = 1, inplace = True)
    
    data = rename_columns(data.copy(), column_identifier, 'deformalize')
        
    return data


#########################################################################################
############################  making historical data ####################################
#########################################################################################


def make_historical_data(data, forecast_horizon, history_length = 1, column_identifier = None,
                         futuristic_covariates = None, future_data_table = None, step = 1, verbose = 0):
    
    future_data = True
    if type(data) != dict :
        if type(data) == str :
            try:
                data = pd.read_csv(data)
            except FileNotFoundError:
                raise FileNotFoundError("File '{0}' does not exist.\n".format(data))
        elif type(data) != pd.DataFrame :
            raise TypeError("The input data must be of type DataFrame, string, or a dict containing temporal and spatial DataFrames or addresses.\n")
        data = rename_columns(data.copy(), column_identifier)
        
        check_validity(data.copy(), input_name = 'data', data_type = 'full', column_identifier = column_identifier)

    elif type(data) == dict:
        if 'temporal_data' in data.keys():
            
            if type(data['temporal_data']) == str:
                try:
                    temporal_data = pd.read_csv(data['temporal_data'])
                except FileNotFoundError:
                    raise FileNotFoundError("File '{0}' does not exist.\n".format(data['temporal_data']))
                    
            elif type(data['temporal_data']) == pd.DataFrame:
                temporal_data = data['temporal_data']
            else:
                raise ValueError("The value of the 'temporal id' key in the data dictionary must be a DataFrame or the address of temporal data.\n")
            temporal_data = rename_columns(temporal_data.copy(), column_identifier)
            check_validity(temporal_data.copy(), input_name = 'temporal_data', 
                           data_type = 'temporal', column_identifier = column_identifier)
            
        else:
            raise Exception("The data on temporal covariates and target variable must be passed to the function using data argument and as a DataFrame, Data address or value of 'temporal_data' key in the dictionary of data. But none is passed.\n")
        

        if ('spatial_data' in data.keys()) and (data['spatial_data'] is not None):
            
            if type(data['spatial_data']) == str:
                try:
                    spatial_data = pd.read_csv(data['spatial_data'])
                except FileNotFoundError:
                    raise ("File '{0}' does not exist.\n".format(data['spatial_data']))
                    
            elif type(data['spatial_data']) == pd.DataFrame:
                spatial_data = data['spatial_data']
            else:
                raise ValueError("The value of the 'spatial id' key in the data dictionary must be a DataFrame or the address of spatial data.\n")
            
            spatial_data = rename_columns(spatial_data.copy(), column_identifier)
            check_validity(spatial_data.copy(), input_name = 'spatial_data',
                           data_type = 'spatial', column_identifier = column_identifier)
            
            if len(list(set(temporal_data['spatial id level 1'].unique())-set(spatial_data['spatial id level 1'].unique())))>0:
                print("\nWarning: Some of the spatial units in the temporal_data, are not recorded in the spatial_data. These spatial units will be ignored.\n")
                temporal_data = temporal_data[temporal_data['spatial id level 1'].isin(spatial_data['spatial id level 1'].unique())]
                
            data = pd.merge(temporal_data, spatial_data, on = 'spatial id level 1', how = 'left')
            
        else:
            data = temporal_data.copy()
    
    # Non_integrated temporal id format
    if 'temporal id' not in data.columns: 
            # find smallest temporal level
            for level in range(1,200):
                if 'temporal id level ' + str(level) in data.columns:
                    smallest_temporal_level = level
                    break
            # add integrated temporal id
            data = add_dummy_integrated_temporal_id(data.copy(), start_level = smallest_temporal_level)
            # remove discrete temporal id columns
            extra_temporal_ids = list(filter(lambda x:x.startswith('temporal id level '), data.columns))
            data.drop(extra_temporal_ids, axis = 1, inplace = True)
            data.rename(columns = {'dummy temporal id':'temporal id'}, inplace = True)
            
            
    if 'target' not in data.columns:
        raise ValueError("There is no column named 'target' in input data")
    
    if column_identifier is None:
        spatial_covariates = list(filter(lambda x:x.startswith('spatial covariate'), data.columns))
        temporal_covariates = list(filter(lambda x:x.startswith('temporal covariate'), data.columns))
    else:
        if 'spatial covariates' in column_identifier.keys():
            spatial_covariates = [item for item in list(column_identifier['spatial covariates']) if item in data.columns]
        else: spatial_covariates = []
        temporal_covariates = [item for item in list(column_identifier['temporal covariates']) if item in data.columns]
        
    all_covariates = list(filter(lambda x: not x.startswith(('temporal id', 'spatial id', 'target', 'Normal target')), data.columns))
    extra_columns = list(set(all_covariates)-set(spatial_covariates + temporal_covariates))
    
    if len(extra_columns) > 0 :
        print("\nWarning: Input data column names must match one of the specified formats:\n{'temporal id', 'temporal id level x', 'spatial id', 'spatial id level x', 'temporal covariate x', 'spatial covariate x', 'target'}, or be specified in column_identifier.\n")
        print("But the names of some of the columns do not match any of the supported formats and is not mentioned in column_identifier:\n{0}\nThis columns will be ignored.\n".format(extra_columns))
    
    ######################## check type of future_data_table
    
    if type(future_data_table) == str:
        try:
            future_data_table = pd.read_csv(future_data_table)
        except FileNotFoundError:
            raise ("File '{0}' does not exist.\n".format(future_data_table))
            
    elif (type(future_data_table) != pd.DataFrame) and (future_data_table is not None):
        raise TypeError("The future_data_table must be a data frame or address of the data frame containing the values of futuristic covariates in the future.")
    
            
    ######################## check type and validity of input history_length and futuristic_covariates #####################
    if type(history_length) == int:
        if history_length == 0 : history_length = 1
        history_length_dict = {covar:history_length for covar in temporal_covariates}
        max_history_length = history_length # maximum history length of all temporal covariates
        
    elif type(history_length) == dict:
        
        key_list = list(history_length.keys())
        for item in key_list:
            if type(item) == tuple:
                for key in item:
                    history_length[key] = history_length[item]
                del history_length[item]
            
        extra_hist_covariates = list(set(history_length.keys()) - set(temporal_covariates))
        
        if (len(extra_hist_covariates) > 0) and (verbose > 0):
            print("\nWarning: The following keys in the history_length do not exist in the input data temporal covariates, and thus will be ignored:\n{0}\n".format(extra_hist_covariates))
        
        history_length = {key:value for key, value in history_length.items() if key not in extra_hist_covariates}
        # The history length of 0 is interpreted to no historical values which is same as history length = 1
        for key, value in history_length.items():
            if value == 0: history_length[key] = 1
        
        history_length_dict = history_length.copy()
        
        # if covariate doesnt mentioned in history_length or futuristic_covariates
        unspecified_history_covariates = list(set(temporal_covariates)-set(history_length_dict.keys()))
        if (futuristic_covariates is not None) and (type(futuristic_covariates) == dict):
            unspecified_history_covariates = list(set(unspecified_history_covariates) - set(futuristic_covariates.keys()))
            
        if len(unspecified_history_covariates)> 0:
            print("\nWarning: The history length of some temporal covariates is not specified in history_length:\n{0}\nFor these covariates the history length of 1 will be considered.\n".format(unspecified_history_covariates))
        for covar in unspecified_history_covariates:
            history_length_dict[covar] = 1
        for covar in history_length_dict.keys():
            if type(history_length_dict[covar]) != int:
                raise ValueError("The specified history length for each covariate in the history_length dict must be of type int.\n")
        max_history_length = max(history_length_dict.values())
        
    else:
        raise TypeError("The history_length must be of type int or dict.\n")
        
    if futuristic_covariates is not None:
        if type(futuristic_covariates) == dict:
            for covar in futuristic_covariates.keys():
                if (len(futuristic_covariates[covar])!=2) or (futuristic_covariates[covar][1]-futuristic_covariates[covar][0]<0):
                    raise ValueError("The temporal interval of each futuristic covariate must be specified in futuristic_covariates dict using a list including start and end of the interval as first and second item.\n")
                elif (type(futuristic_covariates[covar][0])!=int)or(type(futuristic_covariates[covar][1])!=int):
                    raise ValueError("The start and end point of futuristic covariates temporal interval must be of type int.\n")
                elif futuristic_covariates[covar][1] > forecast_horizon:
                    raise ValueError("The end point of futuristic covariates temporal interval must be smaller than forecast_horizon.\n")
                else:
                    history_length_dict['future '+covar] = futuristic_covariates[covar][1] - futuristic_covariates[covar][0] + 1

        else:
            raise TypeError("The futuristic_covariates must be of type dict.\n")
        
        key_list = list(futuristic_covariates.keys())
        for item in key_list:
            if type(item) == tuple:
                for key in item:
                    futuristic_covariates[key] = futuristic_covariates[item]
                del futuristic_covariates[item]
            
        invalid_futuristic_covariates = list(set(futuristic_covariates.keys()) - set(temporal_covariates))
        
        if (len(invalid_futuristic_covariates) > 0) and (verbose > 0):
            print("\nWarning: The following keys in the futuristic_covariates do not exist in the input data temporal covariates, and thus will be ignored:\n{0}\n".format(invalid_futuristic_covariates))
        
        futuristic_covariates = {key:value for key, value in futuristic_covariates.items() if key not in invalid_futuristic_covariates}
        
    ######################## check type and validity of forecast horizon
    
    if type(forecast_horizon) != int:
        raise TypeError("The forecast_horizon must be of type int.")
    elif forecast_horizon == 0:
        forecast_horizon = 1
    
    ############## preparing data
    
    # adding future values of futuristic covariates to the data
    if future_data_table is not None:
        data, future_data_table, futuristic_temporal_units = current_future(data = data.copy(), future_data_table = future_data_table,
                                                              futuristic_covariates = futuristic_covariates,
                                                              column_identifier = column_identifier , mode = 'add')
    else:
        _, _, futuristic_temporal_units = current_future(data = data.copy(),
                                                              future_data_table = None,
                                                              futuristic_covariates = futuristic_covariates,
                                                              column_identifier = column_identifier , mode = 'split')
    
    
    number_of_futuristic_temporal_units = len(futuristic_temporal_units)
    data = data.drop_duplicates(subset = ['temporal id','spatial id level 1']).copy()
    data = data.sort_values(by = ['temporal id','spatial id level 1']).copy()
    
    ####################################### main part #######################################

    if verbose > 0:
        if type(history_length) == int :
            print("\nMaking historical data with the forecast horizon of {0} and history length of {1} is running.\n".format(forecast_horizon, history_length))
        else:
            print("\nMaking historical data with the forecast horizon of {0} is running.\n".format(forecast_horizon))
            
    result = pd.DataFrame()  # we store historical data in this dataframe
    total_number_of_spatial_units = len(data['spatial id level 1'].unique())
    total_number_of_temporal_units = len(data['temporal id'].unique())
    
    if total_number_of_temporal_units + (step*(- max_history_length - forecast_horizon + 1)) <= 0:
        if verbose > 0 :
            print("\nWarning: The specified history length and forecast horizon is too large for the number of recorded temporal units in the input data.\n")
        return None
    
    if futuristic_covariates is None:
        futuristic_covariates = {}
    
    # in this loop we make historical data
    for covar in all_covariates:
        # if covariate is time dependant
        if covar in temporal_covariates:
            
            
            temporal_data_frame = data[[covar]] # selecting column of the covariate that is being processed
            # shift data to the size of futuristic temporal interval end point
            if covar in futuristic_covariates.keys():
                covar_history_length = history_length_dict['future '+covar]
                threshold = futuristic_covariates[covar][0]
    
                while threshold != futuristic_covariates[covar][1]+1:
                    temp = temporal_data_frame.tail((total_number_of_temporal_units + (step*(- max_history_length - threshold + 1)))*total_number_of_spatial_units).reset_index(drop=True)
                    temp.rename(columns={covar: (covar.replace(' ','_') + ' t+' + str(threshold))}, inplace=True) # renaming column
                    result = pd.concat([result, temp], axis=1)
                    threshold += 1
                
            if covar in history_length_dict.keys():    
                covar_history_length = history_length_dict[covar]
                # the first temporal unit of historical data is determined based on the maximum history length of covariates
                # therefore data of covariates with smaller history length should be shifted forward
                if covar_history_length < max_history_length:
                    temporal_data_frame = temporal_data_frame.iloc[(step*(max_history_length-covar_history_length))*total_number_of_spatial_units:]

                threshold = 0
                while threshold != covar_history_length:

                    # if future_data is true, the feature values of the last temporal units with the size of 
                    # forecast_horizon (which their target variable values are unknown) will be considered in historical data
                    if future_data == False:
                        temp = temporal_data_frame.head((total_number_of_temporal_units + (step*(- max_history_length - forecast_horizon + 1)))*total_number_of_spatial_units).copy().reset_index(drop=True)
                    else:
                        temp = temporal_data_frame.head((total_number_of_temporal_units + (step*(- max_history_length + 1)))*total_number_of_spatial_units).copy().reset_index(drop=True)

                    temp.rename(columns={covar: (covar.replace(' ','_') + ' t-' + str(covar_history_length-threshold-1))}, inplace=True) # renaming column  

                    result = pd.concat([result, temp], axis=1)
                    # deleting the values in first day in temporal_data_frame dataframe (similiar to shift)
                    temporal_data_frame = temporal_data_frame.iloc[step*total_number_of_spatial_units:]
                    threshold += 1
     
        # if covariate is independant of time
        elif covar in spatial_covariates:
            
            temporal_data_frame = data[[covar]]
            if future_data == False:
                temp = temporal_data_frame.head((total_number_of_temporal_units + (step*(- max_history_length - forecast_horizon + 1)))*total_number_of_spatial_units).copy().reset_index(drop=True)
            else:
                temp = temporal_data_frame.head((total_number_of_temporal_units + (step*(- max_history_length + 1)))*total_number_of_spatial_units).copy().reset_index(drop=True)
            
            temp.rename(columns={covar: (covar.replace(' ','_'))}, inplace=True)
            if covar == 'Target':
                temp.rename(columns={covar: ('Target_0')}, inplace=True)
            result = pd.concat([result, temp], axis=1)
    
    # next 6 lines is for spatial id code to final dataframe
    temporal_data_frame = data[['spatial id level 1']]
    if future_data == False:
        temp = temporal_data_frame.head((total_number_of_temporal_units + (step*(- max_history_length - forecast_horizon + 1)))*total_number_of_spatial_units).copy().reset_index(drop=True)
    else:
        temp = temporal_data_frame.head((total_number_of_temporal_units + (step*(- max_history_length + 1)))*total_number_of_spatial_units).copy().reset_index(drop=True)
    result.insert(0, 'spatial id', temp)

    # next 7 lines is for adding id of temporal unit (t) to final dataframe
    temporal_data_frame = data[['temporal id']]
    temporal_data_frame = temporal_data_frame[total_number_of_spatial_units*(step*(max_history_length - 1)):]
    if future_data == False:
        temp = temporal_data_frame.head((total_number_of_temporal_units + (step*(- max_history_length - forecast_horizon + 1)))*total_number_of_spatial_units).copy().reset_index(drop=True)
    else:
        temp = temporal_data_frame.head((total_number_of_temporal_units + (step*(- max_history_length + 1)))*total_number_of_spatial_units).copy().reset_index(drop=True)
    result.insert(1, 'temporal id', temp)

    # next 3 lines is for adding target to final dataframe
    temporal_data_frame = data[['target']]
    temporal_data_frame = temporal_data_frame.tail((total_number_of_temporal_units + (step*(- max_history_length - forecast_horizon + 1)))*total_number_of_spatial_units).reset_index(drop=True)
    result.insert(2, 'Target', temporal_data_frame)

    # next 4 lines is for adding normal target to final dataframe if target mode is not normal
    if 'Normal target' in data.columns:
        temporal_data_frame = data[['Normal target']]
        temporal_data_frame = temporal_data_frame.tail((total_number_of_temporal_units + (step*(- max_history_length - forecast_horizon + 1)))*total_number_of_spatial_units).reset_index(drop=True)
        result.insert(3, 'Normal target', temporal_data_frame)
    
    for i in result.columns:
        if (i.endswith('t-0')) and (i not in spatial_covariates):
            result.rename(columns={i: i[:-2]}, inplace=True)
            
    # last samples are related to the futuristic data and has no values for temporal covariates
    # so must be removed from historical data
    if number_of_futuristic_temporal_units > 0:
        number_of_spatial_units = len(result['spatial id'].unique())
        result = result.sort_values(by = ['temporal id', 'spatial id'])
        result = result.iloc[:-(number_of_futuristic_temporal_units*number_of_spatial_units)]
    
    return result

################################################################################################
############################  base function data preprocess ####################################
################################################################################################ 

def preprocess_data(data, forecast_horizon, history_length = 1, column_identifier = None, spatial_scale_table = None,
                    spatial_scale_level = 1, temporal_scale_level = 1,
                    target_mode = 'normal', imputation = True, aggregation_mode = 'mean', augmentation = False,
                    futuristic_covariates = None, future_data_table = None, save_address = None, verbose = 0):
    
    
    
    ####################
    
    spatial_covariates = []
    temporal_covariates = []
    granularity = None
    target_granularity = None
    
    # prepare data (renaming columns, remove extra columns and get spatial and temporal data separately)
    temporal_data, spatial_data = prepare_data(data.copy(), column_identifier)
    
    
    #################### get list of covariates
    
    # if type of each column is specified in column_identifier
    if column_identifier is not None:
        temporal_covariates =  list(set(column_identifier['temporal covariates']) & set(temporal_data.columns))
        if ('spatial covariates' in column_identifier.keys()) and (spatial_data is not None):
            spatial_covariates =  list(set(column_identifier['spatial covariates']) & set(spatial_data.columns))

    # if type of columns are clear based on column name
    else:
        temporal_covariates = list(filter(lambda x: x.startswith('temporal covariate'), temporal_data.columns))
        if spatial_data is not None:
            spatial_covariates = list(filter(lambda x: x.startswith('spatial covariate'), spatial_data.columns))

    
    # if input data has no covariate (spatial or temporal)
    if len(temporal_covariates) == 0:
        if (spatial_data is None) or (len(spatial_covariates) == 0):
            raise ValueError("There is no spatial or temporal covariate included in input data")
            
    # if input data has no target
    if 'target' not in temporal_data.columns:
        raise ValueError("The target variable is not recorded in input data and doesn't specified in column_identifier.\nmissing column: 'target'")

    ############## check the futuristic_covariate input validity
    
    if futuristic_covariates is not None:
        if type(futuristic_covariates) == dict:
            for item in futuristic_covariates.keys():
                if (len(futuristic_covariates[item])!=2) or (futuristic_covariates[item][1]-futuristic_covariates[item][0]<0):
                    raise ValueError("The temporal interval of each futuristic covariate must be specified in futuristic_covariates dict using a list including start and end of the interval as first and second item.\n")
                elif (type(futuristic_covariates[item][0])!=int)or(type(futuristic_covariates[item][1])!=int):
                    raise ValueError("The start and end point of futuristic covariates temporal interval must be of type int.\n")
                elif futuristic_covariates[item][1] > forecast_horizon:
                    raise ValueError("The end point of futuristic covariates temporal interval must be smaller than forecast_horizon.\n")
        else:
            raise TypeError("The futuristic_covariates must be of type dict.\n")
            
        key_list = list(futuristic_covariates.keys())    
        for item in futuristic_covariates.keys():
            if type(item) == tuple:
                for key in item:
                    futuristic_covariates[key] = futuristic_covariates[item]
                del futuristic_covariates[item]

            
        # Removing invalid covariates from data and produce warning if is needed
        invalid_futuristic_covariates = list(set(futuristic_covariates.keys()) - set(temporal_covariates))    
        if (len(invalid_futuristic_covariates) > 0) and (verbose > 0):
            print("\nWarning: The following keys in the futuristic_covariates do not exist in the input data covariates, and thus will be ignored:\n{0}\n".format(invalid_futuristic_covariates))

        futuristic_covariates = {key:value for key, value in futuristic_covariates.items() if key not in invalid_futuristic_covariates}

            
    ############# check the history_length input validity
    
    if type(history_length) == dict :
        for covar in history_length.keys():
            if (type(history_length[covar]) != int) and (covar in temporal_covariates) :
                  raise TypeError("\nThe maximum history length of covariates specified in history_length must be of type int.")
                    
        key_list = list(history_length.keys())
        for item in key_list:
            if type(item) == tuple:
                for key in item:
                    history_length[key] = history_length[item]
                del history_length[item]
            
        extra_hist_covariates = list(set(history_length.keys()) - set(temporal_covariates))
        
        if (len(extra_hist_covariates) > 0) and (verbose > 0):
            print("\nWarning: The following keys in the history_length do not exist in the input data temporal covariates, and thus will be ignored:\n{0}\n".format(extra_hist_covariates))
        
        history_length = {key:value for key, value in history_length.items() if key not in extra_hist_covariates}
        
    ######################## check type and validity of forecast horizon
    
    if type(forecast_horizon) != int:
        raise TypeError("The forecast_horizon must be of type int.")
    elif forecast_horizon == 0:
        forecast_horizon = 1
        
    ######################## check type and temporal_scale_level and augmentation
    
    if (temporal_scale_level < 2) and (augmentation == True):
        raise Exception("The augmentation can only be performed for temporal_scale_level greater than one.")
        
    ######################## check type of future_data_table
    
    if type(future_data_table) == str:
        try:
            future_data_table = pd.read_csv(future_data_table)
        except FileNotFoundError:
            raise FileNotFoundError("File '{0}' does not exist.\n".format(future_data_table))
            
    elif (type(future_data_table) != pd.DataFrame) and (future_data_table is not None):
        raise TypeError("The future_data_table must be a data frame or address of the data frame containing the values of futuristic covariates in the future.")
    
    ############################## Imputation ##############################
    
    if imputation == True :
        
        # removing the rows related to the future data before imputation
        if future_data_table is None:
            temporal_data, future_data_table, _ = current_future(data = temporal_data.copy(), future_data_table = None,
                                                              futuristic_covariates = futuristic_covariates,
                                                              column_identifier = column_identifier , mode = 'split')
        
        if verbose > 0:
            print("-"*45+"\nThe imputation of missing values is running.\n"+"-"*45+"\n")
            
        temporal_data = impute(temporal_data.copy(), None, verbose = 0)
        if spatial_data is not None:
            number_of_raw_spatial_units = spatial_data.shape[0]
            spatial_data.dropna(inplace = True)
            number_of_removed_spatial_units = number_of_raw_spatial_units - spatial_data.shape[0]

            if number_of_removed_spatial_units == number_of_raw_spatial_units:
                raise ValueError("All the spatial units include missing values for spatial covariates. Therefore, no spatial unit remains to make a prediction.\n")
            elif number_of_removed_spatial_units > 0:
                print('\nWarning: The number of {0} spatial units has missing values for spatial covariates and will be removed from the data.\n'.format(number_of_removed_spatial_units))

        # Holding only the spatial units that are common in both data frames
        if spatial_data is not None:
            temporal_data = temporal_data[temporal_data['spatial id level 1'].isin(spatial_data['spatial id level 1'].unique())]
            spatial_data = spatial_data[spatial_data['spatial id level 1'].isin(temporal_data['spatial id level 1'].unique())]
    

    
    # adding future values of futuristic covariates to the data and get the futuristic temporal units
    if future_data_table is not None:
        temporal_data, future_data_table, futuristic_temporal_units = current_future(data = temporal_data.copy(), future_data_table = future_data_table,
                                                              futuristic_covariates = futuristic_covariates,
                                                              column_identifier = column_identifier , mode = 'add')
    else:
        _, _, futuristic_temporal_units = current_future(data = temporal_data.copy(),
                                                              future_data_table = None,
                                                              futuristic_covariates = futuristic_covariates,
                                                              column_identifier = column_identifier , mode = 'split')
    
    ############################## spatial scale transform ##############################
    
    if spatial_scale_level > 1:
        
        if verbose > 0:
            print("-"*65+"\nTransformation of data to the desired spatial scale is running.\n"+"-"*65+"\n")
            
        if type(aggregation_mode) == dict:
            extra_covariates_listed = list(set(aggregation_mode.keys())-set(spatial_covariates + temporal_covariates + ['target']))
            unspecified_covariates = list(set(spatial_covariates + temporal_covariates + ['target'])-set(aggregation_mode.keys()))

            if len(extra_covariates_listed) > 0:
                print("\nWarning : Some of the covariates specified in aggregation_mode are not exist in the data:\n{0}".format(extra_covariates_listed))
                aggregation_mode = {covariate : mode for covariate,mode in aggregation_mode.items() if covariate not in extra_covariates_listed}

            if len(unspecified_covariates)>0:
                if verbose < 2:
                    print("\nWarning : The aggregation_mode is not specified for some of the covariates.\nThe mean operator will be used to aggregate these covariates' values.\n")
                if verbose == 2:
                    print("\nWarning : The aggregation_mode is not specified for some of the covariates:\n{0}\nThe mean operator will be used to aggregate these covariates' values.\n".format(unspecified_covariates))
                for covar in unspecified_covariates:
                      aggregation_mode[covar] = 'mean'

            if spatial_data is not None:
                    spatial_aggregation_mode = {covariate : mode for covariate,mode in aggregation_mode.items() if covariate in spatial_covariates}

            temporal_aggregation_mode = {covariate : mode for covariate,mode in aggregation_mode.items() if covariate in temporal_covariates + ['target']}

        # if aggregation_mode is not dict and is same for all covariates           
        else:
            spatial_aggregation_mode = temporal_aggregation_mode = aggregation_mode
            
        if spatial_scale_table is not None:
            spatial_scale_table = rename_columns(spatial_scale_table.copy(), column_identifier)
        
        # transformation
        temporal_data = spatial_scale_transform(temporal_data.copy(), 'temporal', column_identifier = None, spatial_scale_table = spatial_scale_table, spatial_scale_level = spatial_scale_level, aggregation_mode = temporal_aggregation_mode, verbose = 0)
        if spatial_data is not None:
            spatial_data = spatial_scale_transform(spatial_data.copy(), 'spatial', column_identifier = None, spatial_scale_table = spatial_scale_table, spatial_scale_level = spatial_scale_level, aggregation_mode = spatial_aggregation_mode, verbose = 0)
            # if the desired spatial scale has only one spatial unit, the spatial covariates couldn't be used for prediction
            if len(spatial_data)==1:
                spatial_data = None
                print("\nWarning: The desired spatial scale has only one spatial unit in the data, so spatial covariates' values is same for all the data samples and couldn't be used for prediction.\n")
        
        # to pass the data to the other functions, it must be same as raw data 
        spatial_identifier = list(filter(lambda x: x.startswith(('spatial id')),temporal_data.columns))[0]
        temporal_data.rename(columns = {spatial_identifier:'spatial id level 1'}, inplace = True)
        
        if spatial_data is not None:
            spatial_identifier = list(filter(lambda x: x.startswith(('spatial id')),spatial_data.columns))[0]
            spatial_data.rename(columns = {spatial_identifier:'spatial id level 1'}, inplace = True)
            
    ############################## temporal scale transform ##############################
    
    if temporal_scale_level > 1:
        
        if verbose > 0:
            print("-"*65+"\nTransformation of data to the desired temporal scale is running.\n"+"-"*65+"\n")
            
        # if user prefer to augment data granularity (number of smaller scale temporal units in the bigger scale temporal unit)
        # will be needed for making historical data
        if augmentation == True:
            granularity = find_granularity(temporal_data.copy(), temporal_scale_level)
            
        # filling up the future temporal units null values with inf to recognize the future temporal units
        # after temporal transform and return its values to null for non futuristic covariates
        temporal_data = fill_future_nulls(data = temporal_data.copy(), futuristic_temporal_units = futuristic_temporal_units)
        
        # transformation
        temporal_data = temporal_scale_transform(temporal_data.copy(), temporal_scale_level = temporal_scale_level, augmentation = augmentation, verbose = 0)
        
        # return null values of non futuristic covariates in future temporal units
        temporal_data = temporal_data.replace([np.inf, -np.inf], np.nan)
        
        # to pass the data to the other functions, it must be same as raw data 
        # the next lines reset the names of temporal id level columns to start from level one (only needed for non_integrated temporal id format)
        if augmentation == False:

            current_smallest_temporal_level = 'temporal id level '+str(temporal_scale_level)

            # if non_integrated temporal id format       
            if current_smallest_temporal_level in temporal_data.columns:
                temporal_data.rename(columns = {current_smallest_temporal_level:'temporal id level 1'}, inplace = True)
                for level in range(temporal_scale_level+1, 200):
                    if 'temporal id level '+str(level) in temporal_data.columns:
                        # shift the level by temporal_scale_level units
                        shifted_level = level - temporal_scale_level + 1
                        temporal_data.rename(columns = {'temporal id level '+str(level):'temporal id level '+str(shifted_level)}, inplace = True)
                    else:
                        break
    ############################## target modification ##############################
    
    if (target_mode != 'normal') and (verbose > 0):
            print("-"*35+"\nTarget modification is running.\n"+"-"*35+"\n")
        
    # removing the rows related to the future data before target modification cause it may fill the value of target in future
    if future_data_table is None:
        temporal_data, future_data_table, _ = current_future(data = temporal_data.copy(), future_data_table = None,
                                                          futuristic_covariates = futuristic_covariates,
                                                          column_identifier = column_identifier , mode = 'split')

    if (target_mode == 'moving average'):
        target_granularity = find_granularity(data = temporal_data.copy(), temporal_scale_level = None)

    temporal_data = target_modification(temporal_data, target_mode = target_mode, verbose = 1)
    
    if future_data_table is not None:
        temporal_data, future_data_table, _ = current_future(data = temporal_data.copy(), future_data_table = future_data_table,
                                                              futuristic_covariates = futuristic_covariates,
                                                              column_identifier = column_identifier , mode = 'add')
              
     
    ############################## make historical data ##############################    
    
    if verbose > 0:
        print("-"*35+"\nMaking historical data is running.\n"+"-"*35+"\n")
        
    # next lines is for detecting total number of temporal units in the data
    # to produce warning if it is less than needed number
    if 'temporal id' in temporal_data.columns:
        total_number_of_temporal_units =  len(temporal_data['temporal id'].unique())
    else: 
        # add integrated temporal id
        temporal_data = add_dummy_integrated_temporal_id(temporal_data.copy(), start_level = 1)
        total_number_of_temporal_units =  len(temporal_data['dummy temporal id'].unique())
        temporal_data = temporal_data.drop(['dummy temporal id'],axis = 1)    
    
    # if augmentation == True, for each sample in temporal unit i, if this sample is in index u,
    # the next sample in temporal unit i+1 is placed in index u+granularity. so in making historical data,
    # the step size to go back and get historical values of covariates or move forward to consider future 
    # values of the target variable must be equal to the granularity
                  
    if augmentation == True:
        step = granularity
    else:
        step = 1
    
    if type(history_length) == int:
        
        if total_number_of_temporal_units + (step*(- history_length - forecast_horizon + 1)) <= 0:
            print("\nWarning: The specified history length and forecast horizon is too large for the number of recorded temporal units in the input data.\n")
        
        
        # input data
        if spatial_data is not None:
            make_hist_data = {'temporal_data':temporal_data.copy(),'spatial_data':spatial_data.copy()}
        else:
            make_hist_data = {'temporal_data':temporal_data.copy(),'spatial_data':spatial_data}
                
        historical_data = make_historical_data(data = make_hist_data,
                                               forecast_horizon = forecast_horizon, column_identifier = column_identifier,
                                               history_length = history_length, futuristic_covariates = futuristic_covariates,
                                               future_data_table = None, step = step, verbose = 0)
        if historical_data is not None:
            historical_data = recorrect_hist_data(historical_data, augmentation, granularity, target_mode, target_granularity)

            if save_address is not None:
                try:
                    historical_data.to_csv(save_address+'historical_data h={0}.csv'.format(history_length))
                except FileNotFoundError:
                        print("The address '{0}' is not valid.".format(save_address))

    elif type(history_length) == dict:
        historical_data = {}
        unspecified_covariate = []
                  
        for covar in temporal_covariates:
            if covar not in history_length.keys():
                if futuristic_covariates is None:
                    history_length[covar] = 1
                    unspecified_covariate.append(covar)
                elif covar not in futuristic_covariates.keys():
                    history_length[covar] = 1
                    unspecified_covariate.append(covar)

        if len(unspecified_covariate)>0:
            if verbose < 2:
                print("\nWarning: The maximum history length of some covariates is not specified in history_length. The history length of 1 will be considered for these covariates.\n")
            elif verbose == 2:
                print("\nWarning: The maximum history length of some covariates is not specified in history_length. The history length of 1 will be considered for these covariates:\n{0}\n".format(unspecified_covariate))
        
        # The max history length of 0 is interpreted to no historical values which is same as history length = 1
        for key, value in history_length.items():
            if value == 0: history_length[key] = 1
                
        current_history_length = {covar : 0 for covar in history_length.keys()} 
        
        max_history_length = max(history_length.values())
        
        impossible_histories = []
        for hist in range(1,max_history_length+1):
            if total_number_of_temporal_units + (step*(- hist - forecast_horizon + 1)) <= 0:
                impossible_histories.append(hist)
        if len(impossible_histories) > 0:
            print("Warning: The number of temporal units in the data is not enough to construct a historical data with the specified forecast horizon and the history length(s):\n{0}".format(impossible_histories))
        
        for stage in range(1,max_history_length+1):
            for covar in history_length.keys():
                  if current_history_length[covar]+1 <= history_length[covar]:
                        current_history_length[covar] = current_history_length[covar] + 1
                        
            # input data
            if spatial_data is not None:
                make_hist_data = {'temporal_data':temporal_data.copy(),'spatial_data':spatial_data.copy()}
            else:
                make_hist_data = {'temporal_data':temporal_data.copy(),'spatial_data':None}
                
            historical_data[stage] = make_historical_data(data = make_hist_data,
                                                    forecast_horizon = forecast_horizon, column_identifier = column_identifier,
                                                    history_length = current_history_length,
                                                    futuristic_covariates = futuristic_covariates,
                                                    future_data_table = None,
                                                    step = step, verbose = 0)
            
            if historical_data[stage] is not None:
                
                historical_data[stage] = recorrect_hist_data(historical_data[stage], augmentation, granularity, target_mode, target_granularity)

                if save_address is not None:
                    try:
                        historical_data[stage].to_csv(save_address+'historical_data h=' + str(stage) +'.csv', index = False)
                    except FileNotFoundError:
                            print("The address '{0}' is not valid.".format(save_address))
                            
        historical_data_list =  [value for key,value in historical_data.items()]
        historical_data = historical_data_list
        
    return historical_data

######################

def plot_data(data, spatial_scale_table, temporal_covariate = 'default' ,spatial_scale = 1, spatial_id = None,
              temporal_scale = 1, temporal_range = None, column_identifier = None, month_format_print=False,
              saving_plot_path = None):
    
    ######################### check validity of input arguments    
    if type(data) == str:
        try:
            data = pd.read_csv(data)
        except FileNotFoundError:
            raise FileNotFoundError("File '{0}' does not exist.\n".format(data))
            
    elif type(data) != pd.DataFrame:
        raise TypeError("The input data must be of type DataFrame or string.")
    
    df = rename_columns(data, column_identifier)
    check_validity(df.copy(), input_name = 'data', data_type = 'temporal', column_identifier = column_identifier)
    
    if type(spatial_id) == list:
        if len(spatial_id)>3 :
            print("The number of spatial_ids must be a maximum of 3.")
        spatial_id = spatial_id[:3]
    elif spatial_id is not None:
        raise TypeError("The spatial_id must be of type list.")
    
    ################## spatial scale transform #################
    
    if spatial_scale > 1:
        
        if spatial_scale_table is not None:
            if type(spatial_scale_table) == str:
                try:
                    spatial_scale_table = pd.read_csv(spatial_scale_table)
                except FileNotFoundError:
                    raise FileNotFoundError("File '{0}' does not exist.\n".format(spatial_scale_table))
            spatial_scale_table = rename_columns(spatial_scale_table.copy(), column_identifier)

        df = spatial_scale_transform(df, 'temporal', spatial_scale_table = spatial_scale_table, spatial_scale_level = spatial_scale, aggregation_mode = 'mean')
    
    df.rename(columns = {'spatial id level '+str(spatial_scale):'spatial id level 1'},inplace = True)

    ################ temporal scale transform #####################

    if temporal_scale > 1:
        df = temporal_scale_transform(df, column_identifier=None, temporal_scale_level = temporal_scale
                                     , augmentation=False, verbose=1)


    ################## select desired temporal interval for plot from data ###############
    
    if (type(temporal_range) != dict) and (temporal_range is not None):
        raise TypeError("The temporal_range most be of type dict.")
    
    # if temporal id format is integrated
    if 'temporal id' in df.columns:
        temporal_identifier_column_name = 'temporal id'
        
        scale_format = {'sec':'%Y/%m/%d %H:%M:%S', 'min':'%Y/%m/%d %H:%M', 'hour':'%Y/%m/%d %H', 'day':'%Y/%m/%d', 'week':'%Y/%m/%d', 'month':'%Y/%m', 'year':'%Y'}

        # determining input data temporal scale and transforming temporal id's to timestamp
        df , scale = check_integrated_temporal_id(df)
        
        if temporal_range is not None:
            
            first_day = temporal_range['temporal id'][0]
            last_day = temporal_range['temporal id'][1]
            # get first_day and last_day timestamp
            first_day = datetime.datetime.strptime(first_day,scale_format[scale])
            last_day = datetime.datetime.strptime(last_day,scale_format[scale])
            df = df[(df['temporal id']>=first_day)&(df['temporal id']<=last_day)]

        if month_format_print == True:
            df['temporal id'] = df['temporal id'].apply(lambda x:datetime.datetime.strftime(x,'%Y/%B/%d'))
        else:
            df['temporal id'] = df['temporal id'].apply(lambda x:datetime.datetime.strftime(x,scale_format[scale]))

    # if temporal id format is non_integrated  
    else:
        
        if temporal_range is not None:
            for temporal_id_level , interval in temporal_range.items():
                df = df[(df[temporal_id_level]>=interval[0])&(df[temporal_id_level]<=interval[1])]

        ##### month name for non-integrated ###############
        
        if month_format_print==True:
          if column_identifier is not None:
            for values in column_identifier.values():
                if values == 'month':
                    month_index = [*column_identifier.values()].index(values)
                    temporal_id_x = [*column_identifier.keys()][month_index]
            try:
                df[temporal_id_x] = df[temporal_id_x].apply(lambda x : calendar.month_name[x])
            except(IndexError,TypeError):
                pass
            
        ##### next 5 lines add dummy temporal id for non_integrated temporal id format
        
        if temporal_range is not None:
            list_to_find_smallest_scale = temporal_range.keys()
        else:
            list_to_find_smallest_scale = df.columns
        
        # determine smallest temporal id level
        for level in range(1,200):
            if 'temporal id level '+str(level) in list_to_find_smallest_scale:
                smallest_temporal_level = level
                break

        add_dummy_integrated_temporal_id(df, start_level = level)        
        temporal_identifier_column_name = 'dummy temporal id'       

        
    ############ select desired spatial id #########
    
    # select random spatial ids if spatial id input is not specified
    if spatial_id is None:
        df_spatial_id_values = list(df['spatial id level 1'].unique())
        spatial_id = list([df_spatial_id_values[0]])
    
    df_dict = {}
    
    for j in range(int(len(spatial_id))):
        df0=df['spatial id level 1'] == spatial_id[j]
        df1 = df[df0]
        dict_key = spatial_id[j]
        df_dict[dict_key] = df1
    
    ############ making plot ################

    if temporal_covariate == 'default':
        T_C = df1.keys().tolist()
        T_C = list(filter(lambda x: not x.startswith(('temporal id','spatial id','dummy temporal id')), T_C))

    else:
        T_C=[temporal_covariate]

    for par in T_C:

        fig, ax = plt.subplots()
        for SpatialId, DF in df_dict.items():
            y = DF[par].tolist()
            time = DF[temporal_identifier_column_name].tolist()
            total_temporal_id_number = len(time) + 2
            plt.plot(time, y, label=SpatialId, linewidth=1.0, marker='o', markersize=5)

        # plt.xlabel('time')
        plt.legend()
        plt.ylabel(temporal_covariate)
        plt.gcf().autofmt_xdate()
        plt.ylabel(par)

        # plt.grid()

        default_xtick_size = plt.gcf().get_size_inches()[0]
        plt.gca().margins(x=0.002)
        plt.gcf().canvas.draw()
        tl = plt.gca().get_xticklabels()
        maxsize = max([t.get_window_extent().width for t in tl])
        inch_margin = 0.5  # inch margin
        xtick_size = maxsize / plt.gcf().dpi * total_temporal_id_number + inch_margin
        margin = inch_margin / plt.gcf().get_size_inches()[0]

        plt.gcf().subplots_adjust(left=margin, right=1. - margin)
        if default_xtick_size < xtick_size:
            plt.gcf().set_size_inches(xtick_size, plt.gcf().get_size_inches()[1])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.margins(0.1)
        plt.subplots_adjust(bottom=0.15)
        figure_to_save = plt.gcf()
        plt.show()
        
        if saving_plot_path is not None:
            try:
                figure_to_save.savefig(saving_plot_path + par +' evolution.png', bbox_inches='tight')
            except FileNotFoundError:
                print("The address '{0}' is not valid.".format(saving_plot_path))
        plt.close()
