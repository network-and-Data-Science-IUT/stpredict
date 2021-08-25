import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import pandas as pd
    import numpy as np
    from os import listdir
    from os.path import isfile, join
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import datetime
    from datetime import timedelta
    from dateutil.relativedelta import relativedelta
    import random
    import sys
    import os

def create_time_stamp(data, time_format, required_suffix):
    
    data.loc[:,('temporal id')] = data['temporal id'].astype(str) + required_suffix
    data.loc[:,('temporal id')] = data['temporal id'].apply(lambda x:datetime.datetime.strptime(x,time_format))
    return data

def get_target_temporal_ids(temporal_data, forecast_horizon, granularity):
    
    scale = None
    list_of_supported_formats_string_length = [4,7,10,13,16,19]
    scale_format = {'sec':'%Y/%m/%d %H:%M:%S', 'min':'%Y/%m/%d %H:%M', 'hour':'%Y/%m/%d %H', 'day':'%Y/%m/%d', 'week':'%Y/%m/%d', 'month':'%Y/%m', 'year':'%Y'}
    scale_delta = {'sec':0, 'min':0, 'hour':0, 'day':0, 'week':0, 'month':0, 'year':0}
    
    temporal_data = temporal_data.sort_values(by = ['spatial id','temporal id']).copy()
    temporal_id_instance = str(temporal_data['temporal id'].iloc[0])
    
    if len(temporal_id_instance) in list_of_supported_formats_string_length:
        temporal_format = 'integrated'
    else:
        temporal_format = 'non_integrated'
    
    #################################### find the scale
    
    # try catch is used to detect non-integrated temporal id format in the event of an error and return None as scale
    if temporal_format == 'integrated':
        try:
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

            elif len(temporal_id_instance) == 13:
                scale = 'hour'
                temporal_data = create_time_stamp(temporal_data.copy(), '%Y/%m/%d %H:%M:%S', ':00:00')

            elif len(temporal_id_instance) == 16:
                scale = 'min'
                temporal_data = create_time_stamp(temporal_data.copy(), '%Y/%m/%d %H:%M:%S', ':00')

            elif len(temporal_id_instance) == 19:
                scale = 'sec'
                temporal_data = create_time_stamp(temporal_data.copy(), '%Y/%m/%d %H:%M:%S', '')

        except ValueError:
                temporal_format = 'non_integrated'
    
    ########### get the target time point
    
    if temporal_format == 'integrated':
        
        # The time point of target variable is granularity*forecast_horizon units ahead of the data current temporal id
        scale_delta[scale] = granularity*(forecast_horizon)
        timestamp_delta = relativedelta(years = scale_delta['year'], months = scale_delta['month'], weeks = scale_delta['week'], days = scale_delta['day'], hours = scale_delta['hour'], minutes = scale_delta['min'], seconds = scale_delta['sec'])

        temporal_data['temporal id'] = temporal_data['temporal id'].apply(lambda x: datetime.datetime.strftime((x + timestamp_delta),scale_format[scale]))

                
    if temporal_format == 'non_integrated':
        temporal_data = temporal_data.sort_values(by = ['temporal id','spatial id'])
        total_number_of_spatial_units = len(temporal_data['spatial id'].unique())
        
        # shift units by granularity*forecast_horizon units
        future_data_frame = temporal_data[['temporal id']]
        future_data_frame = future_data_frame.iloc[(total_number_of_spatial_units*granularity*forecast_horizon):,:].reset_index(drop=True)
        temporal_data.insert(0, 'future temporal id', future_data_frame)
        
        # represent future units (where dates are not available) with max_date+x for x in [1,granularity*forecast_horizon]
        max_temporal_id = temporal_data['temporal id'].max()
        future_units = temporal_data.loc[pd.isna(temporal_data['future temporal id']),'temporal id'].unique()
        for index,unit in enumerate(future_units):
            temporal_data.loc[temporal_data['temporal id'] == unit,'future temporal id'] = str(max_temporal_id)+'+'+str(index+1)
        temporal_data['temporal id'] = temporal_data['future temporal id']
        temporal_data = temporal_data.drop(['future temporal id'],axis = 1)
    
    return temporal_data, temporal_format
