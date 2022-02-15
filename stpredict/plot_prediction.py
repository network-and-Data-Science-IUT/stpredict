import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import pandas as pd
    import numpy as np
    from os import listdir
    from os.path import isfile, join, exists
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import datetime
    from datetime import timedelta
    from dateutil.relativedelta import relativedelta
    import random
    import sys
    import os


def create_plot(df, forecast_horizon, granularity, spatial_ids, save_address, plot_type, test_point):
    
    mpl.style.use('default')
    df = df.sort_values(by = ['spatial id','temporal id'])
    
    x_axis_label = 'Target time point'
    
    if spatial_ids is None:
        spatial_ids = list(random.sample(list(df['spatial id']),1))
        
    temporal_ids = list(df['temporal id'].unique())
    
    plt.rc('font', size=60)
    number_of_temporal_ids = len(temporal_ids) + 2
    fig = plt.figure()
        
    for index,spatial_id in enumerate(spatial_ids):
        stage = 'training' if plot_type == 'test' else 'forecast'
        
        if test_point is not None:
            save_file_name = '{0}{2} stage for test point #{3}.pdf'.format(save_address, spatial_id, stage, test_point+1)
        else:
            save_file_name = '{0}{2} stage.pdf'.format(save_address, spatial_id, stage)
        
        
        ax=fig.add_subplot(len(spatial_ids),1, index+1)
        # add the curve of real values of the target variable
        temp_df = df[df['spatial id'] == spatial_id]
        ax.plot(list(temp_df['temporal id']),list(temp_df['real']),label='Real values', marker = 'o', markersize=20, linewidth=3.0, color = 'navy')
        
        # add the curve of predicted values of the target variable in the training, validation and testing set
        if plot_type != 'future':
            temp_train_df = temp_df[temp_df['sort'] == 'train']
            ax.plot(list(temp_train_df['temporal id']),list(temp_train_df['prediction']),label='Training set predicted values', marker = 'o', markersize=20, linewidth=3.0, color = 'green')
            temp_val_df = temp_df[temp_df['sort'] == 'validation']
            if len(temp_val_df)>0:
                ax.plot(list(temp_val_df['temporal id']),list(temp_val_df['prediction']),label='validation set predicted values', marker = 'o', markersize=20, linewidth=3.0, color = 'orange')
            temp_test_df = temp_df[temp_df['sort'] == 'test']
            ax.plot(list(temp_test_df['temporal id']),list(temp_test_df['prediction']),label='Testing set predicted values', marker = 'o', markersize=20, linewidth=3.0, color = 'crimson')

        if plot_type == 'future':
            temp_test_df = temp_df[temp_df['sort'] == 'future']
            ax.plot(list(temp_test_df['temporal id']),list(temp_test_df['prediction']),label='Predicted values', marker = 'o', markersize=20, linewidth=3.0, color = 'orangered')

        ax.grid()
        plt.ylabel('Target')
        plt.xticks(rotation=90)
        # set the size of plot base on number of temporal units and lable fonts
        plt.gca().margins(x=0.002)
        plt.gcf().canvas.draw()
        tl = plt.gca().get_xticklabels()
        maxsize = max([t.get_window_extent().width for t in tl])
        inch_margin = 0.5 # inch margin
        xtick_size = maxsize/plt.gcf().dpi*number_of_temporal_ids+inch_margin
        margin = inch_margin/plt.gcf().get_size_inches()[0]

        plt.gcf().subplots_adjust(left=margin, right=1.-margin)
        plt.gcf().set_size_inches(xtick_size, plt.gcf().get_size_inches()[1])
        if index<len(spatial_ids)-1:
          ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        if index == 0:
          plt.legend()
        ttl = plt.title('spatial id '+str(spatial_id))
        ttl.set_position([.5, 1.05])
        plt.margins(x = 0.01)
        
    
    plt.xlabel(x_axis_label,labelpad = 20)
    plt.gcf().set_size_inches(xtick_size, plt.gcf().get_size_inches()[1]*5*(len(spatial_ids)))
    
    plt.subplots_adjust(hspace=.5)
    plt.tight_layout()
    
    try:
        if not exists(save_address):
            os.makedirs(save_address)
        plt.savefig(save_file_name, bbox_inches='tight', pad_inches=1)
        plt.close()
    except FileNotFoundError:
            print("The address '{0}' is not valid.".format(save_address))
                
def plot_prediction(data, test_type = 'whole-as-one', forecast_horizon = 1, plot_type = 'test', granularity = 1,
                        spatial_ids = None):
    
    
    validation_dir = './prediction/validation process/'
    testing_dir = './prediction/test process/'
    future_dir = './prediction/future prediction/'
    needed_columns = ['temporal id', 'spatial id','real','prediction','sort']
    
    path = validation_dir
    files = [f for f in listdir(path) if isfile(join(path, f))]
    prefix = 'training prediction forecast horizon = {0}, test-point #'.format(forecast_horizon)
    files = [file for file in files if file.startswith(prefix)]
    file_test_points = [int(file.split('test-point #')[1][:-4]) for file in files]
    file_test_points.sort()
    
    if plot_type == 'test':
        address = testing_dir + 'test'
    elif plot_type == 'future':
        address = future_dir + 'future'
        
    data = data.rename(columns={'target temporal id':'temporal id'})

    if test_type == 'whole-as-one':
        
        test_csv_file = testing_dir + 'test prediction forecast horizon = {0}.csv'.format(forecast_horizon)
        train_csv_file = validation_dir + 'training prediction forecast horizon = {0}.csv'.format(forecast_horizon)
        validation_csv_file = validation_dir + 'validation prediction forecast horizon = {0}.csv'.format(forecast_horizon)
        future_csv_file = future_dir + 'future prediction forecast horizon = {0}.csv'.format(forecast_horizon)
            
        if plot_type == 'test':
            test_df = pd.read_csv(test_csv_file)
            selected_model = list(test_df['model name'].unique())[0]
            test_df = test_df.assign(sort = 'test')
            train_df = pd.read_csv(train_csv_file)
            train_df = train_df[train_df['model name'] == selected_model]
            train_df = train_df.assign(sort = 'train')
            if exists(validation_csv_file):
                validation_df = pd.read_csv(validation_csv_file)
                validation_df = validation_df[validation_df['model name'] == selected_model]
                validation_df = validation_df.assign(sort = 'validation')
            else: 
                validation_df = pd.DataFrame(columns = train_df.columns)
            gap_df = data.rename(columns = {'Normal target':'real'})
            gap_df = gap_df.assign(prediction = np.NaN)
            gap_df = gap_df.assign(sort = 'gap')
            gap_df = gap_df[(gap_df['temporal id'] < test_df['temporal id'].min()) & (gap_df['temporal id'] > train_df.append(validation_df)['temporal id'].max())]
            all_df = train_df[needed_columns].append(validation_df[needed_columns]).append(gap_df[needed_columns]).append(test_df[needed_columns])
            
        elif plot_type == 'future':
            future_df = pd.read_csv(future_csv_file)
            future_df = future_df.assign(sort = 'future')
            train_df = data.rename(columns = {'Normal target':'real'})
            train_df = train_df.assign(prediction = np.NaN)
            train_df = train_df.assign(sort = 'train')
            all_df = train_df[needed_columns].append(future_df[needed_columns])
        
        try:
            create_plot(df = all_df, forecast_horizon = forecast_horizon, granularity = granularity, spatial_ids = spatial_ids, 
                        save_address = './plots/', plot_type = plot_type, test_point = None)
        except Exception as e:
            raise Exception('There is a problem in plotting predictions:\n'+str(e))

    if test_type == 'one-by-one':    
        
        test_point_number = len(file_test_points)
        all_test_points_df = pd.read_csv(testing_dir + 'test prediction forecast horizon = {0}.csv'.format(forecast_horizon))
        test_temporal_units = all_test_points_df['temporal id'].unique()
        test_temporal_units.sort()
        test_temporal_units = test_temporal_units[::-1]
        
        if plot_type == 'test':
            for test_point in range(test_point_number):
                test_df = all_test_points_df[all_test_points_df['temporal id'] == test_temporal_units[test_point]]
                selected_model = list(test_df['model name'].unique())[0]
                test_df = test_df.assign(sort='test')
                tp_number = file_test_points[test_point]
                
                train_csv_file = validation_dir + 'training prediction forecast horizon = {0}, test-point #{1}.csv'.format(forecast_horizon,tp_number)
                validation_csv_file = validation_dir + 'validation prediction forecast horizon = {0}, test-point #{1}.csv'.format(forecast_horizon,tp_number)
                
                train_df = pd.read_csv(train_csv_file)
                train_df = train_df[train_df['model name'] == selected_model]
                train_df = train_df.assign(sort = 'train')
                if exists(validation_csv_file):
                    validation_df = pd.read_csv(validation_csv_file)
                    validation_df = validation_df[validation_df['model name'] == selected_model]
                    validation_df = validation_df.assign(sort = 'validation')
                else: 
                    validation_df = pd.DataFrame(columns = train_df.columns)
                gap_df = data.rename(columns = {'Normal target':'real'})
                gap_df = gap_df.assign(prediction = np.NaN)
                gap_df = gap_df.assign(sort = 'gap')
                gap_df = gap_df[(gap_df['temporal id'] < test_df['temporal id'].min()) & (gap_df['temporal id'] > train_df.append(validation_df)['temporal id'].max())]
                all_df = train_df[needed_columns].append(validation_df[needed_columns]).append(gap_df[needed_columns]).append(test_df[needed_columns])

                try:
                    create_plot(df = all_df, forecast_horizon = forecast_horizon, granularity = granularity, spatial_ids = spatial_ids, 
                                save_address = './plots/', plot_type = plot_type, test_point = test_point)
                except Exception as e:
                    print('There is a problem in plotting predictions.')
        
        elif plot_type == 'future':
            
            future_df = pd.read_csv(future_dir + 'future prediction forecast horizon = {0}.csv'.format(forecast_horizon))
            future_df = future_df.assign(sort = 'future')
            train_df = data.rename(columns = {'Normal target':'real'})
            train_df = train_df.assign(prediction = np.NaN)
            train_df = train_df.assign(sort = 'train')
            all_df = train_df[needed_columns].append(future_df[needed_columns])
        
            try:
                create_plot(df = all_df, forecast_horizon = forecast_horizon, granularity = granularity, spatial_ids = spatial_ids, 
                            save_address = './plots/', plot_type = plot_type, test_point = None)
            except Exception:
                print('There is a problem in plotting predictions.')
