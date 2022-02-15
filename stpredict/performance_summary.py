import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from os import listdir
    from os.path import isfile, join, exists
    import datetime
    from datetime import timedelta
    from dateutil.relativedelta import relativedelta
    import random
    import sys
    import os
    import matplotlib as mpl
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.patches as mpatches
    from matplotlib import colors as mcolors
    import matplotlib.ticker as ticker

warnings.filterwarnings("ignore")


# merge the cells in matplotlib table
def mergecells(table, cells, selected_text = 'L'):
    if len(cells)<=1:
        return
    cells_array = [np.asarray(c) for c in cells]
    h = np.array([cells_array[i+1][0] - cells_array[i][0] for i in range(len(cells_array) - 1)])
    v = np.array([cells_array[i+1][1] - cells_array[i][1] for i in range(len(cells_array) - 1)])
    bold = False

    # if it's a horizontal merge, all values for `h` are 0
    if not np.any(h):
        # sort by horizontal coord
        cells = np.array(sorted(list(cells), key=lambda v: v[1]))
        edges = ['BTL'] + ['BT' for i in range(len(cells) - 2)] + ['BTR']
    elif not np.any(v):
        cells = np.array(sorted(list(cells), key=lambda h: h[0]))
        edges = ['TRL'] + ['RL' for i in range(len(cells) - 2)] + ['BRL']
    else:
        raise ValueError("Only horizontal and vertical merges allowed")

    for cell, e in zip(cells, edges):
        table[cell[0], cell[1]].visible_edges = e
        
    txts = [table[cell[0], cell[1]].get_text() for cell in cells]
    tpos = [np.array(t.get_position()) for t in txts]
    
    if any([txt.get_weight() == 'bold' for txt in txts]):
        bold = True
        
    if selected_text == 'L':
        # transpose the text of the left cell
        trans = (tpos[-1] - tpos[0])/2
        # didn't had to check for ha because I only want ha='center'
        txts[0].set_transform(mpl.transforms.Affine2D().translate(*trans))
        if bold == True:
            txts[0].set_weight('bold')
        for txt in txts[1:]:
            txt.set_visible(False)
        
    elif selected_text == 'R':
        # transpose the text of the right cell
        trans = (tpos[0] - tpos[-1])/2
        # didn't had to check for ha because I only want ha='center'
        txts[-1].set_transform(mpl.transforms.Affine2D().translate(*trans))
        if bold == True:
            txts[-1].set_weight('bold')
        for txt in txts[:-1]:
            txt.set_visible(False)
        
# find the consecutive cells with same values and merge them
def merge_same_values(row_number,column,offset,the_table):
    values = [the_table.get_celld()[(i,column)].get_text().get_text() for i in range(offset,row_number+1)]
    current_value = values[0]
    merge_list = []
    for index,value in enumerate(values):
        # model name of test points must not be merged with previous rows
        if column == 2 and (the_table.get_celld()[(index+offset,1)].get_text().get_text() == 'Test'):
            mergecells(the_table, merge_list)
            if index+offset+1 < row_number:
                current_value = the_table.get_celld()[(index+offset+1,column)].get_text().get_text()
                merge_list = []
            
        elif value == current_value:
            merge_list = merge_list+[(index+offset,column)]
                
        else:
            current_value = value
            mergecells(the_table, merge_list)
            merge_list = [(index+offset,column)]
            
    mergecells(the_table, merge_list)
    return the_table

# find the consecutive cells with same values and colours them with the same color
def color_same_values(row_number,column_number,column,offset,the_table,color_dict):
    
    colors = ['#008e99','#424242','#ba4c4d','#858585','#acc269','#7255b2']
    
    # find cells with same text
    values = [the_table.get_celld()[(i,column)].get_text().get_text() for i in range(offset,row_number+1)]
    models = np.unique(values)
    if len(models) == 1 :
        color_dict = {models[0]:'#000000'}
        
    current_value = values[0]
    for index,value in enumerate(values):
        if value == current_value:
            for col in range(column,column_number):
                the_table.get_celld()[(index+offset,col)].get_text().set_color(color_dict[value])
        else:
            current_value = value
            for col in range(column,column_number):
                the_table.get_celld()[(index+offset,col)].get_text().set_color(color_dict[value])
            
    return the_table

# prepare input df by counting features and rounding performance values
def prepare_df(df,base_columns,measure_names,performance_benchmark,test_type,plot_type):
    
    if test_type == 'whole-as-one':
        unique_columns = ['model name', 'history length']
    if test_type == 'one-by-one':
        unique_columns = ['Test point','model name','history length']
        
    df['feature or covariate set'] = df['feature or covariate set'].apply(lambda x:len(x.split(',')))
    if performance_benchmark in ['MAE', 'MAPE', 'MASE', 'MSE', 'AIC', 'BIC', 'likelihood']:
        optimum_performance_df = df.groupby(unique_columns)[[performance_benchmark]].min().reset_index().rename(columns = {performance_benchmark:'best '+performance_benchmark})
    else:
        optimum_performance_df = df.groupby(unique_columns)[[performance_benchmark]].max().reset_index().rename(columns = {performance_benchmark:'best '+performance_benchmark})
    df = pd.merge(df,optimum_performance_df)
    df = df[df[performance_benchmark] == df['best '+performance_benchmark]].drop_duplicates(subset = unique_columns)
    df = df[base_columns + measure_names]
        
    return df

def table_add(table, df, best_loc, measure_names, performance_benchmark, best_loc_flag):
    
    if best_loc_flag == True:
        df = df.reset_index(drop = True)
        df[performance_benchmark] = df[performance_benchmark].astype(float)
        
        if performance_benchmark in ['MAE', 'MAPE', 'MASE', 'MSE', 'AIC', 'BIC', 'likelihood']:
            optimum_performance = df[performance_benchmark].min()
        else:
            optimum_performance = df[performance_benchmark].max()
            
        df_best_loc = list(df[df[performance_benchmark] == optimum_performance].index)[-1]
        best_loc = best_loc + [df_best_loc + len(table)+1]
    
    for measure in measure_names:
        df[measure] = df[measure].apply(lambda x:np.round(x,2))
        df.loc[df[measure]>99999,measure] = df.loc[df[measure]>99999,measure].apply(lambda x:'{:.2E}'.format(x))
    table = table + df.values.tolist()
    
    return table, best_loc

# make tables and save into pdf
def plot_table(train_df,validation_df,test_df,test_type,performance_benchmark,test_point_number):
    
    if test_type == 'whole-as-one':
        base_columns = ['Dataset', 'model name', 'history length', 'feature or covariate set']
    if test_type == 'one-by-one':
        base_columns = ['Test point', 'Dataset', 'model name', 'history length', 'feature or covariate set']
    measure_names = list(filter(lambda x: x not in base_columns, test_df.columns))
    models_number = len(train_df['model name'].unique())
    max_history = len(train_df['history length'].unique())
    
    models = train_df['model name'].unique()
    colors = ['#008e99','#424242','#ba4c4d','#858585','#acc269','#7255b2']
    color_list = colors * ((len(models)//6)+1) 
    color_dict = {model:color_list[i] for i , model in enumerate(models)}

    test_df = prepare_df(test_df,base_columns,measure_names,performance_benchmark,test_type,'table')
    train_df = prepare_df(train_df,base_columns,measure_names,performance_benchmark,test_type,'table')
    if validation_df is not None:
        validation_df = prepare_df(validation_df,base_columns,measure_names,performance_benchmark,test_type,'table')
    
    # the location of best obtained results for each set, in the table 
    best_loc = []
    table = []
    
    if test_type == 'whole-as-one':
        table, best_loc = table_add(table, test_df, best_loc, measure_names, performance_benchmark, True)
        if validation_df is not None:
            table, best_loc = table_add(table, validation_df, best_loc, measure_names, performance_benchmark, True)
        table, best_loc = table_add(table, train_df, best_loc, measure_names, performance_benchmark, True)
        colWidths = [0.15, 0.15, 0.17, 0.15] + [0.15]*len(measure_names)
        for index,measure_name in enumerate(measure_names):
            if measure_name in ['R2_score', 'likelihood']:
                colWidths[index+4] = 0.17
        row_number = len(table)

    if test_type == 'one-by-one':
        table, best_loc = table_add(table, test_df[test_df['Test point'] == 'All test points'], best_loc, measure_names, performance_benchmark, False)
        table = table + [['1','Test','Model','History length','#Features'] + measure_names]
        for test_point in range(1,test_point_number+1):
            table, best_loc = table_add(table, test_df[test_df['Test point'] == str(test_point)], best_loc, measure_names, performance_benchmark, True)
            if validation_df is not None:
                table, best_loc = table_add(table, validation_df[validation_df['Test point'] == str(test_point)], best_loc, measure_names, performance_benchmark, True)
            table, best_loc = table_add(table, train_df[train_df['Test point'] == str(test_point)], best_loc, measure_names, performance_benchmark, True)
        colWidths = [0.15, 0.15, 0.15, 0.17, 0.15] + [0.15]*len(measure_names)
        for index,measure_name in enumerate(measure_names):
            if measure_name in ['R2_score', 'likelihood']:
                colWidths[index+5] = 0.17
        headerColWidths = [0.15, 0.15] + [np.sum(colWidths[2:])/len(measure_names)]*len(measure_names)
        row_number = len(table)
        
    
    # calculate number of pages needed
    npages = row_number // 300
    if row_number % 300 > 0:
        npages += 1
    

    pdf = PdfPages('./plots/performance summary.pdf')
    
    for page in range(npages):
        fig = plt.figure()
        plt.tight_layout()
        ax = fig.gca()
        ax.axis('tight')
        ax.axis('off')
        temp_table = table[page*300:(page+1)*300]
        page_best_locs = [index - (page*300) for index in best_loc if index in range(page*300,(page+1)*300)]
        
        ylim = 0.08278145695364238*(len(temp_table))
        xlim = np.sum(colWidths) + 0.7
        bbox = (0,0,xlim,ylim)
        
        if test_type == 'whole-as-one':
            colLabels = ['Dataset','Model','History length','#Features'] + measure_names

        
        if test_type == 'one-by-one' and page == 0:
            colLabels = ['Test point', 'Dataset','','',''] + measure_names
        elif test_type == 'one-by-one' and page > 0:
            colLabels = ['Test point', 'Dataset','Model','History length','#Features'] + measure_names
        
        
        if test_type == 'one-by-one' and page == 0:
            # make a table to represent header and test set results with different column number
            headerdata = [[temp_table[0][i] for i in [0,1]+[i+5 for i in range(len(measure_names))]]] + [['']*len(headerColWidths)]*(len(temp_table)-1)
            headertable = ax.table(cellText=headerdata, colLabels=['Test point', 'Dataset'] + measure_names, 
                                   colWidths=headerColWidths, loc='center', colColours = ['lightgray']*len(headerColWidths),
                                   cellLoc='center', edges = 'open', bbox = bbox)
            for i in range(len(headerColWidths)):
                headertable.get_celld()[(0,i)].visible_edges = 'closed'
                headertable.get_celld()[(0,i)].set_text_props(weight='bold')
                headertable.get_celld()[(1,i)].visible_edges = 'closed'
            headertable.auto_set_font_size(False)
            headertable.set_fontsize(9)
            headertable.scale(1.5, 1.5)
            fig.canvas.draw()
        
        
        the_table = ax.table(cellText=temp_table, colLabels=colLabels, colWidths=colWidths, loc='center', cellLoc='center', bbox = bbox)
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(9)
        the_table.scale(1.5, 1.5)
        fig.canvas.draw()

        if test_type == 'whole-as-one':
            
            for i in range(1,len(colLabels)):
                for j in page_best_locs:
                    the_table.get_celld()[(j,i)].set_text_props(weight='bold')
                    
            # merge the cells containing dataset name in first column
            offset = 2 if page == 0 else 1
            the_table = merge_same_values(row_number = len(temp_table),
                                  column = 0, offset = offset, the_table = the_table)
            # merge the cells containing model name in second column 
            offset = 2 if page == 0 else 1
            the_table = color_same_values(row_number = len(temp_table), column_number = len(colLabels),
                                  column = 1, offset = offset, the_table = the_table, color_dict = color_dict)
            the_table = merge_same_values(row_number = len(temp_table),
                                  column = 1, offset = offset, the_table = the_table)
            
            
        if test_type == 'one-by-one':
            
            for i in range(2,len(colLabels)):
                for j in page_best_locs:
                    the_table.get_celld()[(j,i)].set_text_props(weight='bold')
                    
            # merge the cells containing test point number in first column
            offset = 2 if page == 0 else 1
            the_table = merge_same_values(row_number = len(temp_table),
                                          column = 0, offset = offset, the_table = the_table)
            # merge the cells containing dataset name in second column
            the_table = merge_same_values(row_number = len(temp_table),
                                          column = 1, offset = offset, the_table = the_table)
            # merge the cells containing model name in third column
            offset = 3 if page == 0 else 1
            the_table = color_same_values(row_number = len(temp_table), column_number = len(colLabels),
                                  column = 2, offset = offset, the_table = the_table, color_dict = color_dict)
            the_table = merge_same_values(row_number = len(temp_table),
                                          column = 2, offset = offset, the_table = the_table)
        
        # colour header
        for i in range(len(colLabels)):
            the_table.get_celld()[(0,i)].set_facecolor('lightgray')
            the_table.get_celld()[(0,i)].set_text_props(weight='bold')
            
            if (page == 0) and (test_type == 'one-by-one' and i>1):
                the_table.get_celld()[(2,i)].set_facecolor('lightgray')#'#40466e'
                the_table.get_celld()[(2,i)].set_text_props(weight='bold')#, color='w')
                
        
        
        # set the_table header invisible to make header table visible
        if test_type == 'one-by-one' and page == 0:
            for i in range(len(colLabels)):
                the_table.get_celld()[(0,i)].set_alpha(0)
                the_table.get_celld()[(1,i)].set_alpha(0)
                if i>=2:
                    the_table.get_celld()[(0,i)].get_text().set_visible(False)
                    the_table.get_celld()[(1,i)].get_text().set_visible(False)
                
            
        
        pdf.savefig(fig , dpi=300, bbox_inches='tight', pad_inches=1)
        plt.close()

    pdf.close()
    
# make barplots and save into pdf

def plot_bar(train_df,validation_df,test_type,performance_benchmark,test_point_number):
    
    if test_type == 'whole-as-one':
        base_columns = ['Dataset', 'model name', 'history length', 'feature or covariate set']
    if test_type == 'one-by-one':
        base_columns = ['Test point', 'Dataset', 'model name', 'history length', 'feature or covariate set']
    measure_names = list(filter(lambda x: x not in base_columns, train_df.columns))
    models = train_df['model name'].unique()
    models_number = len(models)
    max_history = len(train_df['history length'].unique())
    
    colors = ['#008e99','#424242','#ba4c4d','#858585','#acc269','#7255b2']
    color_list = colors * ((len(models)//6)+1) 
    color_dict = {model:color_list[i] for i , model in enumerate(models)}
        
    train_df = prepare_df(train_df,base_columns,measure_names,performance_benchmark,test_type,'bar')
    if validation_df is not None:
        validation_df = prepare_df(validation_df,base_columns,measure_names,performance_benchmark,test_type,'bar')
    
    if test_type == 'whole-as-one':
        test_point_number = 1
    
    models_per_plot = 6
    # calculate number of pages needed
    npages = models_number // (models_per_plot*5)
    if models_number % (models_per_plot*5) > 0:
        npages += 1
    
    barWidth = 0.015 if validation_df is not None else 0.030
    pos = 0.195*np.arange(max_history)
    
        
    # initialize pdf
    pdf = PdfPages('./plots/performance bar plots.pdf')
    
    for test_point in range(1,test_point_number+1):
        
        mpl.rcParams.update({'font.size': 12})
        
        for page in range(npages):
            
            current_models = train_df['model name'].unique()[page*5*models_per_plot:(page+1)*5*models_per_plot]
            num_subplots = (len(current_models)//models_per_plot)+int()
            if len(current_models)%models_per_plot > 0:
                num_subplots += 1
                
            fig, axes = plt.subplots(num_subplots,1)
            if num_subplots == 1:
                axes = np.array([axes])
            
            if test_type == 'one-by-one':
                plt.suptitle('Test point number {0}'.format(test_point), fontweight='bold', fontsize='16', y=1.05)
                
            for subplot in range(num_subplots):
                
                subplot_models = current_models[subplot*models_per_plot:(subplot+1)*models_per_plot]
                ax = axes[subplot]
                ax.grid(axis = 'y', linestyle = '--')
                ax.set_axisbelow(True)
                log_flag = False
                    
                for index,model in enumerate(subplot_models):

                    # Set position of bar on X axis
                    if validation_df is not None:
                        current_train_pos=[x + (2*index)*barWidth for x in pos]
                        current_validation_pos=[x + (2*index+1)*barWidth for x in pos]
                    else:
                        current_train_pos=[x + index*barWidth for x in pos]

                    if test_type == 'whole-as-one':
                        train_performance = list(train_df.loc[train_df['model name'] == model,performance_benchmark])
                        if validation_df is not None:
                            validation_performance = list(validation_df.loc[validation_df['model name'] == model,performance_benchmark])

                    if test_type == 'one-by-one':
                        train_performance = list(train_df.loc[(train_df['Test point'] == test_point)&(train_df['model name'] == model),performance_benchmark])
                        if validation_df is not None:
                            validation_performance = list(validation_df.loc[(validation_df['Test point'] == test_point)&(validation_df['model name'] == model),performance_benchmark])
                    
                    if validation_df is not None:
                        performance_list = list(train_performance)+list(validation_performance)
                    else:
                        performance_list = list(train_performance)
                    if max(performance_list)-min(performance_list)>10000 and min(performance_list)>=0:
                        log_flag = True

                    # Make the plot
                    if validation_df is not None:
                        ax.bar(current_train_pos, train_performance, color=color_dict[model], width=barWidth, edgecolor='white',hatch='//////')
                        ax.bar(current_validation_pos, validation_performance, color=color_dict[model], width=barWidth, edgecolor='white', label=model)
                    else:
                        ax.bar(current_train_pos, train_performance, color=color_dict[model], width=barWidth, edgecolor='white', label=model)
                    

                ax.set_ylabel(performance_benchmark,fontweight='bold')
                if validation_df is not None:
                    ax.set_xticks(ticks = [r + (len(subplot_models)-0.5)*barWidth for r in pos])
                else:
                    ax.set_xticks(ticks = [r + ((len(subplot_models)/2)-0.5)*barWidth for r in pos])
                ax.set_xticklabels(labels = [str(history) for history in range (1,max_history+1)])
                if validation_df is not None:
                    ax.set_title('The predictive models performance', fontweight='bold')
                else:
                    ax.set_title('The predictive models performance on\nthe training dataset', fontweight='bold')
                    
                if subplot == 0 and validation_df is not None:
                    train_patch = mpatches.Patch(edgecolor='gray', facecolor='w', label='Training', hatch='//////')
                    validation_patch = mpatches.Patch(color='gray', label='Validation')
                    leg1 = ax.legend(handles=[train_patch,validation_patch], loc='upper left', bbox_to_anchor=(1, 1))
                    ax.add_artist(leg1)

                # Create legend & Show graphic
                ax.legend(loc='lower left', bbox_to_anchor=(1, 0))
                
                
                y_min = min(ax.get_yticks())
                y_max = max(ax.get_yticks())
                if y_max - y_min <= 1:
                    step = 0.1
                    ax.set_yticks(np.arange(y_min, y_max, step))
                else:
                    step = (y_max-y_min)/10
                    ax.set_yticks(np.arange(y_min, y_max, step))
                    
                if log_flag == True:
                    ax.set_yscale('log')
                
            
            plt.xlabel('History Length', fontweight='bold')
            if max_history>5:
                plt.gcf().set_size_inches(plt.gcf().get_size_inches()[0]*(max_history/5), plt.gcf().get_size_inches()[1]*num_subplots)
            else:
                plt.gcf().set_size_inches(plt.gcf().get_size_inches()[0]*(1.5), plt.gcf().get_size_inches()[1]*num_subplots)
                
            plt.tight_layout()
            pdf.savefig(fig , dpi=300, bbox_inches='tight', pad_inches=1)
            plt.close()
            
    pdf.close()

# save summary of performance reports

def performance_summary(forecast_horizon,test_type,performance_benchmark):
    
    validation_dir = './performance/validation process/'
    testing_dir = './performance/test process/'
    if not exists('./plots/'):
        os.makedirs('./plots/')


    path = validation_dir
    files = [f for f in listdir(path) if isfile(join(path, f))]
    prefix = 'training performance report forecast horizon = {0}, test-point #'.format(forecast_horizon)
    files = [file for file in files if file.startswith(prefix)]
    file_test_points = [int(file.split('test-point #')[1][:-4]) for file in files]
    file_test_points.sort()

    if test_type == 'whole-as-one':
        
        test_csv_file = testing_dir + 'test performance report forecast horizon = {0}.csv'.format(forecast_horizon)
        train_csv_file = validation_dir + 'training performance report forecast horizon = {0}.csv'.format(forecast_horizon)
        validation_csv_file = validation_dir + 'validation performance report forecast horizon = {0}.csv'.format(forecast_horizon)

        test_df = pd.read_csv(test_csv_file)
        test_df.insert(0, 'Dataset', 'Test')

        train_df = pd.read_csv(train_csv_file)
        train_df.insert(0, 'Dataset', 'Training')
        
        if exists(validation_csv_file):
            validation_df = pd.read_csv(validation_csv_file)
            validation_df.insert(0, 'Dataset', 'Validation')
        else:
            validation_df = None
        
        try:
            plot_table(train_df,validation_df,test_df,test_type,performance_benchmark,None)
        except Exception:
            print('There is a problem in creating the results table.')

    elif test_type == 'one-by-one':

        test_point_number = len(file_test_points)    
        test_df = pd.read_csv(testing_dir + 'test performance report forecast horizon = {0}.csv'.format(forecast_horizon))
        test_df = test_df.rename(columns = {'test point':'Test point'})
        test_df.insert(1, 'Dataset', 'Test')


        train_df = pd.DataFrame()
        validation_df = pd.DataFrame()
        for test_point in range(1,test_point_number+1):

            tp_number = file_test_points[test_point-1]
            
            train_csv_file = validation_dir + 'training performance report forecast horizon = {0}, test-point #{1}.csv'.format(forecast_horizon,tp_number)
            validation_csv_file = validation_dir + 'validation performance report forecast horizon = {0}, test-point #{1}.csv'.format(forecast_horizon,tp_number)

            temp_train_df = pd.read_csv(train_csv_file)
            temp_train_df.insert(0, 'Test point', str(test_point))
            temp_train_df.insert(1, 'Dataset', 'Training')
            train_df = train_df.append(temp_train_df)
            
            if exists(validation_csv_file):
                temp_validation_df = pd.read_csv(validation_csv_file)
                temp_validation_df.insert(0, 'Test point', str(test_point))
                temp_validation_df.insert(1, 'Dataset', 'Validation')
                validation_df = validation_df.append(temp_validation_df)
            else:
                validation_df = None

        try:
            plot_table(train_df,validation_df,test_df,test_type,performance_benchmark,test_point_number)
        except Exception:
            print('There is a problem in creating the results table.')
        


# plot the performance bars for training and validation
def performance_bar_plot(forecast_horizon,test_type,performance_benchmark):

    validation_dir = './performance/validation process/'
    testing_dir = './performance/test process/'
    if not exists('./plots/'):
        os.makedirs('./plots/')

    path = validation_dir
    files = [f for f in listdir(path) if isfile(join(path, f))]
    prefix = 'training performance report forecast horizon = {0}, test-point #'.format(forecast_horizon)
    files = [file for file in files if file.startswith(prefix)]
    file_test_points = [int(file.split('test-point #')[1][:-4]) for file in files]
    file_test_points.sort()

    if test_type == 'whole-as-one':
        
        train_csv_file = validation_dir + 'training performance report forecast horizon = {0}.csv'.format(forecast_horizon)
        validation_csv_file = validation_dir + 'validation performance report forecast horizon = {0}.csv'.format(forecast_horizon)
        
        train_df = pd.read_csv(train_csv_file)
        train_df.insert(0, 'Dataset', 'Training')
        
        if exists(validation_csv_file):
            validation_df = pd.read_csv(validation_csv_file)
            validation_df.insert(0, 'Dataset', 'Validation')
        else: 
            validation_df = None
            
        try:
            plot_bar(train_df,validation_df,test_type,performance_benchmark,None)
        except Exception:
            print('There is a problem in plotting the results.')

    elif test_type == 'one-by-one':

        test_point_number = len(file_test_points)    
        test_df = pd.read_csv(testing_dir + 'test performance report forecast horizon = {0}.csv'.format(forecast_horizon))
        test_df.insert(0, 'Test point', 'Overall')
        test_df.insert(1, 'Dataset', 'Test')


        train_df = pd.DataFrame()
        validation_df = pd.DataFrame()
        for test_point in range(1,test_point_number+1):
            
            tp_number = file_test_points[test_point-1]
            
            train_csv_file = validation_dir + 'training performance report forecast horizon = {0}, test-point #{1}.csv'.format(forecast_horizon,tp_number)
            validation_csv_file = validation_dir + 'validation performance report forecast horizon = {0}, test-point #{1}.csv'.format(forecast_horizon,tp_number)

            temp_train_df = pd.read_csv(train_csv_file)
            temp_train_df.insert(0, 'Test point', test_point)
            temp_train_df.insert(1, 'Dataset', 'Training')
            train_df = train_df.append(temp_train_df)
            
            if exists(validation_csv_file):
                temp_validation_df = pd.read_csv(validation_csv_file)
                temp_validation_df.insert(0, 'Test point', test_point)
                temp_validation_df.insert(1, 'Dataset', 'Validation')
                validation_df = validation_df.append(temp_validation_df)
            else:
                validation_df = None

        try:
            plot_bar(train_df,validation_df,test_type,performance_benchmark,test_point_number)
        except Exception:
            print('There is a problem in plotting the results.')