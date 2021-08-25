import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import sys
    import pandas as pd


def get_target_quantities(data: pd.DataFrame):

    if not isinstance(data, pd.DataFrame):
        sys.exit('data input format is not valid')

    target_mode, target_granularity, granularity = None, 1, 1
    target_column_name = list(filter(lambda x: x.startswith('Target '), data.columns.values))[0]
    temp = target_column_name.split('(')[-1][:-1]
    if temp.startswith('augmented'):
        granularity = int(temp.split(' ')[2])
        temp = temp[temp.index('-') + 2:]
    if temp.startswith('normal'):
        target_mode = 'normal'
    elif temp.startswith('cumulative'):
        target_mode = 'cumulative'
    elif temp.startswith('differential'):
        target_mode = 'differential'
    elif temp.startswith('moving'):
        target_mode = 'moving_average'
        target_granularity = int(temp.split(' ')[3])
    else:
        sys.exit("Target column name is not valid.")
    data.rename(columns={target_column_name: target_column_name.split(' ')[0]}, inplace=True)
    return target_mode, target_granularity, granularity, data

