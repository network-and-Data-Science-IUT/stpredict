import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import sys
    import pandas as pd
    from .get_target_quantities import get_target_quantities


def get_future_data(data: list,
                    forecast_horizon: int):

    if not (isinstance(data, list) and all(isinstance(d, pd.DataFrame) for d in data)):
        sys.exit('data input format is not valid')

    if not (isinstance(forecast_horizon, int) and forecast_horizon >= 1):
        sys.exit('data input format is not valid')

    _, _, granularity, _ = get_target_quantities(data[0].copy())

    data = [d.sort_values(by=['temporal id', 'spatial id']) for d in data]

    future_data = [d.iloc[-(forecast_horizon * granularity * len(d['spatial id'].unique())):].copy() for d in data]
    data = [d.iloc[:-(forecast_horizon * granularity * len(d['spatial id'].unique()))].copy() for d in data]

    return data, future_data

