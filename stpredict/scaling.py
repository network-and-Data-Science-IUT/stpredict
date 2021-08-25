import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    import pandas as pd 
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

def data_scaling(train_data, test_data, feature_scaler = 'logarithmic', target_scaler = 'logarithmic'):
    
    '''
    target_scaler = 'logarithmic' or 'normalize' or 'standardize' or None : the scaler used to scale target variable
    feature_scaler = 'logarithmic' or 'normalize' or 'standardize' or None : the scaler used to scale features
    '''
    if feature_scaler is not None:
        
        columns = list(train_data.columns)
        features = [column for column in columns if column not in ['spatial id','temporal id','Target','Normal target']]
        train_data_features = train_data[features]
        test_data_features = test_data[features]
        
        if feature_scaler == 'logarithmic':
            for feature in features:
                test_data.loc[:,(feature)] = list(np.sign(test_data[feature])*(np.log((np.abs(test_data[feature]) + 1).astype(float))))
                train_data.loc[:,(feature)] = list(np.sign(train_data[feature])*(np.log((np.abs(train_data[feature]) + 1).astype(float))))
            

        else:
            if feature_scaler == 'standardize':       
                scaleObject = StandardScaler()
            if feature_scaler == 'normalize':        
                scaleObject = MinMaxScaler()
            
            scaleObject.fit(train_data_features)
            train_data_features = scaleObject.transform(train_data_features)
            test_data_features = scaleObject.transform(test_data_features)
            
            test_data = pd.concat([test_data[['spatial id', 'temporal id', 'Target','Normal target']].reset_index(drop = True), pd.DataFrame(test_data_features, columns = features).reset_index(drop = True)], axis=1)
            train_data = pd.concat([train_data[['spatial id', 'temporal id', 'Target','Normal target']].reset_index(drop = True), pd.DataFrame(train_data_features, columns = features).reset_index(drop = True)], axis=1)
        
        
    if target_scaler is not None:
        
        if target_scaler == 'logarithmic':
            test_data.loc[:,('Target')] = list(np.sign(test_data['Target'])*(np.log((np.abs(test_data['Target']) + 1).astype(float))))
            train_data.loc[:,('Target')] = list(np.sign(train_data['Target'])*(np.log((np.abs(train_data['Target']) + 1).astype(float))))
            
        else:
            if target_scaler == 'standardize':       
                scaleObject = StandardScaler()
            if target_scaler == 'normalize':        
                scaleObject = MinMaxScaler()
        
            train_data_target = np.array(train_data['Target']).reshape(len(train_data), 1)
            test_data_target = np.array(test_data['Target']).reshape(len(test_data), 1)
            scaleObject.fit(train_data_target)
            train_data_target = scaleObject.transform(train_data_target)
            test_data_target = scaleObject.transform(test_data_target)
            
            test_data.loc[:,('Target')] = list(test_data_target)
            train_data.loc[:,('Target')] = list(train_data_target)
            
    return train_data, test_data

def target_descale(scaled_data, base_data, scaler = 'logarithmic'):
    
    '''
    scaled_data : List of target variable predicted values in test set (scaled)
    base_data : List of target variable values in train set before scaling
    scaler = 'logarithmic' or 'normalize' or 'standardize' or None
    '''
    if scaler is None:
        return scaled_data
    
    base_data = np.array(base_data).reshape(-1, 1)
    scaled_data = np.array(scaled_data).reshape(-1, 1)
    
    if scaler == 'logarithmic':
        output_data = np.exp(scaled_data) - 1
        output_data = np.sign(scaled_data)*(np.exp(np.abs(scaled_data)) - 1)
        
    elif scaler == 'standardize':        
        scaleObject = StandardScaler()
        scaleObject.fit_transform(base_data)
        output_data = scaleObject.inverse_transform(scaled_data)
        
    elif scaler == 'normalize':        
        scaleObject = MinMaxScaler()
        scaleObject.fit_transform(base_data)
        output_data = scaleObject.inverse_transform(scaled_data)
    else:
        sys.exit("The target scaler is not valid.")
    
    return list(output_data.ravel())
