import pandas as pd
import numpy as np
import q_settings as std
import q_subfile as var
import q_models as q


def create_time_features(df, target=None):
    from datetime import datetime as dt
    
    df['date'] = df.index
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    
    X = df.drop(['date'], axis=1)
    if target:
        y = df[target]
        X = X.drop([target], axis=1)
        return X, y

    return X

"""
ExtraTreeClassifier
"""
def extratreeclassifier(X,y):
    from sklearn.ensemble import ExtraTreesClassifier
    model = ExtraTreesClassifier()
    fit = model.fit(X,y)
    dfscores = pd.DataFrame(fit.feature_importances_)
    dfcolumns = pd.DataFrame(X.columns)
    featurescores = pd.concat([dfscores,dfcolumns],axis=1)
    featurescores.columns = ['Score','Specs']
    featurescores = featurescores.nlargest(10,'Score')
    return featurescores


"""
Chi Square Test
"""
def selectkbest(X,y):
    import numpy as np
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    bestfeatures = SelectKBest(score_func=chi2, k=10)
    fit = bestfeatures.fit(X, y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    featurescores = pd.concat([dfscores,dfcolumns],axis=1)
    featurescores.columns = ['Score','Specs']
    featurescores = featurescores.nlargest(10,'Score')
    return featurescores


"""
Combined Feature Importance
"""
def combined_feature_selection_(X,y):
    X = X.astype(int)
    y = y.astype(int)
    feature_1 = extratreeclassifier(X,y)
    feature_2 = selectkbest(X,y)
    merged_importance = pd.merge(feature_1, feature_2, how='inner',left_on='Specs',right_on='Specs')
    col_importance = merged_importance[['Specs','Score_x']]
    print("Selected Features:",str(len(col_importance)))
    return round(col_importance,2)



def feature_selection_(vInput):
    
    data = pd.read_csv(vInput)
    data = data.set_index(pd.DatetimeIndex(data['key']))
    
    
    #data = df.loc[df['Calendar Country'].isin(var.std.vCalenderCountry) & df['Business Unit'].isin(var.std.vBusinesUnit) & df['Magnitude_Product_Group_Name'].isin(var.std.vMagnitudeLevel)]
    data = data.loc[data['Country_Code'].isin(var.std.vCalenderCountry)]
    data = data.loc[data['Business Unit_code'].isin(var.std.vBusinesUnit)]
    data = data.loc[data['Magnitude_Product_Group_Name'].isin(var.std.vMagnitudeLevel)]
    data = data.loc[data['Customer Business Sector'].isin(var.std.vCBusinessSectorLevel)]
    
    
    #data.drop(['key'],axis=1, inplace=True)
    data = data.sort_index()
    data.drop(['key','Country_Code', 'Business Unit_code','Magnitude_Product_Group_Name','Customer Business Sector'],axis=1, inplace=True)
    data = data.resample('M').mean()
    
    # X & Y SPLIT
    X, Y = create_time_features(data, 'Sales')
    X = X.fillna(0)
    X[X < 0] = 0
    
    col_importance = combined_feature_selection_(X,Y)
    col_importance.to_csv(var.std.vOutputFolder + "FEATURE" + ".csv")

          
    return col_importance
