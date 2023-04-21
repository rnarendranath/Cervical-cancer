from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from catboost import CatBoostClassifier, Pool
import numpy as np

def ml_pp():
    data = pd.read_csv("kag_risk_factors_cervical_cancer.csv")
    data = data.replace('?', pd.NA)

    # dropping entire column because of so many unknown values
    data = data.drop(['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'], axis=1)
    data = data.apply(pd.to_numeric)
    # filling data which has unknown values
    data = data.fillna(data.median())
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X = X.drop(['Citology', 'Hinselmann', 'Schiller'], axis=1)
    # Taking 10 features
    X_new = X[['Age', 'Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 'Smokes',
               'Smokes (years)', 'Hormonal Contraceptives', 'Hormonal Contraceptives (years)', 'IUD', 'IUD (years)']]
    # splitting data
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.25, random_state=0)
    print('Frequencies of classes:')
    print('Value:\t\t0\t\t1')
    uniq, freq = np.unique(y_train, return_counts=True)
    perc = freq * 100 / sum(freq)
    print(f'Counts:\t\t{freq[0]}\t\t{freq[1]}')
    print(f'Percentage:\t{perc[0]:.2f}\t\t{perc[1]:.2f}')
    # MinMaxScaling
    mm = MinMaxScaler()
    # feeding the independent data into the scaler
    X_train = mm.fit_transform(X_train)
    X_test = mm.fit_transform(X_test)
    return (X_train, X_test, y_train, y_test)

def catboost(X_train, X_test, y_train, y_test):
    # train model
    model = CatBoostClassifier()
    model.fit(X_train, y_train)
    return model