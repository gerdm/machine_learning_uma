import pandas as pd
import numpy as np
import os
from numpy import argmin
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from zipfile import ZipFile
import requests
from numpy.linalg import inv
from io import BytesIO


def estimate_theta(X, y, lmbda): return np.linalg.inv(X @ X.T + lmbda) @ X @ y
def l2_norm(theta, X, y): return np.linalg.norm(theta.T @ X - y.T) ** 2
def standarize_col(df, col): return (df[col] - df[col].mean()) / df[col].std()

def clean_neighborhood(neighborhood):
    """
    Clean the name of the neighborhoods stripping every space
    and replacing '/' for an underscore
    """
    return neighborhood.replace(" ", "").replace("/", "_")


def clean_room(room_type):
    """
    Clean the name of the neighborhoods stripping every '/'
    and replacing spaces for an underscore
    """
    return room_type.replace("/", "").replace(" ", "_")


def load_airbnb(plot_map=True):
    amsterdam_url = ("https://s3.amazonaws.com/"
                     "tomslee-airbnb-data-2/amsterdam.zip")
    print("Descargando información...")
    r = requests.get(amsterdam_url)
    airbnb = pd.DataFrame()

    with ZipFile(BytesIO(r.content)) as azip:
        for file in azip.namelist():
            print("Cargando archivo {file}".format(file=file), end="\r")
            infile = azip.open(file).read()
            infile = pd.read_csv(BytesIO(infile))
            airbnb = airbnb.append(infile)
            del infile
    
    print("Limpiando datos...")
    drop_cols = ["last_modified", "survey_id", "name",
                 "location", "bathrooms", "borough",
                 "city", "country", "host_id", "room_id"]
    airbnb.drop(drop_cols, axis=1, inplace=True)
    
    # Remove prices above 95th quantile and drop all NaN's
    airbnb.dropna(inplace=True)
    q95 = airbnb.price.quantile(q=0.95)
    airbnb.query("price < {q95}".format(q95=q95), inplace=True)
    
    # Clean categorical variables and make dummies for One Hot Encoding
    airbnb["neighborhood"] = airbnb["neighborhood"].apply(clean_neighborhood)
    airbnb["room_type"] = airbnb["room_type"].apply(clean_room)
    airbnb = pd.get_dummies(airbnb)
    # Include bias 
    airbnb.insert(0, "bias", 1)

    # Standarize numerical columns
    std_cols = ["accommodates", "bedrooms", "minstay", "overall_satisfaction", "reviews"]
    for col in std_cols:
        airbnb[col] = standarize_col(airbnb, col)
    
    if plot_map:
        airbnb.plot(x="latitude", y="longitude", c="price",
                    colormap = plt.cm.viridis,
                    kind="scatter", s=10, figsize=(10,7))
        plt.show()

    # Drop geolocation
    airbnb.drop(["latitude", "longitude"], axis=1, inplace=True)
    target_cols = airbnb.drop("price", axis=1).columns.values.tolist()
    return airbnb, target_cols


def plot_cost(regs, costs):
    optix = np.argmin(costs)
    plt.plot(regs, costs)
    plt.scatter(regs, costs)
    plt.scatter(regs[optix], costs[optix],
                color="tab:red", marker="x", s=150)
    plt.grid(alpha=0.6)
    plt.yscale("log")
    plt.ylabel(r"$J(\theta_i|X^{cv})$", fontsize=15)
    plt.xlabel(r"$\lambda_i$", fontsize=15)
    plt.show()
    
    
def train_cv_test_split(X, test_size=0.2, cv_size=0.2, verbose=False):
    train, test = train_test_split(X, test_size=0.2, random_state=1643)
    train, cv = train_test_split(train, test_size=0.2, random_state=1643)
    if not verbose:
        print("Número de observaciones dentro del 'training set' {:,}".format(train.shape[0]))
        print("Número de observaciones dentro del 'cv set' {:>11,}".format(cv.shape[0]))
        print("Número de observaciones dentro del 'test set' {:>9,}".format(test.shape[0]))
    return train, cv, test


def get_feature_targets(train, cv, test, target_col="price"):
    X_train, y_train = train.drop(target_col, axis=1).values, train[target_col].values.reshape(-1, 1)
    X_cv, y_cv = cv.drop(target_col, axis=1).values, cv[target_col].values.reshape(-1, 1)
    X_test, y_test = test.drop(target_col, axis=1).values, test[target_col].values.reshape(-1, 1)
    X_train, X_cv, X_test = X_train.T, X_cv.T, X_test.T
    
    features_targets = {"features": [X_train, X_cv, X_test],
                        "targets":  [y_train, y_cv, y_test]}
    
    return features_targets


def create_k_folds(data, nfolds, target_col="price"):
    """
    Data una seríe de datos n X m con 'n' observaciones
    y 'm' features y una columna objetivo, esta función
    crea una lista de tuples con 'nfolds' pares (un tuple)
    de X (training features) y 'y' (objetivo)
    
    Parametros
    ----------
    data: Pandas.Dataframe con n filas y m columnas
    nfolds: int. Número de 'folds' a separar los datos
    target_col: str. Nombre de la columna a seleccionar
        como objetivo
        
    Regresa
    -------
    lista de 'nfolds' tuples con (Xi, yi); Xi features
    yi targets
    """
    foldsize = int(data.shape[0] / nfolds)
    folds = []
    for k in range(nfolds):
        if k + 1!= nfolds:
            fold = data.iloc[k * foldsize: (k+1) * foldsize,:]
        else:
            fold = data.iloc[k * foldsize:]
        
        X_fold = fold.drop(target_col, axis=1).T.values
        y_fold = fold[target_col].values.reshape(-1, 1)
        folds.append((X_fold, y_fold))
    
    return folds


def make_train_fold(folds, k):
    """
    Dada una lista de tuples de pares (Xi, yi);  Xi in R(m X n) y
    yi in R(n X 1) y un índice 'k'. Regresa un solo par (X, y)
    considerando cada tuple dentro de folds excepto por el k-ésimo
    índice. **Función para poder crear un training set usando el método
    k-fold cv**
    
    Parámetros
    ----------
    folds: lista de tuples (Xi, yi) donde Xi in R(m X n) y
        yi in R(n x 1); m 'features' y n 'training examples'
    k: int. Índice menor a len(folds) de ejemplos a omitir
    
    Regresa 
    -------
    tuple: (X, y) concatenando cada elemento dentro de folds,
        excepto por el k-ésimo índice.
    """
    X = np.c_[tuple(fold[0] for ix, fold in enumerate(folds) if ix != k)]
    y = np.r_[tuple(fold[1] for ix, fold in enumerate(folds) if ix != k)]
    return X, y


if __name__ == "__main__":
    pass
