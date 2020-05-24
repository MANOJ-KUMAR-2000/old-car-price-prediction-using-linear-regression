#######################-PREDICTING OLD CAR PRICE USING LINEAR REGRESSION-#######################
"""
author : MANOJ KUMAR S
date   : 24.01.2020
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns


headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors",
           "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width", "height",
           "curb-weight", "engine-type", "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke",
           "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"]

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
DATA = pd.read_csv(url, header=None)
DATA.columns = headers

# pre-processing DATA
DATA.replace("?", np.nan, inplace=True)
DATA['normalized-losses'] = DATA['normalized-losses'].astype('float64')
DATA['price'] = DATA['price'].astype('float64')
DATA['bore'] = DATA['bore'].astype('float64')
DATA['stroke'] = DATA['stroke'].astype('float64')
DATA['horsepower'] = DATA['horsepower'].astype('float64')
DATA['peak-rpm'] = DATA['peak-rpm'].astype('float64')
DATA.replace(np.nan, DATA.mean(), inplace=True)

# finding corrlation within DATA
corr_df = DATA.corr("pearson")
corr_price = list(corr_df['price'])
feature = list(corr_df.columns)
corr_l = list(zip(corr_price, feature))
corr_l.sort()
y, x = list(zip(*corr_l))

    
def verify_by_graph():
    # corrlation graph
    plt.bar(x,y, width = 0.8)
    plt.title("corrlation graph")
    plt.xlabel("* features *")
    plt.ylabel("* corrlation value *")
    plt.show()

    # DISTRIBUTION GRAPH
    ax1 = sns.distplot(Y_TEST, hist=False, color='r', label='actual price')
    sns.distplot(predicted_price, hist=False, color='b', label='predicted price', ax=ax1)
    plt.title("COMPARISION OF PREDICTED AND ACTUAL PRICE")
    plt.show()

    # REGRESSIONLINE GRAPh
    sns.regplot(x="engine-size", y='price', data=DATA)
    plt.title("REGRESSION LINE between engine-size and price")
    plt.show()


# engine-size has high corrlation
# splitting DATA for training and testing
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(DATA[['engine-size']],
                                                    DATA[['price']], test_size=0.4, random_state=0)


# training data by LINEAR REGRESSION
lr = LinearRegression()
lr.fit(X_TRAIN, Y_TRAIN)

# predicting price by test datas
predicted_price = lr.predict(X_TEST)


#predicting price for user 
value = float(input("ENTER THE ENGINE-SIZE : "))
i = np.array(value).reshape(1,-1)
n_p = lr.predict(i)
print("PREDICTED PRICE : ", n_p[0][0])

verify_by_graph()
