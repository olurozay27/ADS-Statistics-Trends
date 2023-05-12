# -*- coding: utf-8 -*-
"""
Created on Fri May 12 18:28:59 2023

@author: DELL
"""

#import modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn import cluster
import sklearn.metrics as skmet
from sklearn.linear_model import LogisticRegression
import cluster_tools as ct


def world_bank_data(filename):
    """
    Reads in data from a World Bank data file in CSV format.

    Args:
        filename (str): The name of the file to read.
        
    Returns:
        A pandas DataFrame containing the data from the file.
    """
    # Read in the CSV file using pandas
    df = pd.read_csv(filename, skiprows=4)
    return df


def read_world_bank_data():
    """
    Reads in the World Bank data for China's urban population and urban population growth, cleans the data,
    and returns a DataFrame with the relevant information.

    Returns:
    df_curve (pandas.DataFrame): DataFrame with the following columns:
                                 - Date: the year of the observation
                                 - Urban population: the urban population in Nigeria for that year
    """
    # Read in the data
    df_new = pd.read_csv('API_19_DS2_en_csv_v2_5361599.csv', skiprows=4)

    # Clean the data and extract urban population
    df_new = df_new[(df_new["Country Name"] == "China") & (
        df_new["Indicator Name"].isin(["Urban population", "Urban population growth (annual %)"]))]
    df_new = df_new.drop(['Country Name', 'Country Code',
                         'Indicator Name', 'Indicator Code', 'Unnamed: 66'], axis=1)
    df_curve = df_new.transpose()
    df_curve = df_curve.rename(
        columns={13225: "Urban population", 13226: "Urban population growth (annual %)"})
    df_curve = df_curve.reset_index()
    df_curve.columns = ['Date', 'Urban population',
                        'Urban population growth (annual %)']
    df_curve = df_curve.dropna()
    df_curve['Urban population'] = df_curve['Urban population'].astype(int)
    df_curve['Date'] = df_curve['Date'].astype(int)
    df_curve = df_curve.set_index('Date')
    return df_curve


df = world_bank_data("API_NY.GDP.PCAP.CD_DS2_en_csv_v2_5447781.csv")
df = df.drop(["Country Code", "Indicator Name", "Indicator Code"], axis=1)
df = df.dropna(axis=1, how='all')
df = df.set_index('Country Name').T
df_Gdp = df.transpose()
df_Gdp = df_Gdp.loc[['Nigeria', 'China', 'India', 'Aruba', 'Japan']]
df_Gdp = df_Gdp.transpose()
print(df_Gdp)
print(df_Gdp.describe())

ct.map_corr(df_Gdp)
# scatter plot
pd.plotting.scatter_matrix(df_Gdp, figsize=(6.0, 6.0))
plt.tight_layout()  # helps to avoid overlap of labels
plt.show()
# the plot shows that China  and Japan has the lowest correlation

# copy the dataframe to prevent changes of the original dataframe(df_new)
df_fit = df_Gdp[['China', 'Japan']].copy()

df_fit, df_min, df_max = ct.scaler(df_fit)
print(df_fit.describe())

# loop over trial numbers of clusters calculating the silhouette
for ic in range(2, 10):
    # set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=ic)
    kmeans.fit(df_fit)
    # extract labels and calculate silhoutte score
    labels = kmeans.labels_
    print(ic, skmet.silhouette_score(df_fit, labels))

# Using 3 number of clusters
nc = 3  # number of cluster centres
kmeans = cluster.KMeans(n_clusters=nc)
kmeans.fit(df_fit)
# extract labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_
plt.figure(figsize=(6.0, 6.0))
plt.scatter(df_Gdp["China"], df_Gdp["Japan"], c=labels, cmap="tab10")
# colour map Accent selected to increase contrast between colours
# show cluster centres
# rescale and show cluster centres
scen = ct.backscale(cen, df_min, df_max)
xc = cen[:, 0]
yc = cen[:, 1]
plt.scatter(xc, yc, c="k", marker="d", s=80)
# c = colour, s = size
plt.xlabel("China Gdp")
plt.ylabel("Japan Gdp")
plt.title("clusters plot of GDP of china and Japan ")
plt.show()

# produce a curve fit using logistic function
df_curve = read_world_bank_data()
print(df_curve)

# Define the logistic function


def logistic(x, L, k, x0):
    """
    Calculates the logistic function for given input values.

    Args:
    x (float or array-like): Input value(s) for the logistic function.
    L (float): Maximum value of the function.
    k (float): Growth rate of the function.
    x0 (float): Midpoint of the function.

    Returns:
    float or ndarray: The output of the logistic function for the input value(s).
    """
    return L / (1 + np.exp(-k*(x-x0)))


# Define the data
x_data = df_curve.index.values
y_data = df_curve['Urban population'].values

# Define initial guesses for the parameters
p0 = [np.max(y_data), 0.1, np.median(x_data)]

# Fit the data to the logistic function
popt, pcov = curve_fit(logistic, x_data, y_data, p0)

# Plot the data and the fitted curve
plt.plot(x_data, y_data, 'o', label='data')
plt.plot(x_data, logistic(x_data, *popt), label='fit')
plt.xlabel('Year')
plt.ylabel('Urban population')
plt.title('Urban pop growth_China (curv_fit)')
plt.legend()
plt.show()

# Fit logistic regression model
X = df_curve.index.values.reshape(-1, 1)
y = df_curve['Urban population']
model = LogisticRegression()
model.fit(X, y)

# Make predictions for 2030 and 2050
years = [2030]
for year in years:
    pred = model.predict(np.array(year).reshape(-1, 1))[0]
    print(f"Predicted urban population for China in {year}: {int(pred):,}")
