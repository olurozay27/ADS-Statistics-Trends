# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 01:48:01 2023

@author: FABAYO OLUWATOLA SILAS
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import data from the csv file


def read_data(a, b):
    """
    Reads and imports files from comma seperated values, to a python DataFrame

    Arguments:
    a: string, The name of the csv file which is to be read
    b: integer, indicates the number of rows on the csv file to be
    skipped

    Returns:
    data: A pandas dataframe with all values from the excel file
    data_t: The transposed pandas dataframe
    """
    data = pd.read_csv(a, skiprows=b)
    data = data.drop(['Country Code', 'Indicator Code'], axis=1)
    transposed_data = data.set_index(
        data['Country Name']).T.reset_index().rename(columns={'index': 'Year'})
    transposed_data = transposed_data.set_index('Year').dropna(axis=1)
    transposed_data = transposed_data.drop(['Country Name'])
    return data, transposed_data


a = 'API_19_DS2_en_csv_v2_5336371.csv'
b = 4

data, transposed_data = read_data(a, b)

# Slicing the dataframe to get data for the indicators of interest


def indicator_set(c, d, e, f, g, h):
    """
    Reads and selects precise indicators from world bank dataframe,
    to a python DataFrame

    Arguments:
    c - h: 6 selected indicators from the dataset

    Returns:
    ind: A pandas dataframe with specific indicators selected
    """
    ind = data[data['Indicator Name'].isin([c, d, e, f, g, h])]

    return ind


c = 'CO2 emissions (kt)'
d = 'Electricity production from nuclear sources (% of total)'
e = 'Urban population (% of total population)'
f = 'Access to electricity (% of population)'
g = 'Renewable energy consumption (% of total final energy consumption)'
h = 'Electric power consumption (kWh per capita)'

ind = indicator_set(c, d, e, f, g, h)

# Slicing the dataframe to get data for the countries of interest


def country_set(countries):
    """
    Reads and selects country of interest from world bank dataframe,
    to a python DataFrame

    Arguments:
    countries: A list of countries selected from the dataframe
    Returns:
    specific_count: A pandas dataframe with specific countries selected
    """
    specific_count = ind[ind['Country Name'].isin(countries)]
    specific_count = specific_count.dropna(axis=1)
    specific_count = specific_count.reset_index(drop=True)
    return specific_count


# Selecting the countries specifically
countries = ['Qatar', 'China', 'Canada', 'New Zealand',
             'Russian Federation', 'Italy', 'United States', 'United Kingdom']

specific_count = country_set(countries)

# STATISTICS OF THE DATA
stats_desc = specific_count.groupby(["Country Name", "Indicator Name"])
print(stats_desc.describe())


def skew(dist):
    """ Calculates the centralised and normalised skewness of dist. """

    # calculates average and std, dev for centralising and normalising
    aver = np.mean(dist)
    std = np.std(dist)

    # now calculate the skewness
    value = np.sum(((dist-aver) / std)**3) / len(dist-1)

    return value


def kurtosis(dist):
    """ Calculates the centralised and normalised excess kurtosis of dist. """

    # calculates average and std, dev for centralising and normalising
    aver = np.mean(dist)
    std = np.std(dist)

    # now calculate the kurtosis
    value = np.sum(((dist-aver) / std)**4) / len(dist-1) - 3.0

    return value


def grp_countries_ind(indicator):
    """
    Selects and groups countries based on the specific indicators,
    to a python DataFrame

    Arguments:
    indicator: Choosing the indicator

    Returns:
    grp_ind_con: A pandas dataframe with specific countries selected
    """
    grp_ind_con = specific_count[specific_count["Indicator Name"] == indicator]
    grp_ind_con = grp_ind_con.set_index('Country Name', drop=True)
    grp_ind_con = grp_ind_con.transpose().drop('Indicator Name')
    grp_ind_con[countries] = grp_ind_con[countries].apply(
        pd.to_numeric, errors='coerce', axis=1)
    return grp_ind_con


# Giving each indicator a dataframe
CO2_emission = grp_countries_ind("CO2 emissions (kt)")
elec_prod = grp_countries_ind(
    "Electricity production from nuclear sources (% of total)")
urban_pop = grp_countries_ind("Urban population (% of total population)")
elec_access = grp_countries_ind("Access to electricity (% of population)")
ren_ene = grp_countries_ind(
    "Renewable energy consumption (% of total final energy consumption)")
ene_use = grp_countries_ind("Electric power consumption (kWh per capita)")

print(skew(CO2_emission))
print(kurtosis(CO2_emission))
