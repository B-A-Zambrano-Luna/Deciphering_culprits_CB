from scipy.signal import convolve
from scipy import signal
import numpy as np
import pandas as pd

Emptyvalue = -999


def Microcystin_change(value):
    if value == "No data":
        return 0
    elif value == "<0.05":
        return 0.05
    elif value == "<0.10":
        return 0.10
    else:
        return value


def Total_CyB(value):
    if value == "No data":
        return 0
    else:
        return value


def year_used(data, year):
    start_date = pd.to_datetime(year + '-01-31')
    end_date = pd.to_datetime(year + '-12-31')
    selected_data = data[start_date:end_date]
    return selected_data


def mean_over_lakes(data, labels):
    # Group by the index (Date) and calculate the mean for each date
    return data[labels].groupby(data[labels].index).mean()


def transform(value, label):
    if label == 'Total cyanobacterial cell count (cells/mL)':
        return float(value)*(14.12376*(10**(-9)))*(10**3)

    elif np.isnan(value):
        return Emptyvalue

    else:
        return value


def extractData(dataset, years, labels, lake):

    C1Name = dataset['Station'].str.contains(lake.upper())
    nameStations = dataset[C1Name]['Station'].unique()
    DATALABELS = {}
    dataset['SampleDate'] = pd.to_datetime(
        dataset['SampleDate'], format='mixed')
    dataset['day_of_year'] = dataset['SampleDate'].dt.dayofyear
    for label in labels:
        DATALABELS[label] = {}
        # Con1 = dataset['VariableName'] == label
        stationName = None
        maxSamples = 0
        for station in nameStations:
            ConS0 = dataset['Station'] == station
            if dataset[label][ConS0].count() >= maxSamples:
                maxSamples = dataset[label][ConS0].count()
                # print(label, maxSamples, station)
                stationName = station
        ConS0 = dataset['Station'] == stationName
        # print('########', stationName)
        datasetlabel = dataset[ConS0]
        for year in years:
            year_start = pd.Timestamp(year=int(year), month=1, day=1)
            year_end = pd.Timestamp(year=int(year) + 1, month=1, day=1)

            Con2 = datasetlabel['SampleDate'] >= year_start
            Con3 = datasetlabel['SampleDate'] < year_end

            DataLabel = datasetlabel.loc[Con2 & Con3,
                                         ['day_of_year', label]]

            for index, row in DataLabel.iterrows():
                # print(row[label], label, type(row[label]))
                day = int(row['day_of_year'])
                if day in DATALABELS[label]:
                    DATALABELS[label][day][years.index(
                        year)] = transform(row[label], label)
                else:
                    DATALABELS[label][day] = np.ones(len(years))*Emptyvalue
                    DATALABELS[label][day][years.index(
                        year)] = transform(row[label], label)

    return DATALABELS


def covolution_smooth(data, kernel=signal.windows.hann(5)/sum(signal.windows.hann(5))):
    return convolve(data, kernel, mode='same')


def signal_filter(data, signal_0):
    a, b = signal_0
    return signal.filtfilt(b, a, data)
