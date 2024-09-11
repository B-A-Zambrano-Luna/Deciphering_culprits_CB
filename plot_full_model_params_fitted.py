from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import fullmodel_v1_8
import pandas as pd
from Extract_dataV2 import extractData
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
from datetime import datetime, timedelta
# Define your ODE function

# Initial conditions
M_0 = 0
A_0 = 0.05
B_0 = 0.004  # 0.004
Q_B_0 = 0.01
Q_A_0 = 0.01
P_0 = 0.006  # 10
D_0 = 0.0239  # 0.0239
Y_0 = 0.025
W_0 = 0.0025
O_0 = 6
v_A_0 = 0
v_D_0 = 0
v_Y_0 = 0
v_W_0 = 0


model_CyB = fullmodel_v1_8.modelCyB()

initial_conditions = [M_0, B_0, A_0,
                      Q_B_0,
                      Q_A_0, P_0,
                      D_0,
                      Y_0, W_0,
                      v_A_0, v_D_0,
                      v_Y_0, v_W_0, O_0]


# Data
data = pd.read_csv(
    "merged_water_quality_data.csv", low_memory=False)
lake_name = "PIGEON LAKE"
labels = ['MICROCYSTIN, TOTAL',
          'PHOSPHORUS TOTAL DISSOLVED',
          'OXYGEN DISSOLVED (FIELD METER)',
          'Total cyanobacterial cell count (cells/mL)',
          'TEMPERATURE WATER']
# labels = ['Total cyanobacterial cell count (cells/mL)',
#           'TEMPERATURE WATER']
# years = ['2018', '2019', '2020', '2021', '2022', '2023']
years = ['2021']
coment = "_v4_"

data_fit = extractData(data, years, labels, lake_name)

# # import parameters


def read_params():
    model_CyB = fullmodel_v1_8.modelCyB()
    model_CyB.initial = initial_conditions
    name_data_param = "fitting_parameters_full_variables_v1"
    name_data = './FittedParameters/' + \
        'fitting_parameters_full_variables_v2PIGEON LAKE'+coment + '_final' + ".csv"

    params_fit = pd.read_csv(name_data)

    # params_fit = pd.read_csv(
    #     "fitting_parameters_full_variables_v12021Pine Lake_v2__final.csv")
    unknow_params = ["e_BD", "alpha_B", "alpha_D",
                     "alpha_Y",
                     "tau_B", "tau_D", "tau_Y",
                     "a_A", "a_D", "sigma_A",
                     "sigma_D", "x_A", "x_D"]

    model_CyB.initial[1] = params_fit["B_0"][0]
    model_CyB.initial[2] = params_fit["A_0"][0]
    model_CyB.initial[6] = params_fit["D_0"][0]

    for name in unknow_params:
        model_CyB.params[name] = params_fit[name][0]

    # Death Daphnia
    model_CyB.params['n_D'] = 0.0623
    model_CyB.params['e_BD'] = 0.8
    model_CyB.params["tau_B"] = 1.23
    model_CyB.params["alpha_B"] = 0.0035
    model_CyB.params['r_Y'] = 2
    model_CyB.params['r_W'] = 1
    model_CyB.params['Ext_Y'] = 0.025*13
    model_CyB.params['Ext_W'] = 0.025
    model_CyB.params['p'] = 0.9
    model_CyB.params['p_in'] = 0.03  # 0.09601  # 0.125  # (0.009601/2)*0

    # With Toxine
    model_CyB.toxines = True and (model_CyB.initial[1] > 0)
    # New time scale
    # New time scale
    day_start = pd.to_datetime("2023-05-01").day_of_year

    days_end = pd.to_datetime("2023-09-30").day_of_year

    model_CyB.t_0 = 0  # May 1, 2023

    model_CyB.t_f = days_end - day_start

    model_CyB.delta_t = model_CyB.t_f*3

    model_CyB.set_linetime()

    # Temperature and Zm
    path = './ERA5-Land/' + years[-1]
    TempZmData = pd.read_csv(path+lake_name + 'WaterTemperature.csv')
    TempZmData['Date'] = pd.to_datetime(
        TempZmData['Date'], format='mixed')

    tempSamp = TempZmData['lake_mix_layer_temperature']
    Zmsample = (TempZmData['lake_mix_layer_depth_min'] +
                TempZmData['lake_mix_layer_depth_max'])*0.5
    days = TempZmData['Date'].dt.day_of_year

    days = np.array(days) - day_start

    model_CyB.get_interpTemp(tempSamp, days)
    model_CyB.get_interpZm(Zmsample, days)
    return model_CyB


# Fit parameters


model_CyB = read_params()
t_data = model_CyB.t
# Solution model
all_plot = False
somePlots = True
fill = False

solution, info = model_CyB.solver()

path = './New data/Images/Year v_1/'
name = 'Full_model'+".pdf"

# model_CyB.print_solv(title='', all_plot=True,
#                      save=False,
#                      save_path=path+name,
#                      dpi=RESOLUTION)


# y_data = data_fit["Total cyanobacterial cell count (cells/mL)"].values


M_values, B_values, A_values, \
    Q_B_values, Q_A_values, P_values, \
    D_values, Y_values, W_values, \
    v_A_values, v_D_values, v_Y_values, \
    v_W_values, O_values = solution.T


# Persentanges

# M_max = M_values.max()
# B_max = B_values.max()
# v_Y_max = v_Y_values.max()
# v_W_max = v_W_values.max()
# O_min = O_values[40*3:].min()


def generate_dates(year):
    dates = []
    for month in range(5, 10):  # January to December
        next_month = month + 1 if month < 12 else 1  # Handle December
        # Handle December crossing into the next year
        next_year = year + 1 if month == 12 else year

        max_day = (datetime(next_year, next_month, 1) - timedelta(days=1)).day
        for day in range(1, max_day + 1, 1):
            start_date = datetime(year, month, day)
            # Adding one day to get the end date
            dates.append(start_date.strftime("%Y-%m-%d"))
    return dates


day_start = pd.to_datetime("2023-05-01").day_of_year

# Plot fitting data
for label in labels:
    sns.set_style('ticks')
    sns.plotting_context("paper", font_scale=1.5)  # Adjust font size as needed
    fig, axs = plt.subplots(1, 1, figsize=(11 / 2.54, 11 / 2.54))
    # axs = axs.ravel()
    tB = np.array(
        [day - day_start for day in data_fit[label]])
    yB = [data_fit[label][day][0]
          for day in data_fit[label]]
    axs.scatter(tB, yB, color=(
        250/255, 134/255, 0/255), label="Observed Data")

    if label == 'MICROCYSTIN, TOTAL':
        ylabel = "M $[\mu g/L]$ "
        y_values = M_values
    elif label == 'PHOSPHORUS TOTAL DISSOLVED':
        ylabel = "P $[mg P/L]$ "
        y_values = P_values
    elif label == 'OXYGEN DISSOLVED (FIELD METER)':
        ylabel = "O $[mg O/L]$ "
        y_values = O_values
    elif label == 'Total cyanobacterial cell count (cells/mL)':
        ylabel = "B $[mg C/L]$ "
        y_values = B_values
    elif label == 'TEMPERATURE WATER':
        ylabel = "T $[C^{\circ}]$ "
        y_values = model_CyB.Temp(model_CyB.t)

    axs.plot(model_CyB.t, y_values, color=(19/255, 103/255, 131/255))

    # plt.xlabel("Time (days)")
    plt.xlabel("")
    plt.ylabel(ylabel)
    plt.title('')
    y_formatter = ScalarFormatter(useMathText=True, useOffset=False)
    y_formatter.set_powerlimits((-3, 4))
    y_formatter.orderOfMagnitude = 4
    axs.yaxis.set_major_formatter(y_formatter)

    for spine in axs.spines.values():
        spine.set_color('black')
    axs.tick_params(axis='both', which='both', bottom=True, top=True,
                    left=True, right=True, direction='in', length=4, width=1, colors='black')

    # axs.set_ylim(B_values.min()-0.001, B_values.max()*(1+0.05))
    # axs.set_xlim(model_CyB.t.min(), model_CyB.t.max())

    # Dates as x axis
    dates = generate_dates(2021)
    x_ticks = axs.xaxis.get_ticklocs()

    if len(dates) > len(x_ticks):
        x_stape = round(len(dates) / len(x_ticks), 0)
        x_labels = [dates[int(x_val)] for x_val in x_ticks[:-1]]
        axs.xaxis.set_ticklabels(x_labels,
                                 rotation=30)
    else:
        x_labels = dates
        axs.xaxis.set_ticklabels(x_labels,
                                 rotation=30)

    # plt.legend()
    plt.tight_layout()

# Other variables Zm
sns.set_style('ticks')
sns.plotting_context("paper", font_scale=1.5)  # Adjust font size as needed
fig, axs = plt.subplots(1, 1, figsize=(11 / 2.54, 11 / 2.54))
# axs = axs.ravel()
tB = np.array(
    [day - day_start for day in data_fit[label]])

y_values = np.array([model_CyB.Zm(t) for t in model_CyB.t])

axs.plot(model_CyB.t, y_values, color=(19/255, 103/255, 131/255))

# plt.xlabel("Time (days)")
plt.xlabel("")
plt.ylabel("Zm $[m]")
plt.title('')
y_formatter = ScalarFormatter(useMathText=True, useOffset=False)
y_formatter.set_powerlimits((-3, 4))
y_formatter.orderOfMagnitude = 4
axs.yaxis.set_major_formatter(y_formatter)

for spine in axs.spines.values():
    spine.set_color('black')
axs.tick_params(axis='both', which='both', bottom=True, top=True,
                left=True, right=True, direction='in', length=4, width=1, colors='black')

# axs.set_ylim(B_values.min()-0.001, B_values.max()*(1+0.05))
# axs.set_xlim(model_CyB.t.min(), model_CyB.t.max())

# Dates as x axis
dates = generate_dates(2023)
x_ticks = axs.xaxis.get_ticklocs()

if len(dates) > len(x_ticks):
    x_stape = round(len(dates) / len(x_ticks), 0)
    x_labels = [dates[int(x_val)] for x_val in x_ticks[:-1]]
    axs.xaxis.set_ticklabels(x_labels,
                             rotation=30)
else:
    x_labels = dates
    axs.xaxis.set_ticklabels(x_labels,
                             rotation=30)

# plt.legend()
plt.tight_layout()


# Daphnia

sns.set_style('ticks')
sns.plotting_context("paper", font_scale=1.5)  # Adjust font size as needed
fig, axs = plt.subplots(1, 1, figsize=(11 / 2.54, 11 / 2.54))
# axs = axs.ravel()
tB = np.array(
    [day - day_start for day in data_fit[label]])

y_values = D_values

axs.plot(model_CyB.t, y_values, color=(19/255, 103/255, 131/255))

# plt.xlabel("Time (days)")
plt.xlabel("")
plt.ylabel("Daphnia")
plt.title('')
y_formatter = ScalarFormatter(useMathText=True, useOffset=False)
y_formatter.set_powerlimits((-3, 4))
y_formatter.orderOfMagnitude = 4
axs.yaxis.set_major_formatter(y_formatter)

for spine in axs.spines.values():
    spine.set_color('black')
axs.tick_params(axis='both', which='both', bottom=True, top=True,
                left=True, right=True, direction='in', length=4, width=1, colors='black')

# axs.set_ylim(B_values.min()-0.001, B_values.max()*(1+0.05))
# axs.set_xlim(model_CyB.t.min(), model_CyB.t.max())

# Dates as x axis
dates = generate_dates(2023)
x_ticks = axs.xaxis.get_ticklocs()

if len(dates) > len(x_ticks):
    x_stape = round(len(dates) / len(x_ticks), 0)
    x_labels = [dates[int(x_val)] for x_val in x_ticks[:-1]]
    axs.xaxis.set_ticklabels(x_labels,
                             rotation=30)
else:
    x_labels = dates
    axs.xaxis.set_ticklabels(x_labels,
                             rotation=30)

# plt.legend()
plt.tight_layout()

# Algae

sns.set_style('ticks')
sns.plotting_context("paper", font_scale=1.5)  # Adjust font size as needed
fig, axs = plt.subplots(1, 1, figsize=(11 / 2.54, 11 / 2.54))
# axs = axs.ravel()
tB = np.array(
    [day - day_start for day in data_fit[label]])

y_values = A_values

axs.plot(model_CyB.t, y_values, color=(19/255, 103/255, 131/255))

# plt.xlabel("Time (days)")
plt.xlabel("")
plt.ylabel("Algea")
plt.title('')
y_formatter = ScalarFormatter(useMathText=True, useOffset=False)
y_formatter.set_powerlimits((-3, 4))
y_formatter.orderOfMagnitude = 4
axs.yaxis.set_major_formatter(y_formatter)

for spine in axs.spines.values():
    spine.set_color('black')
axs.tick_params(axis='both', which='both', bottom=True, top=True,
                left=True, right=True, direction='in', length=4, width=1, colors='black')

# axs.set_ylim(B_values.min()-0.001, B_values.max()*(1+0.05))
# axs.set_xlim(model_CyB.t.min(), model_CyB.t.max())

# Dates as x axis
dates = generate_dates(2023)
x_ticks = axs.xaxis.get_ticklocs()

if len(dates) > len(x_ticks):
    x_stape = round(len(dates) / len(x_ticks), 0)
    x_labels = [dates[int(x_val)] for x_val in x_ticks[:-1]]
    axs.xaxis.set_ticklabels(x_labels,
                             rotation=30)
else:
    x_labels = dates
    axs.xaxis.set_ticklabels(x_labels,
                             rotation=30)

# plt.legend()
plt.tight_layout()


# Yellow Perch

sns.set_style('ticks')
sns.plotting_context("paper", font_scale=1.5)  # Adjust font size as needed
fig, axs = plt.subplots(1, 1, figsize=(11 / 2.54, 11 / 2.54))
# axs = axs.ravel()
tB = np.array(
    [day - day_start for day in data_fit[label]])

y_values = Y_values

axs.plot(model_CyB.t, y_values, color=(19/255, 103/255, 131/255))

# plt.xlabel("Time (days)")
plt.xlabel("")
plt.ylabel("Yellow Perch")
plt.title('')
y_formatter = ScalarFormatter(useMathText=True, useOffset=False)
y_formatter.set_powerlimits((-3, 4))
y_formatter.orderOfMagnitude = 4
axs.yaxis.set_major_formatter(y_formatter)

for spine in axs.spines.values():
    spine.set_color('black')
axs.tick_params(axis='both', which='both', bottom=True, top=True,
                left=True, right=True, direction='in', length=4, width=1, colors='black')

# axs.set_ylim(B_values.min()-0.001, B_values.max()*(1+0.05))
# axs.set_xlim(model_CyB.t.min(), model_CyB.t.max())

# Dates as x axis
dates = generate_dates(2023)
x_ticks = axs.xaxis.get_ticklocs()

if len(dates) > len(x_ticks):
    x_stape = round(len(dates) / len(x_ticks), 0)
    x_labels = [dates[int(x_val)] for x_val in x_ticks[:-1]]
    axs.xaxis.set_ticklabels(x_labels,
                             rotation=30)
else:
    x_labels = dates
    axs.xaxis.set_ticklabels(x_labels,
                             rotation=30)

# plt.legend()
plt.tight_layout()

# Walleye

sns.set_style('ticks')
sns.plotting_context("paper", font_scale=1.5)  # Adjust font size as needed
fig, axs = plt.subplots(1, 1, figsize=(11 / 2.54, 11 / 2.54))
# axs = axs.ravel()
tB = np.array(
    [day - day_start for day in data_fit[label]])

y_values = W_values

axs.plot(model_CyB.t, y_values, color=(19/255, 103/255, 131/255))

# plt.xlabel("Time (days)")
plt.xlabel("")
plt.ylabel("Walleye")
plt.title('')
y_formatter = ScalarFormatter(useMathText=True, useOffset=False)
y_formatter.set_powerlimits((-3, 4))
y_formatter.orderOfMagnitude = 4
axs.yaxis.set_major_formatter(y_formatter)

for spine in axs.spines.values():
    spine.set_color('black')
axs.tick_params(axis='both', which='both', bottom=True, top=True,
                left=True, right=True, direction='in', length=4, width=1, colors='black')

# axs.set_ylim(B_values.min()-0.001, B_values.max()*(1+0.05))
# axs.set_xlim(model_CyB.t.min(), model_CyB.t.max())

# Dates as x axis
dates = generate_dates(2023)
x_ticks = axs.xaxis.get_ticklocs()

if len(dates) > len(x_ticks):
    x_stape = round(len(dates) / len(x_ticks), 0)
    x_labels = [dates[int(x_val)] for x_val in x_ticks[:-1]]
    axs.xaxis.set_ticklabels(x_labels,
                             rotation=30)
else:
    x_labels = dates
    axs.xaxis.set_ticklabels(x_labels,
                             rotation=30)

# plt.legend()
plt.tight_layout()

# Save plots
# path = './New data/Images/Year v_1/'
# name = 'Full_model_fit'+FORMAT
# plt.savefig(path+name, dpi=RESOLUTION, bbox_inches='tight')

# plt.show()
