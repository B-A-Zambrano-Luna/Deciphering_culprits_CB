from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import fullmodel_v1_8
import pandas as pd
from Extract_dataV2 import extractData
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime, timedelta
# Define your ODE function

# Initial conditions
M_0 = 0
A_0 = 0.05
B_0 = 0.004  # 0.004
Q_B_0 = 0.01
Q_A_0 = 0.01
# P_0 = 0.4  # 0.075  # 10
D_0 = 0.0239  # 0.0239
Y_0 = 0.025*10
W_0 = 0.0025*10
# O_0 = 5
v_A_0 = 0
v_D_0 = 0
v_Y_0 = 0
v_W_0 = 0


# Data
LakeYear = {'PIGEON LAKE': ['2021'],
            'PINE LAKE': ['2017'],
            'MONONA LAKE': ['2015'],
            'MENDOTA LAKE': ['2018']}


lake_name = "MENDOTA LAKE"


years = LakeYear[lake_name]

yearname = ''
for year in years:
    yearname = yearname + str(year) + '_'

if lake_name == 'PIGEON LAKE' and years[-1] == '2021':
    coment = "_v6_3Var_"
else:
    coment = "_v6_3Var_" + yearname

if lake_name != "PINE LAKE":
    coment = "_v7_3Var_" + yearname


if years[-1] == '2018' and lake_name == 'MENDOTA LAKE':
    data = pd.read_csv(
        "Dataset_US.csv", low_memory=False)

    labels = ['Microcystin (nM)',
              'OXYGEN DISSOLVED (FIELD METER)',
              'Total cyanobacterial cell count (cells/mL)']
elif lake_name == 'MONONA LAKE':

    data = pd.read_csv(
        "Combined_Data_for_MO_Merged.csv", low_memory=False)

    labels = ['OXYGEN DISSOLVED (FIELD METER)',
              'Total cyanobacterial cell count (cells/mL)',
              'TEMPERATURE WATER']

elif lake_name in ['PIGEON LAKE', 'PINE LAKE']:
    data = pd.read_csv(
        "merged_water_quality_data.csv", low_memory=False)

    labels = ['MICROCYSTIN, TOTAL',
              'PHOSPHORUS TOTAL DISSOLVED',
              'OXYGEN DISSOLVED (FIELD METER)',
              'Total cyanobacterial cell count (cells/mL)',
              'TEMPERATURE WATER']

data_fit = extractData(data, years, labels, lake_name)

# # import parameters

if lake_name in ['MONONA LAKE']:
    rectTemp = 3.0
    B_scale = 1000*0.5 #0.1
    A_scale = 0.01 #100
    D_scale = 1 #0.1
    P_0 = 0.01
    P_in = 0.01
    O_0 = 5

elif lake_name in ['MENDOTA LAKE']:
    rectTemp = 3.0 #3.0
    B_scale = 0.15 #0.1
    A_scale = 0.05 #10
    D_scale = 5 #10
    P_0 = 0.075 #0.075
    P_in = 0.015
    O_0 = 5

elif lake_name in ['PIGEON LAKE']:
    rectTemp = -1.75 # 0
    B_scale = 0.04 #1
    A_scale = 0.01 #1
    D_scale = 0.075 #1
    P_0 = 0.05
    P_in = 0.01
    O_0 = 6

elif lake_name in ['PINE LAKE']:
    rectTemp = -2.75
    B_scale = 1.0
    A_scale = 2.0
    D_scale = 0.1
    P_0 = 0.4
    P_in = 0.2
    O_0 = 5


model_CyB = fullmodel_v1_8.modelCyB()

initial_conditions = [M_0, B_0, A_0,
                      Q_B_0,
                      Q_A_0, P_0,
                      D_0,
                      Y_0, W_0,
                      v_A_0, v_D_0,
                      v_Y_0, v_W_0, O_0]

print('...Processing', lake_name)


def read_params():
    model_CyB = fullmodel_v1_8.modelCyB()
    model_CyB.initial = initial_conditions
    name_data_param = "fitting_parameters_full_variables_v1"
    name_data = './FittedParameters/' + \
        'fitting_parameters_full_variables_v2'+lake_name + coment + '_final' + ".csv"
    # name_data = './FittedParameters/' + \
    #     'fitting_parameters_full_variables_v2CHESTERMERE LAKE_v4_3Var__backup.csv'
    params_fit = pd.read_csv(name_data)

    # params_fit = pd.read_csv(
    #     "fitting_parameters_full_variables_v12021Pine Lake_v2__final.csv")
    unknow_params = ["alpha_D", "alpha_Y",
                     "tau_D", "tau_Y",
                     "a_A",
                     "sigma_A",
                     "x_A"]

    model_CyB.initial[1] = params_fit["B_0"][0]*B_scale
    print('B(0)=', model_CyB.initial[1])
    model_CyB.initial[2] = params_fit["A_0"][0]*A_scale
    print('A(0)=', model_CyB.initial[2])
    model_CyB.initial[6] = params_fit["D_0"][0]*D_scale
    print('D(0)=', model_CyB.initial[6])

    for name in unknow_params:
        model_CyB.params[name] = params_fit[name][0]
        print('....Parameters')
        print(name, params_fit[name][0])
    # Death Daphnia

    # 0.015  # 0.03 Pigeon lake  # 0.09601  # 0.125
    model_CyB.params['p_in'] = P_in
    # model_CyB.params["delta_B-"] = 2.4
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

    tempSamp = TempZmData['lake_mix_layer_temperature']-rectTemp
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

# path = './New data/Images/Year v_1/'
# name = 'Full_model'+".pdf"

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


SaveFigures = True
yearname = ''
for year in years:
    yearname = yearname + str(year) + '_'
    
path = "./Figures/"+lake_name + '/'+yearname
os.makedirs(path, exist_ok=True)
RESOLUTION = 900


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
day_end = pd.to_datetime("2023-09-30").day_of_year
# Plot fitting data
for label in labels:
    sns.set_style('ticks')
    sns.plotting_context("paper", font_scale=1.5)  # Adjust font size as needed
    fig, axs = plt.subplots(1, 1, figsize=(11 / 2.54, 11 / 2.54))
    # axs = axs.ravel()
    tB = np.array(
        [day - day_start for day in data_fit[label] if day >= day_start and day <= day_end])
    yB = [data_fit[label][day][0]
          for day in data_fit[label] if day >= day_start and day <= day_end]

    if len(yB) == 0:
        labelData = None
    else:
        labelData = "Field measurements"
    axs.scatter(tB, yB, color=(
        250/255, 134/255, 0/255), label=labelData)

    if label in ['MICROCYSTIN, TOTAL', 'Microcystin (nM)']:
        name = 'MICROCYSTIN'
        ylabel = "Microcystin-LR ($\mu g/L$)"
        y_values = M_values
    elif label == 'PHOSPHORUS TOTAL DISSOLVED':
        name = 'PHOSPHORUS'
        ylabel = "Dissolved phosphorus ($mg P/L$)"
        y_values = P_values
    elif label == 'OXYGEN DISSOLVED (FIELD METER)':
        name = 'OXYGEN'
        ylabel = "Dissolved $O_2$ ($mg O_2/L$)"
        y_values = O_values
    elif label == 'Total cyanobacterial cell count (cells/mL)':
        name = 'CB'
        ylabel = "Cyanobacterial biomass ($mg C/L$)"
        y_values = B_values
    elif label == 'TEMPERATURE WATER':
        name = 'TEMPERATURE WATER'
        ylabel = "Water temperature ($C^{\circ}$)"
        y_values = model_CyB.Temp(model_CyB.t)

    axs.plot(model_CyB.t, y_values, color=(19/255, 103/255, 131/255))

    # plt.xlabel("Time (days)")
    plt.xlabel("")
    plt.ylabel(ylabel)
    plt.title('')
    plt.legend()
    y_formatter = ScalarFormatter(useMathText=True, useOffset=False)
    y_formatter.set_powerlimits((-3, 4))
    y_formatter.orderOfMagnitude = 4
    axs.yaxis.set_major_formatter(y_formatter)

    for spine in axs.spines.values():
        spine.set_color('black')
    axs.tick_params(axis='both', which='both', bottom=True, top=True,
                    left=True, right=True, direction='in', length=4, width=1, colors='black')

    # axs.set_ylim(y_values.min()-0.001, y_values.max()*(1+0.05))
    # axs.set_xlim(model_CyB.t.min(), model_CyB.t.max())

    # Dates as x axis
    dates = generate_dates(int(year))
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
    if SaveFigures:
        plt.savefig(path + name + ".pdf", dpi=RESOLUTION, bbox_inches='tight')


if not 'PHOSPHORUS TOTAL DISSOLVED' in labels:
    sns.set_style('ticks')
    sns.plotting_context("paper", font_scale=1.5)  # Adjust font size as needed
    fig, axs = plt.subplots(1, 1, figsize=(11 / 2.54, 11 / 2.54))
    # axs = axs.ravel()
    tB = np.array(
        [day - day_start for day in data_fit[label]])

    y_values = P_values

    axs.plot(model_CyB.t, y_values, color=(19/255, 103/255, 131/255))

    # plt.xlabel("Time (days)")
    plt.xlabel("")
    plt.ylabel("Dissolved phosphorus ($mg P/L$)")
    plt.title('')
    y_formatter = ScalarFormatter(useMathText=True, useOffset=False)
    y_formatter.set_powerlimits((-3, 4))
    y_formatter.orderOfMagnitude = 4
    axs.yaxis.set_major_formatter(y_formatter)

    for spine in axs.spines.values():
        spine.set_color('black')
    axs.tick_params(axis='both', which='both', bottom=True, top=True,
                    left=True, right=True, direction='in', length=4, width=1, colors='black')

    # axs.set_ylim(y_values.min()-0.001, y_values.max()*(1+0.05))
    # axs.set_xlim(model_CyB.t.min(), model_CyB.t.max())

    # Dates as x axis
    # dates = generate_dates(2023)
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
    if SaveFigures:
        plt.savefig(path + 'PHOSPHORUS' + ".pdf",
                    dpi=RESOLUTION, bbox_inches='tight')


if not 'MICROCYSTIN, TOTAL' in labels and not 'Microcystin (nM)' in labels:

    sns.set_style('ticks')
    sns.plotting_context("paper", font_scale=1.5)  # Adjust font size as needed
    fig, axs = plt.subplots(1, 1, figsize=(11 / 2.54, 11 / 2.54))
    # axs = axs.ravel()
    if lake_name in ['MONONA LAKE']:
        dataMicrosystin = pd.read_csv('MicrocystinMONONALAKE.csv')

        CyB = dataMicrosystin['Lake'] == 'Monona'

        tM = np.array([model_CyB.t[15],
                       model_CyB.t[45*3],
                       model_CyB.t[75*3],
                       model_CyB.t[105*3],
                       model_CyB.t[135*3]])
        yM = dataMicrosystin[CyB]['MCLR'][:5].astype(float) / 100
        if len(yB) == 0:
            labelData = None
        else:
            labelData = "Field measurements (2012)"
        axs.scatter(tM, yM, color=(
            250/255, 134/255, 0/255), label=labelData)

    y_values = M_values

    axs.plot(model_CyB.t, y_values, color=(19/255, 103/255, 131/255))

    # plt.xlabel("Time (days)")
    plt.xlabel("")
    plt.ylabel("Microcystin-LR ($\mu g/L$)")
    plt.title('')
    y_formatter = ScalarFormatter(useMathText=True, useOffset=False)
    y_formatter.set_powerlimits((-3, 4))
    y_formatter.orderOfMagnitude = 4
    axs.yaxis.set_major_formatter(y_formatter)

    for spine in axs.spines.values():
        spine.set_color('black')
    axs.tick_params(axis='both', which='both', bottom=True, top=True,
                    left=True, right=True, direction='in', length=4, width=1, colors='black')

    # axs.set_ylim(y_values.min()-0.001, y_values.max()*(1+0.05))
    # axs.set_xlim(model_CyB.t.min(), model_CyB.t.max())

    # Dates as x axis
    # dates = generate_dates(2023)
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
    if SaveFigures:
        plt.savefig(path + 'MICROCYSTIN' + ".pdf",
                    dpi=RESOLUTION, bbox_inches='tight')


if not 'TEMPERATURE WATER' in labels:
    sns.set_style('ticks')
    sns.plotting_context("paper", font_scale=1.5)  # Adjust font size as needed
    fig, axs = plt.subplots(1, 1, figsize=(11 / 2.54, 11 / 2.54))
    # axs = axs.ravel()
    tB = np.array(
        [day - day_start for day in data_fit[label]])

    y_values = np.array([model_CyB.Temp(t) for t in model_CyB.t])

    axs.plot(model_CyB.t, y_values, color=(19/255, 103/255, 131/255))

    # plt.xlabel("Time (days)")
    plt.xlabel("")
    plt.ylabel("Water temperature ($C^{\circ}$)")
    plt.title('')
    y_formatter = ScalarFormatter(useMathText=True, useOffset=False)
    y_formatter.set_powerlimits((-3, 4))
    y_formatter.orderOfMagnitude = 4
    axs.yaxis.set_major_formatter(y_formatter)

    for spine in axs.spines.values():
        spine.set_color('black')
    axs.tick_params(axis='both', which='both', bottom=True, top=True,
                    left=True, right=True, direction='in', length=4, width=1, colors='black')

    axs.set_ylim(y_values.min()-0.001, y_values.max()*(1+0.05))
    axs.set_xlim(model_CyB.t.min(), model_CyB.t.max())

    # Dates as x axis
    # dates = generate_dates(2021)
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
    if SaveFigures:
        plt.savefig(path + 'TEMPERATURE WATER' + ".pdf",
                    dpi=RESOLUTION, bbox_inches='tight')


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
plt.ylabel("Epilimnion depth ($m$)")
plt.title('')
y_formatter = ScalarFormatter(useMathText=True, useOffset=False)
y_formatter.set_powerlimits((-3, 4))
y_formatter.orderOfMagnitude = 4
axs.yaxis.set_major_formatter(y_formatter)

for spine in axs.spines.values():
    spine.set_color('black')
axs.tick_params(axis='both', which='both', bottom=True, top=True,
                left=True, right=True, direction='in', length=4, width=1, colors='black')

axs.set_ylim(y_values.min()-0.001, y_values.max()*(1+0.05))
axs.set_xlim(model_CyB.t.min(), model_CyB.t.max())

# Dates as x axis
# dates = generate_dates(2021)
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
if SaveFigures:
    plt.savefig(path + 'Epilimnion' + ".pdf",
                dpi=RESOLUTION, bbox_inches='tight')

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
plt.ylabel("Daphnia biomass ($mg C/L$)")
plt.title('')
y_formatter = ScalarFormatter(useMathText=True, useOffset=False)
y_formatter.set_powerlimits((-3, 4))
y_formatter.orderOfMagnitude = 4
axs.yaxis.set_major_formatter(y_formatter)

for spine in axs.spines.values():
    spine.set_color('black')
axs.tick_params(axis='both', which='both', bottom=True, top=True,
                left=True, right=True, direction='in', length=4, width=1, colors='black')

# axs.set_ylim(y_values.min()-0.001, y_values.max()*(1+0.05))
# axs.set_xlim(model_CyB.t.min(), model_CyB.t.max())

# Dates as x axis
# dates = generate_dates(2023)
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
if SaveFigures:
    plt.savefig(path + 'Daphnia' + ".pdf", dpi=RESOLUTION, bbox_inches='tight')

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
plt.ylabel("Algae biomass ($mg C/L$)")
plt.title('')
y_formatter = ScalarFormatter(useMathText=True, useOffset=False)
y_formatter.set_powerlimits((-3, 4))
y_formatter.orderOfMagnitude = 4
axs.yaxis.set_major_formatter(y_formatter)

for spine in axs.spines.values():
    spine.set_color('black')
axs.tick_params(axis='both', which='both', bottom=True, top=True,
                left=True, right=True, direction='in', length=4, width=1, colors='black')

# axs.set_ylim(y_values.min()-0.001, y_values.max()*(1+0.05))
# axs.set_xlim(model_CyB.t.min(), model_CyB.t.max())

# Dates as x axis
# dates = generate_dates(2023)
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
if SaveFigures:
    plt.savefig(path + 'Algea' + ".pdf", dpi=RESOLUTION, bbox_inches='tight')

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
plt.ylabel("Yellow perch biomass ($mg C/L$)")
plt.title('')
y_formatter = ScalarFormatter(useMathText=True, useOffset=False)
y_formatter.set_powerlimits((-3, 4))
y_formatter.orderOfMagnitude = 4
axs.yaxis.set_major_formatter(y_formatter)

for spine in axs.spines.values():
    spine.set_color('black')
axs.tick_params(axis='both', which='both', bottom=True, top=True,
                left=True, right=True, direction='in', length=4, width=1, colors='black')

# axs.set_ylim(y_values.min()-0.001, y_values.max()*(1+0.05))
# axs.set_xlim(model_CyB.t.min(), model_CyB.t.max())

# Dates as x axis
# dates = generate_dates(2023)
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
if SaveFigures:
    plt.savefig(path + 'YellowPerch' + ".pdf",
                dpi=RESOLUTION, bbox_inches='tight')

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
plt.ylabel("Walleye biomass ($mg C/L$)")
plt.title('')
y_formatter = ScalarFormatter(useMathText=True, useOffset=False)
y_formatter.set_powerlimits((-3, 4))
y_formatter.orderOfMagnitude = 4
axs.yaxis.set_major_formatter(y_formatter)

for spine in axs.spines.values():
    spine.set_color('black')
axs.tick_params(axis='both', which='both', bottom=True, top=True,
                left=True, right=True, direction='in', length=4, width=1, colors='black')

# axs.set_ylim(y_values.min()-0.001, y_values.max()*(1+0.05))
# axs.set_xlim(model_CyB.t.min(), model_CyB.t.max())

# Dates as x axis
# dates = generate_dates(2023)
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
if SaveFigures:
    plt.savefig(path + 'WallEye' + ".pdf", dpi=RESOLUTION, bbox_inches='tight')

# Save plots
# path = './New data/Images/Year v_1/'
# name = 'Full_model_fit'+FORMAT
# plt.savefig(path+name, dpi=RESOLUTION, bbox_inches='tight')

# plt.show()
