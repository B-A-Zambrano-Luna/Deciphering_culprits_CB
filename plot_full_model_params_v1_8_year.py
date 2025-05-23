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
import os
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
    # print('B(0)=', model_CyB.initial[1])
    model_CyB.initial[2] = params_fit["A_0"][0]*A_scale
    # print('A(0)=', model_CyB.initial[2])
    model_CyB.initial[6] = params_fit["D_0"][0]*D_scale
    # print('D(0)=', model_CyB.initial[6])

    for name in unknow_params:
        model_CyB.params[name] = params_fit[name][0]

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



solution, info = model_CyB.solver()


M_values, B_values, A_values, \
    Q_B_values, Q_A_values, P_values, \
    D_values, Y_values, W_values, \
    v_A_values, v_D_values, v_Y_values, \
    v_W_values, O_values = solution.T


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


print("Model_v1", info["message"])


####### Plots ###############

def extract_value(label):
    return float(label.split()[0])


threhold = 0


####### Color_map##########
map_color = "tab10"
map_color = "inferno"
map_color = 'cool'
color_bar = False
WithTitle = False


# Solution model
fill = False
all_plot = False

somePlots = True

# sns.set_style("dark")
# sns.set_style('darkgrid')
sns.set_style('ticks')
sns.plotting_context("paper", font_scale=1.5)
BASE_COLORS = '#FF4500'  # '#B22222'
LINE_COLOR = '#F0A145'  # '#B7770E'  # '#E1A332'
SAVE_PLOT = True
Print_Values = False
dates = generate_dates(2023)
path = './Figures/Variations/' + lake_name + '/' + yearname
os.makedirs(path, exist_ok=True)
############### Dimensions ######################
FigsizeAll = (11 / 2.54, 11 / 2.54)
# FigsizeSome = (7.5 / 2.54, 10.5 / 2.54)
FigsizeSome = (17.8 / 2.54, 10.5 / 2.54)
FORMAT = '.pdf'
FONTSIZE = 9  # 14 Two Figures  # 7 Several plots
RESOLUTION = 800
NoBins = 25
SpaceDates = 2  # Space between dates to be plotted base on NoBins
Start_day = model_CyB.t_0  # Check dates
End_day = model_CyB.t_f
TIMEStap = (Start_day*3, End_day*3+1)  # Time to plot
DATES = [date[5:] for date in dates][Start_day:End_day+1]
tick_len = 4  # Length for ticks with labels
other_tick_len = 2  # Length for other ticks
hspace = 0.35
wspace = 0.6
FONTSIZETITLE = 7  # 13  # Several Plots
LegendWidth = 2
LINEWIDTH = 0.75

# Which Plot
plot_zmKb = False  # Using
box_psition_zmKb = (1.065, 0.98)
plot_z_m = False  # Using
box_psition_z_m = (1.095, 0.98)
plot_d_E = False  # Using
box_psition_d_E = (1.115, 0.98)
plot_phos = False  # Using
box_psition_phos = (1.115, 0.98)
# Different Temperatures peaks
plot_temp_peak = False  # Using
box_psition_temp_peak = (1.095, 0.98)

############ body burning ######################
# bodyburnig and increace temp in peaks
plot_temp_peak_body = False  # Using
box_psition_body = (1.095, 0.98)

############# Fishes ####################
LINEWIDTHBODY = LINEWIDTH + 0.3 #0.15
FONTSIZEBODY = FONTSIZE + 4
FONTSIZETITLEBODY = FONTSIZETITLE + 6
# bodyburnig and increace temp in peaks
plot_temp_peak_fish = True  # Using
box_psition_fish = (1.095, 0.98)
plot_temp_peak_fish_body = False  # Using
box_psition_fish_body = (1.095, 0.98)


######## Some function to measure ##########


def days_before_after_toxin(M_values, threhold):
    # Find indices where elements are greater than b
    greater_than_b_indices = np.where(M_values > threhold)[0]

    if len(greater_than_b_indices) == 0:
        return None, None  # No elements greater than b

    # First index where element is greater than b
    x_0 = greater_than_b_indices[0]

    # Last index where element is greater than b
    x_1 = greater_than_b_indices[-1]

    return int(x_0/3), int(x_1/3)


def Avarage_max_peaks(B, x_0, x_1, model_CyB):
    if x_0 == None or x_1 == None:
        return 0
    else:
        avar = np.trapz(B[x_0*3:x_1*3],
                        model_CyB.t[x_0*3:x_1*3])\
            / (x_1 - x_0)
        return avar


# My New Stuff
legend_handles = []

if plot_zmKb:
    model_CyB = read_params()
    # Set the y-axis formatter to use scientific notation
    y_formatter = ScalarFormatter(useMathText=True, useOffset=False)
    y_formatter.set_powerlimits((-3, 4))  # Adjust the power limits as needed
    y_formatter.orderOfMagnitude = 4  # Set the exponent to 4

    # zmKb_values = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5]
    zmKb_values = [0.9, 0.85, 0.8, 0.75, 0.7,
                   0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3]

    zmKb_values = np.append(np.round(np.arange(0.3, 0.9+0.05, 0.05), 2), 0.3)
    solution_bg = {}

    if all_plot:
        # Create subplots
        fig, axs = plt.subplots(7, 2, figsize=FigsizeAll)
        axs = axs.ravel()
    elif not all_plot and not somePlots:
        # Create subplots
        fig, axs = plt.subplots(4, 2, figsize=FigsizeAll)
        axs = axs.ravel()
    elif somePlots:
        # Create subplots
        fig, axs = plt.subplots(2, 2, figsize=FigsizeSome)
        axs = axs.ravel()

    colors = cm.GnBu(np.linspace(0.3, 1, len(zmKb_values)))
    max_values = [0, 0, 0, 0]
    min_values = [10, 10, 10, 10]
    for i, b in enumerate(zmKb_values):
        model_CyB.params['z_mK_bg'] = b

        # modelCyB.max_temp = 25.9
        solution, info = model_CyB.solver()
        M_values, B_values, A_values, \
            Q_B_values, Q_A_values, P_values, \
            D_values, Y_values, W_values, \
            v_A_values, v_D_values, v_Y_values, \
            v_W_values, O_values = solution.T

        # Maximum Toxin
        if Print_Values:
            if b == 0.3:
                x_0, x_1 = days_before_after_toxin(M_values, 10)
                avar_0 = Avarage_max_peaks(
                    B_values, x_0, x_1, model_CyB)

                print("Avar Base", avar_0, 'Phos:', b)
                max_M_0 = M_values.max()

                print("Maximum M base: ", max_M_0, 'Phos:', b)

            max_O = O_values[130*3:250*3].max()
            min_O = O_values[130*3:250*3].min()
            max_M = M_values.max()
            print('Oxigen reduction: ', (min_O-max_O)/max_O, 'Phos:', b)
            print("Maximum M Change: ", (max_M-max_M_0)/max_M_0, 'Phos:', b)
            # Print days of bloom toxine
            x_0, x_1 = days_before_after_toxin(M_values, 10)
            avar = Avarage_max_peaks(
                B_values, x_0, x_1, model_CyB)
            if avar_0 != 0:
                print("Avarage CyB:", (avar-avar_0)/avar_0, 'Phos:', b)
            else:
                print("Avarage CyB (base case):", avar_0, 'Phos:', b)
            print('Days bloom Toxine:', x_0, 'to', x_1, 'Phos:', b)

        if all_plot:
            solution_values = [M_values, B_values, A_values, Q_B_values,
                               Q_A_values, P_values, D_values, Y_values,
                               W_values, v_A_values, v_D_values, v_Y_values,
                               v_W_values, O_values]
            sns.set_style("whitegrid")
            variables = ["M $[\mu g/L]$", "B $[mgC/L]$ ",
                         'A $[mgC/L]$', '$Q_B$ $[mgP/mg C]$',
                         '$Q_A$ $[mgP/mg C]$', 'P $[mgP/L]$',
                         'D $[mgC/L]$', 'Y $[mgC/L]$',
                         "W $[mgC/L]$",
                         "$v_A$ $[\mu g /mg C]$", "$v_D$ $[\mu g /mg C]$", "$v_Y [\mu g /mg C]$",
                         "$v_W$ $[\mu g /mg C]$", "O $[mg O_2/L]$"]
        elif not all_plot and not somePlots:
            solution_values = [M_values, B_values, A_values,
                               P_values, D_values, Y_values,
                               W_values, O_values]
            sns.set_style("whitegrid")
            variables = ["M $[\mu g/L]$", "B $[mgC/L]$ ",
                         'A $[mgC/L]$', 'P $[mgP/L]$',
                         'D $[mgC/L]$', 'Y $[mgC/L]$',
                         "W $[mgC/L]$", "O $[mg O_2/L]$"]

        elif somePlots:
            solution_values = [M_values, B_values, A_values,
                               O_values]

            # sns.set_style("whitegrid")
            variables = ["Microcystin-LR ($\mu g/L$)", "Cyanobacterial biomass ($mg C/L$)",
                         'Algal biomass ($mg C/L$)', "Dissolved $O_2$ ($mg O_2/L$)"]
        for j, var in enumerate(solution_values):
            axs[j].yaxis.set_major_formatter(y_formatter)

            axs[j].tick_params(axis='y', labelsize=FONTSIZE,
                               pad=0.5)
            axs[j].tick_params(axis='x', labelsize=FONTSIZE,
                               pad=0.5)
            for spine in axs[j].spines.values():
                spine.set_color('black')  # Set all spines color to black
            axs[j].tick_params(axis='both', which='both', bottom=True, top=False,
                               left=True, right=False, direction='out',
                               length=tick_len, width=0.7, colors='black')

            if variables[j] == "CyB":
                var = B_values
                axs[j].set_ylabel(f"mgC/L {variables[j]}",
                                  fontsize=FONTSIZETITLE)
            else:
                axs[j].set_ylabel(variables[j],
                                  fontsize=FONTSIZETITLE)

            max_values[j] = max(
                max_values[j], var[TIMEStap[0]:TIMEStap[1]].max())
            min_values[j] = min(
                min_values[j], var[TIMEStap[0]:TIMEStap[1]].min())
            # axs[j].set_ylim(min_values[j]*(1-0.05), max_values[j]*(1+0.1))
            # axs[j].set_xlim(model_CyB.t[TIMEStap[0]:TIMEStap[1]].min(),
            #                 model_CyB.t[TIMEStap[0]:TIMEStap[1]].max())
            # axs[j].yaxis.set_major_formatter(y_formatter)

            axs[j].xaxis.set_major_locator(
                MaxNLocator(integer=True, nbins=NoBins))

            x_ticks = axs[j].xaxis.get_ticklocs()

            # x_ticks = np.array([i*60 for i in range(int(365/60))])

            if len(dates) >= len(x_ticks):
                max_tick = x_ticks.max()
                No_days = len(DATES)
                # x_labels_dates = [DATES[int((x_val/max_tick)*364)]
                #             for x_val in x_ticks[:: SpaceDates ]]

                x_labels = [''] * len(x_ticks)

                # x_labels[::SpaceDates] = [DATES[int((x_val/max_tick)*No_days)]
                #                           for x_val in x_ticks[:: SpaceDates]]
                x_labels[::SpaceDates] = [dates[int(x_val)][5:]
                                          for x_val in x_ticks[:: SpaceDates]]
                axs[j].set_xticks(x_ticks)

                axs[j].xaxis.set_ticklabels(x_labels,
                                            rotation=90,
                                            fontsize=FONTSIZE)

                # Adjust the length of ticks with labels
                for tick in axs[j].xaxis.get_major_ticks():
                    if tick.label1.get_text():  # Check if tick has a label
                        tick.tick1line.set_markersize(tick_len)
                        # tick.tick2line.set_markersize(tick_len)
                    else:
                        tick.tick1line.set_markersize(other_tick_len)
                        # tick.tick2line.set_markersize(other_tick_len)

            else:
                x_labels = dates
                axs[j].xaxis.set_ticklabels(x_labels,
                                            rotation=90,
                                            fontsize=FONTSIZE)
            axs[j].set_xlabel('Time (MM-DD)', fontsize=FONTSIZETITLE,
                              labelpad=2)
            axs[j].grid(False)
            color_index = i % len(colors)

            if b != 0.3:
                line, = axs[j].plot(model_CyB.t[TIMEStap[0]:TIMEStap[1]],
                                    var[TIMEStap[0]:TIMEStap[1]],
                                    color=colors[color_index],
                                    label=f"{b:.2f} m", linewidth=LINEWIDTH)  # Collect lines for legend

            elif b == 0.3:
                line, = axs[j].plot(model_CyB.t[TIMEStap[0]:TIMEStap[1]],
                                    var[TIMEStap[0]:TIMEStap[1]],
                                    color=BASE_COLORS,
                                    label=f"{b:.2f} m (base)", linewidth=LINEWIDTH)
            # legend_handles.append(line)  # Append line for legend
    # Add color bar
    if color_bar:
        cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])  # Adjust position here
        sm = plt.cm.ScalarMappable(cmap="cool", norm=plt.Normalize(
            vmin=min(zmKb_values), vmax=max(zmKb_values)))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Algae Growth Factor')

    # axs[0].axhline(y=10, color=LINE_COLOR, linestyle='dotted')
    # axs[-1].set_xlabel('Time (days)')

    handles, labels = axs[0].get_legend_handles_labels()
    # box_psition = (0.98, 1.05)
    legend = fig.legend(handles[:-1], labels, loc='outside upper center',
                        bbox_to_anchor=box_psition_zmKb,
                        fancybox=True, shadow=False, ncol=1,
                        title='Turbidity', fontsize=FONTSIZETITLE)
    legend.get_title().set_ha('center')
    plt.tight_layout()
    if SAVE_PLOT:
        # Save plots
        # path = './New data/Images/Year v_1/'
        name = 'Backgound Light Attenuation' + FORMAT

        plt.savefig(path+name, dpi=RESOLUTION, bbox_inches='tight')
    plt.show()


# Different values of Z_m


if plot_z_m:
    model_CyB = read_params()
    # Set the y-axis formatter to use scientific notation
    y_formatter = ScalarFormatter(useMathText=True, useOffset=False)
    y_formatter.set_powerlimits((-3, 4))  # Adjust the power limits as needed
    y_formatter.orderOfMagnitude = 4  # Set the exponent to 4

    # p_values = [0.01, 0.03, 0.05, 0.07]
    Z_values = list(np.append(np.round(np.arange(0.25, 3, 0.25), 2), 0.0))

    if all_plot:
        # Create subplots
        fig, axs = plt.subplots(7, 2, figsize=FigsizeAll)
        axs = axs.ravel()
    elif not all_plot and not somePlots:
        # Create subplots
        fig, axs = plt.subplots(4, 2, figsize=FigsizeAll)
        axs = axs.ravel()

    elif somePlots:
        # Create subplots
        fig, axs = plt.subplots(2, 2, figsize=FigsizeSome)
        axs = axs.ravel()

    max_values = [0, 0, 0, 0]
    min_values = [10, 10, 10, 10]

    # Base case
    solution, info = model_CyB.solver()
    M_values, B_values, A_values, \
        Q_B_values, Q_A_values, P_values, \
        D_values, Y_values, W_values, \
        v_A_values, v_D_values, v_Y_values, \
        v_W_values, O_values = solution.T
    if Print_Values:
        # Maximum Toxin
        x_0, x_1 = days_before_after_toxin(M_values, 10)
        avar_0 = Avarage_max_peaks(
            B_values, x_0, x_1, model_CyB)
        print("Avar Base", avar_0, 'z_m:', 7)
        max_M_0 = M_values.max()
        print("Maximum M base: ", max_M_0, 'z_m:', 7)

    for z_m in Z_values:

        model_CyB.auxZm = z_m

        # Different temperature

        # modelCyB.max_temp = 25.9
        solution, info = model_CyB.solver()
        M_values, B_values, A_values, \
            Q_B_values, Q_A_values, P_values, \
            D_values, Y_values, W_values, \
            v_A_values, v_D_values, v_Y_values, \
            v_W_values, O_values = solution.T

        # Maximum Toxin
        # if z_m == 7:
        #     x_0, x_1 = days_before_after_toxin(M_values, 10)
        #     avar_0 = Avarage_max_peaks(
        #         B_values, x_0, x_1, model_CyB)
        #     print("Avar Base", avar_0, 'z_m:', z_m)
        #     max_M_0 = M_values.max()

        #     print("Maximum M base: ", max_M_0, 'z_m:', z_m)
        if Print_Values:
            max_O = O_values[130*3:250*3].max()
            min_O = O_values[130*3:250*3].min()
            max_M = M_values.max()
            print('Oxigen reduction: ', (min_O-max_O)/max_O, 'z_m:', z_m)
            print("Maximum M Change: ", (max_M-max_M_0)/max_M_0, 'z_m:', z_m)
            # Print days of bloom toxine
            x_0, x_1 = days_before_after_toxin(M_values, 10)
            avar = Avarage_max_peaks(
                B_values, x_0, x_1, model_CyB)
            if avar_0 != 0:
                print("Avarage CyB:", (avar-avar_0)/avar_0, 'z_m:', z_m)
            else:
                print("Avarage CyB (base case):", avar_0, 'z_m:', z_m)
            print('Days bloom Toxine:', x_0, 'to', x_1, 'z_m:', z_m)

        colors = cm.GnBu(np.linspace(0.3, 1, len(Z_values)))
        if all_plot:
            solution_values = [M_values, B_values, A_values, Q_B_values,
                               Q_A_values, P_values, D_values, Y_values,
                               W_values, v_A_values, v_D_values, v_Y_values,
                               v_W_values, O_values]
            sns.set_style("whitegrid")
            # colors = sns.color_palette(map_color, n_colors=len(p_values))

            variables = ["M $[\mu g/L]$", "B $[mgC/L]$ ",
                         'A $[mgC/L]$', '$Q_B$ $[mgP/mg C]$',
                         '$Q_A$ $[mgP/mg C]$', 'P $[mgP/L]$',
                         'D $[mgC/L]$', 'Y $[mgC/L]$',
                         "W $[mgC/L]$",
                         "$v_A$ $[\mu g /mg C]$", "$v_D$ $[\mu g /mg C]$", "$v_Y [\mu g /mg C]$",
                         "$v_W$ $[\mu g /mg C]$", "O $[mg O_2/L]$"]
        elif not all_plot and not somePlots:
            solution_values = [M_values, B_values, A_values,
                               P_values, D_values, Y_values,
                               W_values, O_values]
            # sns.set_style("whitegrid")
            # colors = sns.color_palette(map_color, n_colors=len(p_values) * 2)

            variables = ["M $[\mu g/L]$", "B $[mgC/L]$ ",
                         'A $[mgC/L]$', 'P $[mgP/L]$',
                         'D $[mgC/L]$', 'Y $[mgC/L]$',
                         "W $[mgC/L]$", "O $[mg O_2/L]$"]

        elif somePlots:
            solution_values = [M_values, B_values, A_values,
                               O_values]
            # sns.set_style("whitegrid")
            # colors = sns.color_palette(map_color, n_colors=len(p_values) * 2)

            # variables = ["M $[\mu g/L]$", "B $[mgC/L]$ ",
            #              'A $[mgC/L]$', "O $[mg O_2/L]$"]
            variables = ["Microcystin-LR ($\mu g/L$)", "Cyanobacterial biomass ($mgC/L$)",
                         'Algal biomass ($mg C/L$)', "Dissolved $O_2$ ($mg O_2/L$)"]
        color_index = Z_values.index(z_m) % len(colors)
        for j, var in enumerate(solution_values):

            axs[j].yaxis.set_major_formatter(y_formatter)

            axs[j].tick_params(axis='y', labelsize=FONTSIZE,
                               pad=0.5)
            axs[j].tick_params(axis='x', labelsize=FONTSIZE,
                               pad=0.5)
            for spine in axs[j].spines.values():
                spine.set_color('black')  # Set all spines color to black
            axs[j].tick_params(axis='both', which='both', bottom=True, top=False,
                               left=True, right=False, direction='out',
                               length=tick_len, width=0.7, colors='black')

            if variables[j] == "CyB":
                var = B_values
                axs[j].set_ylabel(f"mgC/L {variables[j]}",
                                  fontsize=FONTSIZETITLE-1)
            else:
                axs[j].set_ylabel(variables[j],
                                  fontsize=FONTSIZETITLE)

            max_values[j] = max(
                max_values[j], var[TIMEStap[0]:TIMEStap[1]].max())
            min_values[j] = min(
                min_values[j], var[TIMEStap[0]:TIMEStap[1]].min())
            # axs[j].set_ylim(min_values[j]*(1-0.05), max_values[j]*(1+0.1))
            # axs[j].set_xlim(model_CyB.t[TIMEStap[0]:TIMEStap[1]].min(),
            #                 model_CyB.t[TIMEStap[0]:TIMEStap[1]].max())
            # axs[j].yaxis.set_major_formatter(y_formatter)

            axs[j].xaxis.set_major_locator(
                MaxNLocator(integer=True, nbins=NoBins))

            x_ticks = axs[j].xaxis.get_ticklocs()

            # x_ticks = np.array([i*60 for i in range(int(365/60))])

            if len(dates) >= len(x_ticks):
                max_tick = x_ticks.max()
                No_days = len(DATES)
                # x_labels_dates = [DATES[int((x_val/max_tick)*364)]
                #             for x_val in x_ticks[:: SpaceDates ]]

                x_labels = [''] * len(x_ticks)

                # x_labels[::SpaceDates] = [DATES[int((x_val/max_tick)*No_days)]
                #                           for x_val in x_ticks[:: SpaceDates]]
                x_labels[::SpaceDates] = [dates[int(x_val)][5:]
                                          for x_val in x_ticks[:: SpaceDates]]
                axs[j].set_xticks(x_ticks)

                axs[j].xaxis.set_ticklabels(x_labels,
                                            rotation=90,
                                            fontsize=FONTSIZE)

                # Adjust the length of ticks with labels
                for tick in axs[j].xaxis.get_major_ticks():
                    if tick.label1.get_text():  # Check if tick has a label
                        tick.tick1line.set_markersize(tick_len)
                        # tick.tick2line.set_markersize(tick_len)
                    else:
                        tick.tick1line.set_markersize(other_tick_len)
                        # tick.tick2line.set_markersize(other_tick_len)

            else:
                x_labels = dates
                axs[j].xaxis.set_ticklabels(x_labels,
                                            rotation=90,
                                            fontsize=FONTSIZE)
            axs[j].set_xlabel('Time (MM-DD)', fontsize=FONTSIZETITLE,
                              labelpad=2)
            axs[j].grid(False)
            # color_index = j % len(colors)

            if z_m != 0:
                line, = axs[j].plot(model_CyB.t, var, color=colors[color_index],
                                    label=f"+{z_m:.2f} m", linewidth=LINEWIDTH)

            elif z_m == 0:
                line, = axs[j].plot(model_CyB.t, var, color=BASE_COLORS,
                                    label=f"+{z_m:.2f} m (base)", linewidth=LINEWIDTH)

            if fill:
                axs[j].fill_between(model_CyB.t, var, var.min(),
                                    color=colors[color_index], alpha=0.05)

    # axs[0].axhline(y=10, color=LINE_COLOR, linestyle='dotted')

    if WithTitle:
        plt.suptitle(lake_name + ' for Different phosphorous values')
    handles, labels = axs[0].get_legend_handles_labels()
    # Create a sorted order based on the numeric values in the labels
    sorted_indices = sorted(
        range(len(labels)), key=lambda i: extract_value(labels[i]))
    legend = fig.legend([handles[i] for i in sorted_indices],
                        [labels[i] for i in sorted_indices],
                        loc='outside upper center',
                        bbox_to_anchor=box_psition_z_m,
                        fancybox=True, shadow=False, ncol=1,
                        title='Epilimnion depth\n comparison with\n the base depth')
    legend.get_title().set_ha('center')
    # plt.setp(legend.get_title(),  multialignment='left', rotation=90)

    # Adjust layout for better readability
    plt.tight_layout()
    if SAVE_PLOT:
        # Save plots
        # path = './New data/Images/Year v_1/'
        name = 'Full_model_z_m'+FORMAT
        plt.savefig(path+name, dpi=RESOLUTION, bbox_inches='tight')
    # Show the plot
    plt.show()


# Different values of d_E


if plot_d_E:
    model_CyB = read_params()
    # Set the y-axis formatter to use scientific notation
    y_formatter = ScalarFormatter(useMathText=True, useOffset=False)
    y_formatter.set_powerlimits((-3, 4))  # Adjust the power limits as needed
    y_formatter.orderOfMagnitude = 4  # Set the exponent to 4

    # p_values = [0.01, 0.03, 0.05, 0.07]
    d_values = list(
        np.around(np.arange(0.02+0.01, 0.08+0.05, 0.01), 2))
    d_values.append(0.02)
    if all_plot:
        # Create subplots
        fig, axs = plt.subplots(7, 2, figsize=FigsizeAll)
        axs = axs.ravel()
    elif not all_plot and not somePlots:
        # Create subplots
        fig, axs = plt.subplots(4, 2, figsize=FigsizeAll)
        axs = axs.ravel()

    elif somePlots:
        # Create subplots
        fig, axs = plt.subplots(2, 2, figsize=FigsizeSome)
        axs = axs.ravel()

    max_values = [0, 0, 0, 0]
    min_values = [10, 10, 10, 10]

    # Base case
    solution, info = model_CyB.solver()
    M_values, B_values, A_values, \
        Q_B_values, Q_A_values, P_values, \
        D_values, Y_values, W_values, \
        v_A_values, v_D_values, v_Y_values, \
        v_W_values, O_values = solution.T
    if Print_Values:
        # Maximum Toxin
        x_0, x_1 = days_before_after_toxin(M_values, 10)
        avar_0 = Avarage_max_peaks(
            B_values, x_0, x_1, model_CyB)
        print("Avar Base", avar_0, 'd_E:', 0.02)
        max_M_0 = M_values.max()
        print("Maximum M base: ", max_M_0, 'd_E:', 0.02)

    for d_E in d_values:

        model_CyB.params['d_E'] = d_E

        # Different temperature

        # modelCyB.max_temp = 25.9
        solution, info = model_CyB.solver()
        M_values, B_values, A_values, \
            Q_B_values, Q_A_values, P_values, \
            D_values, Y_values, W_values, \
            v_A_values, v_D_values, v_Y_values, \
            v_W_values, O_values = solution.T

        # Maximum Toxin
        # if z_m == 7:
        #     x_0, x_1 = days_before_after_toxin(M_values, 10)
        #     avar_0 = Avarage_max_peaks(
        #         B_values, x_0, x_1, model_CyB)
        #     print("Avar Base", avar_0, 'z_m:', z_m)
        #     max_M_0 = M_values.max()

        #     print("Maximum M base: ", max_M_0, 'z_m:', z_m)
        if Print_Values:
            max_O = O_values[130*3:250*3].max()
            min_O = O_values[130*3:250*3].min()
            max_M = M_values.max()
            print('Oxigen reduction: ', (min_O-max_O)/max_O, 'd_E:', d_E)
            print("Maximum M Change: ", (max_M-max_M_0)/max_M_0, 'd_E:', d_E)
            # Print days of bloom toxine
            x_0, x_1 = days_before_after_toxin(M_values, 10)
            avar = Avarage_max_peaks(
                B_values, x_0, x_1, model_CyB)
            if avar_0 != 0:
                print("Avarage CyB:", (avar-avar_0)/avar_0, 'd_E:', d_E)
            else:
                print("Avarage CyB (base case):", avar_0, 'd_E:', d_E)
            print('Days bloom Toxine:', x_0, 'to', x_1, 'd_E:', d_E)

        colors = cm.GnBu(np.linspace(0.3, 1, len(d_values)))
        if all_plot:
            solution_values = [M_values, B_values, A_values, Q_B_values,
                               Q_A_values, P_values, D_values, Y_values,
                               W_values, v_A_values, v_D_values, v_Y_values,
                               v_W_values, O_values]
            sns.set_style("whitegrid")
            # colors = sns.color_palette(map_color, n_colors=len(p_values))

            variables = ["M $[\mu g/L]$", "B $[mgC/L]$ ",
                         'A $[mgC/L]$', '$Q_B$ $[mgP/mg C]$',
                         '$Q_A$ $[mgP/mg C]$', 'P $[mgP/L]$',
                         'D $[mgC/L]$', 'Y $[mgC/L]$',
                         "W $[mgC/L]$",
                         "$v_A$ $[\mu g /mg C]$", "$v_D$ $[\mu g /mg C]$", "$v_Y [\mu g /mg C]$",
                         "$v_W$ $[\mu g /mg C]$", "O $[mg O_2/L]$"]
        elif not all_plot and not somePlots:
            solution_values = [M_values, B_values, A_values,
                               P_values, D_values, Y_values,
                               W_values, O_values]
            # sns.set_style("whitegrid")
            # colors = sns.color_palette(map_color, n_colors=len(p_values) * 2)

            variables = ["M $[\mu g/L]$", "B $[mgC/L]$ ",
                         'A $[mgC/L]$', 'P $[mgP/L]$',
                         'D $[mgC/L]$', 'Y $[mgC/L]$',
                         "W $[mgC/L]$", "O $[mg O_2/L]$"]

        elif somePlots:
            solution_values = [M_values, B_values, A_values,
                               O_values]
            sns.set_style("whitegrid")
            # colors = sns.color_palette(map_color, n_colors=len(p_values) * 2)

            # variables = ["M $[\mu g/L]$", "B $[mgC/L]$ ",
            #              'A $[mgC/L]$', "O $[mg O_2/L]$"]
            variables = ["Microcystin-LR ($\mu g/L$)", "Cyanobacterial biomass ($mg C/L$)",
                         'Algal biomass ($mg C/L$)', "Dissolved $O_2$ ($mg O_2/L$)"]
        color_index = d_values.index(d_E) % len(colors)
        for j, var in enumerate(solution_values):

            axs[j].yaxis.set_major_formatter(y_formatter)

            axs[j].tick_params(axis='y', labelsize=FONTSIZE,
                               pad=0.5)
            axs[j].tick_params(axis='x', labelsize=FONTSIZE,
                               pad=0.5)
            for spine in axs[j].spines.values():
                spine.set_color('black')  # Set all spines color to black
            axs[j].tick_params(axis='both', which='both', bottom=True, top=False,
                               left=True, right=False, direction='out',
                               length=tick_len, width=0.7, colors='black')

            if variables[j] == "CyB":
                var = B_values
                axs[j].set_ylabel(f"mgC/L {variables[j]}",
                                  fontsize=FONTSIZETITLE-1)
            else:
                axs[j].set_ylabel(variables[j],
                                  fontsize=FONTSIZETITLE)

            max_values[j] = max(
                max_values[j], var[TIMEStap[0]:TIMEStap[1]].max())
            min_values[j] = min(
                min_values[j], var[TIMEStap[0]:TIMEStap[1]].min())
            # axs[j].set_ylim(min_values[j]*(1-0.05), max_values[j]*(1+0.1))
            # axs[j].set_xlim(model_CyB.t[TIMEStap[0]:TIMEStap[1]].min(),
            #                 model_CyB.t[TIMEStap[0]:TIMEStap[1]].max())
            # axs[j].yaxis.set_major_formatter(y_formatter)

            axs[j].xaxis.set_major_locator(
                MaxNLocator(integer=True, nbins=NoBins))

            x_ticks = axs[j].xaxis.get_ticklocs()

            # x_ticks = np.array([i*60 for i in range(int(365/60))])

            if len(dates) >= len(x_ticks):
                max_tick = x_ticks.max()
                No_days = len(DATES)
                # x_labels_dates = [DATES[int((x_val/max_tick)*364)]
                #             for x_val in x_ticks[:: SpaceDates ]]

                x_labels = [''] * len(x_ticks)

                # x_labels[::SpaceDates] = [DATES[int((x_val/max_tick)*No_days)]
                #                           for x_val in x_ticks[:: SpaceDates]]
                x_labels[::SpaceDates] = [dates[int(x_val)][5:]
                                          for x_val in x_ticks[:: SpaceDates]]
                axs[j].set_xticks(x_ticks)

                axs[j].xaxis.set_ticklabels(x_labels,
                                            rotation=90,
                                            fontsize=FONTSIZE)

                # Adjust the length of ticks with labels
                for tick in axs[j].xaxis.get_major_ticks():
                    if tick.label1.get_text():  # Check if tick has a label
                        tick.tick1line.set_markersize(tick_len)
                        # tick.tick2line.set_markersize(tick_len)
                    else:
                        tick.tick1line.set_markersize(other_tick_len)
                        # tick.tick2line.set_markersize(other_tick_len)

            else:
                x_labels = dates
                axs[j].xaxis.set_ticklabels(x_labels,
                                            rotation=90,
                                            fontsize=FONTSIZE)
            axs[j].set_xlabel('Time (MM-DD)', fontsize=FONTSIZETITLE,
                              labelpad=2)
            axs[j].grid(False)

            if d_E != 0.02:
                line, = axs[j].plot(model_CyB.t, var, color=colors[color_index],
                                    label=f"{d_E:.2f} m/day", linewidth=LINEWIDTH)

            elif d_E == 0.02:
                line, = axs[j].plot(model_CyB.t, var, color=BASE_COLORS,
                                    label=f"{d_E:.2f} m/day (base)", linewidth=LINEWIDTH)

            if fill:
                axs[i].fill_between(model_CyB.t, var, var.min(),
                                    color=colors[color_index], alpha=0.05)

    # axs[0].axhline(y=10, color=LINE_COLOR, linestyle='dotted')
    # Set common x-axis label and title
    # axs[-1].set_xlabel('Time (days)')
    if WithTitle:
        plt.suptitle(lake_name + ' for Different phosphorous values')
    handles, labels = axs[0].get_legend_handles_labels()
    # Create a sorted order based on the numeric values in the labels
    sorted_indices = sorted(
        range(len(labels)), key=lambda i: extract_value(labels[i]))
    legend = fig.legend([handles[i] for i in sorted_indices],
                        [labels[i] for i in sorted_indices],
                        loc='outside upper center',
                        bbox_to_anchor=box_psition_d_E,
                        fancybox=True, shadow=False, ncol=1,
                        title='Water exchange rate')
    legend.get_title().set_ha('center')
    # plt.setp(legend.get_title(),  multialignment='left', rotation=90)

    # Adjust layout for better readability
    plt.tight_layout()
    if SAVE_PLOT:
        # Save plots
        # path = './New data/Images/Year v_1/'
        name = 'Full_model_d_E'+FORMAT
        plt.savefig(path+name, dpi=RESOLUTION, bbox_inches='tight')
    # Show the plot
    plt.show()


# Different values of phosphorus


if plot_phos:
    model_CyB = read_params()
    # Set the y-axis formatter to use scientific notation
    y_formatter = ScalarFormatter(useMathText=True, useOffset=False)
    y_formatter.set_powerlimits((-3, 4))  # Adjust the power limits as needed
    y_formatter.orderOfMagnitude = 4  # Set the exponent to 4

    # p_values = [0.01, 0.03, 0.05, 0.07]
    # p_values = [0.01, 0.02, 0.03, 0.04, 0.06, 0.07, 0.1, 0.2,
    #             0.3, 0.4, P_0]
    # Original list
    p_values = [0.01, 0.02, 0.03, 0.04, 0.06, 0.07, 0.1, 0.2, 0.3, 0.4, P_0]

    # Remove duplicates while preserving order, except for P_0
    p_values0 = [item for item in p_values if item != P_0]

    # Add P_0 at the end
    p_values0.append(P_0)
    p_values = p_values0
    # p_values = [0.1, 0.2, 0.3, 0.4]
    # p_values = [0.5, 0.6, 0.7, 0.8]

    solution_pho = {}

    # variables = ["M", "B", 'A', 'Q_B', 'Q_A', 'P', 'D',
    #              'Y', "W", "v_A", "v_D", "v_Y", "v_W", "O"]

    if all_plot:
        # Create subplots
        fig, axs = plt.subplots(7, 2, figsize=FigsizeAll)
        axs = axs.ravel()
    elif not all_plot and not somePlots:
        # Create subplots
        fig, axs = plt.subplots(4, 2, figsize=FigsizeAll)
        axs = axs.ravel()

    elif somePlots:
        # Create subplots
        fig, axs = plt.subplots(2, 2, figsize=FigsizeSome)
        axs = axs.ravel()

    max_values = [0, 0, 0, 0]
    min_values = [10, 10, 10, 10]

    # Maximum Toxin
    if Print_Values:
        x_0, x_1 = days_before_after_toxin(M_values, 10)
        avar_0 = Avarage_max_peaks(
            B_values, x_0, x_1, model_CyB)
        print("Avar Base", avar_0, 'Phos:', 0.7)
        max_M_0 = M_values.max()

        print("Maximum M base: ", max_M_0, 'Phos:', 0.7)

    for p in p_values:
        model_CyB.initial[5] = p

        # Different temperature

        # modelCyB.max_temp = 25.9
        solution, info = model_CyB.solver()
        M_values, B_values, A_values, \
            Q_B_values, Q_A_values, P_values, \
            D_values, Y_values, W_values, \
            v_A_values, v_D_values, v_Y_values, \
            v_W_values, O_values = solution.T

        # max_O = O_values[130*3:250*3].max()
        # min_O = O_values[130*3:250*3].min()
        # max_M = M_values.max()
        # print('Oxigen reduction: ', (min_O-max_O)/max_O, 'Phos:', p)
        # print("Maximum M Change: ", (max_M-max_M_0)/max_M_0, 'Phos:', p)
        # Print days of bloom toxine
        if Print_Values:
            x_0, x_1 = days_before_after_toxin(M_values, 10)
            avar = Avarage_max_peaks(
                B_values, x_0, x_1, model_CyB)
            if avar_0 != 0:
                print("Avarage CyB:", (avar-avar_0)/avar_0, 'Phos:', p)
            else:
                print("Avarage CyB (base case):", avar_0, 'Phos:', p)
            print('Days bloom Toxine:', x_0, 'to', x_1, 'Phos:', p)

        if all_plot:
            solution_values = [M_values, B_values, A_values, Q_B_values,
                               Q_A_values, P_values, D_values, Y_values,
                               W_values, v_A_values, v_D_values, v_Y_values,
                               v_W_values, O_values]
            # sns.set_style("whitegrid")
            # colors = sns.color_palette(map_color, n_colors=len(p_values))
            colors = cm.GnBu(np.linspace(0.3, 1, len(p_values)))
            variables = ["M $[\mu g/L]$", "B $[mgC/L]$ ",
                         'A $[mgC/L]$', '$Q_B$ $[mgP/mg C]$',
                         '$Q_A$ $[mgP/mg C]$', 'P $[mgP/L]$',
                         'D $[mgC/L]$', 'Y $[mgC/L]$',
                         "W $[mgC/L]$",
                         "$v_A$ $[\mu g /mg C]$", "$v_D$ $[\mu g /mg C]$", "$v_Y [\mu g /mg C]$",
                         "$v_W$ $[\mu g /mg C]$", "O $[mg O_2/L]$"]
        elif not all_plot and not somePlots:
            solution_values = [M_values, B_values, A_values,
                               P_values, D_values, Y_values,
                               W_values, O_values]
            # sns.set_style("whitegrid")
            # colors = sns.color_palette(map_color, n_colors=len(p_values) * 2)
            colors = cm.GnBu(np.linspace(0.3, 1, len(p_values)))
            variables = ["M $[\mu g/L]$", "B $[mgC/L]$ ",
                         'A $[mgC/L]$', 'P $[mgP/L]$',
                         'D $[mgC/L]$', 'Y $[mgC/L]$',
                         "W $[mgC/L]$", "O $[mg O_2/L]$"]

        elif somePlots:
            solution_values = [M_values, B_values, A_values,
                               O_values]
            # sns.set_style("whitegrid")
            # colors = sns.color_palette(map_color, n_colors=len(p_values) * 2)
            colors = cm.GnBu(np.linspace(0.3, 1, len(p_values)))
            # variables = ["M $[\mu g/L]$", "B $[mgC/L]$ ",
            #              'A $[mgC/L]$', "O $[mg O_2/L]$"]
            variables = ["Microcystin-LR ($\mu g/L$)", "Cyanobacterial biomass ($mgC/L$)",
                         'Algal biomass ($mgC/L$)', "Dissolved $O_2$ ($ mg O_2/L$)"]
        color_index = p_values.index(p) % len(colors)
        for j, var in enumerate(solution_values):

            axs[j].yaxis.set_major_formatter(y_formatter)

            axs[j].tick_params(axis='y', labelsize=FONTSIZE,
                               pad=0.5)
            axs[j].tick_params(axis='x', labelsize=FONTSIZE,
                               pad=0.5)
            for spine in axs[j].spines.values():
                spine.set_color('black')  # Set all spines color to black
            axs[j].tick_params(axis='both', which='both', bottom=True, top=False,
                               left=True, right=False, direction='out',
                               length=tick_len, width=0.7, colors='black')

            if variables[j] == "CyB":
                var = B_values
                axs[j].set_ylabel(f"mgC/L {variables[j]}",
                                  fontsize=FONTSIZETITLE-1)
            else:
                axs[j].set_ylabel(variables[j],
                                  fontsize=FONTSIZETITLE)

            max_values[j] = max(
                max_values[j], var[TIMEStap[0]:TIMEStap[1]].max())
            min_values[j] = min(
                min_values[j], var[TIMEStap[0]:TIMEStap[1]].min())
            # axs[j].set_ylim(min_values[j]*(1-0.05), max_values[j]*(1+0.1))
            # axs[j].set_xlim(model_CyB.t[TIMEStap[0]:TIMEStap[1]].min(),
            #                 model_CyB.t[TIMEStap[0]:TIMEStap[1]].max())
            # axs[j].yaxis.set_major_formatter(y_formatter)

            axs[j].xaxis.set_major_locator(
                MaxNLocator(integer=True, nbins=NoBins))

            x_ticks = axs[j].xaxis.get_ticklocs()

            # x_ticks = np.array([i*60 for i in range(int(365/60))])

            if len(dates) >= len(x_ticks):
                max_tick = x_ticks.max()
                No_days = len(DATES)
                # x_labels_dates = [DATES[int((x_val/max_tick)*364)]
                #             for x_val in x_ticks[:: SpaceDates ]]

                x_labels = [''] * len(x_ticks)

                # x_labels[::SpaceDates] = [DATES[int((x_val/max_tick)*No_days)]
                #                           for x_val in x_ticks[:: SpaceDates]]
                x_labels[::SpaceDates] = [dates[int(x_val)][5:]
                                          for x_val in x_ticks[:: SpaceDates]]
                axs[j].set_xticks(x_ticks)

                axs[j].xaxis.set_ticklabels(x_labels,
                                            rotation=90,
                                            fontsize=FONTSIZE)

                # Adjust the length of ticks with labels
                for tick in axs[j].xaxis.get_major_ticks():
                    if tick.label1.get_text():  # Check if tick has a label
                        tick.tick1line.set_markersize(tick_len)
                        # tick.tick2line.set_markersize(tick_len)
                    else:
                        tick.tick1line.set_markersize(other_tick_len)
                        # tick.tick2line.set_markersize(other_tick_len)

            else:
                x_labels = dates
                axs[j].xaxis.set_ticklabels(x_labels,
                                            rotation=90,
                                            fontsize=FONTSIZE)
            axs[j].set_xlabel('Time (MM-DD)', fontsize=FONTSIZETITLE,
                              labelpad=2)
            axs[j].grid(False)

            if p != P_0:
                line, = axs[j].plot(model_CyB.t, var, color=colors[color_index],
                                    label=f"{p:.2f} mgP/L", linewidth=LINEWIDTH)  # Collect lines for legend

            elif p == P_0:
                line, = axs[j].plot(model_CyB.t, var, color=BASE_COLORS,
                                    label=f"{p:.2f} mgP/L (base)", linewidth=LINEWIDTH)

            if fill:
                axs[j].fill_between(model_CyB.t, var, var.min(),
                                    color=colors[color_index], alpha=0.05)

    # axs[0].axhline(y=10, color=LINE_COLOR, linestyle='dotted')
    # Set common x-axis label and title
    # axs[-1].set_xlabel('Time (days)')
    if WithTitle:
        plt.suptitle(lake_name + ' for Different phosphorous values')
    handles, labels = axs[0].get_legend_handles_labels()

    sorted_indices = sorted(
        range(len(labels)), key=lambda i: extract_value(labels[i]))
    legend = fig.legend([handles[i] for i in sorted_indices],
                        [labels[i] for i in sorted_indices],
                        loc='outside upper center',
                        bbox_to_anchor=box_psition_phos,
                        fancybox=True, shadow=False, ncol=1,
                        title='Initial phosphorus')
    legend.get_title().set_ha('center')
    # plt.setp(legend.get_title(),  multialignment='left', rotation=90)

    # Adjust layout for better readability
    plt.tight_layout()
    if SAVE_PLOT:
        # Save plots
        # path = './New data/Images/Year v_1/'
        name = 'Full_model_phos_levels'+FORMAT
        plt.savefig(path+name, dpi=RESOLUTION, bbox_inches='tight')

    # Show the plot
    plt.show()


# Different Temperatures peaks

if plot_temp_peak:
    # Set the y-axis formatter to use scientific notation
    y_formatter = ScalarFormatter(useMathText=True, useOffset=False)
    y_formatter.set_powerlimits((-3, 4))  # Adjust the power limits as needed
    y_formatter.orderOfMagnitude = 4  # Set the exponent to 4

    # Temp_increase = [0, 0.4, 0.6, 0.7, 0.8, 1.0, 1.1, 1.4,
    #                  1.7, 2.6, 3.4]

    Temp_increase = np.append(np.round(np.arange(0.3, 3.4+0.3, 0.3), 1), 0)
    Temp_increase = list(Temp_increase)
    # colors = sns.color_palette(map_color, n_colors=len(Temp_increase))
    colors = cm.GnBu(np.linspace(0.3, 1, len(Temp_increase)))
    solution_Temp = {}

    if all_plot:
        # Create subplots
        fig, axs = plt.subplots(7, 2, figsize=FigsizeAll)
        axs = axs.ravel()
    elif not all_plot and not somePlots:
        # Create subplots
        fig, axs = plt.subplots(4, 2, figsize=FigsizeAll)
        axs = axs.ravel()
    elif somePlots:
        # Create subplots
        fig, axs = plt.subplots(2, 2, figsize=FigsizeSome)
        axs = axs.ravel()
    # Change function of temperature
    max_values = [0, 0, 0, 0]
    min_values = [10, 10, 10, 10]
    for Temp in Temp_increase:
        model_CyB = read_params()

        # Different temperature
        model_CyB.auxTemp = Temp
        # modelCyB.max_temp = 25.9
        solution, info = model_CyB.solver()
        M_values, B_values, A_values, \
            Q_B_values, Q_A_values, P_values, \
            D_values, Y_values, W_values, \
            v_A_values, v_D_values, v_Y_values, \
            v_W_values, O_values = solution.T
        if Print_Values:
            # Maximum Toxin
            if Temp == 0:
                x_0, x_1 = days_before_after_toxin(M_values, 10)
                avar_0 = Avarage_max_peaks(
                    B_values, x_0, x_1, model_CyB)
                print("Avar Base", avar_0, 'Temp:', Temp)
                max_M_0 = M_values.max()

                print("Maximum M base: ", max_M_0, 'Temp:', Temp)

            max_O = O_values[130*3:250*3].max()
            min_O = O_values[130*3:250*3].min()
            max_M = M_values.max()
            print('Oxigen reduction: ', (min_O-max_O)/max_O, 'Temp:', Temp)
            print("Maximum M Change: ", (max_M-max_M_0)/max_M_0, 'Temp:', Temp)
            # Print days of bloom toxine
            x_0, x_1 = days_before_after_toxin(M_values, 10)
            avar = Avarage_max_peaks(
                B_values, x_0, x_1, model_CyB)
            print("Avarage CyB:", (avar-avar_0)/avar_0, 'Temp:', Temp)
            print('Days bloom Toxine:', x_0, 'to', x_1, 'Temp:', Temp)

        if all_plot:
            solution_values = [M_values, B_values, A_values, Q_B_values,
                               Q_A_values, P_values, D_values, Y_values,
                               W_values, v_A_values, v_D_values, v_Y_values,
                               v_W_values, O_values]
            sns.set_style("whitegrid")
            # colors = sns.color_palette(map_color, n_colors=len(p_values) * 2)
            # colors = cm.GnBu(np.linspace(0.3, 1, len(Temp_increase)))
            variables = ["M $[\mu g/L]$", "B $[mgC/L]$ ",
                         'A $[mgC/L]$', '$Q_B$ $[mgP/mg C]$',
                         '$Q_A$ $[mgP/mg C]$', 'P $[mgP/L]$',
                         'D $[mgC/L]$', 'Y $[mgC/L]$',
                         "W $[mgC/L]$",
                         "$v_A$ $[\mu g /mg C]$", "$v_D$ $[\mu g /mg C]$", "$v_Y [\mu g /mg C]$",
                         "$v_W$ $[\mu g /mg C]$", "O $[mg O_2/L]$"]
        elif not all_plot and not somePlots:
            solution_values = [M_values, B_values, A_values,
                               P_values, D_values, Y_values,
                               W_values, O_values]
            # sns.set_style("whitegrid")
            # colors = sns.color_palette(
            #     map_color, n_colors=len(Temp_increase)*2)

            variables = ["M $[\mu g/L]$", "B $[mgC/L]$ ",
                         'A $[mgC/L]$', 'P $[mgP/L]$',
                         'D $[mgC/L]$', 'Y $[mgC/L]$',
                         "W $[mgC/L]$", "O $[mg O_2/L]$"]
        elif somePlots:
            solution_values = [M_values, B_values, A_values,
                               O_values]
            variables = ["Microcystin-LR $(\mu g/L$)", "Cyanobacterial biomass ($mg C/L$)",
                         'Algal biomass  ($mg C/L$)', "Dissolved $O_2$  ($ mg O_2/L$)"]
        for i, var in enumerate(solution_values):
            # var = solution.T[variables.index(variables[i])]

            axs[i].yaxis.set_major_formatter(y_formatter)

            axs[i].tick_params(axis='y', labelsize=FONTSIZE,
                               pad=0.5)
            axs[i].tick_params(axis='x', labelsize=FONTSIZE,
                               pad=0.5)

            for spine in axs[i].spines.values():
                spine.set_color('black')  # Set all spines color to black

            axs[i].tick_params(axis='both', which='both', bottom=True, top=False,
                               left=True, right=False, direction='out',
                               length=tick_len, width=0.7, colors='black')

            axs[i].set_ylabel(variables[i], fontsize=FONTSIZETITLE,
                              labelpad=0)

            max_values[i] = max(
                max_values[i], var[TIMEStap[0]:TIMEStap[1]].max())
            min_values[i] = min(
                min_values[i], var[TIMEStap[0]:TIMEStap[1]].min())

            # axs[i].set_ylim(min_values[i]*(1-0.05), max_values[i] * (1+0.1))
            # axs[i].set_xlim(model_CyB.t[TIMEStap[0]:TIMEStap[1]].min(),
            #                 model_CyB.t[TIMEStap[0]:TIMEStap[1]].max())

            axs[i].xaxis.set_major_locator(
                MaxNLocator(integer=True, nbins=NoBins))

            x_ticks = axs[i].xaxis.get_ticklocs()

            if len(dates) >= len(x_ticks):
                max_tick = x_ticks.max()
                No_days = len(DATES)
                # x_labels_dates = [DATES[int((x_val/max_tick)*364)]
                #             for x_val in x_ticks[:: SpaceDates ]]

                x_labels = [''] * len(x_ticks)

                # x_labels[::SpaceDates] = [DATES[int((x_val/max_tick)*No_days)]
                #                           for x_val in x_ticks[:: SpaceDates]]
                x_labels[::SpaceDates] = [dates[int(x_val)][5:]
                                          for x_val in x_ticks[:: SpaceDates]]
                axs[i].set_xticks(x_ticks)

                axs[i].xaxis.set_ticklabels(x_labels,
                                            rotation=90,
                                            fontsize=FONTSIZE)

                # Adjust the length of ticks with labels
                for tick in axs[i].xaxis.get_major_ticks():
                    if tick.label1.get_text():  # Check if tick has a label
                        tick.tick1line.set_markersize(tick_len)
                        # tick.tick2line.set_markersize(tick_len)
                    else:
                        tick.tick1line.set_markersize(other_tick_len)
                        # tick.tick2line.set_markersize(other_tick_len)
            else:
                x_labels = dates
                axs[i].xaxis.set_ticklabels(x_labels,
                                            rotation=30)
            axs[i].set_xlabel('Time (MM-DD)', fontsize=FONTSIZETITLE,
                              labelpad=2)
            axs[i].grid(False)
            color_index = Temp_increase.index(Temp) % len(colors)

            if Temp != 0:
                line, = axs[i].plot(model_CyB.t, var, color=colors[color_index],
                                    label=f"+{Temp}", linewidth=LINEWIDTH)
            elif Temp == 0:
                line, = axs[i].plot(model_CyB.t, var, color=BASE_COLORS,
                                    label=f"+{Temp} (base)", linewidth=LINEWIDTH)

    # Set common x-axis label and title
    # axs[0].axhline(y=10, color=LINE_COLOR, linestyle='dotted')
    # axs[-1].set_xlabel('Time (days)')
    if WithTitle:
        plt.suptitle(lake_name + ' increase of maximum peaks')
    handles, labels = axs[0].get_legend_handles_labels()

    sorted_indices = sorted(
        range(len(labels)), key=lambda i: extract_value(labels[i]))
    legend = fig.legend([handles[i] for i in sorted_indices],
                        [labels[i] for i in sorted_indices],
                        loc='outside upper center',
                        bbox_to_anchor=box_psition_temp_peak,
                        fancybox=True, shadow=False, ncol=1,
                        title='Increase\n in $\degree C$')
    legend.get_title().set_ha('center')
    # Adjust the spacing between subplots
    # plt.subplots_adjust(hspace=hspace, wspace=wspace)
    plt.tight_layout()
    if SAVE_PLOT:
        # Save plots
        # path = './New data/Images/Year v_1/'
        name = 'Full_model_temp_peaks_short_long_term_v2'+FORMAT
        plt.savefig(path+name, dpi=RESOLUTION, bbox_inches='tight')
        print('Image was printed')
    # Show the plot
    plt.show()


################## Body burning ################################


# Different Temperatures peaks bodyborning

if plot_temp_peak_body:
    # Set the y-axis formatter to use scientific notation
    y_formatter = ScalarFormatter(useMathText=True, useOffset=False)
    y_formatter.set_powerlimits((-3, 4))  # Adjust the power limits as needed
    y_formatter.orderOfMagnitude = 4  # Set the exponent to 4

    Temp_increase = np.round(np.arange(0.3, 3.4+0.3, 0.3), 1)
    Temp_increase = list(Temp_increase)
    Temp_increase.append(0)
    # colors = sns.color_palette(map_color, n_colors=len(Temp_increase))
    colors = cm.GnBu(np.linspace(0.3, 1, len(Temp_increase)))
    solution_Temp = {}

    ############## joind figure ################

    fig, axs = plt.subplots(2, 4, figsize=FigsizeSome)

    ######### no joind Figure ##################
    # fig, axs = plt.subplots(2, 2, figsize=FigsizeSome)

    axs = axs.ravel()
    # Change function of temperature

    ########## no joind figures ###########
    max_values = [0, 0, 0, 0]
    min_values = [10, 10, 10, 10]

    ####### joind figures ##############

    max_values = [0, 0, 0, 0, 0, 0, 0, 0]
    min_values = [10, 10, 10, 10, 10, 10, 10, 10]

    for Temp in Temp_increase:
        model_CyB = read_params()
        # if Temp != 0:
        #     # Data Temperature
        #     days_temp = np.array(
        #         [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
        #     temp_water = np.array(
        #         [7.35, 7.3, 7.91, 9.4, 11.67, 13.59, 15.49, 16.02, 14.08, 11.29, 9.26, 8.07])

        #     delta_temp = Temp

        #     for i in range(temp_water.shape[0]):
        #         if temp_water[i] > 11:
        #             temp_water[i] = temp_water[i] + delta_temp

        #     # Define the logistic function
        #     def temperature(t, K, T, t_0, k_0):
        #         return K*np.exp(-T*(t-t_0)**2)+k_0

        #     # Set bounds for positive values
        #     bounds = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])

        #     # Fit the logistic function to the data
        #     initial_guess = [15, 0, 200, 7]
        #     params, covariance = curve_fit(temperature,
        #                                    days_temp,
        #                                    temp_water,
        #                                    p0=initial_guess,
        #                                    bounds=bounds)

        #     model_CyB.max_temp = round(params[0], 5)
        #     model_CyB.min_temp = round(params[3], 5)
        #     model_CyB.max_temp_time = round(params[2]-20, 5)
        #     model_CyB.T = round(params[1], 5)
        model_CyB.auxTemp = Temp
        # Different temperature

        # modelCyB.max_temp = 25.9
        solution, info = model_CyB.solver()
        M_values, B_values, A_values, \
            Q_B_values, Q_A_values, P_values, \
            D_values, Y_values, W_values, \
            v_A_values, v_D_values, v_Y_values, \
            v_W_values, O_values = solution.T

        # # Maximum Toxin
        # if Temp == 0:
        #     x_0, x_1 = days_before_after_toxin(M_values, 10)
        #     avar_0 = Avarage_max_peaks(
        #         B_values, x_0, x_1, model_CyB)
        #     print("Avar Base", avar_0, 'Temp:', Temp)
        #     max_M_0 = M_values.max()

        #     print("Maximum M base: ", max_M_0, 'Temp:', Temp)

        # max_O = O_values[130*3:250*3].max()
        # min_O = O_values[130*3:250*3].min()
        # max_M = M_values.max()
        # print('Oxigen reduction: ', (min_O-max_O)/max_O, 'Temp:', Temp)
        # print("Maximum M Change: ", (max_M-max_M_0)/max_M_0, 'Temp:', Temp)
        # # Print days of bloom toxine
        # x_0, x_1 = days_before_after_toxin(M_values, 10)
        # avar = Avarage_max_peaks(
        #     B_values, x_0, x_1, model_CyB)
        # print("Avarage CyB:", (avar-avar_0)/avar_0, 'Temp:', Temp)
        # print('Days bloom Toxine:', x_0, 'to', x_1, 'Temp:', Temp)

        ############## Joind Figure #########################

        solution_values = [M_values, B_values,
                           A_values, O_values,
                           Y_values, v_Y_values,
                           W_values, v_W_values]

        variables = ["Microcystin-LR $(\mu g/L$)",
                     "Cyanobacterial biomass ($mg C/L$)",
                     'Algal biomass  ($mg C/L$)',
                     "Dissolved $O_2$  ($ mg O_2/L$)",
                     'Yellow perch biomass ($mg C/L$)',
                     'Body burden of yellow perch',
                     'Walleye biomass ($mg C/L$)',
                     'Body burden of walleye']

        ################ no joind Figure ###################

        # solution_values = [Y_values, W_values, v_Y_values,
        #                    v_W_values]

        # variables = ['Yellow perch biomass ($mg C/L$)',
        #              'Walleye biomass ($mg C/L$)',
        #              'Body burden of yellow perch',
        #              'Body burden of walleye']

        for i, var in enumerate(solution_values):
            axs[i].yaxis.set_major_formatter(y_formatter)

            if i == 5:
                axs[i].tick_params(axis='y', labelsize=FONTSIZE,
                                   pad=0.3)
            else:
                axs[i].tick_params(axis='y', labelsize=FONTSIZE,
                                   pad=0.5)

            axs[i].tick_params(axis='x', labelsize=FONTSIZE,
                               pad=0.5)

            for spine in axs[i].spines.values():
                spine.set_color('black')  # Set all spines color to black

            axs[i].tick_params(axis='both', which='both', bottom=True, top=False,
                               left=True, right=False, direction='out',
                               length=tick_len, width=0.7, colors='black')

            axs[i].set_ylabel(variables[i], fontsize=FONTSIZETITLE,
                              labelpad=0)

            max_values[i] = max(
                max_values[i], var[TIMEStap[0]:TIMEStap[1]].max())
            min_values[i] = min(
                min_values[i], var[TIMEStap[0]:TIMEStap[1]].min())

            # axs[i].set_ylim(min_values[i]*(1-0.05), max_values[i] * (1+0.08))
            # axs[i].set_xlim(model_CyB.t[TIMEStap[0]:TIMEStap[1]].min(),
            #                 model_CyB.t[TIMEStap[0]:TIMEStap[1]].max())

            axs[i].xaxis.set_major_locator(
                MaxNLocator(integer=True, nbins=NoBins))

            x_ticks = axs[i].xaxis.get_ticklocs()

            if len(dates) >= len(x_ticks):
                max_tick = x_ticks.max()
                No_days = len(DATES)
                # x_labels_dates = [DATES[int((x_val/max_tick)*364)]
                #             for x_val in x_ticks[:: SpaceDates ]]

                x_labels = [''] * len(x_ticks)

                # x_labels[::SpaceDates] = [DATES[int((x_val/max_tick)*No_days)]
                #                           for x_val in x_ticks[:: SpaceDates]]
                x_labels[::SpaceDates] = [dates[int(x_val)][5:]
                                          for x_val in x_ticks[:: SpaceDates]]
                axs[i].set_xticks(x_ticks)

                axs[i].xaxis.set_ticklabels(x_labels,
                                            rotation=90,
                                            fontsize=FONTSIZE)

                # Adjust the length of ticks with labels
                for tick in axs[i].xaxis.get_major_ticks():
                    if tick.label1.get_text():  # Check if tick has a label
                        tick.tick1line.set_markersize(tick_len)
                        # tick.tick2line.set_markersize(tick_len)
                    else:
                        tick.tick1line.set_markersize(other_tick_len)
                        # tick.tick2line.set_markersize(other_tick_len)
            else:
                x_labels = dates
                axs[i].xaxis.set_ticklabels(x_labels,
                                            rotation=30)
            axs[i].set_xlabel('Time (MM-DD)', fontsize=FONTSIZETITLE,
                              labelpad=2)
            axs[i].grid(False)
            color_index = Temp_increase.index(Temp) % len(colors)

            if Temp != 0:
                line, = axs[i].plot(model_CyB.t, var, color=colors[color_index],
                                    label=f"+{Temp:.2f}", linewidth=LINEWIDTH)
            elif Temp == 0:
                line, = axs[i].plot(model_CyB.t, var, color=BASE_COLORS,
                                    label=f"+{Temp:.2f} (base)", linewidth=LINEWIDTH)

    # Set common x-axis label and title
    # axs[0].axhline(y=10, color=LINE_COLOR, linestyle='dotted')
    # axs[-1].set_xlabel('Time (days)')
    if WithTitle:
        plt.suptitle(lake_name + ' increase of maximum peaks')
    handles, labels = axs[0].get_legend_handles_labels()

    sorted_indices = sorted(
        range(len(labels)), key=lambda i: extract_value(labels[i]))
    legend = fig.legend([handles[i] for i in sorted_indices],
                        [labels[i] for i in sorted_indices],
                        loc='outside upper center',
                        bbox_to_anchor=box_psition_body,
                        fancybox=True, shadow=False, ncol=1,
                        title='Increase\n in $\degree C$')

    legend.get_title().set_ha('center')
    # Adjust the spacing between subplots
    # plt.subplots_adjust(hspace=hspace, wspace=wspace)
    plt.tight_layout()
    if SAVE_PLOT:
        # Save plots
        # path = './New data/Images/Year v_1/'
        name = 'Full_model_temp_peaks_short_long_term_body'+FORMAT
        plt.savefig(path+name, dpi=RESOLUTION, bbox_inches='tight')
        print('Image Save as', name + FORMAT)
    # Show the plot
    plt.show()



##################### Fishes ########################

# Different Temperatures peaks bodyborning


if plot_temp_peak_fish:
    # Set the y-axis formatter to use scientific notation
    y_formatter = ScalarFormatter(useMathText=True, useOffset=False)
    y_formatter.set_powerlimits((-3, 4))  # Adjust the power limits as needed
    y_formatter.orderOfMagnitude = 4  # Set the exponent to 4

    # Temp_increase = [0, 0.1, 0.2, 0.3, 0.4]
    # Temp_increase = [0, 0.6, 0.7, 1.0, 1.1, 1.4]
    # Temp_increase = [0, 0.4, 0.8, 1.7, 2.6, 3.4]

    # Temp_increase = [0, 0.1, 0.2, 0.3,
    # 0.4, 0.6, 0.7, 0.8]

    Temp_increase = np.append(np.round(np.arange(0.3, 3.4+0.3, 0.3), 1), 0)
    Temp_increase = list(Temp_increase)

    # colors = sns.color_palette(map_color, n_colors=len(Temp_increase))
    colors = cm.GnBu(np.linspace(0.3, 1, len(Temp_increase)))
    solution_Temp = {}

    fig, axs = plt.subplots(1, 2, figsize=FigsizeSome)

    axs = axs.ravel()
    # Change function of temperature
    max_values = [0, 0, 0, 0]
    min_values = [10, 10, 10, 10]
    for Temp in Temp_increase:
        model_CyB = read_params()
        # if Temp != 0:
        #     # Data Temperature
        #     days_temp = np.array(
        #         [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
        #     temp_water = np.array(
        #         [7.35, 7.3, 7.91, 9.4, 11.67, 13.59, 15.49, 16.02, 14.08, 11.29, 9.26, 8.07])

        #     delta_temp = Temp

        #     for i in range(temp_water.shape[0]):
        #         if temp_water[i] > 11:
        #             temp_water[i] = temp_water[i] + delta_temp

        #     # Define the logistic function
        #     def temperature(t, K, T, t_0, k_0):
        #         return K*np.exp(-T*(t-t_0)**2)+k_0

        #     # Set bounds for positive values
        #     bounds = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])

        #     # Fit the logistic function to the data
        #     initial_guess = [15, 0, 200, 7]
        #     params, covariance = curve_fit(temperature,
        #                                    days_temp,
        #                                    temp_water,
        #                                    p0=initial_guess,
        #                                    bounds=bounds)

        #     model_CyB.max_temp = round(params[0], 5)
        #     model_CyB.min_temp = round(params[3], 5)
        #     model_CyB.max_temp_time = round(params[2]-20, 5)
        #     model_CyB.T = round(params[1], 5)
        model_CyB.auxTemp = Temp
        # Different temperature

        # modelCyB.max_temp = 25.9
        solution, info = model_CyB.solver()
        M_values, B_values, A_values, \
            Q_B_values, Q_A_values, P_values, \
            D_values, Y_values, W_values, \
            v_A_values, v_D_values, v_Y_values, \
            v_W_values, O_values = solution.T

        solution_values = [Y_values,
                           W_values]

        variables = ['Yellow perch ($mg C/L$)',
                     'Walleye ($mg C/L$)']
        color_index = Temp_increase.index(Temp) % len(colors)
        for j, var in enumerate(solution_values):

            axs[j].yaxis.set_major_formatter(y_formatter)

            axs[j].tick_params(axis='y', labelsize=FONTSIZEBODY,
                               pad=0.5)
            axs[j].tick_params(axis='x', labelsize=FONTSIZEBODY,
                               pad=0.5)
            for spine in axs[j].spines.values():
                spine.set_color('black')  # Set all spines color to black
            axs[j].tick_params(axis='both', which='both', bottom=True, top=False,
                               left=True, right=False, direction='out',
                               length=tick_len, width=0.7, colors='black')

            if variables[j] == "CyB":
                var = B_values
                axs[j].set_ylabel(f"mgC/L {variables[j]}",
                                  fontsize=FONTSIZETITLEBODY-1)
            else:
                axs[j].set_ylabel(variables[j],
                                  fontsize=FONTSIZETITLEBODY)

            max_values[j] = max(
                max_values[j], var[TIMEStap[0]:TIMEStap[1]].max())
            min_values[j] = min(
                min_values[j], var[TIMEStap[0]:TIMEStap[1]].min())
            # axs[j].set_ylim(min_values[j]*(1-0.05), max_values[j]*(1+0.1))
            # axs[j].set_xlim(model_CyB.t[TIMEStap[0]:TIMEStap[1]].min(),
            #                 model_CyB.t[TIMEStap[0]:TIMEStap[1]].max())
            # axs[j].yaxis.set_major_formatter(y_formatter)

            axs[j].xaxis.set_major_locator(
                MaxNLocator(integer=True, nbins=NoBins))

            x_ticks = axs[j].xaxis.get_ticklocs()

            # x_ticks = np.array([i*60 for i in range(int(365/60))])

            if len(dates) >= len(x_ticks):
                max_tick = x_ticks.max()
                No_days = len(DATES)
                # x_labels_dates = [DATES[int((x_val/max_tick)*364)]
                #             for x_val in x_ticks[:: SpaceDates ]]

                x_labels = [''] * len(x_ticks)

                # x_labels[::SpaceDates] = [DATES[int((x_val/max_tick)*No_days)]
                #                           for x_val in x_ticks[:: SpaceDates]]
                x_labels[::SpaceDates] = [dates[int(x_val)][5:]
                                          for x_val in x_ticks[:: SpaceDates]]
                axs[j].set_xticks(x_ticks)

                axs[j].xaxis.set_ticklabels(x_labels,
                                            rotation=90,
                                            fontsize=FONTSIZEBODY)

                # Adjust the length of ticks with labels
                for tick in axs[j].xaxis.get_major_ticks():
                    if tick.label1.get_text():  # Check if tick has a label
                        tick.tick1line.set_markersize(tick_len)
                        # tick.tick2line.set_markersize(tick_len)
                    else:
                        tick.tick1line.set_markersize(other_tick_len)
                        # tick.tick2line.set_markersize(other_tick_len)

            else:
                x_labels = dates
                axs[j].xaxis.set_ticklabels(x_labels,
                                            rotation=90,
                                            fontsize=FONTSIZEBODY)
            axs[j].set_xlabel('Time (MM-DD)', fontsize=FONTSIZETITLEBODY,
                              labelpad=2)
            axs[j].grid(False)

            if Temp != 0:
                line, = axs[j].plot(model_CyB.t, var, color=colors[color_index],
                                    label=f"+{Temp:.2f}", linewidth=LINEWIDTHBODY)

            elif Temp == 0:
                line, = axs[j].plot(model_CyB.t, var, color=BASE_COLORS,
                                    label=f"+{Temp:.2f} (base)", linewidth=LINEWIDTHBODY)

            if fill:
                axs[j].fill_between(model_CyB.t, var, var.min(),
                                    color=colors[color_index], alpha=0.05)

    # Set common x-axis label and title
    # axs[-1].set_xlabel('Time (days)')
    if WithTitle:
        plt.suptitle(lake_name + ' increase of maximum peaks')
    handles, labels = axs[0].get_legend_handles_labels()
    if color_bar:
        # Create a ScalarMappable object
        sm = plt.cm.ScalarMappable(cmap=map_color, norm=plt.Normalize(
            vmin=min(Temp_increase), vmax=max(Temp_increase)))

        # Add color bar to the figure
        plt.colorbar(sm, ax=axs, cmap=colors)
    else:
        handles, labels = axs[0].get_legend_handles_labels()
        # box_psition = (0.96, 0.88)
        sorted_indices = sorted(
            range(len(labels)), key=lambda i: extract_value(labels[i]))
        legend = fig.legend([handles[i] for i in sorted_indices],
                            [labels[i] for i in sorted_indices],
                            loc='outside upper center',
                            bbox_to_anchor=box_psition_fish,
                            fancybox=True, shadow=False, ncol=1,
                            title='Increase\n in $\degree C$')

        legend.get_title().set_ha('center')
    plt.tight_layout()
    if SAVE_PLOT:
        # Save plots
        # path = './New data/Images/Year v_1/'
        name = 'Full_model_temp_peaks_short_long_term_fish'+FORMAT
        plt.savefig(path+name, dpi=RESOLUTION, bbox_inches='tight')
    # Show the plot
    plt.show()


if plot_temp_peak_fish_body:
    # Set the y-axis formatter to use scientific notation
    y_formatter = ScalarFormatter(useMathText=True, useOffset=False)
    y_formatter.set_powerlimits((-3, 4))  # Adjust the power limits as needed
    y_formatter.orderOfMagnitude = 4  # Set the exponent to 4

    # Temp_increase = [0, 0.1, 0.2, 0.3, 0.4]
    # Temp_increase = [0, 0.6, 0.7, 1.0, 1.1, 1.4]
    # Temp_increase = [0, 0.4, 0.8, 1.7, 2.6, 3.4]

    # Temp_increase = [0, 0.1, 0.2, 0.3,
    # 0.4, 0.6, 0.7, 0.8]

    Temp_increase = np.append(np.round(np.arange(0.3, 3.4+0.3, 0.3), 1), 0)
    Temp_increase = list(Temp_increase)

    # colors = sns.color_palette(map_color, n_colors=len(Temp_increase))
    colors = cm.GnBu(np.linspace(0.3, 1, len(Temp_increase)))
    solution_Temp = {}

    fig, axs = plt.subplots(1, 2, figsize=FigsizeSome)

    axs = axs.ravel()
    # Change function of temperature
    max_values = [0, 0, 0, 0]
    min_values = [10, 10, 10, 10]
    for Temp in Temp_increase:
        model_CyB = read_params()
        # if Temp != 0:
        #     # Data Temperature
        #     days_temp = np.array(
        #         [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
        #     temp_water = np.array(
        #         [7.35, 7.3, 7.91, 9.4, 11.67, 13.59, 15.49, 16.02, 14.08, 11.29, 9.26, 8.07])

        #     delta_temp = Temp

        #     for i in range(temp_water.shape[0]):
        #         if temp_water[i] > 11:
        #             temp_water[i] = temp_water[i] + delta_temp

        #     # Define the logistic function
        #     def temperature(t, K, T, t_0, k_0):
        #         return K*np.exp(-T*(t-t_0)**2)+k_0

        #     # Set bounds for positive values
        #     bounds = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])

        #     # Fit the logistic function to the data
        #     initial_guess = [15, 0, 200, 7]
        #     params, covariance = curve_fit(temperature,
        #                                    days_temp,
        #                                    temp_water,
        #                                    p0=initial_guess,
        #                                    bounds=bounds)

        #     model_CyB.max_temp = round(params[0], 5)
        #     model_CyB.min_temp = round(params[3], 5)
        #     model_CyB.max_temp_time = round(params[2]-20, 5)
        #     model_CyB.T = round(params[1], 5)
        model_CyB.auxTemp = Temp
        # Different temperature

        # modelCyB.max_temp = 25.9
        solution, info = model_CyB.solver()
        M_values, B_values, A_values, \
            Q_B_values, Q_A_values, P_values, \
            D_values, Y_values, W_values, \
            v_A_values, v_D_values, v_Y_values, \
            v_W_values, O_values = solution.T

        solution_values = [v_Y_values,
                           v_W_values]

        variables = ['Body burden of yellow perch',
                     'Body burden of walleye']
        color_index = Temp_increase.index(Temp) % len(colors)
        for j, var in enumerate(solution_values):

            axs[j].yaxis.set_major_formatter(y_formatter)

            axs[j].tick_params(axis='y', labelsize=FONTSIZEBODY,
                               pad=0.5)
            axs[j].tick_params(axis='x', labelsize=FONTSIZEBODY,
                               pad=0.5)
            for spine in axs[j].spines.values():
                spine.set_color('black')  # Set all spines color to black
            axs[j].tick_params(axis='both', which='both', bottom=True, top=False,
                               left=True, right=False, direction='out',
                               length=tick_len, width=0.7, colors='black')

            if variables[j] == "CyB":
                var = B_values
                axs[j].set_ylabel(f"mgC/L {variables[j]}",
                                  fontsize=FONTSIZETITLEBODY-1)
            else:
                axs[j].set_ylabel(variables[j],
                                  fontsize=FONTSIZETITLEBODY)

            max_values[j] = max(
                max_values[j], var[TIMEStap[0]:TIMEStap[1]].max())
            min_values[j] = min(
                min_values[j], var[TIMEStap[0]:TIMEStap[1]].min())
            # axs[j].set_ylim(min_values[j]*(1-0.05), max_values[j]*(1+0.1))
            # axs[j].set_xlim(model_CyB.t[TIMEStap[0]:TIMEStap[1]].min(),
            #                 model_CyB.t[TIMEStap[0]:TIMEStap[1]].max())
            # axs[j].yaxis.set_major_formatter(y_formatter)

            axs[j].xaxis.set_major_locator(
                MaxNLocator(integer=True, nbins=NoBins))

            x_ticks = axs[j].xaxis.get_ticklocs()

            # x_ticks = np.array([i*60 for i in range(int(365/60))])

            if len(dates) >= len(x_ticks):
                max_tick = x_ticks.max()
                No_days = len(DATES)
                # x_labels_dates = [DATES[int((x_val/max_tick)*364)]
                #             for x_val in x_ticks[:: SpaceDates ]]

                x_labels = [''] * len(x_ticks)

                # x_labels[::SpaceDates] = [DATES[int((x_val/max_tick)*No_days)]
                #                           for x_val in x_ticks[:: SpaceDates]]
                x_labels[::SpaceDates] = [dates[int(x_val)][5:]
                                          for x_val in x_ticks[:: SpaceDates]]
                axs[j].set_xticks(x_ticks)

                axs[j].xaxis.set_ticklabels(x_labels,
                                            rotation=90,
                                            fontsize=FONTSIZEBODY)

                # Adjust the length of ticks with labels
                for tick in axs[j].xaxis.get_major_ticks():
                    if tick.label1.get_text():  # Check if tick has a label
                        tick.tick1line.set_markersize(tick_len)
                        # tick.tick2line.set_markersize(tick_len)
                    else:
                        tick.tick1line.set_markersize(other_tick_len)
                        # tick.tick2line.set_markersize(other_tick_len)

            else:
                x_labels = dates
                axs[j].xaxis.set_ticklabels(x_labels,
                                            rotation=90,
                                            fontsize=FONTSIZEBODY)
            axs[j].set_xlabel('Time (MM-DD)', fontsize=FONTSIZETITLEBODY,
                              labelpad=2)
            axs[j].grid(False)

            if Temp != 0:
                line, = axs[j].plot(model_CyB.t, var, color=colors[color_index],
                                    label=f"+{Temp:.2f}", linewidth=LINEWIDTHBODY)

            elif Temp == 0:
                line, = axs[j].plot(model_CyB.t, var, color=BASE_COLORS,
                                    label=f"+{Temp:.2f} (base)", linewidth=LINEWIDTHBODY)

            if fill:
                axs[j].fill_between(model_CyB.t, var, var.min(),
                                    color=colors[color_index], alpha=0.05)

    # Set common x-axis label and title
    # axs[-1].set_xlabel('Time (days)')
    if WithTitle:
        plt.suptitle(lake_name + ' increase of maximum peaks')
    handles, labels = axs[0].get_legend_handles_labels()
    if color_bar:
        # Create a ScalarMappable object
        sm = plt.cm.ScalarMappable(cmap=map_color, norm=plt.Normalize(
            vmin=min(Temp_increase), vmax=max(Temp_increase)))

        # Add color bar to the figure
        plt.colorbar(sm, ax=axs, cmap=colors)
    else:
        handles, labels = axs[0].get_legend_handles_labels()
        # box_psition = (0.96, 0.88)
        sorted_indices = sorted(
            range(len(labels)), key=lambda i: extract_value(labels[i]))
        legend = fig.legend([handles[i] for i in sorted_indices],
                            [labels[i] for i in sorted_indices],
                            loc='outside upper center',
                            bbox_to_anchor=box_psition_fish_body,
                            fancybox=True, shadow=False, ncol=1,
                            title='Increase\n in $\degree C$')

        legend.get_title().set_ha('center')
    plt.tight_layout()
    if SAVE_PLOT:
        # Save plots
        # path = './New data/Images/Year v_1/'
        name = 'Full_model_temp_peaks_fish_body_burden'+FORMAT
        plt.savefig(path+name, dpi=RESOLUTION, bbox_inches='tight')
    # Show the plot
    plt.show()
