import time
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import pandas as pd
from fullmodel_v1_8 import modelCyB
from Extract_dataV2 import extractData
# Define the ODE system
# Initial conditions
M_0 = 0
A_0 = 0.05
B_0 = 0.004  # 0.004
Q_B_0 = 0.01
Q_A_0 = 0.01
P_0 = 0.06  # 10
D_0 = 0.0239  # 0.0239
Y_0 = 0.025
W_0 = 0.0025
O_0 = 6
v_A_0 = 0
v_D_0 = 0
v_Y_0 = 0
v_W_0 = 0


model_CyB = modelCyB()

initial_conditions = [M_0, B_0, A_0,
                      Q_B_0,
                      Q_A_0, P_0,
                      D_0,
                      Y_0, W_0,
                      v_A_0, v_D_0,
                      v_Y_0, v_W_0, O_0]

model_CyB.initial = initial_conditions

model_CyB.toxines = True

# Define the model function to pass to curve_fit

# Data
data = pd.read_csv(
    "merged_water_quality_data.csv", low_memory=False)

lake_name = "PINE LAKE"

# labels = ['MICROCYSTIN, TOTAL',
#           'PHOSPHORUS TOTAL DISSOLVED',
#           'OXYGEN DISSOLVED (FIELD METER)',
#           'Total cyanobacterial cell count (cells/mL)']

labels = ['MICROCYSTIN, TOTAL',
          'OXYGEN DISSOLVED (FIELD METER)',
          'Total cyanobacterial cell count (cells/mL)']


# years = ['2018', '2019', '2020', '2021', '2022', '2023']
years = ['2017']
yearname = ''
for year in years:
    yearname = yearname + str(year) + '_'

coment = "_v6_3Var_"+yearname
data_fit = extractData(data, years, labels, lake_name)

data = None


# New time scale
day_start = pd.to_datetime("2023-05-01").day_of_year

days_end = pd.to_datetime("2023-09-30").day_of_year

model_CyB.t_0 = 0  # May 1, 2023

model_CyB.t_f = days_end - day_start

model_CyB.delta_t = model_CyB.t_f*3

model_CyB.set_linetime()


# Some parameters

# Death Daphnia
model_CyB.params['n_D'] = 0.06
model_CyB.params['e_BD'] = 0.8
model_CyB.params["tau_B"] = 1.23
model_CyB.params["alpha_B"] = 0.0035
model_CyB.params['r_Y'] = 2
model_CyB.params['r_W'] = 1
model_CyB.params['Ext_Y'] = 0.025*13
model_CyB.params['Ext_W'] = 0.025
model_CyB.params['p_in'] = 0.03

# Get Temperature Function
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

# New time

# y_data = data_fit["Total cyanobacterial cell count (cells/mL)Tra"].values


unknow_params = ["alpha_D", "alpha_Y",
                 "tau_D", "tau_Y",
                 "a_A", "a_D",
                 "sigma_A", "sigma_D",
                 "x_A", "x_D",
                 "n_D"]


def model(parameterTuple):
    model_CyB.initial[1] = parameterTuple[0]
    model_CyB.initial[2] = parameterTuple[1]
    model_CyB.initial[6] = parameterTuple[2]

    i = len(parameterTuple) - len(unknow_params)
    for name in unknow_params:
        model_CyB.params[name] = parameterTuple[i]
        i = i+1

    y_model, info = model_CyB.solver()
    M_values, B_values, A_values, \
        Q_B_values, Q_A_values, P_values, \
        D_values, Y_values, W_values, \
        v_A_values, v_D_values, v_Y_values, \
        v_W_values, O_values = model_CyB.solution

    # M_values = M_values*0.05
    print('INFO:', info['message'])
    if info["message"] == "Integration successful.":
        return M_values, B_values, P_values, O_values
    else:
        shapeData = M_values.shape
        return np.ones(shapeData)*(-1), np.ones(shapeData)*(-1), np.ones(shapeData)*(-1), np.ones(shapeData)*(-1)


# Constrain the parameters to be positive by defining bounds
# Lower bounds are 0, upper bounds are positive infinity

# Minimum
minimum_params = {}
minimum_params["alpha_D"] = 0.001
minimum_params["alpha_Y"] = 0.0001
minimum_params["tau_D"] = 0.5
minimum_params["tau_Y"] = 0.5
minimum_params["a_A"] = 0.001
minimum_params["a_D"] = 0.001
minimum_params["sigma_A"] = 0.001
minimum_params["sigma_D"] = 0.001
minimum_params["x_A"] = 0.001
minimum_params["x_D"] = 0.001
minimum_params["n_D"] = 0.0206
# minimum_params['p_in'] = 0.001
# minimum_params["NormM"] = 0.01

# Maximums
maximum_params = {}
maximum_params["alpha_D"] = 0.01
maximum_params["alpha_Y"] = 0.013
maximum_params["tau_D"] = 2
maximum_params["tau_Y"] = 2
maximum_params["a_A"] = 0.1
maximum_params["a_D"] = 0.1
maximum_params["sigma_A"] = 0.01
maximum_params["sigma_D"] = 0.01
maximum_params["x_A"] = 0.01
maximum_params["x_D"] = 0.01
maximum_params["n_D"] = 0.25


param_bounds = [(0, 0.1),  # B(0)
                (0, 0.1),  # A(0)
                (0, 0.1)]  # D(0)
for name in unknow_params:
    param_bounds.append((minimum_params[name], maximum_params[name]))


# initial_guess = np.array(list(minimum_params.values()))
initial_guess = []
for bounds in param_bounds:
    initial_guess.append(bounds[0]+0.001)


name_data_param = "./FittedParameters/fitting_parameters_full_variables_v2"
all_solutions_errors = []


def sumOfSquaredError(parameterTuple, *args):
    M_values, B_values, P_values, O_values = model(parameterTuple)

    # Local error
    def localerror(label, ModelOutput):
        dayslabel = list(data_fit[label].keys())
        dayslabel.sort()
        ans1 = 0
        samples = 0
        for day in dayslabel:
            dataday = data_fit[label][day]
            dataday = dataday[dataday != -999]
            if len(dataday) != 0 and day < days_end:
                samples += 1
                # print(dataday, label, ModelOutput[int((day - day_start)*3)])
                ans1 += ((dataday -
                          ModelOutput[int((day - day_start)*3)])**2).mean()

        # print(ans1, label, day)
        # Prevent division by zero
        return ans1 / samples if samples != 0 else np.nan

    # Microcystin Error
    if 'MICROCYSTIN, TOTAL' in labels:
        errorM = localerror('MICROCYSTIN, TOTAL', M_values)
    else:
        errorM = np.nan
    # Cyanobacteria Error
    if 'Total cyanobacterial cell count (cells/mL)' in labels:
        errorB = localerror(
            'Total cyanobacterial cell count (cells/mL)', B_values)
    else:
        errorB = np.nan
    # Phosphorus Error
    if 'PHOSPHORUS TOTAL DISSOLVED' in labels:
        errorP = localerror('PHOSPHORUS TOTAL DISSOLVED', P_values)
    else:
        errorP = np.nan
    # Oxygen Error
    if 'OXYGEN DISSOLVED (FIELD METER)' in labels:
        errorO = localerror('OXYGEN DISSOLVED (FIELD METER)', O_values)
    else:
        errorO = np.nan

    Erros = np.array([errorM, errorB, errorP, errorO])

    print(Erros)
    ans = np.mean(Erros[~np.isnan(Erros)])
    global best_solution
    global best_error
    # Check if the current solution is better
    new_error = ans
    # Save the current solution and error to the list
    current_solution = np.append(
        parameterTuple, np.array([ans, errorM, errorB, errorP, errorO]))
    all_solutions_errors.append(current_solution)
    # print(new_error, "new_error")
    if ans < best_error:
        best_solution = parameterTuple
        best_error = ans
        errors = np.array([best_error, errorM, errorB, errorP, errorO])
        # Save the best solution to a DataFrame and CSV file
        df = pd.DataFrame([np.append(best_solution, errors)],
                          index=[lake_name],
                          columns=["B_0", "A_0", "D_0"] + unknow_params + ["MSE", "MSEM", "MSEB", "MSEP", "MSEO"])

        df.to_csv(name_data_param + lake_name + coment + "_backup.csv")
        print("Current Solution:", current_solution)
        print("Best Solution:", best_solution)
        print("Best Error:", best_error)
        print("saved")
    return ans


initial_time = time.time()

strategies = ["best1bin", "best1exp",
              "rand1exp", "randtobest1exp",
              "currenttobest1exp", "best2exp",
              "rand2exp", "randtobest1bin",
              "currenttobest1bin", "best2bin",
              "rand2bin", "rand1bin"]

best_solution = None
best_error = 100


def getParameteres(updating="immediate", workers=-1):
    if updating == "immediate":
        result = differential_evolution(
            sumOfSquaredError,
            bounds=param_bounds,
            disp=True,
            seed=0,
            strategy=strategies[0],
            # x0=initial_guess,
            maxiter=25,
            updating=updating,
            # workers=6,
            # callback=callback_1,
            popsize=15,
            polish=False,
            tol=0.01)
    elif updating == "deferred":
        result = differential_evolution(
            sumOfSquaredError,
            bounds=param_bounds,
            disp=True,
            seed=0,
            strategy=strategies[0],
            # x0=initial_guess,
            maxiter=25,
            updating=updating,
            workers=workers,
            # callback=callback_1,
            popsize=15,
            polish=False,
            tol=0.01)
    return result


# if __name__ == '__main__':
#     result = getParameteres(updating="deferred", workers=5)

result = getParameteres(updating="immediate")
df = pd.DataFrame(all_solutions_errors,
                  index=[lake_name] * len(all_solutions_errors),
                  columns=["B_0", "A_0", "D_0"] + unknow_params + ["MSE", "MSEM", "MSEB", "MSEP", "MSEO"])
df.to_csv(name_data_param + lake_name + coment + "_full_poss.csv")

final_time = time.time()

print("Total time=", (final_time-initial_time)/60)

print("MSE=", result.fun)

#################################

params_fitted = result.x

# Extract the fitted parameters
# print("Fitted Parameters:", params_fitted)

# Save parameters

df = pd.DataFrame(np.array([params_fitted]),
                  index=[lake_name],
                  columns=["B_0", "A_0", "D_0"]
                  + unknow_params)

df.to_csv(name_data_param + lake_name + coment
          + "_final" + ".csv")


#################
