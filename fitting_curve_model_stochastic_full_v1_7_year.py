# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 20:17:52 2023

@author: LENOVO
"""

import time
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from fullmodel_v1_7 import modelCyB
import pandas as pd
from Extract_data import extractData
# Define the ODE system
# Initial conditions
M_0 = 0
A_0 = 0.5
B_0 = 0.004  # 0.004
Q_B_0 = 0.01
Q_A_0 = 0.01
P_0 = 0.7  # 10
D_0 = 0.0239  # 0.0239
Y_0 = 0.025
W_0 = 0.0025
O_0 = 7
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
data = pd.read_excel("all-bloom-indicators.xlsx")
year = "2016"
lake_name = "Pigeon Lake"
coment = "_v1_7_"

data = data[data["Waterbody name"] == lake_name]

data_fit = extractData(data, year)

# Calculate the 75th percentile
percentile_threshold = data_fit["Total cyanobacterial cell count (cells/mL)Tra"].quantile(
    0.75)

# Filter values below the 75th percentile
# data_fit = data_fit[data_fit["Total cyanobacterial cell count (cells/mL)Tra"]
#                     < 5]


t_data = data_fit["days"].values

# New time scale
days_before = 15

days_after = 30

model_CyB.t_0 = 0

model_CyB.t_f = t_data.max()-t_data.min() + days_after

model_CyB.t_f = 365

model_CyB.delta_t = model_CyB.t_f*3

model_CyB.set_linetime()


# Some parameters

# Death Daphnia
model_CyB.params['n_D'] = 0.06
model_CyB.params['e_BD'] = 0.8
model_CyB.params["tau_B"] = 1.23
model_CyB.params["alpha_B"] = 0.0035
# model_CyB.params["tau_Y"] = 1.23
# # model_CyB.params["tau_D"] = 0.01
# model_CyB.params["alpha_D"] = 0.003941
# model_CyB.params['phi_D'] = 700
# model_CyB.params['phi_Y'] = 700

model_CyB.params['r_Y'] = 8*0+2
model_CyB.params['r_W'] = 4*0+1
model_CyB.params['Ext_Y'] = 0.025*13
model_CyB.params['Ext_W'] = 0.025
# model_CyB.params['p_in'] = 0.15*2

# Temperature

#model_CyB.max_temp_time = 200.31
#model_CyB.max_temp = 25.9
#model_CyB.freq_temp = -0.0172
# model_CyB.init_temp = t_data.min()

# New time
#t_data = t_data-t_data.min() + days_before

y_data = data_fit["Total cyanobacterial cell count (cells/mL)Tra"].values


unknow_params = ["e_BD", "alpha_B", "alpha_D",
                 "alpha_Y",
                 "tau_B", "tau_D", "tau_Y",
                 "a_A", "a_D", "sigma_A",
                 "sigma_D", "x_A", "x_D",
                 "n_D"]


def model(t, parameterTuple):
    model_CyB.initial[1] = parameterTuple[0]
    model_CyB.initial[2] = parameterTuple[1]
    model_CyB.initial[6] = parameterTuple[2]

    i = len(parameterTuple) - len(unknow_params)
    for name in unknow_params:
        model_CyB.params[name] = parameterTuple[i]
        i = i+1

    y_model, info = model_CyB.solver()
    M_values, B_values, A_values,\
        Q_B_values, Q_A_values, P_values, \
        D_values, Y_values, W_values,\
        v_A_values, v_D_values, v_Y_values, \
        v_W_values, O_values = model_CyB.solution
    print(info['message'])
    if info["message"] == "Integration successful.":
        t = np.array(t, int)
        return B_values[t*3]
    else:
        return np.ones(t.shape)*(-1)


# Constrain the parameters to be positive by defining bounds
# Lower bounds are 0, upper bounds are positive infinity

# Minimum
minimum_params = {}
minimum_params["e_BD"] = 0.65
minimum_params["alpha_B"] = 0.001
minimum_params["alpha_D"] = 0.001
minimum_params["alpha_Y"] = 0.0001
minimum_params["tau_B"] = 0.123
minimum_params["tau_D"] = 0.5
minimum_params["tau_Y"] = 0.5
minimum_params["a_A"] = 0.001
minimum_params["a_D"] = 0.001
minimum_params["sigma_A"] = 0.001
minimum_params["sigma_D"] = 0.001
minimum_params["x_A"] = 0.001
minimum_params["x_D"] = 0.001
minimum_params["n_D"] = 0.001

# Maximums
maximum_params = {}
maximum_params["e_BD"] = 0.8
maximum_params["alpha_B"] = 0.01
maximum_params["alpha_D"] = 0.01
maximum_params["alpha_Y"] = 0.013
maximum_params["tau_B"] = 2
maximum_params["tau_D"] = 2
maximum_params["tau_Y"] = 2
maximum_params["a_A"] = 0.1
maximum_params["a_D"] = 0.1
maximum_params["sigma_A"] = 0.01
maximum_params["sigma_D"] = 0.01
maximum_params["x_A"] = 0.01
maximum_params["x_D"] = 0.01
maximum_params["n_D"] = 0.15

param_bounds = [(0, 0.1),  # B(0)
                (0, 0.1),  # A(0)
                (0, 0.1)]  # D(0)
for name in unknow_params:
    param_bounds.append((minimum_params[name], maximum_params[name]))


# initial_guess = np.array(list(minimum_params.values()))
initial_guess = []
for bounds in param_bounds:
    initial_guess.append(bounds[0]+0.001)


name_data_param = "fitting_parameters_full_variables_v1"
all_solutions_errors = []


def sumOfSquaredError(parameterTuple, *args):
    ans = np.sum((y_data - model(t_data, parameterTuple)) ** 2)/t_data.shape[0]
    # print("MSE=", ans)

    global best_solution
    global best_error
    # Check if the current solution is better
    # print(type(xk), "Type")
    new_error = ans
    # print(new_error, "new_error")
    # Save the current solution and error to the list
    current_solution = np.append(parameterTuple, ans)
    all_solutions_errors.append(current_solution)
    # print(new_error, "new_error")
    if ans < best_error:
        best_solution = parameterTuple
        best_error = ans

        # Save the best solution to a DataFrame and CSV file
        df = pd.DataFrame([np.append(best_solution, best_error)],
                          index=[year],
                          columns=["B_0", "A_0", "D_0"] + unknow_params + ["MSE"])
        df.to_csv('./New data/'+lake_name + '/'+name_data_param + year +
                  lake_name + coment + "_callback.csv")

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


result = differential_evolution(
    sumOfSquaredError,
    bounds=param_bounds,
    disp=True,
    seed=3,
    strategy=strategies[0],
    # x0=initial_guess,
    maxiter=25,
    updating="immediate",
    # callback=callback_1,
    popsize=15,
    polish=False,
    tol=0.01)


df = pd.DataFrame(all_solutions_errors,
                  columns=["B_0", "A_0", "D_0"] + unknow_params + ["MSE"])
df.to_csv(name_data_param + year + lake_name + coment + "_full_poss.csv")

final_time = time.time()

print("Total time=", (final_time-initial_time)/60)

print("MSE=", result.fun)

params_fitted = result.x

# Extract the fitted parameters
print("Fitted Parameters:", params_fitted)

# Plot the observed data and the fitted curve for y_1
t_fit = t_data
y_fit = model(t_fit, params_fitted)


model_CyB.print_solv()
# plt.plot(t_data, y_data, label="Observed Data")
plt.scatter(t_data, y_data, color='red')
plt.plot(t_fit, y_fit, linestyle='--', label="Fitted Curve")
plt.xlabel("Days")
plt.ylabel("CyB mg C")
plt.legend()
plt.title(lake_name)
plt.show()


# Save parameters

df = pd.DataFrame(np.array([params_fitted]),
                  index=[year],
                  columns=["B_0", "A_0", "D_0"]
                  + unknow_params)

df.to_csv('./New data/'+lake_name + '/'
          + name_data_param + year + lake_name+coment
          + "_final" + ".csv")
