import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy.interpolate import Akima1DInterpolator, CubicSpline, PchipInterpolator, interp1d


def generate_dates(year):
    dates = []
    for month in range(1, 13):  # January to December
        next_month = month + 1 if month < 12 else 1  # Handle December
        # Handle December crossing into the next year
        next_year = year + 1 if month == 12 else year

        max_day = (datetime(next_year, next_month, 1) - timedelta(days=1)).day
        for day in range(1, max_day + 1, 1):
            start_date = datetime(year, month, day)
            # Adding one day to get the end date
            dates.append(start_date.strftime("%Y-%m-%d"))
    return dates


# Parameters
params = {
    'I_in': 300,
    'z_mk_A': (7*0.0004)*1000,
    'z_mk_B': (7*0.0004)*1000,
    'z_mK_bg': (7*0.3),
    'H_B': 120,
    'H_A': 120,
    'z_m': 7,
    'Q_m_B': 0.004,
    'Q_M_B': 0.04,
    'Q_m_A': 0.004,
    'Q_M_A': 0.04,
    'rho_m_A': 1,
    'rho_m_B': 1,
    'M_B': 1.5/1000,
    'M_A': 1.5/1000,
    'mu_B': 1,
    'mu_A': 1,
    'mu_r_B': 0.05,
    'mu_r_A': 0.05,
    'v': 0.25,
    'd_E': 0.02,
    'p_in': 0.15,
    'e_BD': 0,
    'e_AD': 0.8,
    'phi': 700,
    'phi_D': 700,
    'phi_Y': 700,
    'alpha_B': 0.00,
    'alpha_A': 0.0035,
    'tau_B': 0,
    'tau_A': 1.23,
    'theta_D': 0.03,
    'n_D': 0.185,
    'theta_Y': 0.064,
    'theta_W': 0.061,
    'e_DY': 0.667,
    'e_YW': 0.788,
    'alpha_D': 0.008,
    'alpha_Y': 0,
    "tau_Y": 0,
    'tau_D': 0,
    'n_Y': 0.0016,
    'n_W': 0.00104,
    "n_A": 0.39,
    "n_B": 0.39,
    'p': 100/14.12376,
    'd_M': 0.02,
    "x_A": 0,
    "x_D": 0,
    "x_Y": 0.00398,
    "x_W": 0.00398,
    "sigma_A": 0,
    "sigma_D": 0,
    "sigma_Y": 0.0062,
    "sigma_W": 0.0062,
    "a_A": 0,
    "a_D": 0,
    "a_Y": 0.1733*(10**(-6)),
    "a_W": 0.1733*(10**(-6)),
    "T_min_B": 5,
    "T_opt_B": 30,
    "T_max_B": 40,
    "T_min_A": 5,
    "T_opt_A": 27,
    "T_max_A": 42,
    "q_air": 0.1,
    "q_d": 0.01,
    "delta_A": 0.02,
    "delta_B+": 0.02+0.01,
    "delta_B-": 0.09545,
    "delta_D": 0.013568,
    "delta_Y": 0.0045,
    "delta_W": 0.005,
    "r_Y": 1,
    "r_W": 1,
    "Ext_Y": 0.4,
    "Ext_W": 0.4
}

# Define functions


# Growth Rate


def eta_A(T, params):
    T_min = params["T_min_A"]
    T_max = params["T_max_A"]
    T_opt = params["T_opt_A"]
    num = (T-T_min)**2*(T-T_max)
    den = (T_opt-T_min)*((T_opt-T_min)*(T-T_opt) -
                         (T_opt-T_max)*(T_opt+T_min-2*T))

    if (T <= T_min) or (T >= T_max):
        return 0.06  # 0.175
    else:
        return num/den*((T >= T_min))*((T <= T_max))


def eta_B(T, params):
    T_min = params["T_min_B"]
    T_max = params["T_max_B"]
    T_opt = params["T_opt_B"]

    num = (T-T_min)**2*(T-T_max)
    den = (T_opt-T_min)*((T_opt-T_min)*(T-T_opt) -
                         (T_opt-T_max)*(T_opt+T_min-2*T))

    if (T <= T_min) or (T >= T_max):
        return 0.06  # 0.175
    else:
        return num/den*((T >= T_min))*((T <= T_max))


def h_A(A, B, params):
    return (1 / (params['z_mk_A'] * A + params['z_mk_B'] * B + params['z_mK_bg'])) * \
        np.log((params['H_A'] + params['I_in']) /
               (params['H_A'] + I(A, B, params)))


def h_B(A, B, params):
    return (1 / (params['z_mk_A'] * A + params['z_mk_B'] * B + params['z_mK_bg'])) * \
        np.log((params['H_B'] + params['I_in']) /
               (params['H_B'] + I(A, B, params)))


def I(B, A, params):
    return params['I_in'] * np.exp(-(params['z_mk_A'] * A + params['z_mk_B'] * B + params['z_mK_bg']))


def rho_A(Q_A, P, params):
    return params['rho_m_A'] * ((params['Q_M_A'] - Q_A) / (params['Q_M_A'] - params['Q_m_A'])) * (P / (params["M_A"] + P))


def rho_B(Q_B, P, params):
    return params['rho_m_B'] * ((params['Q_M_B'] - Q_B) / (params['Q_M_B'] - params['Q_m_B'])) * (P / (params["M_B"] + P))


def f_B(B, A, params):
    return (params['phi'] * params['alpha_B'] * B) / \
        (1 + params['phi'] * params['alpha_B'] * params['tau_B'] *
         B + params['phi'] * params['alpha_A'] * params['tau_A'] * A)


def f_A(B, A, params):
    return (params['phi'] * params['alpha_A'] * A) / \
        (1 + params['phi'] * params['alpha_B'] * params['tau_B'] *
         B + params['phi'] * params['alpha_A'] * params['tau_A'] * A)


def f_D(D, params):
    return (params['phi_D'] * params['alpha_D'] * D) / (1 + params['phi_D'] * params['alpha_D'] * params['tau_D'] * D)


def f_Y(Y, params):
    return (params['phi_Y'] * params['alpha_Y'] * Y) / (1 + params['phi_Y'] * params['alpha_Y'] * params['tau_Y'] * Y)


def N_Y(O):
    return 1/(1+np.exp(3.1209*(O-4.01131)))


def N_W(O):
    return 1/(1 + np.exp(3.516129*(O-4.09359)))


def N_B(T, params):
    return params["n_B"] * (1 - np.exp(-0.01577*(T-29.58466)**2))


def N_A(T, params):
    return params["n_A"] * (1 - np.exp(-0.01002*(T-26.75)**2))


def E_YW(T, params):
    num = T**2*(T-30)
    den = 24*(24*(T-24)+6*(24-2*T))
    return params['e_YW']*(num/den)


def E_YD(T, params):
    num = T**2*(T-30)
    den = 24*(24*(T-24)+6*(24-2*T))
    return params['e_DY']*(num/den)


def delta_W(T, params):
    return params['delta_W']*(1+np.exp(-34.4467*(T-20.2893)))**(-0.00256)


def delta_Y(T, params):
    return params['delta_Y']*(1+np.exp(-34.4467*(T-20.2893)))**(-0.00256)


def delta_D(T, params):
    K = T+273.15
    num = (K/284)*np.exp(6866.7076*(-0.05888-1/K))
    den = 1+np.exp(-0.1148*(-0.05888-1/K))+np.exp(-0.17172*(-0.0913163-1/K))
    return num/den

# Define the system of equations


# Initial conditions
M_0 = 0
A_0 = 0.5
B_0 = 0.0
Q_B_0 = 0.01
Q_A_0 = 0.01
P_0 = 0.7
D_0 = 0.0239
Y_0 = 0.05
W_0 = 0.0025
v_A_0 = 0
v_D_0 = 0
v_Y_0 = 0
v_W_0 = 0
O_0 = 7
initial_conditions = [M_0, B_0, A_0, Q_B_0,
                      Q_A_0, P_0, D_0,
                      Y_0, W_0,
                      v_A_0, v_D_0, v_Y_0, v_W_0, O_0]


class modelCyB(object):

    def __init__(self):
        self.params = params.copy()
        self.solution = []
        self.initial = initial_conditions.copy()
        self.t_0 = 0
        self.t_f = 150
        self.delta_t = 150*3
        self.t = np.linspace(self.t_0, self.t_f, self.delta_t)
        self.toxines = True
        self.max_temp_time = 197.3989-121
        self.max_temp = 12.8017
        self.min_temp = 4.16188
        self.freq_temp = -0.0172
        self.init_temp = 0
        self.T = 0.0005196470766
        self.labels = False
        self.interpTemp = None

    def set_linetime(self):
        self.t = np.linspace(self.t_0, self.t_f, self.delta_t)

    def get_max_Temp(self):
        return self.Temp(self.t).max()

    def get_interpTemp(self, tempSamp, days):
        self.interpTemp = Akima1DInterpolator(
            days, tempSamp)
        # self.interpTemp = PchipInterpolator(days, tempSamp, extrapolate=True)
        # self.interpTemp = interp1d(
        #     days, tempSamp,
        #     fill_value='extrapolate',
        #     kind='next')

        # def Temp(x): return 20
        # self.interpTemp = Temp

    def Temp(self, t, data=None):
        # print('Temp:', self.interpTemp(t), t)
        return self.interpTemp(t)

    def system(self, y, t):
        params = self.params
        M, B, A, Q_B, \
            Q_A, P, D, Y, \
            W, v_A, v_D, v_Y, v_W, O = y

        dMdt = params['p'] * params['mu_B'] * B * \
            (1 - params['Q_m_B'] / Q_B)\
            * h_B(A, B, params)\
            * eta_B(self.Temp(t), params) \
            - params['d_M']*M

        # dMdt = 0

        dBdt = (params['mu_B']
                * (1 - (params['Q_m_B'] / Q_B))
                * h_B(A, B, params)
                * eta_B(self.Temp(t), params)) * B - \
            params['mu_r_B'] * B - \
            (params["d_E"] / params['z_m']) * B\
            - f_B(B, A, params)*D\
            - N_B(self.Temp(t), params)*B*0.01

        dAdt = (params['mu_A']
                * (1 - (params['Q_m_A'] / Q_A))
                * h_A(A, B, params)
                * eta_A(self.Temp(t), params)) * A - \
            params['mu_r_A'] * A - \
            ((params['v'] + params["d_E"]) / params['z_m']) * A \
            - (params["x_A"]*v_A)*A \
            - f_A(B, A, params)*D\
            - N_A(self.Temp(t), params)*A*0.01

        dDdt = params['e_BD'] * np.minimum(1, Q_B / params['theta_D']) * f_B(B, A, params)*D \
            + params['e_AD'] * np.minimum(1, Q_A / params['theta_D']) * f_A(B, A, params)*D \
            - (params['alpha_A'] + params['alpha_B']) * D\
            - (params['n_D'] + params["x_D"]*v_D) * D \
            - (0.02*f_D(D, params))*Y

        dYdt = params['r_Y'] * E_YD(self.Temp(t), params)*((0.2)*f_D(D, params)
                                                           + (1-0.2) * params["Ext_Y"]) * Y * (1-Y/0.4) \
            - params['alpha_D'] * Y - \
            (N_Y(O) + params["x_Y"]*v_Y) * Y\
            - 0.3*f_Y(Y, params)*W
        # print(E_YD(self.Temp(t), params)*f_D(D, params), params['alpha_D'])
        # print(params["x_Y"]*v_Y)

        dWdt = params['r_W'] * E_YW(self.Temp(t), params)*((0.3)*f_Y(Y, params)
                                                           + ((1-0.3)*params["Ext_W"]))*W*(1-W/0.15)\
            - params['alpha_Y'] * W - \
            (N_W(O) + params["x_W"]*v_W) * W
        # 10000
        dQ_Bdt = rho_B(Q_B, P, params) - \
            params['mu_B'] * (Q_B - params['Q_m_B']) * h_B(B, A, params)

        dQ_Adt = rho_A(Q_B, P, params) - \
            params['mu_A'] * (Q_A - params['Q_m_A']) * h_A(B, A, params)

        dPdt = (params["d_E"] / params['z_m']) * (params['p_in'] - P) - \
            rho_A(Q_A, P, params) * A - \
            rho_B(Q_B, P, params) * B

        if round(np.min(A), 6) <= 0:
            dv_Adt = 0
        else:
            dv_Adt = params["a_A"]*M - params["sigma_A"]*v_A \
                - params["mu_A"]*(1 - params["Q_m_A"]/Q_A) * \
                h_A(A, B, params)*v_A

        def Q_MCYST(T, params):
            return 4.703283*params['mu_B'] \
                * (1 - (params['Q_m_B'] / Q_B))\
                * h_B(A, B, params)\
                * eta_B(self.Temp(t), params)  \
                + 3.5685

        if round(np.min(D), 6) <= 0:
            dv_Ddt = 0
        else:
            dv_Ddt = params["a_D"]*M - params["sigma_D"]*v_D \
                + f_A(B, A, params)*v_A\
                + f_B(B, A, params)*Q_MCYST(self.Temp(t), params) \
                - params["e_BD"]*np.minimum(1, Q_B/params["theta_D"])*f_B(B, A, params)*v_D\
                - params["e_AD"] * \
                np.minimum(1, Q_A/params["theta_D"])*f_A(B, A, params)*v_D

        if round(np.min(Y), 6) <= 0:
            dv_Ydt = 0
        else:
            # print(dv_Ydt)

            dv_Ydt = params["a_Y"]*M - params["sigma_Y"]*v_Y\
                + f_D(D, params)*v_D \
                - E_YD(self.Temp(t), params)*f_D(D, params)*v_Y
            # if dv_Ydt > 1:
            #     print(t, Y)
        if round(np.min(W), 6) <= 0:
            dv_Wdt = 0
        else:
            dv_Wdt = params["a_W"]*M - params["sigma_W"]*v_W\
                + f_Y(Y, params)*v_Y\
                - E_YW(self.Temp(t), params)*f_Y(Y, params) * W

        dOdt = params["q_air"] - params["q_d"]*O \
            + params["delta_A"] * A * (1 - (params['Q_m_A'] / Q_A)) * h_A(A, B, params)*eta_A(self.Temp(t), params) \
            + params["delta_B+"]*B*(1 - (params['Q_m_B'] / Q_B)) * h_B(A, B, params)*eta_B(self.Temp(t), params)\
            - params["delta_B-"]*N_B(self.Temp(t), params) * B\
            - delta_D(self.Temp(t), params) * D\
            - delta_Y(self.Temp(t), params) * Y\
            - delta_W(self.Temp(t), params) * W

        return [dMdt,
                dBdt,
                dAdt,
                dQ_Bdt,
                dQ_Adt, dPdt,
                dDdt,
                dYdt,
                dWdt,
                self.toxines*dv_Adt,
                self.toxines*dv_Ddt,
                self.toxines*dv_Ydt,
                self.toxines*dv_Wdt,
                dOdt]

    def solver(self):

        # Simulation time in days
        t = self.t

        # Solve the system of equations
        sol, info = odeint(self.system, self.initial, t,
                           full_output=True)

        # Extract solutions
        self.solution = sol.T
        return sol, info

    def print_solv(self, title="", all_plot=False,
                   save=False, save_path='', dpi=300):

        M_values, B_values, A_values, Q_B_values, \
            Q_A_values, P_values, D_values, Y_values, \
            W_values, v_A_values, v_D_values, v_Y_values, \
            v_W_values, O_values = self.solution
        if all_plot:
            t = np.linspace(self.t_0, self.t_f, self.delta_t)

            colors = sns.color_palette("tab20", n_colors=14)
            sns.set_style('ticks')
            sns.plotting_context("paper", font_scale=1.5)
            # Create subplots
            fig, axs = plt.subplots(7, 2, figsize=(12, 12))
            axs = axs.ravel()

            variables = ["M $[\mu g/L]$", "B $[mgC/L]$ ",
                         'A $[mgC/L]$', '$Q_B$ $[mgP/mg C]$',
                         '$Q_A$ $[mgP/mg C]$', 'P $[mgP/L]$',
                         'D $[mgC/L]$', 'Y $[mgC/L]$',
                         "W $[mgC/L]$",
                         "$v_A$ $[\mu g /mg C]$", "$v_D$ $[\mu g /mg C]$", "$v_Y [\mu g /mg C]$",
                         "$v_W$ $[\mu g /mg C]$", "O $[mg O_2/L]$"]
            labels = ["microcystin-LR", "CyB", 'Algae', 'Q_B', 'Q_A',
                      'Phosphorus', 'Daphnia', 'Yellow Perch',
                      "Walleye", "Body burde Algae",
                      "Body burde Daphnia", "Body burde Yellow Perch",
                      "Body burde Walleye",
                      "Dissolved Oxygen"]

            for i, var in enumerate([M_values, B_values,
                                     A_values, Q_B_values,
                                     Q_A_values, P_values,
                                     D_values, Y_values,
                                     W_values, v_A_values,
                                     v_D_values, v_Y_values,
                                     v_W_values,
                                     O_values]):
                if self.labels:
                    axs[i].plot(t, var, color=colors[i], label=labels[i])
                    axs[i].legend()
                else:
                    # axs[i].plot(t, var, color=colors[i])
                    axs[i].plot(t, var, color=(2/255, 48/255, 74/255))
                axs[i].set_ylim(var.min(), var.max()*(1+0.05))
                axs[i].set_xlim(self.t.min(), self.t.max())
                # Dates as x axis
                x_ticks = axs[i].xaxis.get_ticklocs()
                dates = generate_dates(2021)

                if len(dates) > len(x_ticks):
                    x_stape = round(len(dates) / len(x_ticks), 0)
                    x_labels = [dates[int(x_val)] for x_val in x_ticks[:-1]]
                    axs[i].xaxis.set_ticklabels(x_labels,
                                                rotation=30)
                else:
                    x_labels = dates
                    axs[i].xaxis.set_ticklabels(x_labels,
                                                rotation=30)
                axs[i].set_xlabel('')
                axs[i].set_ylabel(variables[i])
                axs[i].grid(False)

            plt.tight_layout()
            plt.suptitle(title)
            if save:
                plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            plt.show()
        elif not all_plot:
            t = np.linspace(self.t_0, self.t_f, self.delta_t)
            sns.set_style('ticks')
            sns.plotting_context("paper", font_scale=1.5)
            colors = sns.color_palette("tab10", n_colors=14)

            # Create subplots
            fig, axs = plt.subplots(4, 2, figsize=(12, 12))
            axs = axs.ravel()

            variables = ["M $[\mu g/L]$", "B $[mgC/L]$ ",
                         'A $[mgC/L]$', 'P $[mgP/L]$',
                         'D $[mgC/L]$', 'Y $[mgC/L]$',
                         "W $[mgC/L]$", "O $[mg O_2/L]$"]
            labels = ["microcystin-LR", "CyB", 'Algae',
                      'Phosphorus', 'Daphnia', 'Yellow Perch',
                      "Walleye", "Dissolved Oxygen"]

            for i, var in enumerate([M_values, B_values,
                                     A_values, P_values,
                                     D_values, Y_values,
                                     W_values, O_values]):
                if self.labels:
                    axs[i].plot(t, var, color=colors[i], label=labels[i])
                    axs[i].legend()
                else:
                    # axs[i].plot(t, var, color=colors[i])
                    axs[i].plot(t, var, color=(2/255, 48/255, 74/255))

                axs[i].set_ylim(var.min(), var.max()*(1+0.05))
                axs[i].set_xlim(self.t.min(), self.t.max())
                # Dates as x axis
                x_ticks = axs[i].xaxis.get_ticklocs()
                dates = generate_dates(2021)

                if len(dates) > len(x_ticks):
                    x_stape = round(len(dates) / len(x_ticks), 0)
                    x_labels = [dates[int(x_val)] for x_val in x_ticks[:-1]]
                    axs[i].xaxis.set_ticklabels(x_labels,
                                                rotation=30)
                else:
                    x_labels = dates
                    axs[i].xaxis.set_ticklabels(x_labels,
                                                rotation=30)
                axs[i].set_xlabel('')
                axs[i].set_ylabel(variables[i])
                axs[i].grid(False)

            plt.tight_layout()
            plt.suptitle(title)
            if save:
                plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            plt.show()
