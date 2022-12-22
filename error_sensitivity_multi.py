from scipy.stats import rankdata
import pyEDM as edm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import sys
sys.path.append("D:/Environmental_Modelling/3_Semester/aktuelles_thema_marieke/material")
import models
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from collections import Counter
from math import factorial

sns.set_style("whitegrid")

# Function for calculating the WPE

def embed(x, m, d = 1):
    """
    Pennekamp 2019
    """
    n = len(x) - (m-1)*d
    X = np.arange(len(x))
    out = np.array([X[np.arange(n)]]*m)
    a = np.repeat(np.arange(1, m)*d, out.shape[1])
    out[1:,] = out[1:,]+a.reshape(out[1:,].shape)
    out = x[out]

    return out

def entropy(wd):
    """
    in bits
    """
    return -np.sum(list(wd.values())*np.log2(list(wd.values())))


def word_distr(x_emb, tie_method='average'):

    words = [np.array2string(rankdata(x_emb[:, i])) for i in range(x_emb.shape[1])]
    vars = {np.array2string(rankdata(x_emb[:, i])):np.var(x_emb[:,i]) for i in range(x_emb.shape[1])}
    c_old = c = dict(Counter(words))
    c = dict(Counter(words))
    for k, v in c.items():
        c[k] = (v*vars[k])/np.sum(np.array([b*vars[a] for (a,b) in c_old.items()]))
    return c

def weighted_permutation_entropy(x, m, d=1):

    x_emb = embed(x, m=m, d=d)
    wd = word_distr(x_emb)
    denom = np.log2(2 * factorial(m))
    ent = entropy(wd) / denom

    return ent

# Function for calculating the forecast horizon

def fh(yo,yf):
    window = 3
    rho = 0.00075

    d = []
    for j in range(len(yo) - window):
        d.append(np.mean(abs(yo[j:j + window].to_numpy() - yf[j:j + window].to_numpy())))

    t_reached = (np.array(d) > rho)

    if t_reached.sum() == 0:
        fh = len(yo)
    else:
        fh = np.argmax(t_reached)

    return fh


# Function for extracting best parameters for smap

def optim_param(df, param):

    if param not in ["E", "Theta"]:
        raise ValueError 

    global_max = df["rho"].max()
    df_filtered = df[df["rho"]==global_max]

    return int(df_filtered[param].min())

# Calculate mae + forestcast horizon sensitivity for training and testing period with ricker det, ricker sto and smap

sigma_space = np.linspace(0, 0.001, 100)
phi_space = np.linspace(0, 0.001, 100)
r_space = np.linspace(0.05,3.3,100)
variables = ["sigma","phi","r"]

r_default = 0.05

# When settings should be changed uncomment respective default values for phi and sigma

# Setting for purely deterministic observations

#sigma_default = 0
phi_default = 0

# Setting for introduction of low observation and process errors

sigma_default = 0.00001
#phi_default = 0.00001

# Setting for introduction of medium observation and process errors

#sigma_default = 0.0001
#phi_default = 0.0001
 
# Setting for introduction of high observation and process errors

#sigma_default = 0.001
#phi_default = 0.001

r_results_train = []
r_results_test = []

for index in tqdm(range(len(sigma_space))):


        sigma = sigma_default
        phi = phi_default
        r = r_space[index]

        seed = 300

        # Simulate observation data

        iterations = 4 * 52

        init_s = 0.99

        mo_pars = {'lambda1': r, 'K1': 1, 'alpha':1, 'beta':0.00006, 'lambda2': r, 'K2': 1, 'gamma':1, 'delta':0.00005}
        mo_errs = {"sigma":sigma,"phi":phi, "init_u":0}

        so_pars = {"iterations":iterations, "initial_size": (init_s, init_s), "ensemble_size": 1}

        mo = models.Ricker_Multi(seed,set_seed=True)

        mo.parameters(mo_pars, mo_errs)

        yo = mo.simulate(so_pars)['ts'][:,:,0].squeeze()

        # Training

        # Simulate ricker process model deterministic and stochastic for training period

        iterations_train = int(iterations/2)
        iterations_test = int(iterations/2)

        init_s_test = yo[:iterations_train+1][-1]

        mricker_pars = {'lambda': r, 'K': 1}

        mfricker_det_errs = {"sigma": 0.0, "phi": 0.0, "init_u": 0.0}
        mfricker_sto_errs = {"sigma": sigma, "phi": phi, "init_u": 0.0}

        sricker_pars_train = {"iterations": iterations_train,
                            "initial_size": init_s, "ensemble_size": 1}
        sricker_pars_test = {"iterations": iterations_test,
                            "initial_size": init_s_test, "ensemble_size": 1}

        # Deterministic train period

        mricker_det_train = models.Ricker_Single(seed,set_seed=True)
        mricker_det_train.parameters(mricker_pars, mfricker_det_errs)

        yricker_det_train = mricker_det_train.simulate(sricker_pars_train)[
            'ts'][0, :]

        # Deterministic test period based on last observation from train period

        mricker_det_test = models.Ricker_Single(seed,set_seed=True)
        mricker_det_test.parameters(mricker_pars, mfricker_det_errs)

        yricker_det_test = mricker_det_test.simulate(sricker_pars_test)['ts'][0, :]

        # Stochastic train period

        mricker_sto_train = models.Ricker_Single(seed,set_seed=True)
        mricker_sto_train.parameters(mricker_pars, mfricker_sto_errs)

        yricker_sto_train = mricker_sto_train.simulate(sricker_pars_train)['ts'][0, :]

        # Stochastic test period

        mricker_sto_test = models.Ricker_Single(seed,set_seed=True)
        mricker_sto_test.parameters(mricker_pars, mfricker_sto_errs)

        yricker_sto_test = mricker_sto_test.simulate(sricker_pars_test)['ts'][0, :]

        N_df_train = pd.DataFrame(
            {"yo": yo[:iterations_train], "yricker_det_train": yricker_det_train, "yricker_sto_train": yricker_sto_train})
        N_df_test = pd.DataFrame(
            {"yo": yo[iterations_test:], "yricker_det_test": yricker_det_test, "yricker_sto_test": yricker_sto_test})

        smap_df = pd.DataFrame({"index":range(len(yo)),"yo": yo})

        training_end_index = iterations_train

        # EDM train and test period

        optim_E_df_train = edm.EmbedDimension(dataFrame=smap_df, lib=f"1 {training_end_index}", pred=f"1 {training_end_index-1}",
                                            columns="yo", target="yo", showPlot=False)

        optim_E_df_test = edm.EmbedDimension(dataFrame=smap_df, lib=f"1 {training_end_index}", pred=f"{training_end_index+1} {iterations-1}",
                                            columns="yo", target="yo", showPlot=False)

        E_train = optim_param(optim_E_df_train, "E")
        E_test = optim_param(optim_E_df_test, "E")

        optim_theta_df_train = edm.PredictNonlinear(dataFrame=smap_df, lib=f"1 {training_end_index}", pred=f"1 {training_end_index-1}",
                                                    columns="yo", target="yo", E=E_train, showPlot=False)

        optim_theta_df_test = edm.PredictNonlinear(dataFrame=smap_df, lib=f"1 {training_end_index}", pred=f"{training_end_index+1} {iterations-1}",
                                                columns="yo", E=E_test, showPlot=False)

        theta_train = optim_param(optim_theta_df_train, "Theta")
        theta_test = optim_param(optim_theta_df_test, "Theta")

        ysmap_train = edm.SMap(dataFrame=smap_df, lib=f"1 {training_end_index}", pred=f"1 {training_end_index-1}",
                            columns="yo", E=E_train, theta=theta_train)["predictions"]["Predictions"].tolist()

        ysmap_test = edm.SMap(dataFrame=smap_df, lib=f"1 {training_end_index}", pred=f"{training_end_index+1} {iterations-1}",
                            columns="yo", E=E_test, theta=theta_test)["predictions"]["Predictions"].tolist()

        if E_train > 1:

            train_helper = [np.nan for i in range((E_train-1))]

            N_df_train["ysmap_train"] = np.array(train_helper + ysmap_train)

        else:
            N_df_train["ysmap_train"] = np.array(ysmap_train)


        N_df_test["ysmap_test"] = np.array(ysmap_test)

        # Calculate gof measures for smap and ricker

        gof_df_train = N_df_train.copy().dropna()
        gof_df_test = N_df_test.copy().dropna()

        # MAE for train and test dataset 

        gof_df_train["yricker_det_train_mae"] = abs(gof_df_train["yricker_det_train"]-gof_df_train["yo"])
        gof_df_train["yricker_sto_train_mae"] = abs(gof_df_train["yricker_sto_train"]-gof_df_train["yo"])
        gof_df_train["ysmap_train_mae"] = abs(gof_df_train["ysmap_train"]-gof_df_train["yo"])

        yricker_det_train_mae = gof_df_train["yricker_det_train_mae"].mean()
        yricker_sto_train_mae = gof_df_train["yricker_sto_train_mae"].mean()
        ysmap_train_mae = gof_df_train["ysmap_train_mae"].mean()

        gof_df_test["yricker_det_test_mae"] = abs(gof_df_test["yricker_det_test"]-gof_df_test["yo"])
        gof_df_test["yricker_sto_test_mae"] = abs(gof_df_test["yricker_sto_test"]-gof_df_test["yo"])
        gof_df_test["ysmap_test_mae"] = abs(gof_df_test["ysmap_test"]-gof_df_test["yo"])  

        yricker_det_test_mae = gof_df_test["yricker_det_test_mae"].mean()
        yricker_sto_test_mae = gof_df_test["yricker_sto_test_mae"].mean()
        ysmap_test_mae = gof_df_test["ysmap_test_mae"].mean()  

        # Forecast horizon

        yricker_det_train_fh = fh(gof_df_train["yo"],gof_df_train["yricker_det_train"])
        yricker_sto_train_fh = fh(gof_df_train["yo"],gof_df_train["yricker_sto_train"])
        ysmap_train_fh = fh(gof_df_train["yo"],gof_df_train["ysmap_train"])

        yricker_det_test_fh = fh(gof_df_test["yo"],gof_df_test["yricker_det_test"])
        yricker_sto_test_fh = fh(gof_df_test["yo"],gof_df_test["yricker_sto_test"])
        ysmap_test_fh = fh(gof_df_test["yo"],gof_df_test["ysmap_test"]) 

        # WPE

        yo_wpe_train = weighted_permutation_entropy(yo[:training_end_index],5)
        yo_wpe_test = weighted_permutation_entropy(yo[training_end_index:],5)    

        # Combine data

        result_df_train = pd.DataFrame({"type":["train"],
                                        "sigma":[sigma],
                                        "phi":[phi],
                                        "yricker_det_train_mae":[yricker_det_train_mae],
                                        "yricker_sto_train_mae":[yricker_sto_train_mae],
                                        "ysmap_train_mae":[ysmap_train_mae],
                                        "yricker_det_train_fh":[yricker_det_train_fh],
                                        "yricker_sto_train_fh":[yricker_sto_train_fh],
                                        "ysmap_train_fh":[ysmap_train_fh],
                                        "wpe_train":[yo_wpe_train],
                                        "r":[r]})

        result_df_test = pd.DataFrame({"type":["test"],
                                    "sigma":[sigma],
                                    "phi":[phi],
                                    "yricker_det_test_mae":[yricker_det_test_mae],
                                    "yricker_sto_test_mae":[yricker_sto_test_mae],
                                    "ysmap_test_mae":[ysmap_test_mae],
                                    "yricker_det_test_fh":[yricker_det_test_fh],
                                    "yricker_sto_test_fh":[yricker_sto_test_fh],
                                    "ysmap_test_fh":[ysmap_test_fh],
                                    "wpe_test":[yo_wpe_test],
                                    "r":[r]})


        r_results_train.append(result_df_train)
        r_results_test.append(result_df_test)


        # Save a plot for train and test period showing each predictive model and the observations

        fig, axes = plt.subplots(1,2,figsize=(9,6))

        axes[0].plot(N_df_train["yo"],label="yo")
        axes[0].plot(N_df_train["yricker_det_train"],label="yricker_det")
        axes[0].plot(N_df_train["yricker_sto_train"],label="yricker_sto")
        axes[0].plot(N_df_train["ysmap_train"],label="smap")
        axes[0].legend()
        axes[0].set_title("Training")    

        axes[1].plot(N_df_test["yo"],label="yo")
        axes[1].plot(N_df_test["yricker_det_test"],label="yricker_det")
        axes[1].plot(N_df_test["yricker_sto_test"],label="yricker_sto")
        axes[1].plot(N_df_test["ysmap_test"],label="smap")
        axes[1].legend()
        axes[1].set_title("Testing")

        plt.tight_layout()
        fig.suptitle(rf" $\sigma$={sigma}; $\phi$={phi}")
        fig.subplots_adjust(top=0.9)
        fig.savefig(f"D:/Environmental_Modelling/3_Semester/aktuelles_thema_marieke/temp_results/multi_plots/{round(sigma,5)}_{round(phi,5)}_{round(r,5)}.png")
        plt.close()


# Merge results

r_training_results = pd.concat(r_results_train)
r_testing_results = pd.concat(r_results_test)

# Write results to file

r_training_results.to_csv("D:/Environmental_Modelling/3_Semester/aktuelles_thema_marieke/final_results/training_sensitivity_multi_r_high_phi.csv",index=False)
r_testing_results.to_csv("D:/Environmental_Modelling/3_Semester/aktuelles_thema_marieke/final_results/testing_sensitivity_multi_r_high_phi.csv",index=False)
