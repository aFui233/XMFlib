import re
import numpy as np

def eps_kbt(epsilon_ev, temperature):
    """
    Convert interaction energy from eV to dimensionless form (epsilon / kBT).

    Args:
        epsilon_ev (float): Interaction energy in eV.
        temperature (float): Temperature in K.

    Returns:
        float: Dimensionless interaction energy.
    """
    kB = 8.617333262145e-5  # eV/K
    return epsilon_ev / (kB * temperature)

def parse_hidden_layers(model_file):
    match = re.search(r'hl([\d\-]+)', model_file)
    if match:
        hl_str = match.group(1)
        return [int(x) for x in hl_str.split('-')]
    else:
        raise ValueError("hidden_layers信息未在文件名中找到")
    
def data_norm(x_input, dataset):
    X_all = np.stack([dataset[:, 0], dataset[:, 1]], axis=1)
    x_mean = X_all.mean(axis=0)
    x_std = X_all.std(axis=0)

    x_output = (x_input - x_mean) / x_std

    return x_output

def data_denorm(y_input, dataset):
    y_all = dataset[:, 2]
    y_mean = y_all.mean()
    y_std = y_all.std()

    y_output = y_input * y_std + y_mean

    return y_output