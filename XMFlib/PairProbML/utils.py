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