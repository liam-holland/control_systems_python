import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import control as ct
    import numpy as np 
    import matplotlib.pyplot as plt
    import sympy as sp

    from math import pi


    return ct, np, pi, sp


@app.cell
def _(pi):
    # Define constants
    rho_0 = 1.2 # Density of air
    c_0  = 343 # Speed of sound
    Z_0 = rho_0 * c_0 # Charactaristic impedance
    k_g = 0.5 # Feedback gain
    omega_g = 0.4 # Cutoff angular frequency

    # Charaterisitcs of driver and enclosure
    Sd = 0.0344;  
    Rms = 2.9;  
    Bl = 14;  
    Vb = 0.017;
    R_ss = Rms / Sd;
    omega_0 = 2*pi*40;  
    Qms = 5.2;
    Csb = Vb / (rho_0 * c_0**2 * Sd);
    F = Bl / Sd;


    return Csb, F, Qms, R_ss, Sd, Z_0, k_g, omega_0, omega_g


@app.cell
def _(sp):
    # Lets recreate them symbolically

    # Define continuous variable, discrete and sample time
    sym_s, sym_z, sym_Ts = sp.symbols('s z T_s')

    # Define base symbolic constants
    sym_rho0, sym_c0, sym_kg, sym_omegag = sp.symbols('rho_0 c_0 k_g omega_g')

    # Derived constant
    sym_Z0 = sym_rho0 * sym_c0  # Characteristic impedance

    # Define base symbolic characteristics of driver and enclosure
    sym_Sd, sym_Rms, sym_Bl, sym_Vb, sym_omega0, sym_Qms = sp.symbols(
        'S_d R_ms Bl V_b omega_0 Q_ms'
    )

    # Define the resonators symbols
    sym_Rst, sym_omegat, sym_Qt = sp.symbols('R_st omega_t Q_t') 

    # Derived driver and enclosure characteristics
    sym_Rss = sym_Rms / sym_Sd
    sym_Csb = sym_Vb / (sym_rho0 * sym_c0**2 * sym_Sd)
    sym_F = sym_Bl / sym_Sd

    return (
        sym_Csb,
        sym_F,
        sym_Qms,
        sym_Qt,
        sym_Rss,
        sym_Rst,
        sym_Ts,
        sym_c0,
        sym_kg,
        sym_omega0,
        sym_omegag,
        sym_omegat,
        sym_rho0,
        sym_s,
        sym_z,
    )


@app.cell
def _(
    sym_Qms,
    sym_Qt,
    sym_Rss,
    sym_Rst,
    sym_c0,
    sym_kg,
    sym_omega0,
    sym_omegag,
    sym_omegat,
    sym_rho0,
    sym_s,
):
    # Define equations
    sym_Zss = (sym_Rss * (sym_s**2 + sym_s*sym_omega0 / sym_Qms + sym_omega0**2) / (sym_s * sym_omega0/sym_Qms) )
    sym_Zst = (sym_Rst * (sym_s**2 + sym_s*sym_omegat / sym_Qt + sym_omegat**2) / (sym_s * sym_omegat/sym_Qt) )
    sym_G = sym_rho0 * sym_c0 * sym_kg * ( sym_omegag / (sym_s + sym_omegag) )


    return sym_G, sym_Zss, sym_Zst


@app.cell
def _():
    # Numerical values example, I could then plug this in

    # Define your numerical values (e.g., from your speaker/control system)
    # num_vals = {
    #     K: 1.5,
    #     omega_n: 251.33,  # roughly 2 * pi * 40 Hz
    #     zeta: 0.6,
    # }
    return


@app.cell
def _(sym_Csb, sym_F, sym_G, sym_Zss, sym_Zst, sym_s):
    # Transfer functions
    sym_H1 = 1/ sym_F * ( 1 - (sym_Zss + sym_G) / sym_Zst  )
    sym_H2 = sym_s * sym_Csb * sym_G/ sym_F

    return (sym_H1,)


@app.cell
def _(sym_Ts, sym_s, sym_y, sym_z):
    sym_H1.subs(sym_s, 2/sym_Ts * (sym_z-1)/(sym_z + 1))

    sym_H1 = sym_y
    return (sym_H1,)


@app.cell
def _(sp):
    y_H1 = sp.solve()
    return


@app.cell
def _(ct):
    # Define a continous transfer function variable
    s = ct.tf('s')
    return (s,)


@app.cell
def _(Qms, R_ss, Sd, ct, omega_0, s):
    # Specific Passive impedance
    Z_ss = ct.minreal(R_ss * (s**2 + s*omega_0/Qms + omega_0**2) / (s * omega_0/Qms))

    # Total mechanical passive impedance
    Z_ms = ct.minreal(Z_ss * Sd)
    return (Z_ss,)


@app.cell
def _(Z_0, ct, np, pi, s):
    # There are my resonators

    N       = 1
    Rst     = np.array([Z_0])           # specific resistance for each resonator
    ft      = np.array([200])          # target frequency in Hz for each resonator
    omega_t = 2 * pi * ft      # resonant frequency
    Qt      = np.array([0.25])            # quality factor for each resonator

    admittance_sum = 0;

    for i in range(N):
        admittance_sum = admittance_sum + (1/Rst[i]) * (s * omega_t[i]/Qt[i]) / (s**2 + s*omega_t[i]/Qt[i] + omega_t[i]**2)

    # The target impedance is the inverse of admittance
    Zst = ct.minreal(1 / admittance_sum)
    return (Zst,)


@app.cell
def _():
    Fs = 44100 # Sample rate
    Ts = 1/Fs # Sample time
    return (Ts,)


@app.cell
def _(Csb, F, Z_0, Z_ss, Zst, ct, k_g, omega_g, s):

    G = Z_0 * k_g * ( omega_g / ( s + omega_g) )

    H_1 = ct.minreal( 1/F * ( 1 - (Z_ss + G)/Zst) )
    H_2 = ct.minreal( s * Csb * G / F)
    return H_1, H_2


@app.cell
def _(H_2, ct):
    H_2_plot = ct.bode_plot(H_2)
    H_2_plot.__getattribute__("figure")
    return


@app.cell
def _(H_1, H_2, Ts, ct):
    H_1_discrete = ct.sample_system(H_1,Ts=Ts)
    H_2_discrete = ct.sample_system(H_2,Ts=Ts)
    return H_1_discrete, H_2_discrete


@app.cell
def _(H_1_discrete, ct):
    H1_d_p = ct.bode_plot(H_1_discrete)
    H1_d_p.__getattribute__("figure")
    return


@app.cell
def _(H_1_discrete, H_2_discrete):
    print(H_1_discrete)
    print(H_2_discrete)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
