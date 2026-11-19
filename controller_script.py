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
    from scipy import signal
    from math import pi

    return ct, mo, np, pi, plt, signal, sp


@app.cell
def _(sp):
    # A1. Symbols
    sym_s = sp.Symbol("s", positive=True)
    sym_rho0, sym_c0, sym_kg, sym_omegag = sp.symbols(
        "rho_0 c_0 k_g omega_g", positive=True
    )
    sym_Sd, sym_Rms, sym_Bl, sym_Vb, sym_omega0, sym_Qms = sp.symbols(
        "S_d R_ms Bl V_b omega_0 Q_ms", positive=True
    )
    sym_Rst, sym_omegat, sym_Qt = sp.symbols("R_st omega_t Q_t", positive=True)

    # Derived quantities
    sym_Rss = sym_Rms / sym_Sd
    sym_Csb = sym_Vb / (sym_rho0 * sym_c0**2 * sym_Sd)
    sym_F = sym_Bl / sym_Sd
    return (
        sym_Bl,
        sym_Csb,
        sym_F,
        sym_Qms,
        sym_Qt,
        sym_Rms,
        sym_Rss,
        sym_Rst,
        sym_Sd,
        sym_Vb,
        sym_c0,
        sym_kg,
        sym_omega0,
        sym_omegag,
        sym_omegat,
        sym_rho0,
        sym_s,
    )


@app.cell
def _(
    sym_Csb,
    sym_F,
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
    # A2. Continuous-time transfer functions
    sym_Zss = (
        sym_Rss
        * (sym_s**2 + sym_s * sym_omega0 / sym_Qms + sym_omega0**2)
        / (sym_s * sym_omega0 / sym_Qms)
    )
    sym_Zst = (
        sym_Rst
        * (sym_s**2 + sym_s * sym_omegat / sym_Qt + sym_omegat**2)
        / (sym_s * sym_omegat / sym_Qt)
    )
    sym_G = sym_rho0 * sym_c0 * sym_kg * (sym_omegag / (sym_s + sym_omegag))

    sym_H1 = (1 / sym_F) * (1 - (sym_Zss + sym_G) / sym_Zst)
    sym_H2 = sym_s * sym_Csb * sym_G / sym_F
    return sym_G, sym_H1, sym_H2, sym_Zss, sym_Zst


@app.cell
def _(mo, sp, sym_G, sym_H1, sym_H2, sym_Zss, sym_Zst):
    # A3. Show the symbolic equations (this is where the symbolic part stops)
    mo.md(
        f"""
        ## Symbolic model (continuous time)

        $$Z_{{ss}}(s) = {sp.latex(sym_Zss)}$$

        $$Z_{{st}}(s) = {sp.latex(sym_Zst)}$$

        $$G(s) = {sp.latex(sym_G)}$$

        $$H_1(s) = {sp.latex(sym_H1)}$$

        $$H_2(s) = {sp.latex(sym_H2)}$$
        """
    )
    return


@app.cell
def _(sp, sym_H1, sym_H2, sym_s):
    # A4. Collect each H as one rational function of s and keep its coefficient
    # expressions (highest power first). Cheap, and not displayed.
    def s_coeffs(H):
        num, den = sp.fraction(sp.cancel(sp.together(H)))
        pn = sp.Poly(sp.expand(num), sym_s)
        pd = sp.Poly(sp.expand(den), sym_s)
        return pn.all_coeffs(), pd.all_coeffs()

    H1_s_num, H1_s_den = s_coeffs(sym_H1)
    H2_s_num, H2_s_den = s_coeffs(sym_H2)
    return H1_s_den, H1_s_num, H2_s_den, H2_s_num


@app.cell
def _(pi):
    # B1. Physical constants & driver / enclosure parameters
    rho_0 = 1.2  # Density of air
    c_0 = 343  # Speed of sound
    Z_0 = rho_0 * c_0  # Characteristic impedance
    k_g = 0.5  # Feedback gain
    f_g = 200 
    omega_g = 2 * pi * f_g  # Cutoff angular frequency (rad/s)

    Sd = 0.0344  # Effective piston area
    Rms = 2.9  # Mechanical resistance
    Bl = 14  # Force factor
    Vb = 0.017  # Enclosure volume
    omega_0 = 2 * pi * 40  # Resonance frequency (40 Hz)
    Qms = 5.2  # Mechanical Q factor

    # Resonator (single resonator example)
    Rst_val = Z_0
    omega_t_val = 2 * pi * 200
    Qt_val = 0.25

    # Sample rate
    Fs = 44100
    return (
        Bl,
        Fs,
        Qms,
        Qt_val,
        Rms,
        Rst_val,
        Sd,
        Vb,
        c_0,
        k_g,
        omega_0,
        omega_g,
        omega_t_val,
        rho_0,
    )


@app.cell
def _(
    Bl,
    H1_s_den,
    H1_s_num,
    H2_s_den,
    H2_s_num,
    Qms,
    Qt_val,
    Rms,
    Rst_val,
    Sd,
    Vb,
    c_0,
    ct,
    k_g,
    omega_0,
    omega_g,
    omega_t_val,
    rho_0,
    sym_Bl,
    sym_Qms,
    sym_Qt,
    sym_Rms,
    sym_Rst,
    sym_Sd,
    sym_Vb,
    sym_c0,
    sym_kg,
    sym_omega0,
    sym_omegag,
    sym_omegat,
    sym_rho0,
):
    # B2. Symbols -> numbers, giving the continuous-time transfer functions
    param_vals = {
        sym_rho0: rho_0,
        sym_c0: c_0,
        sym_kg: k_g,
        sym_omegag: omega_g,
        sym_Sd: Sd,
        sym_Rms: Rms,
        sym_Bl: Bl,
        sym_Vb: Vb,
        sym_omega0: omega_0,
        sym_Qms: Qms,
        sym_Rst: Rst_val,
        sym_omegat: omega_t_val,
        sym_Qt: Qt_val,
    }

    def to_float(coeffs):
        return [float(c.subs(param_vals)) for c in coeffs]

    num1_c, den1_c = to_float(H1_s_num), to_float(H1_s_den)
    num2_c, den2_c = to_float(H2_s_num), to_float(H2_s_den)

    H_1_ct = ct.TransferFunction(num1_c, den1_c)
    H_2_ct = ct.TransferFunction(num2_c, den2_c)

    return H_1_ct, H_2_ct


@app.cell
def _(Fs, H_1_ct, ct, np):
    # All this does is axtract the array from the object
    example = ct.c2d(H_1_ct,1/Fs,method ="tustin")
    get_num_array = np.asarray(example.__getattribute__("num_array")[0][0])

    print(get_num_array)
    return


@app.cell
def _(Fs, H_1_ct, H_2_ct, ct, np):
    # Bilinear transform
    def coeffs(H):
        Hd = ct.c2d(H, 1 / Fs, method="tustin")
    
        n = np.asarray(Hd.__getattribute__("num_array")[0][0], dtype=float)
        d = np.asarray(Hd.__getattribute__("den_array")[0][0], dtype=float)

        return n , d  , Hd

    # I could also use scipy instead of control
    # def print_filter_summary(H_ct, Fs, name="H"):
    #     # 1. Discretize using scipy or control directly
    #     # (Scipy returns b, a arrays natively)
    #     num, den = signal.cont2discrete(
    #         (H_ct.num[0][0], H_ct.den[0][0]), dt=1 / Fs, method="bilinear"
    #     )

    #     # Normalize by a[0] so a[0] = 1.0
    #     b = num.flatten() / den[0]
    #     a = den.flatten() / den[0]


    # Remember H[z] =  Y[z] / X[z] = num / den

    # Coefficients for the input samples and output samples
    in_1, out_1, H_1_d = coeffs(H_1_ct)
    in_2, out_2, H_2_d = coeffs(H_2_ct)

    ## Now write the coeffcients for the y[n-?] and x[n-?] samples in table form and in Latex form  



    return in_1, in_2, out_1, out_2


@app.cell
def _(in_1, in_2, mo, out_1):
    # C code for a per-sample process function (direct form I, double precision)
    def c_filter(name, b, a):
        n = len(b) - 1
        arr = lambda v: ", ".join(repr(float(x)) for x in v)
        return f"""#define {name}_ORDER {n}

    static const double {name}_B[{n + 1}] = {{ {arr(b)} }};
    static const double {name}_A[{n + 1}] = {{ {arr(a)} }};  /* A[0] == 1 */

    typedef struct {{
        double u[{name}_ORDER];  /* u[0] = u[k-1], u[1] = u[k-2], ... */
        double y[{name}_ORDER];  /* y[0] = y[k-1], y[1] = y[k-2], ... */
    }} {name}_State;

    static inline double {name}_process({name}_State *s, double u0)
    {{
        double y0 = {name}_B[0] * u0;
        for (int i = 1; i <= {name}_ORDER; i++)
            y0 += {name}_B[i] * s->u[i - 1] - {name}_A[i] * s->y[i - 1];
        for (int i = {name}_ORDER - 1; i > 0; i--) {{
            s->u[i] = s->u[i - 1];
            s->y[i] = s->y[i - 1];
        }}
        s->u[0] = u0;
        s->y[0] = y0;
        return y0;
    }}
    """

    c_code = c_filter("H1", in_1, out_1) + "\n" + c_filter("H2", in_2, out_1)
    mo.md(f"## Generated C\n\n```c\n{c_code}```")
    return


@app.cell
def _(Fs, H_1_ct, H_2_ct, in_1, in_2, np, out_1, out_2, plt):
    # D1. Continuous vs discrete frequency response
    f = np.logspace(1, np.log10(20000), 600)
    w = 2 * np.pi * f

    def disc_resp(b, a):
        zinv = np.exp(-1j * w / Fs)
        return sum(bj * zinv**j for j, bj in enumerate(b)) / sum(
            ai * zinv**i for i, ai in enumerate(a)
        )

    fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for col, (name, Hc, bd, ad) in enumerate(
        [("H1", H_1_ct, in_1, out_1), ("H2", H_2_ct, in_2, out_2)]
    ):
        Hs = Hc(1j * w)
        Hz = disc_resp(bd, ad)
        axs[0, col].semilogx(f, 20 * np.log10(np.abs(Hs)), label="continuous")
        axs[0, col].semilogx(f, 20 * np.log10(np.abs(Hz)), "--", label="Tustin")
        axs[1, col].semilogx(f, np.unwrap(np.angle(Hs)) * 180 / np.pi)
        axs[1, col].semilogx(f, np.unwrap(np.angle(Hz)) * 180 / np.pi, "--")
        axs[0, col].set_title(name)
        axs[0, col].set_ylabel("Magnitude (dB)")
        axs[1, col].set_ylabel("Phase (deg)")
        axs[1, col].set_xlabel("Frequency (Hz)")
        axs[0, col].grid(True, which="both", alpha=0.3)
        axs[1, col].grid(True, which="both", alpha=0.3)
    axs[0, 0].legend()
    fig.tight_layout()
    fig
    return


@app.cell
def _(in_1, mo, np, out_1, signal):
    # D2. Mirror of the generated C loop, in float64 and float32, vs lfilter
    def process_loop(b, a, u, dtype):
        n = len(b) - 1
        b = np.asarray(b, dtype=dtype)
        a = np.asarray(a, dtype=dtype)
        hu = np.zeros(n, dtype=dtype)
        hy = np.zeros(n, dtype=dtype)
        out = np.zeros(len(u), dtype=dtype)
        for k, u0 in enumerate(u.astype(dtype)):
            y0 = b[0] * u0
            for i in range(1, n + 1):
                y0 = y0 + b[i] * hu[i - 1] - a[i] * hy[i - 1]
            hu[1:] = hu[:-1].copy()
            hy[1:] = hy[:-1].copy()
            hu[0], hy[0] = u0, y0
            out[k] = y0
        return out

    _rng = np.random.default_rng(0)
    _u = _rng.standard_normal(4000)
    _ref = signal.lfilter(in_1, out_1, _u)
    _scale = np.max(np.abs(_ref))

    err64 = np.max(np.abs(process_loop(in_1, out_1, _u, np.float64) - _ref)) / _scale
    err32 = np.max(np.abs(process_loop(in_1, out_1, _u, np.float32) - _ref)) / _scale
    mo.md(
        f"""
        ### H1 loop vs `lfilter` (relative to peak output)
        - float64: {err64:.1e}
        - float32: {err32:.1e}
        """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
