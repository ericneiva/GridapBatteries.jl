import pybamm
import numpy as np
from src import DiffusionOnly

model = DiffusionOnly(coord_sys="spherical polar")

def D_s(x):
    a = [0.0, -0.9231, -0.4066, -0.9930, 0,0]
    b = [-13.96, 0.3216, 0.4532, 0.8098, 0.0]
    c = [0.0, 2.534e-3, 3.926e-3, 9.924e-2, 1.0]

    D_ref = (
        10
        ** (
            a[0] * x + b[0]
            + a[1] * pybamm.exp(-((x - b[1]) ** 2) / c[1])
            + a[2] * pybamm.exp(-((x - b[2]) ** 2) / c[2])
            + a[3] * pybamm.exp(-((x - b[3]) ** 2) / c[3])
            + a[4] * pybamm.exp(-((x - b[4]) ** 2) / c[4])
        )
        #* 2.7  # correcting factor (see O'Regan et al 2021)
    ) 
    return D_ref / 25e-12

def D_e(x):
    p = [1.01e3, 1.01, -1.56e3, -4.87e2]
    T = 300
    D_ref = p[0] * pybamm.exp(p[1] * x) * pybamm.exp(p[2] / T) * pybamm.exp(p[3] * x / T) * 1.0e-10
    return D_ref / 25e-12

parameters = pybamm.ParameterValues({
    "Electrode diffusivity [m2.s-1]": D_s,
    "Electrolyte diffusivity [m2.s-1]": D_e,
    "Reaction rate constant [A.m-2]": 1e-3,
    "Maximum concentration [mol.m-3]": 1,
    "Initial electrode concentration [mol.m-3]": 0.5,
    "Initial electrolyte concentration [mol.m-3]": 1,
    "Transfer coefficient": 0,
    "Electrode thickness [m]": 0.3,
    "Electrolyte thickness [m]": 0.15 * 20,
})

sim = pybamm.Simulation(model, parameter_values=parameters, var_pts={model.r_s: 100, model.r_e: 100})
sol = sim.solve(np.linspace(0, 2, 1000))