#
# Contact lens curing temperature dependent model
#
import pybamm
import numpy as np

def smooth_heaviside(x, s):
    transition = 1 / (1 + np.exp( (1 / x + 1 / (x - s)) * s))
    # return transition
    return (x > 0) * (x < s) * transition + (x > s) * 1

def smooth_max(x, s=1e-4):
    return smooth_heaviside(x, 1e-4) * x
    # return pybamm.maximum(x, 0)
    # return (x + (x**x + s) ** (1 / 2)) / 2


class DiffusionOnly(pybamm.models.base_model.BaseModel):
    def __init__(self, name="diffusion only", coord_sys="cartesian"):
        super().__init__(name=name)

        ######################
        # Variables
        ######################
        c_s = pybamm.Variable(
            "Electrode concentration [mol.m-3]", domains={"primary": "electrode"}
        )
        c_e = pybamm.Variable(
            "Electrolyte concentration [mol.m-3]", domains={"primary": "electrolyte"}
        )
        self.r_s = pybamm.SpatialVariable(
            "r_s", domain="electrode", coord_sys=coord_sys,
        )
        self.r_e = pybamm.SpatialVariable(
            "r_e", domain="electrolyte", coord_sys=coord_sys,
        )

        ######################
        # Parameters
        ######################
        k = pybamm.Parameter("Reaction rate constant [A.m-2]")
        c_s_max = pybamm.Parameter("Maximum concentration [mol.m-3]")
        c_s0 = pybamm.Parameter("Initial electrode concentration [mol.m-3]")
        c_e0 = pybamm.Parameter("Initial electrolyte concentration [mol.m-3]")
        t_plus = pybamm.Parameter("Transfer coefficient")

        def D_s(c_s):
            return pybamm.FunctionParameter(
                "Electrode diffusivity [m2.s-1]",
                {
                    "Electrode concentration [mol.m-3]": c_s,
                },
            )

        def D_e(c_e):
            return pybamm.FunctionParameter(
                "Electrolyte diffusivity [m2.s-1]",
                {
                    "Electrolyte concentration [mol.m-3]": c_e,
                },
            )

        ######################
        # Governing equations
        ######################
        dcsdt = pybamm.div(D_s(c_s) * pybamm.grad(c_s))
        dcedt = pybamm.div(D_e(c_e) * pybamm.grad(c_e))
        self.rhs = {c_s: dcsdt, c_e: dcedt}

        c_s_surf = pybamm.BoundaryValue(c_s, "right")
        c_e_surf = pybamm.BoundaryValue(c_e, "left")
        rbc_s = (
            k
            * (
                smooth_max(c_e_surf / c_e0)
                * smooth_max(c_s_surf / c_s_max)
                * smooth_max(1 - c_s_surf / c_s_max)
            )
            ** (1 / 2)
            / D_s(smooth_max(c_s_surf))
        )
        lbc_e = (
            (1 - t_plus)
            * k
            * (
                smooth_max(c_e_surf / c_e0)
                * smooth_max(c_s_surf / c_s_max)
                * smooth_max(1 - c_s_surf / c_s_max)
            )
            ** (1 / 2)
            / D_e(smooth_max(c_e_surf))
        )
        self.boundary_conditions = {
            c_s: {"left": (pybamm.Scalar(0), "Neumann"), "right": (rbc_s, "Neumann")},
            c_e: {"left": (lbc_e, "Neumann"), "right": (pybamm.Scalar(1), "Dirichlet")},
        }

        self.initial_conditions = {c_s: c_s0, c_e: c_e0}

        c_s_surf = pybamm.BoundaryValue(c_s, "right")
        c_e_surf = pybamm.BoundaryValue(c_e, "left")

        ######################
        # (Some) variables
        ######################
        self.variables = {
            "Electrode concentration [mol.m-3]": c_s,
            "Electrolyte concentration [mol.m-3]": c_e,
            "Concentration [mol.m-3]": pybamm.concatenation(c_s, c_e),
            "Surface electrode concentration [mol.m-3]": c_s_surf,
            "Surface electrolyte concentration [mol.m-3]": c_e_surf,
            "Time [s]": pybamm.t,
            "Time [min]": pybamm.t / 60,
            "r_s [m]": self.r_s,
            "r_s [um]": self.r_s * 1e6,
            "r_e [m]": self.r_e,
            "r_e [um]": self.r_e * 1e6,
        }

    @property
    def default_geometry(self):
        L_s = pybamm.Parameter("Electrode thickness [m]")
        L_e = pybamm.Parameter("Electrolyte thickness [m]")
        return pybamm.Geometry(
            {
                "electrode": {self.r_s: {"min": pybamm.Scalar(0), "max": L_s}},
                "electrolyte": {self.r_e: {"min": L_s, "max": L_s + L_e}},
            }
        )

    @property
    def default_submesh_types(self):
        return {
            "electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "electrolyte": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
        }

    @property
    def default_var_pts(self):
        return {self.r_s: 50, self.r_e: 50}

    @property
    def default_spatial_methods(self):
        return {
            "electrode": pybamm.FiniteVolume(),
            "electrolyte": pybamm.FiniteVolume(),
        }

    @property
    def default_solver(self):
        # return pybamm.IDAKLUSolver()
        return pybamm.CasadiSolver("fast", rtol=1e-6, atol=1e-6)
