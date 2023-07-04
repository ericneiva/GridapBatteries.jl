#
# Contact lens curing temperature dependent model
#
import pybamm


class DiffusionOnly(pybamm.models.base_model.BaseModel):
    def __init__(self, name="diffusion only"):
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
        self.x_s = pybamm.SpatialVariable(
            "x_s", domain="electrode", coord_sys="cartesian"
        )
        self.x_e = pybamm.SpatialVariable(
            "x_e", domain="electrolyte", coord_sys="cartesian"
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
            -k
            * ((c_e_surf / c_e0) * (c_s_surf / c_s_max) * (1 - c_s_surf / c_s_max))
            ** (1 / 2)
            / D_s(c_s_surf)
        )
        lbc_e = -(
            (1 - t_plus)
            * k
            * ((c_e_surf / c_e0) * (c_s_surf / c_s_max) * (1 - c_s_surf / c_s_max))
            ** (1 / 2)
            / D_e(c_e_surf)
        )
        self.boundary_conditions = {
            c_s: {"left": (pybamm.Scalar(0), "Neumann"), "right": (rbc_s, "Neumann")},
            c_e: {"left": (lbc_e, "Neumann"), "right": (pybamm.Scalar(0), "Neumann")},
        }

        self.initial_conditions = {c_s: c_s0, c_e: c_e0}

        ######################
        # (Some) variables
        ######################
        self.variables = {
            "Electrode concentration [mol.m-3]": c_s,
            "Electrolyte concentration [mol.m-3]": c_e,
            "Time [s]": pybamm.t,
            "Time [min]": pybamm.t / 60,
            "x_s [m]": self.x_s,
            "x_s [um]": self.x_s * 1e6,
            "x_e [m]": self.x_e,
            "x_e [um]": self.x_e * 1e6,
        }

    @property
    def default_geometry(self):
        L_s = pybamm.Parameter("Electrode thickness [m]")
        L_e = pybamm.Parameter("Electrolyte thickness [m]")
        return pybamm.Geometry(
            {
                "electrode": {self.x_s: {"min": pybamm.Scalar(0), "max": L_s}},
                "electrolyte": {self.x_e: {"min": L_s, "max": L_s + L_e}},
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
        return {self.x_s: 50, self.x_e: 50}

    @property
    def default_spatial_methods(self):
        return {
            "electrode": pybamm.FiniteVolume(),
            "electrolyte": pybamm.FiniteVolume(),
        }

    @property
    def default_solver(self):
        # return pybamm.IDAKLUSolver()
        return pybamm.CasadiSolver("fast")
