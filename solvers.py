from scipy.optimize import fsolve # fsolve(f, x0=x0, fprime=jacobian)
import casadi as ca

def fsolve_casadi(casadi_obj, sym_x, sym_p, x0, p, jac_fun = None):
    jac = jac_fun
    fun = casadi_obj
    if isinstance(casadi_obj, ca.SX) or isinstance(casadi_obj, ca.MX):
        fun = ca.Function()
