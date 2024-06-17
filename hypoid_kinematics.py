from casadi.casadi import exp
import numpy as np
import casadi as ca
import screwCalculus as sc
from scipy.optimize import fsolve
from math import sqrt, pi, atan, cos, sin, acos, asin, tan
from utils import *
from hypoid_utils import *

"""
This package encloses the basic functions related to the gleason's facemilling hypoid generator kinematics.
It contains cradle-style (9 DOF) kinematic functions and gridning wheel (tool) geometry functions."""

def gear_tool_kinem(joint_values):

    ggt0 = np.array([[0, 0, -1, 0],
                     [0, 1, 0, 0],
                     [1, 0, 0, 0],
                     [0, 0, 0, 1]
                     ])
    
    Y = np.zeros((6,9))
    Y[:, 0] = np.array([0,0,0,0,0,-1])
    Y[:, 1] = np.array([0,0,-1,0,0,0])
    Y[:, 2] = np.array([0,0,0,0,1,0])
    Y[:, 3] = np.array([1,0,0,0,0,0])
    Y[:, 4] = np.array([0,1,0,0,0,0])
    Y[:, 5] = np.array([0,0,0,-1,0,0])
    Y[:, 6] = np.array([0,0,1,0,0,0])
    Y[:, 7] = np.array([0,0,0,1,0,0])
    Y[:, 8] = np.array([0,0,0,0,0,1])

    q = joint_values[[6,7,8,4,3,5,0,2,1]]

    helix = np.array([0., np.inf, 0, np.inf, np.inf, 0, np.inf, 0, 0])
    return sc.FWkin_globalPOE(ggt0, Y, q, helix)

def gear_tool_twist(joint_values, joint_vel):

    ggt0 = np.array([[0, 0, -1, 0],
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1]
                    ])
    
    Y = np.zeros((6,9))
    Y[:, 0] = np.array([0,0,0,0,0,-1])
    Y[:, 1] = np.array([0,0,-1,0,0,0])
    Y[:, 2] = np.array([0,0,0,0,1,0])
    Y[:, 3] = np.array([1,0,0,0,0,0])
    Y[:, 4] = np.array([0,1,0,0,0,0])
    Y[:, 5] = np.array([0,0,0,-1,0,0])
    Y[:, 6] = np.array([0,0,1,0,0,0])
    Y[:, 7] = np.array([0,0,0,1,0,0])
    Y[:, 8] = np.array([0,0,0,0,0,1])

    q = joint_values[[6,7,8,4,3,5,0,2,1]]
    helix = np.array([0., np.inf, 0, np.inf, np.inf, 0, np.inf, 0, 0])
    q_dot = joint_vel[[6,7,8,4,3,5,0,2,1]]

    J = sc.bodyJac_globalPOE(ggt0, Y, q, helix)

    return sc.hat(J@q_dot)

def gear_tool_twist_spatial(joint_values, joint_vel):

    # offset matrix
    ggt0 = np.array([
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
        ])
    
    # unitary twists
    Y = np.zeros((6,9))
    Y[:, 0] = np.array([0,0,0,0,0,-1])
    Y[:, 1] = np.array([0,0,-1,0,0,0])
    Y[:, 2] = np.array([0,0,0,0,1,0])
    Y[:, 3] = np.array([1,0,0,0,0,0])
    Y[:, 4] = np.array([0,1,0,0,0,0])
    Y[:, 5] = np.array([0,0,0,-1,0,0])
    Y[:, 6] = np.array([0,0,1,0,0,0])
    Y[:, 7] = np.array([0,0,0,1,0,0])
    Y[:, 8] = np.array([0,0,0,0,0,1])

    # joint sequence and helix lead (prismatic = inf, revolute = 0, helical = finite>0)
    q = joint_values[[6,7,8,4,3,5,0,2,1]]
    helix = np.array([0., np.inf, 0, np.inf, np.inf, 0, np.inf, 0, 0])
    q_dot = joint_vel[[6,7,8,4,3,5,0,2,1]]

    J = sc.spatialJac_globalPOE(Y, q, helix)

    return sc.hat(J@q_dot)

def machine_kinematics(machineParMatrix):
    """
    Numeric function that builds the homogeneous transorm function and the spatial twists, given a set of machine settings
    """
    # kinematic map
    ggt = lambda phi: gear_tool_kinem ( 
        machineParMatrix @ np.array([1,phi,phi**2,phi**3,phi**4,phi**5,phi**6,phi**7])
        )
    
    # body twist
    Vgt_body = lambda phi: gear_tool_twist(machineParMatrix @ np.array([1,phi,phi**2,phi**3,phi**4,phi**5,phi**6,phi**7]),\
                                machineParMatrix @ np.array([0,   1, 2.*phi, 3.*phi**2, 4.*phi**3, 5.*phi**4, 6.*phi**5, 7.*phi**6])
                                )
    
    # spatial twist
    Vgt_spatial = lambda phi: gear_tool_twist_spatial(machineParMatrix @ np.array([1,phi,phi**2,phi**3,phi**4,phi**5,phi**6,phi**7]),\
                                       machineParMatrix @ np.array([0,   1, 2.*phi, 3.*phi**2, 4.*phi**3, 5.*phi**4, 6.*phi**5, 7.*phi**6])
                                       )
    
    return ggt, Vgt_body, Vgt_spatial

def casadi_machine_kinematics(member, systemHand, casadi_type = 'SX'):
    """
    Casadi variant of the machine_kinematics function. It allows to define the kinematic map with symbolic machine settings
    """
    casadi_var = getattr(ca, casadi_type)
    machine_matrix = casadi_var.sym('M', 9, 8)
    auxiliary_mat = machine_matrix
    [cMat, sMat] = manageMachinePar(member, systemHand)
    cMat = cMat*sMat
    roll = auxiliary_mat[6,1]
    auxiliary_mat = cMat*auxiliary_mat
    auxiliary_mat[6, 2:-1] = auxiliary_mat[6, 2:-1] * roll

    phi = casadi_var.sym('phi')

    ggt, Vgt, Vgt_spatial = machine_kinematics(auxiliary_mat)

    ggt = ca.Function('ggt', [machine_matrix, phi], [ggt(phi)])
    Vgt = ca.Function('Vgt', [machine_matrix, phi], [Vgt(phi)])
    Vgt_spatial = ca.Function('Vgt_spatial', [machine_matrix, phi], [Vgt_spatial(phi)])
    return ggt, Vgt, Vgt_spatial

def tool_casadi_blade(flank, settings):
    
    # reading parameter settings
    Rp = settings[0]  # point radius
    rho = settings[1]  # spherical radius
    rhof = settings[2]  # fillet radius
    alphap = settings[5] * np.pi / 180  # blade angle

    # Pre-computing of geometric intermediate parameters
    if flank.lower() == 'concave':
        # CONCAVE parameters
        Czblade = np.sin(alphap) * rho  # Ccz e Ccx coord. x e z del centro dell'arco di circ. del BLADE
        Cxblade = Rp + rho * np.cos(alphap)
        theta_iniz_blade = np.arcsin((Czblade + rhof) / (rhof + rho))  # angolo di inizio arco
        theta_fin_ER = np.pi / 2 - theta_iniz_blade
        Xf = Cxblade - (rho + rhof) * np.cos(theta_iniz_blade)
        L = theta_fin_ER * rhof
        points = lambda csi, theta: pointsCoNcave(csi, theta)
        normals = lambda csi, theta: normalsCoNcave(csi, theta)

    elif flank.lower() == 'convex':
        # CONVEX parameters
        Czblade = np.sin(alphap) * rho
        Cxblade = Rp - rho * np.cos(alphap)
        theta_iniz_blade = np.arcsin((Czblade + rhof) / (rhof + rho))
        theta_fin_ER = np.pi / 2 - theta_iniz_blade
        Xf = Cxblade + (rho + rhof) * np.cos(theta_iniz_blade)
        L = theta_fin_ER * rhof
        points = lambda csi, theta: pointsConveX(csi, theta)
        normals = lambda csi, theta: normalsConveX(csi, theta)

    else:
        raise ValueError("tool flank was not specified correctly")

    end1 = theta_fin_ER * rhof

    # OUTPUT FUNCTIONS
    def pointsCoNcave(csi, theta):  # CONCAVE
        bool1 = csi < end1
        bool2 = csi >= end1

        angle = bool1 * (csi / rhof) + bool2 * ((csi - end1) / rho + theta_iniz_blade)

        sa = np.sin(angle)
        ca = np.cos(angle)

        Xtoolcurv = bool1 * (Xf + rhof * sa) + bool2 * (Cxblade - rho * ca)
        Ztoolcurv = bool1 * (rhof * (ca - 1)) + bool2 * (Czblade - rho * sa)

        Xtool = Xtoolcurv * np.cos(theta)
        Ytool = Xtoolcurv * np.sin(theta)
        Ztool = Ztoolcurv

        return np.array([Xtool, Ytool, Ztool])

    def normalsCoNcave(csi, theta):
        bool1 = csi < end1
        bool2 = csi >= end1
        boolTip = (csi >= end1)

        angle = bool1 * (csi / rhof) + bool2 * ((csi - end1) / rho + theta_iniz_blade)

        sa = np.sin(angle)
        ca = np.cos(angle)

        nXtoolcurv = bool1 * (sa) + boolTip * (ca)
        nZtoolcurv = bool1 * (ca) + boolTip * (sa)

        nXtool = nXtoolcurv * np.cos(theta)
        nYtool = nXtoolcurv * np.sin(theta)
        nZtool = nZtoolcurv

        return np.array([nXtool, nYtool, nZtool])

    def pointsConveX(csi, theta):  # CONVEX
        bool1 = csi < end1
        bool2 = csi >= end1

        angle = bool1 * (csi / rhof) + bool2 * ((csi - end1) / rho + theta_iniz_blade)

        sa = np.sin(angle)
        ca = np.cos(angle)

        Xtoolcurv = bool1 * (Xf - rhof * sa) + bool2 * (Cxblade + rho * ca)
        Ztoolcurv = bool1 * (rhof * (ca - 1)) + bool2 * (Czblade - rho * sa)

        Xtool = Xtoolcurv * np.cos(theta)
        Ytool = Xtoolcurv * np.sin(theta)
        Ztool = Ztoolcurv

        return np.array([Xtool, Ytool, Ztool])

    def normalsConveX(csi, theta):
        bool1 = csi < end1
        bool2 = csi >= end1
        boolTip = (csi >= end1)

        angle = bool1 * (csi / rhof) + bool2 * ((csi - end1) / rho + theta_iniz_blade)

        sa = np.sin(angle)
        ca = np.cos(angle)

        nXtoolcurv = bool1 * (-sa) + boolTip * (-ca)
        nZtoolcurv = bool1 * (ca) + boolTip * (sa)

        nXtool = nXtoolcurv * np.cos(theta)
        nYtool = nXtoolcurv * np.sin(theta)
        nZtool = nZtoolcurv

        return np.array([nXtool, nYtool, nZtool])

    return points, normals, L

def tool_casadi(flank, settings, topremCheck = True):
    # Rp: Concave Cutter point radius IF 'concave'
    #     Convex Cutter point radius  IF 'convex'
    #
    # rhof: Edge radius.
    #
    # rhob: Blend radius.
    #
    # alphap: Blade angle.
    #
    # rho: Blade radius.
    #
    # stf: Toprem depth   IF 'toprem'.
    #      Flankrem depth IF 'flankrem.
    #
    # the "casadi" version of this function manages the the profile curves transition with boolean operators instead of if/elses

    # Verifica parametri compatibili con la generazione del profilo
    Rp = settings[0]  # point radius
    rho = settings[1]  # spherical radius
    rhof = settings[2]  # fillet radius
    rhotop = settings[3]  # toprem radius
    rhoflank = settings[4]  # flankrem radius
    alphap = settings[5] * np.pi / 180  # blade angle
    stfflank = settings[6]  # flankrem depth
    stftop = settings[7]  # toprem depth
    alphaTop = settings[8] * np.pi / 180  # toprem angle
    alphaFlank = settings[9] * np.pi / 180  # flankrem angle
    
    if topremCheck == False:
        Czblade = rho * np.sin(alphap)
        theta_iniz_blade = np.arcsin((Czblade + rhof) / (rho + rhof))
        stftop = rhof - rhof * np.sin(theta_iniz_blade)  # ***!!!*** THIS CONSIDERING alphaTop = 0
        rhotop = 10
    
    # Inizializzazione parametri geometrici
    if flank.strip().lower() == 'concave':
        # Parametri CONCAVE
        
        Czblade = np.sin(alphap) * rho  # Ccz e Ccx coord. x e z del centro dell'arco di circ. del BLADE
        Cxblade = Rp + rho * np.cos(alphap)
        theta_blade_fin = np.arcsin((Czblade + stfflank) / rho)
        theta_iniz_blade = np.arcsin((Czblade + stftop) / rho)  # angolo di inizio arco
        
        theta_fin_Top = theta_iniz_blade - alphaTop
        Cztop = -stftop + rhotop * np.sin(theta_fin_Top)
        Cxtop = Cxblade - rho * np.cos(theta_iniz_blade) + rhotop * np.cos(theta_fin_Top)
        theta_iniz_top = np.arcsin((Cztop + rhof) / (rhof + rhotop))
        
        theta_fin_ER = np.pi / 2 - theta_iniz_top
        Xf = Cxtop - rhotop * np.cos(theta_iniz_top) - rhof * np.sin(theta_fin_ER)
        
        theta_iniz_flank = theta_blade_fin + alphaFlank
        Cxflank = Cxblade + np.cos(theta_iniz_flank) * rhoflank - rho * np.cos(theta_blade_fin)
        Czflank = -stfflank + np.sin(theta_iniz_flank) * rhoflank
        
        points = lambda csi, theta: pointsCoNcave(csi, theta)
        normals = lambda csi, theta: normalsCoNcave(csi, theta)
    
    elif flank.strip().lower() == "convex":
        # Parametri CONVEX
        
        Czblade = np.sin(alphap) * rho
        Cxblade = Rp - rho * np.cos(alphap)
        theta_iniz_blade = np.arcsin((Czblade + stftop) / rho)
        theta_blade_fin = np.arcsin((Czblade + stfflank) / rho)
        
        theta_fin_Top = theta_iniz_blade - alphaTop
        Cztop = -stftop + rhotop * np.sin(theta_fin_Top)
        Cxtop = Cxblade + rho * np.cos(theta_iniz_blade) - rhotop * np.cos(theta_fin_Top)
        theta_iniz_top = np.arcsin((Cztop + rhof) / (rhof + rhotop))
        
        theta_fin_ER = np.pi / 2 - theta_iniz_top
        theta_iniz_flank = theta_blade_fin + alphaFlank
        Cxflank = Cxblade - np.cos(theta_iniz_flank) * rhoflank + rho * np.cos(theta_blade_fin)
        Czflank = -stfflank + np.sin(theta_iniz_flank) * rhoflank
        Xf = Cxtop + rhotop * np.cos(theta_iniz_top) + rhof * np.sin(theta_fin_ER)

        points = lambda csi, theta: pointsConveX(csi, theta)
        normals = lambda csi, theta: normalsConveX(csi, theta)
    
    else:
        raise ValueError("Incorrect flank specified")
    
    L = theta_fin_ER * rhof
    end1 = theta_fin_ER * rhof
    end2 = (theta_fin_Top - theta_iniz_top) * rhotop + end1
    end3 = (theta_blade_fin - theta_iniz_blade) * rho + end2
    
    # FUNZIONI
    
    def pointsCoNcave(csi, theta):
        bool1 = csi < end1
        bool2 = (csi >= end1) & (csi < end2)
        bool3 = (csi >= end2) & (csi < end3)
        bool4 = csi >= end3
        
        angle = bool1 * (csi / rhof) + \
                bool2 * ((csi - end1) / rhotop + theta_iniz_top) + \
                bool3 * ((csi - end2) / rho + theta_iniz_blade) + \
                bool4 * ((csi - end3) / rhoflank + theta_iniz_flank)
        
        sa = np.sin(angle)
        ca = np.cos(angle)
        
        Xtoolcurv = bool1 * (Xf + rhof * sa) + \
            bool2 * (Cxtop - rhotop * ca) + \
            bool3 * (Cxblade - rho * ca) + \
            bool4 * (Cxflank - rhoflank * ca)
        
        Ztoolcurv = bool1 * (rhof * (ca - 1)) + \
            bool2 * (Cztop - rhotop * sa) + \
            bool3 * (Czblade - rho * sa) + \
            bool4 * (Czflank - rhoflank * sa)
        
        Xtool = Xtoolcurv * np.cos(theta)
        Ytool = Xtoolcurv * np.sin(theta)
        Ztool = Ztoolcurv
        
        return np.array([Xtool, Ytool, Ztool])
    
    def normalsCoNcave(csi, theta):
        bool1 = csi < end1
        bool2 = (csi >= end1) & (csi < end2)
        bool3 = (csi >= end2) & (csi < end3)
        bool4 = csi >= end3
        boolTip = (csi >= end1)
        
        angle = bool1 * (csi / rhof) + \
                bool2 * ((csi - end1) / rhotop + theta_iniz_top) + \
                bool3 * ((csi - end2) / rho + theta_iniz_blade) + \
                bool4 * ((csi - end3) / rhoflank + theta_iniz_flank)
        
        sa = np.sin(angle)
        ca = np.cos(angle)
        
        nXtoolcurv = bool1 * (sa) + \
                     boolTip * (ca)
        
        nZtoolcurv = bool1 * (ca) + \
                     boolTip * (sa)
        
        nXtool = nXtoolcurv * np.cos(theta)
        nYtool = nXtoolcurv * np.sin(theta)
        nZtool = nZtoolcurv
        
        return np.array([nXtool, nYtool, nZtool])
    
    def pointsConveX(csi, theta):
        bool1 = csi < end1
        bool2 = (csi >= end1) & (csi < end2)
        bool3 = (csi >= end2) & (csi < end3)
        bool4 = csi >= end3
        
        angle = bool1 * (csi / rhof) + \
                bool2 * ((csi - end1) / rhotop + theta_iniz_top) + \
                bool3 * ((csi - end2) / rho + theta_iniz_blade) + \
                bool4 * ((csi - end3) / rhoflank + theta_iniz_flank)
        
        sa = np.sin(angle)
        ca = np.cos(angle)
        
        Xtoolcurv = bool1 * (Xf - rhof * sa) + \
            bool2 * (Cxtop + rhotop * ca) + \
            bool3 * (Cxblade + rho * ca) + \
            bool4 * (Cxflank + rhoflank * ca)
        
        Ztoolcurv = bool1 * (rhof * (ca - 1)) + \
            bool2 * (Cztop - rhotop * sa) + \
            bool3 * (Czblade - rho * sa) + \
            bool4 * (Czflank - rhoflank * sa)
        
        Xtool = Xtoolcurv * np.cos(theta)
        Ytool = Xtoolcurv * np.sin(theta)
        Ztool = Ztoolcurv
        
        return np.array([Xtool, Ytool, Ztool])
    
    def normalsConveX(csi, theta):
        bool1 = csi < end1
        bool2 = (csi >= end1) & (csi < end2)
        bool3 = (csi >= end2) & (csi < end3)
        bool4 = csi >= end3
        boolTip = (csi >= end1)
        
        angle = bool1 * (csi / rhof) + \
                bool2 * ((csi - end1) / rhotop + theta_iniz_top) + \
                bool3 * ((csi - end2) / rho + theta_iniz_blade) + \
                bool4 * ((csi - end3) / rhoflank + theta_iniz_flank)
        
        sa = np.sin(angle)
        ca = np.cos(angle)
        
        nXtoolcurv = bool1 * (-sa) + \
                     boolTip * (-ca)
        
        nZtoolcurv = bool1 * (ca) + \
                     boolTip * (sa)
        
        nXtool = nXtoolcurv * np.cos(theta)
        nYtool = nXtoolcurv * np.sin(theta)
        nZtool = nZtoolcurv
        
        return np.array([nXtool, nYtool, nZtool])
    
    return points, normals, L

def casadi_tool_fun(flank, toprem = None, flankrem = None, casadi_type = 'SX'):
    casadi_var = getattr(ca, casadi_type)
    tool_settings = casadi_var.sym('T', 10, 1)
    # tool surface variables
    csi = casadi_var.sym('csi');
    theta = casadi_var.sym('theta');
    if toprem == None and flankrem == None:
        p, n, L = tool_casadi_blade(flank, tool_settings)
    else:
        p, n, L = tool_casadi(flank, tool_settings, topremCheck = True)

    p_fun = ca.Function('p',  [tool_settings, ca.vertcat(csi, theta)], [p(csi, theta)])
    n_fun = ca.Function('n',  [tool_settings, ca.vertcat(csi, theta)], [n(csi, theta)])
    L_fun = ca.Function('p',  [tool_settings], [L])
    return p_fun, n_fun, L_fun

def parametric_tool_casadi(flank, Rp, RHO, alpha, edgeRadius, csi, theta):
    [pTool, nTool, L] = casadi_tool_fun(flank, toprem = None, flankrem = None)
    toolvec = np.array([Rp, RHO, edgeRadius, 9000, 9000, alpha, 100, 0, 0, 0])
    point = pTool(toolvec, np.array([csi,theta]))
    normal = nTool(toolvec, np.array([csi, theta]))
    return point, normal

def main():
    toolvec = [50, 150, 1, 0, 300, 20, 0, 0, 0, 0]
    p_fun, n_fun, L_fun = casadi_tool_fun('concave', None, None, 'SX')
    print(p_fun(toolvec, np.array([[1,0], [1,0]])))
    print(n_fun(toolvec, np.array([[1,0], [1,0]])))
    pp, nn = parametric_tool_casadi('concave', toolvec[0], toolvec[1], toolvec[5], toolvec[2], 1, 1)
    print(pp.full())

if __name__ == '__main__':
    main()


