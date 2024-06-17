from casadi.casadi import exp
import numpy as np
import casadi as ca
from scipy.optimize import fsolve
from math import sqrt, pi, atan, cos, sin, acos, asin, tan
from utils import *


# import our packages
from hypoid_kinematics import *
from hypoid import *
from hypoid_utils import *
import screwCalculus as sc

""" Note: after completing the packages consider merging "hypoid_functions" with "hypoid_utils" """


def approxToolIdentification_casadi(data, member, RHO = None):

    RHOinput = RHO
    
    if member.lower() == 'pinion':
        cutterFieldName, subCutterFieldName, commonFieldName, subCommonFieldName, _, _ =\
            get_data_field_names('pinion', 'concave')
        cutterConvexFieldName, subConvexCutterFieldName, _, _, _, _ =\
            get_data_field_names('pinion', 'convex')
    else:
        cutterFieldName, subCutterFieldName, commonFieldName, subCommonFieldName, _, _ =\
              get_data_field_names('gear', 'concave')
        cutterConvexFieldName, subConvexCutterFieldName, _, _, _, _ =\
              get_data_field_names('gear', 'convex')
    
    # extract data from struct
    nT         = data[commonFieldName][f'{subCommonFieldName}NTEETH']
    edgeRadius = data[cutterFieldName][f'{subCutterFieldName}EDGERADIUS']
    rc0        = data[commonFieldName][f'{subCommonFieldName}MEANCUTTERRAIDUS']            # mean cutter radius
    alphanD    = data['SystemData']['NOMINALDRIVEPRESSUREANGLE']                             # drive side tooth pressure angle
    alphanC    = data['SystemData']['NOMINALCOASTPRESSUREANGLE']                             # coast side tooth pressure angle
    hamc       = data[commonFieldName][f'{subCommonFieldName}MEANCHORDALADDENDUM']         # tooth mean chordal addendum
    ham        = data[commonFieldName][f'{subCommonFieldName}MEANADDENDUM']               # tooth mean addendum
    t          = data[commonFieldName][f'{subCommonFieldName}MEANNORMALCHORDALTHICKNESS']   # tooth normal chordal thickness
    Rm         = data[commonFieldName][f'{subCommonFieldName}MEANCONEDIST']                # mean cone distance
    pitchapex  = data[commonFieldName][f'{subCommonFieldName}PITCHAPEX']
    delta      = data[commonFieldName][f'{subCommonFieldName}PITCHANGLE']
    betam      = data[commonFieldName][f'{subCommonFieldName}SPIRALANGLE']
    RHO        = data[cutterFieldName][f'{subCutterFieldName}RHO']
    hand       = data['SystemData']['HAND']
    mmn        = data['SystemData']['NORMALMODULE']

    if data[cutterFieldName][f'{subCutterFieldName}TYPE'].lower() == 'straight':
        RHO = 20e3
    
    if RHOinput is not None:
        RHO = RHOinput
        data[cutterFieldName][f'{subCutterFieldName}RHO'] = RHO

    if (hand.lower() == 'right' and member.lower() == 'gear') or (hand.lower() == 'left' and member.lower() == 'pinion'):
        Tng = lambda phi2: sc.TrotZ(phi2)@sc.TtZ(-pitchapex)@sc.TrotY(delta*pi/180)@sc.TtZ(Rm)@sc.TrotX(betam*pi/180)
        signThick = +1
        rotguess = 2*pi/nT
    else:
        Tng = lambda phi2: sc.TrotZ(phi2)@sc.TtZ(-pitchapex)@sc.TrotY(delta*pi/180)@sc.TtZ(Rm)@sc.TrotX(-betam*pi/180)
        signThick = -1
        rotguess = -2*pi/nT

    if member.lower() == 'gear':
        pressAngCvx = alphanD # drive should always be convex for gear and concave for pinion, but it may happen to have inverse situations
        pressAngCnv = alphanC 
    else:
        pressAngCnv = alphanD
        pressAngCvx = alphanC

    machinePar = assignMachinePar(data, member, 'concave')
    cMat, sMat = manageMachinePar(member, hand)
    machineParMatrix = cMat*sMat*machinePar

    # Kinematic computation
    ggt, Vgt, Vgt_spatial = machine_kinematics(machineParMatrix)

    triplet = initial_guess_from_data(data, member, 'convex')
    csiguessCVX = mmn*2
    thetaguessCVX = triplet[1]
    phiguessCVX = triplet[2]
    triplet = initial_guess_from_data(data, member, 'concave')
    csiguessCNV = mmn*2
    thetaguessCNV = triplet[1]
    phiguessCNV = triplet[2]

    x0 = [rc0-t, pressAngCvx, pressAngCnv, csiguessCNV, thetaguessCNV, csiguessCVX, thetaguessCVX, csiguessCNV, thetaguessCNV, csiguessCVX, thetaguessCVX,\
    rotguess, phiguessCNV, phiguessCVX, phiguessCNV, phiguessCVX, rc0+t]
    csiLow = mmn*0.1
    csiMax = mmn
    lb = [x0[0]-rc0/4, x0[1] - 10, x0[2] - 10, csiLow, x0[4] - pi/2, csiLow, x0[6] - pi/2, csiLow, x0[8] - pi/2, csiLow, x0[10] - pi/2, x0[11] - 5*pi/nT, x0[12] - pi/2, x0[13] - pi/2, x0[14] - pi/2, x0[15] - pi/2, x0[16]-rc0/4]
    ub = [x0[0]+rc0/4, x0[1] + 10, x0[2] + 10, x0[3] + csiMax, x0[4] + pi/2, x0[5] + csiMax , x0[6] + pi/2, x0[7] + csiMax, x0[8] + pi/2, x0[9] + csiMax, x0[10] + pi/2, x0[11] + 5*pi/nT, x0[12] + pi/2, x0[13] + pi/2, x0[14] + pi/2, x0[15] + pi/2, x0[16]+rc0/4]
    lb = np.array(lb)
    ub = np.array(ub)
    x0 = np.array(x0)

    x_unscaled = ca.SX.sym('x', 17+3*4+3*4, 1)
    x = x_unscaled * (ub - lb) + lb
    Rpcvx = x[0]
    alphacvx = x[1]
    alphacnv = x[2]
    csiI = x[3]  # concave
    thetaI = x[4]
    csiO = x[5]  # convex
    thetaO = x[6]
    csiIprime = x[7]
    thetaIprime = x[8]
    csiOprime = x[9]
    thetaOprime = x[10]
    phi2 = x[11]
    phiEnvIprime = x[12]
    phiEnvOprime = x[13]
    phiEnvI = x[14]
    phiEnvO = x[15]
    Rpcnv = x[16]
    pO = x[17:20]
    pI = x[20:23]
    pOp = x[23:26]
    pIp = x[26:29]
    nO = x[17:20]
    nI = x[20:23]
    nOp = x[23:26]
    nIp = x[26:29]

    toolO, toolNO = parametric_tool_casadi('convex', Rpcvx, RHO, alphacvx, edgeRadius, csiO, thetaO)
    toolI, toolNI = parametric_tool_casadi('concave', Rpcnv, RHO, alphacnv, edgeRadius, csiI, thetaI)

    T = sc.rigidInverse(Tng(phi2))
    pointO = T @ sc.TrotZ(signThick * 2 * np.pi / nT) @ ggt(phiEnvO) @ ca.vertcat(toolO, 1)
    pointI = T @ ggt(phiEnvI) @ ca.vertcat(toolI, 1)

    toolOprime, toolNOprime = parametric_tool_casadi('convex', Rpcvx, RHO, alphacvx, edgeRadius, csiOprime, thetaOprime)
    toolIprime, toolNIprime = parametric_tool_casadi('concave', Rpcnv, RHO, alphacnv, edgeRadius, csiIprime, thetaIprime)
    pointOprime = ggt(phiEnvOprime) @ ca.vertcat(toolOprime, 1)
    normalOprime = ggt(phiEnvOprime) @ ca.vertcat(toolNOprime, 0)
    pointIprime = ggt(phiEnvIprime) @ ca.vertcat(toolIprime, 1)
    normalIprime = ggt(phiEnvIprime) @ ca.vertcat(toolNIprime, 0)

    pointOprime = T @ sc.TrotZ(signThick * 2 * np.pi / nT) @ pointOprime
    normalOprime = T @ sc.TrotZ(signThick * 2 * np.pi / nT) @ normalOprime
    pointIprime = T @ pointIprime
    normalIprime = T @ normalIprime

    out = ca.SX(17, 1)
    out[0:3] = (pO - np.array([-(hamc - ham), +signThick * t / 2, 0]))
    out[3:6] = (pI - np.array([-(hamc - ham), -signThick * t / 2, 0]))
    out[6] = pOp[2]
    out[7] = pOp[0] * nOp[1] - pOp[1] * nOp[0]
    out[8] = ca.cos(pressAngCvx*pi/180) * ca.sqrt(normalOprime[0]**2 + normalOprime[1]**2) + signThick * normalOprime[1]
    out[9] = pIp[2]
    out[10] = pIp[0] * nIp[1] - pIp[1] * nIp[0]
    out[11] = ca.cos(pressAngCnv*pi/180) * ca.sqrt(normalIprime[0]**2 + normalIprime[1]**2) - signThick * normalIprime[1]
    out[12] = ca.vertcat(toolNOprime, 1).T @ Vgt(phiEnvOprime) @ ca.vertcat(toolOprime, 1)
    out[13] = ca.vertcat(toolNIprime, 1).T @ Vgt(phiEnvIprime) @ ca.vertcat(toolIprime, 1)
    out[14] = ca.vertcat(toolNO, 1).T @ Vgt(phiEnvO) @ ca.vertcat(toolO, 1)
    out[15] = ca.vertcat(toolNI, 1).T @ Vgt(phiEnvI) @ ca.vertcat(toolI, 1)
    out[16] = Rpcnv - (2 * rc0 - Rpcvx)
    out[17:20] = pO-pointO[0:3]
    out[17:20] = pI-pointI[0:3]
    out[17:20] = pOp-pointOprime[0:3]
    out[17:20] = pIp-pointIprime[0:3]
    out[17:20] = nOp-pointOprime[0:3]
    out[17:20] = nIp-pointIprime[0:3]

    fun_test = ca.Function('ft', [x_unscaled], [sqrt(normalOprime[0]**2 + normalOprime[1]**2)])
    opts = IPOPT_global_options()
    problem = {'x': x_unscaled, 'g': out, 'f': 0}
    solver = ca.nlpsol('S', 'ipopt', problem, opts)

    x0 = (x0 - lb)/(ub - lb)

    solution  = solver(x0 = x0, lbx = lb*0, ubx = lb*0+1, ubg = 0, lbg = 0)
    res = solution['x'].full().squeeze()
    res = res*(ub-lb) + lb

    data[cutterFieldName][f'{subCutterFieldName}RHO'] = RHO
    data[cutterConvexFieldName][f'{subConvexCutterFieldName}RHO'] = RHO
    data[cutterConvexFieldName][f'{subConvexCutterFieldName}POINTRADIUS'] = res[0]
    data[cutterConvexFieldName][f'{subConvexCutterFieldName}BLADEANGLE'] = res[1]
    data[cutterFieldName][f'{subCutterFieldName}BLADEANGLE'] = res[2]
    data[cutterFieldName][f'{subCutterFieldName}POINTRADIUS'] = res[-1]
    edge_radius = (res[-1] - res[0])/2.5
    data[cutterFieldName][f'{subCutterFieldName}EDGERADIUS'] = edge_radius
    data[cutterConvexFieldName][f'{subConvexCutterFieldName}EDGERADIUS'] = edge_radius
    triplet_concave = [res[3], res[4], res[14]]
    triplet_convex = [res[5], res[6], res[15]]

    return data, triplet_concave, triplet_convex

def AGMAcomputationHypoid(Hand, taper, initialConeData, toothInitialData, Method = 1, rc0 = None, GearGenType = "Generated"):
    
    if rc0 is None or rc0 is np.nan:
        rc0 = 0
        rc0Flag = None

    uniformToothCoeff = 1

    # extract initial data for cone parameters determination
    SIGMA = initialConeData["SIGMA"]*pi/180         # shaft angle
    a = initialConeData["a"]                        # hypoid offset
    z1 = initialConeData["z1"]                      # pinion teeth
    z2 = initialConeData["z2"]                      # gear teeth
    u = initialConeData["u"]                        # transmission ratio
    de2 = initialConeData["de2"]                    # gear outer diameter
    b2 = initialConeData["b2"]                      # facewidth
    betam1 = initialConeData["betam1"]*pi/180       # pinion spiral angle

    # extract initial data for tooth dimensions
    alphadD = toothInitialData["alphaD"]*pi/180     # nominal design pressure angle drive side
    alphadC =  toothInitialData["alphaC"]*pi/180    # nominal design pressure angle coast side
    falphalim =  toothInitialData["falphalim"]      # influence factor of limit pressure angle
    khap = toothInitialData["khap"]                 # addendum factor
    khfp = toothInitialData["khfp"]                 # dedendum factor
    xhm1 = toothInitialData["xhm1"]                 # profile shift coefficient
    jen  = toothInitialData["jen"]                  # outer normal backlash
    xsmn = toothInitialData["xsmn"]                 # thickness modification coefficient

    DeltaSIGMA = SIGMA-pi/2                

    # method 0 (spiral bevel gears)
    delta1 = atan(sin(SIGMA)/(cos(SIGMA) + u))    # pinion pith angle
    delta2 = SIGMA - delta1                       # gear pitch angle
    Re2 = de2/(2.*sin(delta2))                    # outer cone distance (equal for both members)
    Re1 = Re2                                     # outer cone distance (equal for both members)
    Rm2 = Re2 - b2/2                              # mean cone distance (equal for both members)
    Rm1 = Rm2 
    betam2 = betam1                              # spiral angle (equal for both members)
    cbe2 = 0.5                                   # face width factor (equal for both members)

    if Method == 1: # hypoid gears
        betaDelta1 = betam1
        DeltaSIGMA = SIGMA - pi/2
        deltaint2 = atan(u*cos(DeltaSIGMA)/2/(1-u*sin(DeltaSIGMA)))
        rmpt2 = (de2 - b2*sin(deltaint2))/2
        epsiprime = asin(a*sin(deltaint2)/rmpt2)
        K1 = tan(betaDelta1)*sin(epsiprime) + cos(epsiprime)
        rmn1 = rmpt2*K1/u
        ni0 = atan(a/(rmpt2*(tan(deltaint2)*cos(DeltaSIGMA) - sin(DeltaSIGMA)) + rmn1)) 
        
        # starting guess value for the iterative process of the gear offset angle in axial plane

        # intermediate pinion offset angle in axial plane
        eps2 = lambda ni: (a - rmn1*sin(ni))/rmpt2 
        # intermediate pinion pitch angle
        deltaint1 = lambda ni: atan(sin(ni)/(tan(eps2(ni))*cos(DeltaSIGMA)) + tan(DeltaSIGMA)*cos(ni)) 
        # intermediate pinion offset angle in pitch plane
        eps2prime = lambda ni: asin( sin(eps2(ni))*cos(DeltaSIGMA)/cos(deltaint1(ni)) ) 
        # intermediate pinion mean spirla angle
        betamint1 = lambda ni: atan( (K1 - cos(eps2prime(ni)) )/sin(eps2prime(ni)) ) 
        # increment in hypoid dimension factor
        deltaK = lambda ni: sin(eps2prime(ni))*( tan(betaDelta1) - tan(betamint1(ni)) )
        # pin mean radius increment
        Deltarmpt1 = lambda ni: rmpt2*deltaK(ni)/u 
        # pinion offset angle in axial plane
        eps1 = lambda ni: asin(sin(eps2(ni)) - Deltarmpt1(ni)/rmpt2*sin(ni)) 
        # pinion pitch angle
        delta1 = lambda ni: atan(sin(ni)/(tan(eps1(ni))*cos(DeltaSIGMA)) + tan(DeltaSIGMA)*cos(ni))
        # pinion offset angle in pitch plane
        eps1prime = lambda ni: asin( sin(eps1(ni))*cos(DeltaSIGMA)/cos(delta1(ni)) )
        # pinion spiral angle
        betam1 = lambda ni: atan( (K1 + deltaK(ni) - cos(eps1prime(ni)))/sin(eps1prime(ni)) )
        # gear spiral angle
        betam2 = lambda ni: betam1(ni) - eps1prime(ni)
        # gear pitch angle
        delta2 = lambda ni: atan( sin(eps1(ni))/(tan(ni)*cos(DeltaSIGMA)) + cos(eps1(ni))*tan(DeltaSIGMA) )
        # gear mean cone distance
        Rm2 = lambda ni: rmpt2/(sin(delta2(ni)))
        # pinion mean cone distance
        Rm1 = lambda ni: (rmn1 + Deltarmpt1(ni))/(sin(delta1(ni)))
        # mean pinion radius
        rmpt1 = lambda ni: Rm1(ni)*sin(delta1(ni))
        # limit pressure angle
        alphalim = lambda ni: atan( -tan(delta1(ni))*tan(delta2(ni))/cos(eps1prime(ni))*( (Rm1(ni)*sin(betam1(ni)) - Rm2(ni)*sin(betam2(ni))) / (Rm1(ni)*tan(delta1(ni)) + Rm2(ni)*tan(delta2(ni))) ) ) 

        rc0 = lambda ni: rc0 # user assigned rc0 value

        # else use suggested mean cutter radius if user didn't specify any
        if rc0Flag == None:
            match taper.lower():
                case "standard":
                    rc0 = lambda ni: (Rm2(ni) + 1.1*Rm2(ni)*sin(betam2(ni)))/2
                case "uniform":
                    rc0 = lambda ni: uniformToothCoeff*Rm2(ni)*sin(betam2(ni))
                case "duplex":
                    rc0 = lambda ni: (Rm2(ni) + 1.1*Rm2(ni)*sin(betam2(ni)))/2
                case "trl":
                    rc0 = lambda ni: (Rm2(ni) + 1.1*Rm2(ni)*sin(betam2(ni)))/2

        rhombeta = lambda ni: rc0(ni)

        rholim = lambda ni: 1/cos(alphalim(ni))*(tan(betam1(ni)) - tan(betam2(ni)))/(-tan(alphalim(ni))*(tan(betam1(ni))/(Rm1(ni)*tan(delta1(ni))) + tan(betam2(ni))/(Rm2(ni)*tan(delta2(ni)))) + 1/(Rm1(ni)*cos(betam1(ni))) - 1/(Rm2(ni)*cos(betam2(ni))))

        Delta = lambda ni: abs(rhombeta(ni)/rholim(ni) - 1)
        ni = ni0
        if Delta(ni0) > 0.01: # check if ni0 value satisfies the constraints
            Delta = lambda ni: abs(rhombeta(ni)/rholim(ni) - 1) - 0.005
            ni, opts , flag, msg = fsolve(Delta, ni0, full_output = True)
            if flag != 1:
                raise ValueError('Iterative algorithm did not converge properly!')
            
        rc0 = rc0(ni)
        Rm1 = Rm1(ni) # pinion mean cone distance
        Rm2 = Rm2(ni) # gear mean cone distance
        betam2 = betam2(ni) # gear spiral angle
        betam1 = betam1(ni) # pinion spiral angle
        delta2 = delta2(ni) # gear pitch angle
        delta1 = delta1(ni) # pinion pitch angle
        cbe2 = (de2/2/sin(delta2) - Rm2)/b2 # facewidth factor

    # Determination of basic data
    dm1 = 2*Rm1*sin(delta1)# pinion mean pitch diameter
    dm2 = 2*Rm2*sin(delta2)# gear mean pitch diameter
    zetam = asin(2*a/(dm2 + dm1*cos(delta2)/cos(delta1))) # offset angle in pinion axial plane
    zetamp = asin(sin(zetam)*sin(DeltaSIGMA)/cos(delta1)) # offset angle in the pitch plane
    ap = Rm2*sin(zetamp) # offset in pitch plane
    mmn = 2*Rm2*sin(delta2)*cos(betam2)/z2 # mean normal module
    alphalim = -atan( tan(delta1)*tan(delta2)/cos(zetamp)*( (Rm1*sin(betam1) - Rm2*sin(betam2)) / (Rm1*tan(delta1) + Rm2*tan(delta2)) ) ) # limit pressure angle
    alphanD = alphadD + falphalim*alphalim #generated normal pressure angle drive side
    alphanC = alphadC - falphalim*alphalim # generated normla pressure angle coast side
    alphaeD = alphanD - alphalim # effective pressure angle drive side (useless ?)
    alphaeC = alphanC + alphalim # effective pressure angle coast side
    alphan = (alphanD + alphanC)/2 # mean normal pressure angle
    Re2 = Rm2 + cbe2*b2 # outer pitch cone distance
    Ri2 = Re2 - b2 # inner pitch cone distance
    de2 = 2*Re2*sin(delta2) # gear outer pitch diameter
    di2 = 2*Ri2*sin(delta2) # gear inner pitch diameter
    met = de2/z2 # outer transverse module
    be2 = Re2 - Rm2 # gear facewidth from calculation point to outside
    bi2 = Rm2 - Ri2 # gear facewidth from calculation point to inside
    tzm2 = dm1*sin(delta2)/(2*cos(delta1)) - 0.5*cos(zetam)*tan(DeltaSIGMA)*(dm2 + dm1*cos(delta2)/cos(delta1)) # crossing point to calculation point along gear axis
    tzm1 = dm2/2*cos(zetam)*cos(DeltaSIGMA) - tzm2*sin(DeltaSIGMA) # crossing point to calculation point along pinion axis
    tz1 =  Rm1*cos(delta1) - tzm1 # pitch apex beyond crossing point along axis, pinion
    tz2 = Rm2*cos(delta2) - tzm2 # pitch apex beyond crossing point along axis, gear

    # determination of tooth depth at calculation point
    hmw = 2*mmn*khap # mean working depth
    ham2 = mmn*(khap - xhm1) # gear mean addendum
    hfm2 = mmn*(khfp + xhm1) # gear mean dedendum
    ham1 = mmn*(khap + xhm1) # pinion mean addendum
    hfm1 = mmn*(khfp - xhm1) # pinion mean dedendum
    c = mmn*(khfp - khap) # clearance
    hm = ham1 + hfm1 # mean whole depth (equal for both pinion and gear)
    hm = mmn*(khap + khfp) # mean whole depth (same formula as spur gears : tooth height = module*(1+1.25))
    
    # determination of dedendum angles
    match taper:
        case "Standard":
            sumthetafS =  atan(hfm1/Rm2) +  atan(hfm2/Rm2)  # sum dedendum angles
            thetaa2 =  atan(hfm1/Rm2)                     # addendum gear
            thetaf2 = sumthetafS - thetaa2               # dedendum gear
        case "Uniform":
            sumthetafU = 0 
            thetaa2 = 0 
            thetaf2 = 0 
        case "Duplex":
            sumthetafC = (pi*met/Re2/tan(alphan)/cos(betam2))*(1 - Rm2*sin(betam2)/rc0) 
            thetaa2 = sumthetafC*ham2/hmw 
            thetaf2 = sumthetafC - thetaa2 
        case "TRL":
            sumthetafM = min(   atan(hfm1/Rm2) +  atan(hfm2/Rm2)  ,  (pi*met/Re2/tan(alphan)/cos(betam2))*(1 - Rm2*sin(betam2)/rc0)  ) 
            thetaa2 = sumthetafM*ham2/hmw 
            thetaf2 = sumthetafM - thetaa2 
     
    if toothInitialData["thetaa2"] is not None:
        thetaa2 = toothInitialData["thetaa2"]
        thetaf2 = toothInitialData["thetaf2"]

    # determination of root angles and face angles

    deltaa2 = delta2 + thetaa2 # face angle gear
    deltaf2 = delta2 - thetaf2 # root angle gear
    phiR =  atan( a*tan(DeltaSIGMA)*cos(deltaf2)/(Rm2*cos(thetaf2) - tz2*cos(deltaf2) ) ) # auxiliary angle for calculating pinion offset angle in root plane
    phiO =  atan(a*tan(DeltaSIGMA)*cos(deltaa2)/(Rm2*cos(thetaa2) - tz2*cos(deltaa2)) ) # auxiliary angle for calculating pinion offset angle in face plane
    zetaR =  asin(a*cos(phiR)* sin(deltaf2)/(Rm2*cos(thetaf2) - tz2*cos(deltaf2))) - phiR # pinion offset angle in root plane
    zetaO =  asin(a*cos(phiO)*sin(deltaa2)/(Rm2*cos(thetaa2) - tz2*cos(deltaa2))) - phiO # pinion offset angle in face plane
    deltaa1 =  asin(sin(DeltaSIGMA)*sin(deltaf2) + cos(DeltaSIGMA)*cos(deltaf2)*cos(zetaR)) # pinion face angle
    deltaf1 =  asin(sin(DeltaSIGMA)*sin(deltaa2) + cos(DeltaSIGMA)*cos(deltaa2)*cos(zetaO)) # pinion root angle
    thetaa1 = deltaa1 - delta1 # pinion addendum angle
    thetaf1 = delta1 - deltaf1 # pinion dedendum angle
    tzF2 = tz2 - (Rm2*sin(thetaa2) - ham2*cos(thetaa2))/sin(deltaa2) # gear face apex beyon crossing point along axi
    tzR2 = tz2 + (Rm2*sin(thetaf2) - hfm2*cos(thetaf2))/sin(deltaf2) # gear root apex beyond crossing point  along axi
    tzF1 = (a*sin(zetaR)*cos(deltaf2) - tzR2*sin(deltaf2) - c)/sin(deltaa1) # pinion face apex beyond crossing point
    tzR1 = (a*sin(zetaO)*cos(deltaa2) - tzF2*sin(deltaa2) - c)/sin(deltaf1) # pinion root apex beyond crossing point

    # determination of pinion face width
    # method 0 values
    b1 = b2
    be1 = cbe2*b1
    bi1 = b1 - be1
    if Method == 1: # hypoid values
        bp1 = sqrt(Re2**2 - ap**2) - sqrt(Ri2**2 - ap**2) # pinion facewidth in pitchplane
        b1A = sqrt(Rm2**2 - ap**2) - sqrt(Ri2**2 - ap**2) # pinion facewidth from calculation point to front crown
        lambdaprime = atan(sin(zetamp)*cos(delta2)/(u*cos(delta1) + cos(delta2)*cos(zetamp))) # auxiliary angle
        breri1 = b2*cos(lambdaprime)/cos(zetamp - lambdaprime) # pinion facewidth
        Deltabx1 = hmw*sin(zetaR)*(1 - 1/u) # pinion facewidth increment along pinion axis
        Deltagxe = cbe2*breri1/cos(thetaa1)*cos(deltaa1) + Deltabx1 - (hfm2 - c)*sin(delta1) # increment along pinion axis from calculation point to outside
        Deltagxi = (1 - cbe2)*breri1*cos(deltaa1)/cos(thetaa1) + Deltabx1 + (hfm2 - c)*sin(delta1) # increment along pinion axis from calculation point to inside
        be1 = (Deltagxe + ham1*sin(delta1))/cos(deltaa1)*cos(thetaa1) # pinion face width from calculation point to outside
        bi1 = ( Deltagxi - ham1*sin(delta1) )/ ( cos(delta1) - tan(thetaa1)*sin(delta1) ) # pinion facewidth from calculation point to inside
        b1 = bi1 + be1 # pinion facewidth along pitch cone


    ## determination of inner and outer spiral angles
    #pinion
    Re21 = sqrt(Rm2**2 + be1**2 + 2*Rm2*be1*cos(zetamp)) # gear cone distance of outer pinion boundary point 
    Ri21 = sqrt(Rm2**2 + bi1**2 - 2*Rm2*bi1*cos(zetamp)) # gear cone distance of inner pinion boundary point
    betae21 = asin( (2*Rm2*rc0*sin(betam2) - Rm2**2 + Re21**2)/(2*Re21*rc0) ) # gear spiral angle at outer boundary point
    betai21 = asin( (2*Rm2*rc0*sin(betam2) - Rm2**2 + Ri21**2) / (2*Ri21*rc0) ) # gear spiral anfle at inner boundary point
    zetaep21 = asin(ap/Re21) # pinion offset angle in pitch plane at outer boundary point
    zetaip21 = asin(ap/Ri21) # pinion offset angle at pitch plane at inner boundary point
    betae1 = betae21 + zetaep21 # outer pinion spiral angle
    betai1 = betai21 + zetaip21 # inner pinion spiral angle
    # gear
    betae2 = asin( (2*Rm2*rc0*sin(betam2) - Rm2**2 + Re2**2) / (2*Re2*rc0) ) # outer gear spiral angle
    betai2 = asin( (2*Rm2*rc0*sin(betam2) - Rm2**2 + Ri2**2) / (2*Ri2*rc0) ) # inner gear spiral angle
    ## determination of tooth depth
    hae1 = ham1 + be1*tan(thetaa1) # pinion outer addendum
    hae2 = ham2 + be2*tan(thetaa2) # gear outer addendum
    hfe1 = hfm1 + be1*tan(thetaf1) # pinion outer dedendum
    hfe2 = hfm2 + be2*tan(thetaf2) # gear outer dedendum
    he1 = hae1 + hfe1 # pinion outer whole depth
    he2 = hae2 + hfe2 # gear outer whole depth
    hai1 = ham1 - bi1*tan(thetaa1) # pinion inner addendum
    hai2 = ham2 - bi2*tan(thetaa2) # gear inner addendum
    hfi1 = hfm1 - bi1*tan(thetaf1) # pinion inner dedendum
    hfi2 = hfm2 - bi2*tan(thetaf2) # gear inner dedendum
    hi1 = hai1 + hfi1 # pinion inner whole depth
    hi2 = hai2 + hfi2 # gear inner whole depth
    ## determination of tooth thickness, pag 234 pdf    pag 44 ISO journal
    #alphan PAG 235
    xsm1 = xsmn - jen*Rm2*cos(betam2)/( 4*mmn*cos(alphan)*Re2*cos(betae2) ) # pinion thickness modification coefficient 
    smn1 = 0.5*mmn*pi + 2*mmn*(xsm1 + xhm1*tan(alphan)) # pinion mean normal circular tooth thickness smn1
    xsm2 = -xsmn - jen*Rm2*cos(betam2)/( 4*mmn*cos(alphan)*Re2*cos(betae2) ) # gear thickness modification coefficient
    smn2 = 0.5*mmn*pi + 2*mmn*(xsm2 - xhm1*tan(alphan)) # gear mean normal circular tooth thickness
    smt1 = smn1/(cos(betam1)) # pinion mean transverse circular thickness
    smt2 = smn2/(cos(betam2)) # gear mean transverse circular thickness
    dmn1 = dm1/((1 - sin(betam1)**2*cos(alphan)**2)*cos(delta1)) # pinion normal diameter
    dmn2 = dm2/((1 - sin(betam2)**2*cos(alphan)**2)*cos(delta2)) # gear normal diameter
    smnc1 = dmn1*sin(smn1/dmn1) # pinion mean normal chordal tooth thickness
    smnc2 = dmn2*sin(smn2/dmn2) # gear mean normal chordal tooth thickness
    hamc1 = ham1 + 0.5*dmn1*cos(delta1)*(1 - cos(smn1/dmn1)) # pinion mean chordal addendum
    hamc2 = ham2 + 0.5*dmn2*cos(delta2)*(1 - cos(smn2/dmn2)) # gear mean chordal addendum
    ## determination of remaining dimension
    Re1 = Rm1 + be1 # pinion outer pitch cone distance
    Ri1 = Rm1 - bi1 # pinion inner pitch cone distance
    de1 = 2*Re1*sin(delta1) # pinion outer pitch diameter
    di1 = 2*Ri1*sin(delta1) # pinion inner pitch diameter
    dae1 = de1 + 2*hae1*cos(delta1) # pinion outside addendum diameter 
    dae2 = de2 + 2*hae2*cos(delta2) # gear outside addendum diameter
    dfe1 = de1 - 2*hfe1*cos(delta1) # pinion outside dedendum diameter
    dfe2 = de2 - 2*hfe2*cos(delta2) # gear outside dedendum diameter
    dai1 = di1 + 2*hai1*cos(delta1) # pinion inside addendum diameter
    dai2 = di2 + 2*hai2*cos(delta2) # gear inside addendum diameter
    dfi1 = di1 - hfi1*cos(delta1) # pinion inside dedendum diameter
    dfi2 = di2 - hfi2*cos(delta2) # pinion inside dedendum diameter
    txo1 = tzm1 + be1*cos(delta1) - hae1*sin(delta1) # pinion crossing point to crown along axis
    txo2 = tzm2 + be2*cos(delta2) - hae2*sin(delta2) # gear crossing point to crown along axis
    txi1 = tzm1 - bi1*cos(delta1) - hai1*sin(delta1) # pinion crossing point to front crown along axis
    txi2 = tzm2 - bi2*cos(delta2) - hai2*sin(delta2) # gear crossing point to front crown along axis
    ht1 = (tzF1 + txo1)/cos(deltaa1)*sin(thetaa1 + thetaf1) - (tzR1 - tzF1)*sin(deltaf1) # pinion whole depth perpendicular to the root cone
    ## undercut check
    # pinion at the moment we choose the inner pitch cone distance as check
    # point which in most cases might be the most critical one (verify this conjecture)
    Rx1 = Ri1 # point to be checked for undercut internal pitch cone distance
    Rx2 = sqrt(Rm2**2 + (Rm1 - Rx1)**2 - 2*Rm2*(Rm1 - Rx1)*cos(zetamp)) # gear cone distance
    betax2 = asin((2*Rm2*rc0*sin(betam2) - Rm2**2 + Rx2**2)/(2*Rx2*rc0)) # gear spiral angle at check point
    zetaxp2 = asin(ap/Rx2) # pinion offset angle in pitch plane at check point
    betax1 = betax2 + zetaxp2 # pinion spiral angle at check point 
    dx1 = 2*Rx1*sin(delta1) # pinion pitch diameter at check point
    dx2 = 2*Rx2*sin(delta2) # gear pitch diameter at check point
    mxn = dx2/z2*cos(betax2) # mean normal module at check point
    dEx1 = dx2*z1*cos(betax2)/z2/cos(betax1) # pinion effective diameter at check point
    REx1 = dEx1/2/sin(delta1) # pinion appropriate cone distance
    znx1 = z1/((1-sin(betax1)**2*cos(alphan)**2)*cos(betax1)*cos(delta1)) # intermediate value
    alphalimx = -atan(    tan(delta1)*tan(delta2)/(cos(zetamp))*(  (REx1*sin(betax1) - Rx2*sin(betax2))/(REx1*tan(delta1) + Rx2*tan(delta2))  )    ) # limit pressure angle at checkpoint
    alphaeDx = alphanD - alphalimx # effective pressure anfle at check point drive side
    alphaeCx = alphanC + alphalimx # effective pressure angle at check point coast side
    alphaeminx = min(alphaeDx, alphaeCx) # smaller effective pressure angle
    ## determination of mionimum profile shift coefficient at calculation point on the pinion
    khapx = khap + (Rx2 - Rm2)*tan(thetaa2)/mmn # working tool addendum at checkpoint
    xhx1 = 1.1*khapx - (znx1*mxn*sin(alphaeminx)**2)/(2*mmn) # pinion minimum profile shift coefficient at checkpoint
    xhmminx1 = xhx1 + (dEx1 - dx1)*cos(delta1)/(2*mmn) # pinion minimum profile shift coefficient at calculation point 

    ## generating basic design data structure
    basicDesignData = initialize_design_data()

    baseconethickPin = initialConeData["pinBaseThick"]
    baseconethickGear = initialConeData["gearBaseThick"]
    baseconApexPin = tzR1 - baseconethickPin/cos(pi/2-deltaf1)
    baseconApexGear = tzR2 - baseconethickGear/cos(pi/2-deltaf2)

    pinion_commonFieldName, pinion_subCommonFieldName  = get_data_field_names('pinion', 'concave', fields = 'common')
    gear_commonFieldName, gear_subCommonFieldName = get_data_field_names('gear', 'concave', fields = 'common')

    ## System data
    basicDesignData["SystemData"]["HAND"] = Hand
    basicDesignData["SystemData"]["shaft_angle"] = SIGMA*180/pi
    basicDesignData["SystemData"]["taperType"] = taper
    basicDesignData["SystemData"]["transmissionType"] = 'Hypoid'
    basicDesignData["SystemData"]["ratio"] = u
    basicDesignData["SystemData"]["hypoidOffset"] = a
    basicDesignData["SystemData"]["NOMINALDRIVEPRESSUREANGLE"] = alphanD*180/pi
    basicDesignData["SystemData"]["NOMINALCOASTPRESSUREANGLE"] = alphanC*180/pi
    basicDesignData["SystemData"]["NORMALMODULE"] = mmn

    basicDesignData[pinion_commonFieldName] = {
        f'{pinion_subCommonFieldName}GenType' : 'GENERATED',
        f'{pinion_subCommonFieldName}NTEETH' : z1,
        f'{pinion_subCommonFieldName}SPIRALANGLE' : betam1*180/pi,
        f'{pinion_subCommonFieldName}OUTERCONEDIST' : Re1,
        f'{pinion_subCommonFieldName}MEANCONEDIST' : Rm1,
        f'{pinion_subCommonFieldName}INNERCONEDIST': Ri1,
        f'{pinion_subCommonFieldName}MEANNORMALCHORDALTHICKNESS' : smnc1,
        f'{pinion_subCommonFieldName}MEANADDENDUM' : ham1,
        f'{pinion_subCommonFieldName}MEANCHORDALADDENDUM' : hamc1,
        f'{pinion_subCommonFieldName}FACEWIDTH' : b1,
        f'{pinion_subCommonFieldName}FACEANGLE' : deltaa1*180/pi,
        f'{pinion_subCommonFieldName}BACKANGLE' : delta1*180/pi,
        f'{pinion_subCommonFieldName}FRONTANGLE' : delta1*180/pi,
        f'{pinion_subCommonFieldName}PITCHANGLE' : delta1*180/pi,
        f'{pinion_subCommonFieldName}BASECONEANGLE' : deltaf1*180/pi,
        f'{pinion_subCommonFieldName}PITCHAPEX' : tz1,
        f'{pinion_subCommonFieldName}FACEAPEX' : tzF1,
        f'{pinion_subCommonFieldName}ROOTAPEX' : tzR1,
        f'{pinion_subCommonFieldName}BASECONEAPEX' : baseconApexPin,
        f'{pinion_subCommonFieldName}MEANCUTTERRAIDUS' : rc0,
        }
    
    basicDesignData[gear_commonFieldName] = {
        f'{gear_subCommonFieldName}GenType' : GearGenType,
        f'{gear_subCommonFieldName}NTEETH' : z2,
        f'{gear_subCommonFieldName}SPIRALANGLE' : betam2*180/pi,
        f'{gear_subCommonFieldName}OUTERCONEDIST' : Re2,
        f'{gear_subCommonFieldName}MEANCONEDIST' : Rm2,
        f'{gear_subCommonFieldName}INNERCONEDIST' : Ri2,
        f'{gear_subCommonFieldName}MEANNORMALCHORDALTHICKNESS' : smnc2,
        f'{gear_subCommonFieldName}MEANADDENDUM' : ham2,
        f'{gear_subCommonFieldName}MEANCHORDALADDENDUM' : hamc2,
        f'{gear_subCommonFieldName}FACEWIDTH' : b2,
        f'{gear_subCommonFieldName}FACEANGLE' : deltaa2*180/pi,
        f'{gear_subCommonFieldName}BACKANGLE' : delta2*180/pi,
        f'{gear_subCommonFieldName}FRONTANGLE' : delta2*180/pi,
        f'{gear_subCommonFieldName}PITCHANGLE' : delta2*180/pi,
        f'{gear_subCommonFieldName}BASECONEANGLE' : deltaf2*180/pi,
        f'{gear_subCommonFieldName}PITCHAPEX' : tz2,
        f'{gear_subCommonFieldName}FACEAPEX' : tzF2,
        f'{gear_subCommonFieldName}ROOTAPEX' : tzR2,
        f'{gear_subCommonFieldName}BASECONEAPEX' : baseconApexGear,
        f'{gear_subCommonFieldName}MEANCUTTERRAIDUS' : rc0,
        }
    
    handSign = +1
    phicnv = alphanC
    phicvx = alphanD
    if Hand.lower() == 'left':
        handSign = -1
        phicnv = alphanD
        phicvx = alphanC

    slidingBasePin = Rm1*sin(thetaf1) - hfm1*cos(thetaf1)
    slidingBaseGear = Rm2*sin(thetaf2) - hfm2*cos(thetaf2)
    machCtrbckPin = tzR1 - slidingBasePin/sin(deltaf1)
    machCtrbckGear = tzR2 - slidingBaseGear/sin(deltaf2)
    tz2 + (Rm2*sin(thetaf2) - hfm2*cos(thetaf2))/sin(deltaf2)
    

    machine_field, sub_machine_field = get_data_field_names('gear', 'concave', fields = 'machine')
    ## generated gear
    # concave gear
    basicDesignData[machine_field][f'{sub_machine_field}RADIALSETTING'] = sqrt((Rm2*cos(thetaf2))**2 + rc0**2 - 2*Rm2*cos(thetaf2)*rc0*sin(betam2))
    basicDesignData[machine_field][f'{sub_machine_field}CRADLEANGLE'] = atan(   rc0*cos(betam2)/(Rm2*cos(thetaf2) - rc0*sin(betam2))   )*180/pi
    basicDesignData[machine_field][f'{sub_machine_field}ROOTANGLE'] = deltaf2*180/pi
    basicDesignData[machine_field][f'{sub_machine_field}SLIDINGBASE'] = slidingBaseGear
    basicDesignData[machine_field][f'{sub_machine_field}MACHCTRBACK'] = machCtrbckGear
    basicDesignData[machine_field][f'{sub_machine_field}RATIOROLL'] = cos(thetaf2)/sin(delta2)

    machine_field, sub_machine_field = get_data_field_names('gear', 'convex', fields = 'machine')
    # convex gear
    basicDesignData[machine_field][f'{sub_machine_field}RADIALSETTING'] = sqrt((Rm2*cos(thetaf2))**2 + rc0**2 - 2*Rm2*cos(thetaf2)*rc0*sin(betam2))
    basicDesignData[machine_field][f'{sub_machine_field}CRADLEANGLE'] = atan(   rc0*cos(betam2)/(Rm2*cos(thetaf2) - rc0*sin(betam2))   )*180/pi
    basicDesignData[machine_field][f'{sub_machine_field}ROOTANGLE'] = deltaf2*180/pi
    basicDesignData[machine_field][f'{sub_machine_field}SLIDINGBASE'] = slidingBaseGear
    basicDesignData[machine_field][f'{sub_machine_field}MACHCTRBACK'] = machCtrbckGear
    basicDesignData[machine_field][f'{sub_machine_field}RATIOROLL'] = cos(thetaf2)/sin(delta2)

    # concave pinion
    machine_field, sub_machine_field = get_data_field_names('pinion', 'concave', fields = 'machine')
    basicDesignData[machine_field][f'{sub_machine_field}RADIALSETTING'] = sqrt((Rm1*cos(thetaf1))**2 + rc0**2 - 2*Rm1*cos(thetaf1)*rc0*sin(betam1))
    basicDesignData[machine_field][f'{sub_machine_field}CRADLEANGLE'] = atan(   rc0*cos(betam1)/(Rm1*cos(thetaf1) - rc0*sin(betam1))   )*180/pi
    basicDesignData[machine_field][f'{sub_machine_field}ROOTANGLE'] = deltaf1*180/pi
    basicDesignData[machine_field][f'{sub_machine_field}SLIDINGBASE'] = slidingBasePin
    basicDesignData[machine_field][f'{sub_machine_field}MACHCTRBACK'] = machCtrbckPin
    basicDesignData[machine_field][f'{sub_machine_field}RATIOROLL'] = cos(thetaf1)/sin(delta1)

    # convex pinion
    machine_field, sub_machine_field = get_data_field_names('pinion', 'convex', fields = 'machine')
    basicDesignData[machine_field][f'{sub_machine_field}RADIALSETTING'] = sqrt((Rm1*cos(thetaf1))**2 + rc0**2 - 2*Rm1*cos(thetaf1)*rc0*sin(betam1))
    basicDesignData[machine_field][f'{sub_machine_field}CRADLEANGLE'] = atan(   rc0*cos(betam1)/(Rm1*cos(thetaf1) - rc0*sin(betam1))   )*180/pi
    basicDesignData[machine_field][f'{sub_machine_field}ROOTANGLE'] = deltaf1*180/pi
    basicDesignData[machine_field][f'{sub_machine_field}SLIDINGBASE'] = slidingBasePin
    basicDesignData[machine_field][f'{sub_machine_field}MACHCTRBACK'] = machCtrbckPin
    basicDesignData[machine_field][f'{sub_machine_field}RATIOROLL'] = cos(thetaf1)/sin(delta1)

    # cutter data
    PQ = (hfm2 - ham2)/2
    P0N = hfm2*cos(thetaf2) - PQ*cos(thetaf2)
    Np = z2*cos(thetaf2)/sin(delta2)
    pointRadiusConcave = rc0 + smnc2/2
    pointRadiusConvex = rc0 - smnc2/2
    edgeRadius = 0.15*met

    # gear cutter
    # concave
    cutter_field, sub_cutter_field = get_data_field_names('gear', 'concave', fields = 'cutter')
    basicDesignData[cutter_field][f"{sub_cutter_field}POINTRADIUS"] = pointRadiusConcave
    basicDesignData[cutter_field][f"{sub_cutter_field}BLADEANGLE"] = phicnv*180/pi
    basicDesignData[cutter_field][f"{sub_cutter_field}EDGERADIUS"] = edgeRadius
    basicDesignData[cutter_field][f"{sub_cutter_field}TYPE"] = 'CURVED'
    basicDesignData[cutter_field][f"{sub_cutter_field}TopremTYPE"] = 'NONE'
    basicDesignData[cutter_field][f"{sub_cutter_field}FlankremTYPE"] = 'NONE'
    basicDesignData[cutter_field][f"{sub_cutter_field}RHO"] = 800
    basicDesignData[cutter_field][f"{sub_cutter_field}TopremDEPTH"] = None
    basicDesignData[cutter_field][f"{sub_cutter_field}TopremRADIUS"] = None
    basicDesignData[cutter_field][f"{sub_cutter_field}FlankremDEPTH"] = None
    basicDesignData[cutter_field][f"{sub_cutter_field}FlankremRADIUS"] = None
    # convex
    cutter_field, sub_cutter_field = get_data_field_names('gear', 'convex', fields = 'cutter')
    basicDesignData[cutter_field][f"{sub_cutter_field}POINTRADIUS"] = pointRadiusConvex
    basicDesignData[cutter_field][f"{sub_cutter_field}BLADEANGLE"] = phicvx*180/pi
    basicDesignData[cutter_field][f"{sub_cutter_field}EDGERADIUS"] = edgeRadius
    basicDesignData[cutter_field][f"{sub_cutter_field}TYPE"] = 'CURVED'
    basicDesignData[cutter_field][f"{sub_cutter_field}TopremTYPE"] = 'NONE'
    basicDesignData[cutter_field][f"{sub_cutter_field}FlankremTYPE"] = 'NONE'
    basicDesignData[cutter_field][f"{sub_cutter_field}RHO"] = 800
    basicDesignData[cutter_field][f"{sub_cutter_field}TopremDEPTH"] = None
    basicDesignData[cutter_field][f"{sub_cutter_field}TopremRADIUS"] = None
    basicDesignData[cutter_field][f"{sub_cutter_field}FlankremDEPTH"] = None
    basicDesignData[cutter_field][f"{sub_cutter_field}FlankremRADIUS"] = None

    # pinion cutter
    rc0P = rc0
    if rc0Flag == None:
        match taper.lower():
            case "standard":
                rc0P = (Rm1 + 1.1*Rm1*sin(betam1))/2
            case "uniform":
                rc0P = uniformToothCoeff*Rm1*sin(betam1)
            case "duplex":
                rc0P = (Rm1 + 1.1*Rm1*sin(betam1))/2
            case "trl":
                rc0P = (Rm1 + 1.1*Rm1*sin(betam1))/2

    pointRadiusConcave = rc0P + smnc1/2
    pointRadiusConvex = rc0P - smnc1/2
    # concave
    cutter_field, sub_cutter_field = get_data_field_names('pinion', 'concave', fields = 'cutter')
    basicDesignData[cutter_field][f"{sub_cutter_field}POINTRADIUS"] = pointRadiusConcave
    basicDesignData[cutter_field][f"{sub_cutter_field}BLADEANGLE"] = phicnv*180/pi
    basicDesignData[cutter_field][f"{sub_cutter_field}EDGERADIUS"] = edgeRadius
    basicDesignData[cutter_field][f"{sub_cutter_field}TYPE"] = 'STRAIGHT'
    basicDesignData[cutter_field][f"{sub_cutter_field}TopremTYPE"] = 'NONE'
    basicDesignData[cutter_field][f"{sub_cutter_field}FlankremTYPE"] = 'NONE'
    basicDesignData[cutter_field][f"{sub_cutter_field}RHO"] = None
    basicDesignData[cutter_field][f"{sub_cutter_field}TopremDEPTH"] = None
    basicDesignData[cutter_field][f"{sub_cutter_field}TopremRADIUS"] = None
    basicDesignData[cutter_field][f"{sub_cutter_field}FlankremDEPTH"] = None
    basicDesignData[cutter_field][f"{sub_cutter_field}FlankremRADIUS"] = None
    # convex
    cutter_field, sub_cutter_field = get_data_field_names('pinion', 'convex', fields = 'cutter')
    basicDesignData[cutter_field][f"{sub_cutter_field}POINTRADIUS"] = pointRadiusConvex
    basicDesignData[cutter_field][f"{sub_cutter_field}BLADEANGLE"] = phicvx*180/pi
    basicDesignData[cutter_field][f"{sub_cutter_field}EDGERADIUS"] = edgeRadius
    basicDesignData[cutter_field][f"{sub_cutter_field}TYPE"] = 'STRAIGHT'
    basicDesignData[cutter_field][f"{sub_cutter_field}TopremTYPE"] = 'NONE'
    basicDesignData[cutter_field][f"{sub_cutter_field}FlankremTYPE"] = 'NONE'
    basicDesignData[cutter_field][f"{sub_cutter_field}RHO"] = None
    basicDesignData[cutter_field][f"{sub_cutter_field}TopremDEPTH"] = None
    basicDesignData[cutter_field][f"{sub_cutter_field}TopremRADIUS"] = None
    basicDesignData[cutter_field][f"{sub_cutter_field}FlankremDEPTH"] = None
    basicDesignData[cutter_field][f"{sub_cutter_field}FlankremRADIUS"] = None

    return basicDesignData

def shaftSegmentComputation(data):
    gear_data = data['GearCommonData']
    pinion_data = data['PinionCommonData']
    
    # compute gear data
    O_g = gear_data['gearOUTERCONEDIST']
    gamma_g = gear_data['gearPITCHANGLE']
    pA_g = gear_data['gearPITCHAPEX']
    bA_g = gear_data['gearBASECONEAPEX']
    gammab_g = gear_data['gearBASECONEANGLE']
    Fw_g = gear_data['gearFACEWIDTH']
    backA_g = gear_data['gearBACKANGLE']
    frontA_g = gear_data['gearFRONTANGLE']
    
    PB_g = np.array([O_g * np.cos(np.deg2rad(gamma_g)) - pA_g, 
                     O_g * np.sin(np.deg2rad(gamma_g))])  # [z, R]
    PF_g = np.array([(O_g - Fw_g) * np.cos(np.deg2rad(gamma_g)) - pA_g, 
                     (O_g - Fw_g) * np.sin(np.deg2rad(gamma_g))])  # [z, R]
    
    zB_g = (-np.tan(np.deg2rad(gammab_g)) * bA_g + PB_g[1] + np.tan(np.deg2rad(90 - backA_g)) * PB_g[0]) / \
            (np.tan(np.deg2rad(gammab_g)) + np.tan(np.deg2rad(90 - backA_g)))
    RB_g = np.tan(np.deg2rad(gammab_g)) * zB_g + np.tan(np.deg2rad(gammab_g)) * bA_g
    zA_g = (-np.tan(np.deg2rad(gammab_g)) * bA_g + PF_g[1] + np.tan(np.deg2rad(90 - frontA_g)) * PF_g[0]) / \
            (np.tan(np.deg2rad(gammab_g)) + np.tan(np.deg2rad(90 - frontA_g)))
    RA_g = np.tan(np.deg2rad(gammab_g)) * zA_g + np.tan(np.deg2rad(gammab_g)) * bA_g
    
    gear_data['gearShaftzB'] = zB_g
    gear_data['gearShaftRB'] = RB_g
    gear_data['gearShaftzA'] = zA_g
    gear_data['gearShaftRA'] = RA_g
    gear_data['gearShaftDiA'] = RA_g * (2 - 0.85)  # diametro interno a ridosso del punto A
    gear_data['gearShaftDiB'] = RB_g * (2 - 0.85)  # diametro interno a ridosso del punto B

    # compute pinion data
    O_p = pinion_data['pinOUTERCONEDIST']
    gamma_p = pinion_data['pinPITCHANGLE']
    pA_p = pinion_data['pinPITCHAPEX']
    bA_p = pinion_data['pinBASECONEAPEX']
    gammab_p = pinion_data['pinBASECONEANGLE']
    Fw_p = pinion_data['pinFACEWIDTH']
    backA_p = pinion_data['pinBACKANGLE']
    frontA_p = pinion_data['pinFRONTANGLE']
    
    PB_p = np.array([O_p * np.cos(np.deg2rad(gamma_p)) - pA_p, 
                     O_p * np.sin(np.deg2rad(gamma_p))])  # [z, R]
    PF_p = np.array([(O_p - Fw_p) * np.cos(np.deg2rad(gamma_p)) - pA_p, 
                     (O_p - Fw_p) * np.sin(np.deg2rad(gamma_p))])  # [z, R]
    
    zB_p = (-np.tan(np.deg2rad(gammab_p)) * bA_p + PB_p[1] + np.tan(np.deg2rad(90 - backA_p)) * PB_p[0]) / \
            (np.tan(np.deg2rad(gammab_p)) + np.tan(np.deg2rad(90 - backA_p)))
    RB_p = np.tan(np.deg2rad(gammab_p)) * zB_p + np.tan(np.deg2rad(gammab_p)) * bA_p
    zA_p = (-np.tan(np.deg2rad(gammab_p)) * bA_p + PF_p[1] + np.tan(np.deg2rad(90 - frontA_p)) * PF_p[0]) / \
            (np.tan(np.deg2rad(gammab_p)) + np.tan(np.deg2rad(90 - frontA_p)))
    RA_p = np.tan(np.deg2rad(gammab_p)) * zA_p + np.tan(np.deg2rad(gammab_p)) * bA_p
    
    pinion_data['pinShaftzB'] = zB_p
    pinion_data['pinShaftRB'] = RB_p
    pinion_data['pinShaftzA'] = zA_p
    pinion_data['pinShaftRA'] = RA_p
    pinion_data['pinShaftDiA'] = RA_p * (2 - 0.85)  # diametro interno a ridosso del punto A
    pinion_data['pinShaftDiB'] = RB_p * (2 - 0.85)  # diametro interno a ridosso del punto B
    
    return data

def main():

    SystemData = {
        'HAND': "Right",
        'taper' : "Standard",
        'hypoidOffset' : 25
    }

    coneData = {
        'SIGMA' : 90,
        'a' : SystemData['hypoidOffset'],
        'z1' : 9,
        'u' : 3.7,
        'de2': 225,
        'b2' : 38.8,
        'betam1' : 45,
        'rc0' : 75,
        'gearBaseThick' : 15,
        'pinBaseThick' : 8,
    }

    coneData['z2'] = round(coneData['u']*coneData['z1'])
    coneData['u'] = coneData['z2']/coneData['z1']

    toothData = {
        'alphaD' : 21,
        'alphaC' : 20,
        'falphalim' : 1,
        'khap' : 1,
        'khfp' : 1.25,
        'xhm1' : 0.45,
        'jen' : 0.1,
        'xsmn' : 0.05,
        'thetaa2' : None,
        'thetaf2' : None
    }

    data = AGMAcomputationHypoid(SystemData['HAND'], SystemData['taper'], coneData, toothData)
    data = shaftSegmentComputation(data)
    data["GearCommonData"]["GearGenType"] = 'generated'
    gearBlank = assignBlankPar(data, 'gear')
    pinionBlank = assignBlankPar(data, 'pinion')
    dictprint(data)        
    print(gearBlank)
    print(pinionBlank)  
    data, trpl_cnv, trpl_cvx = approxToolIdentification_casadi(data, 'gear', RHO = 500)
    data, trpl_cnv, trpl_cvx = approxToolIdentification_casadi(data, 'pinion', RHO = 500)
    print(trpl_cnv, trpl_cvx)
    dictprint(data)
    return

def main2():
    a = np.array([
        [0.8143,    0.8308,    0.0759,    0.3371,    0.6892,    0.9961,    0.0844,    0.1361],
        [0.2435,    0.5853,    0.0540,    0.1622,    0.7482,    0.0782,    0.3998,    0.8693],
        [0.9293,    0.5497,    0.5308,    0.7943,    0.4505,    0.4427,    0.2599,    0.5797],
        [0.3500,    0.9172,    0.7792,    0.3112,    0.0838,    0.1067,    0.8001,    0.5499],
        [0.1966,    0.2858,    0.9340,    0.5285,    0.2290,    0.9619,    0.4314,    0.1450],
        [0.2511,    0.7572,    0.1299,    0.1656,    0.9133,    0.0046,    0.9106,    0.8530],
        [0.6160,    0.7537,    0.5688,    0.6020,    0.1524,    0.7749,    0.1818,    0.6221],#########roll
        [0.4733,    0.3804,    0.4694,    0.2630,    0.8258,    0.8173,    0.2638,    0.3510],
        [0.3517,    0.5678,    0.0119,    0.6541,    0.5383,    0.8687,    0.1455,    0.5132]
    ])
    [cmat, smat] = manageMachinePar('pinion', 'right')
    anew = a*cmat*smat
    roll = anew[6,1]
    anew[6, 2:-1] = anew[6, 2:-1]*roll
    ggt, vgt, vgt_spatial = machine_kinematics(anew)
    x = ca.SX.sym('x')
    print(ggt(2))
    print(vgt(2))
    print(vgt_spatial(2))
    ggt, vgt, vgt_spatial = casadi_machine_kinematics('pinion', 'right')
    print(ggt(a, 2).full())

if __name__ == "__main__":
    main()
