import numpy as np
from math import sqrt, pi, atan, cos, sin, acos, asin, tan
from utils import *

def get_data_field_names(member, flank):
    if member.lower() == 'pinion' and flank.lower() == 'concave':
        cutterFieldName = 'PinionConcaveCutterData'
        subCutterFieldName = 'pinCutCnv'
        commonFieldName = 'PinionCommonData'
        subCommonFieldName = 'pin'
        machineFieldName = 'PinionConcaveMachineSettings'
        subMachineFieldName = 'pinCnv'
    elif member.lower() == 'pinion' and flank.lower() == 'convex':
        cutterFieldName = 'PinionConvexCutterData'
        subCutterFieldName = 'pinCutCvx'
        commonFieldName = 'PinionCommonData'
        subCommonFieldName = 'pin'
        machineFieldName = 'PinionConvexMachineSettings'
        subMachineFieldName = 'pinCvx'
    elif member.lower() == 'gear' and flank.lower() == 'concave':
        cutterFieldName = 'GearConcaveCutterData'
        subCutterFieldName = 'gearCutCnv'
        commonFieldName = 'GearCommonData'
        subCommonFieldName = 'gear'
        machineFieldName = 'GearConcaveMachineSettings'
        subMachineFieldName = 'gearCnv'
    elif member.lower() == 'gear' and flank.lower() == 'convex':
        cutterFieldName = 'GearConvexCutterData'
        subCutterFieldName = 'gearCutCvx'
        commonFieldName = 'GearCommonData'
        subCommonFieldName = 'gear'
        machineFieldName = 'GearConvexMachineSettings'
        subMachineFieldName = 'gearCvx'

    return cutterFieldName, subCutterFieldName, commonFieldName, subCommonFieldName,\
            machineFieldName, subMachineFieldName

def assignMachinePar(data, member, flank):
    if member.lower() == 'pinion':
        if flank.lower() == 'concave':
            mainFieldName = 'PinionConcaveMachineSettings'
            subFieldName = 'pinCnv'
        else:
            mainFieldName = 'PinionConvexMachineSettings'
            subFieldName = 'pinCvx'
    else:
        if flank.lower() == 'concave':
            mainFieldName = 'GearConcaveMachineSettings'
            subFieldName = 'gearCnv'
        else:
            mainFieldName = 'GearConvexMachineSettings'
            subFieldName = 'gearCvx'

    S0    = data[mainFieldName][f'{subFieldName}RADIALSETTING']
    sig0  = data[mainFieldName][f'{subFieldName}TILTANGLE']
    z0    = data[mainFieldName][f'{subFieldName}SWIVELANGLE']
    E0    = data[mainFieldName][f'{subFieldName}BLANKOFFSET']
    gam0  = data[mainFieldName][f'{subFieldName}ROOTANGLE']
    D0    = data[mainFieldName][f'{subFieldName}MACHCTRBACK']
    B0    = data[mainFieldName][f'{subFieldName}SLIDINGBASE']
    q0    = data[mainFieldName][f'{subFieldName}CRADLEANGLE']
    m     = data[mainFieldName][f'{subFieldName}RATIOROLL']
    C2    = data[mainFieldName][f'{subFieldName}2C']
    D6    = data[mainFieldName][f'{subFieldName}6D']
    E24   = data[mainFieldName][f'{subFieldName}24E']
    F120  = data[mainFieldName][f'{subFieldName}120F']
    G720  = data[mainFieldName][f'{subFieldName}720G']
    H5040 = data[mainFieldName][f'{subFieldName}5040H']
    B1    = data[mainFieldName][f'{subFieldName}H1']
    B2    = data[mainFieldName][f'{subFieldName}H2']
    B3    = data[mainFieldName][f'{subFieldName}H3']
    B4    = data[mainFieldName][f'{subFieldName}H4']
    B5    = data[mainFieldName][f'{subFieldName}H5']
    B6    = data[mainFieldName][f'{subFieldName}H6']
    E1    = data[mainFieldName][f'{subFieldName}V1']
    E2    = data[mainFieldName][f'{subFieldName}V2']
    E3    = data[mainFieldName][f'{subFieldName}V3']
    E4    = data[mainFieldName][f'{subFieldName}V4']
    E5    = data[mainFieldName][f'{subFieldName}V5']
    E6    = data[mainFieldName][f'{subFieldName}V6']
    S1    = data[mainFieldName][f'{subFieldName}R1']
    S2    = data[mainFieldName][f'{subFieldName}R2']
    S3    = data[mainFieldName][f'{subFieldName}R3']
    S4    = data[mainFieldName][f'{subFieldName}R4']
    S5    = data[mainFieldName][f'{subFieldName}R5']
    S6    = data[mainFieldName][f'{subFieldName}R6']

    if member.lower() == 'gear' and data['GearCommonData']['GearGenType'].lower() == 'formate':
        H = data[mainFieldName][f'{subFieldName}Horizontal']
        V = data[mainFieldName][f'{subFieldName}Vertical']
        q0 = atan(V/H)
        S0 = sqrt(V**2 + H**2)
        q0 = q0*180/pi
        return np.array([
            [  S0,  0,  0,  0,  0,  0,  0,  0],     #joint Tool 2
            [sig0,  0,  0,  0,  0,  0,  0,  0],     #joint Tool 4
            [  z0,  0,  0,  0,  0,  0,  0,  0],     #joint Tool 3
            [  E0,  0,  0,  0,  0,  0,  0,  0],     #joint Gear 1
            [  B0,  0,  0,  0,  0,  0,  0,  0],     #joint Gear 2
            [  q0,  0,  0,  0,  0,  0,  0,  0],     #joint Tool 1
            [   0,  0,  0,  0,  0,  0,  0,  0],     #joint Gear 5
            [  D0,  0,  0,  0,  0,  0,  0,  0],     #joint Gear 4
            [gam0,  0,  0,  0,  0,  0,  0,  0]      #joint Gear 3
            ])

    return np.array([
        [  S0, S1, S2, S3,  S4,   S5,   S6,     0],     #joint Tool 2
        [sig0,  0,  0,  0,   0,    0,    0,     0],     #joint Tool 4
        [  z0,  0,  0,  0,   0,    0,    0,     0],     #joint Tool 3
        [  E0, E1, E2, E3,  E4,   E5,   E6,     0],     #joint Gear 1
        [  B0, B1, B2, B3,  B4,   B5,   B6,     0],     #joint Gear 2
        [  q0,  1,  0,  0,   0,    0,    0,     0],     #joint Tool 1
        [   0,  m, C2, D6, E24, F120, G720, H5040],     #joint Gear 5
        [  D0,  0,  0,  0,   0,    0,    0,     0],     #joint Gear 4
        [gam0,  0,  0,  0,   0,    0,    0,     0]      #joint Gear 3
        ])  

def manageMachinePar(member, systemHand, mode = 'gleason'):
    
    # coefficient settings
    if mode.lower() == 'gleason':
        cMat = np.array([
            [1, 1, 1/2, 1/6, 1/24, 1/120, 1/720, 1/5040],
            [1, 1, 1/2, 1/6, 1/24, 1/120, 1/720, 1/5040],
            [1, 1, 1/2, 1/6, 1/24, 1/120, 1/720, 1/5040],
            [1, 1, 1/2, 1/6, 1/24, 1/120, 1/720, 1/5040],
            [1, 1, 1/2, 1/6, 1/24, 1/120*1/1000, 1/720*1/1000, 1/5040],
            [1 ,1, 1/2, 1/6, 1/24, 1/120, 1/720, 1/5040],
            [0, 1, 1/2, 1/6, 1/24, 1/120, 1/720*1/1000, 1/5040*1/1000],
            [1, 1, 1/2, 1/6, 1/24, 1/120, 1/720, 1/5040],
            [1, 1, 1/2, 1/6, 1/24, 1/120, 1/720, 1/5040]
        ])
        cMat[[1,2,5,8], [0,0,0,0]] = cMat[[1,2,5,8], [0,0,0,0]]*pi/180

    # sign settings

    if member.lower == 'pinion':
        if systemHand.lower() == 'left':
            signMat = np.array([
                [+1, +1, +1, +1, +1, +1, +1, +1], # R radial motion
                [-1, +1, +1, +1, +1, +1, +1, +1], # sig
                [-1, +1, +1, +1, +1, +1, +1, +1], # zeta
                [+1, +1, +1, +1, +1, +1, +1, +1], # E vertical motion
                [+1, +1, +1, +1, +1, +1, +1, +1], # B sliding base (helical motion)
                [-1, +1, +1, +1, +1, +1, +1, +1], # q cradle
                [+0, +1, -1, -1, -1, -1, -1, -1], # roll
                [+1, +1, +1, +1, +1, +1, +1, +1], # machCtr
                [+1, +1, +1, +1 ,+1, +1, +1, +1], #root
                ])
        else: # right hand
            signMat = np.array([
                [+1, +1, +1, -1, +1, +1, +1, +1],
                [+1, +1, +1, +1, +1, +1, +1, +1],
                [+1, +1, +1, +1, +1, +1, +1, +1],
                [+1, +1, -1, +1, -1, +1, +1, +1],
                [+1, -1, +1, -1, +1, -1, +1, -1],
                [-1, +1, +1, +1, +1, +1, +1, +1],
                [+0, +1, +1, -1, +1, -1, +1, -1],
                [+1, +1, +1, +1, +1, +1, +1, +1],
                [+1, +1, +1, +1 ,+1, +1, +1, +1]
            ])
    else: # gear
        if systemHand.lower() == 'left':
            signMat = np.array([
                [+1, +1, +1, +1, +1, +1, +1, +1],
                [-1, +1, +1, +1, +1, +1, +1, +1],
                [+1, +1, +1, +1, +1, +1, +1, +1],
                [-1, +1, +1, +1, +1, +1, +1, +1],
                [+1, +1, +1, +1, +1, +1, +1, +1],
                [+1, +1, +1, +1, +1, +1, +1, +1],
                [+0, +1, -1, -1, -1, -1, -1, -1],
                [+1, +1, +1, +1, +1, +1, +1, +1],
                [+1, +1, +1, +1 ,+1, +1, +1, +1]
            ])
        else: # right hand
            signMat = np.array([
                [+1, +1, +1, -1, +1, +1, +1, +1],
                [+1, +1, +1, +1, +1, +1, +1, +1],
                [+1, +1, +1, +1, +1, +1, +1, +1],
                [+1, +1, -1, +1, -1, +1, +1, +1],
                [+1, -1, +1, -1, +1, -1, +1, -1],
                [-1, +1, +1, +1, +1, +1, +1, +1],
                [+0, +1, +1, -1, +1, -1, +1, -1],
                [+1, +1, +1, +1, +1, +1, +1, +1],
                [+1, +1, +1, +1 ,+1, +1, +1, +1]
            ])
    return cMat, signMat

def assignBlankPar(data, member):
    _, _, commonFieldName, subCommonFieldName,\
           machineFieldName, subMachineFieldName = get_data_field_names(member, 'concave')
    
    A0 = data[commonFieldName][f'{subCommonFieldName}OUTERCONEDIST']
    Fw = data[commonFieldName][f'{subCommonFieldName}FACEWIDTH']
    beta = data[commonFieldName][f'{subCommonFieldName}SPIRALANGLE']

    # front/back cones
    deltaf = data[commonFieldName][f'{subCommonFieldName}FRONTANGLE']
    deltab = data[commonFieldName][f'{subCommonFieldName}BACKANGLE']

    # pitch cone
    gammaP = data[commonFieldName][f'{subCommonFieldName}PITCHANGLE']
    dpa = data[commonFieldName][f'{subCommonFieldName}PITCHAPEX']

    # face cone
    dfa = data[commonFieldName][f'{subCommonFieldName}FACEAPEX']
    gammaF = data[commonFieldName][f'{subCommonFieldName}FACEANGLE']
    
    # root cone
    dra = data[commonFieldName][f'{subCommonFieldName}ROOTAPEX']
    gammaR = data[machineFieldName][f'{subMachineFieldName}ROOTANGLE']

    # base cone
    gammaB = data[commonFieldName][f'{subCommonFieldName}BASECONEANGLE']
    dba = data[commonFieldName][f'{subCommonFieldName}BASECONEAPEX']
    
    RA = data[commonFieldName][f'{subCommonFieldName}ShaftRA']
    
    return (A0, Fw, beta, deltaf, deltab, gammaP, dpa, gammaF, dfa, gammaR, dra, gammaB, dba, RA)

def assignToolPar(data, member, flank, stfFlank = None):
    
    alphaFlank = None
    rho_straight = 2e6
    cutterFieldName, subCutterFieldName, _, _, machineFieldName, subMachineFieldName =\
          get_data_field_names(member, flank)
    
    Rp = data[cutterFieldName][f'{subCutterFieldName}POINTRADIUS']
    alphap = data[cutterFieldName][f'{subCutterFieldName}BLADEANGLE']
    rhof = data[cutterFieldName][f'{subCutterFieldName}EDGERADIUS']
    if data[cutterFieldName][f'{subCutterFieldName}TYPE'].lower() == 'curved':
        rho = data[cutterFieldName][f'{subCutterFieldName}RHO']
    else:
        rho = rho_straight

    if data[cutterFieldName][f'{subCutterFieldName}TopremTYPE'].lower() == 'blended':
        rhoTop = data[cutterFieldName][f'{subCutterFieldName}TopremRADIUS']
        stftop = data[cutterFieldName][f'{subCutterFieldName}TopremDEPTH']

    elif data[cutterFieldName][f'{subCutterFieldName}TopremTYPE'].lower() == 'straight':
        alphaTop = data[cutterFieldName][f'{subCutterFieldName}TopremANGLE']
        stftop = data[cutterFieldName][f'{subCutterFieldName}TopremDEPTH']      
        rhoTop = rho_straight

    else: # None (feature disabled)
        rhoTop = 50 # initialize to a reasonable blend radius in case an identification activates the feature
        Czblade = rho*sin(alphap*pi/180)
        theta_iniz_blade = asin((Czblade + rhof)/(rho + rhof))

        try:
            alphaTop = data['PinionConcaveCutterData'][f'{subCutterFieldName}TopremANGLE']
            msgbox("Tool with angle offset between tip & blade not implemented", 'Error', 1)
            return
        except:
            pass
        
        stftop = rhof - rhof*sin(theta_iniz_blade)
        alphaTop = 0

    if data[cutterFieldName][f'{subCutterFieldName}FlankremTYPE'].lower() == 'blended':
        rhoFlank = data[cutterFieldName][f'{subCutterFieldName}FlankremRADIUS']
        stfFlank = data[cutterFieldName][f'{subCutterFieldName}FlankremDEPTH']
        

    elif data[cutterFieldName][f'{subCutterFieldName}FlankremTYPE'].lower() == 'straight':
        alphaFlank = data[cutterFieldName][f'{subCutterFieldName}FlankremANGLE']
        stfFlank = data[cutterFieldName][f'{subCutterFieldName}FlankremDEPTH']      
        rhoFlank = rho_straight

    else: # None (feature disabled)
        rhoFlank = rho # initialize to a reasonable blend radius in case an identification activates the feature
        if stfFlank == None:
            stfFlank = rho/3
        alphaTop = 0

    return (Rp,rho,rhof,rhoTop,rhoFlank,alphap,stfFlank,stftop,alphaTop,alphaFlank)

def initialguesses(A0, Fw, gammaP, gammaF, member, hand, q0, z0, beta, machCtr, gammaR, S0, Rp, RA):
    """
    A0 outer cone distance
    Fw facewidth
    gammaP pitchangle
    gammaF faceangle
    member 
    hand system hand
    q0 cradle angle
    z0 swivel angle (some Gleason designs have this =/= 0 even if tilt angle is zero)
    beta spiral angle
    machCtr machine center to back setting
    gammaR root angle
    S0 machine radial setting
    Rp tool pointradius
    RA radial distance of the root-toe point
    """
    xF = (A0-0.5*Fw)*sin(gammaF)
    yF = (A0-0.5*Fw)*cos(gammaF)
    xP = (A0-0.5*Fw)*sin(gammaP)
    yP = (A0-0.5*Fw)*cos(gammaP)
    csiout = 2*np.linalg.norm([xF - xP,yF - yP])
    if csiout < 0.01:
        csiout = 5

    thetag = acos( ((machCtr+(A0-1.2*Fw)*cos(gammaP))*cos(gammaR) + RA*sin(gammaR) - S0*cos(q0))/Rp) # angle on the cutter
    
    if member.lower() == 'pinion' and hand.lower() == 'left':
        thetaout = (thetag - q0) + z0
    elif member.lower() == 'pinion' and hand.lower() == 'right':
        thetaout = -(thetag + q0) + z0
    elif member.lower() == 'gear' and hand.lower() == 'left': # right gear
        thetaout = -(thetag + q0) + z0
    elif member.lower() == 'gear' and hand.lower() == 'right':
        thetaout = (thetag - q0) + z0
    
    # try for root cone initial guess
    phiout = 0
    csiout = 0
    return (csiout, thetaout, phiout)

def initial_guess_from_data(data, member, flank): # wrapper for initielguesses() to provide the designData instead of all the parameters
    hand = data['SystemData']['HAND']
    toolvec = assignToolPar(data, member, flank)
    RawMachinePar = assignMachinePar(data, member, flank)
    cMat, signMat = manageMachinePar(member, hand)
    MachineParMatrix = cMat * signMat * RawMachinePar
    m = RawMachinePar[6, 1]
    MachineParMatrix[6, 2:] = MachineParMatrix[6, 2:] * m
    q0 = MachineParMatrix[5, 0]
    z0 = MachineParMatrix[2, 0]
    sig0 = MachineParMatrix[1, 0]
    machCtr = MachineParMatrix[7, 0]
    gammaroot = MachineParMatrix[8, 0]
    S0 = MachineParMatrix[0, 0]  # radial setting
    A0, Fw, beta, deltaf, deltab, gammaP, dpa, gammaF, dfa, gammaR, dra, gammaB, dba, RA = assignBlankPar(data, member)
    triplet_guess = initialguesses(A0, Fw, gammaP, gammaF, member, hand, q0, z0, beta, machCtr, gammaroot, S0, toolvec[0], RA)
    
    return triplet_guess

def initialize_design_data(): # return designData

    commonStrings = ['GearCommonData', 'PinionCommonData']
    subCommonStrings = ['gear', 'pin']
    cutterStrings = ['GearConcaveCutterData','GearConvexCutterData','PinionConcaveCutterData','PinionConvexCutterData']
    subCutterStrings = ['gearCutCnv', 'gearCutCvx', 'pinCutCnv', 'pinCutCvx']
    machineStrings = ['GearConcaveMachineSettings', 'GearConvexMachineSettings', 'PinionConcaveMachineSettings', 'PinionConvexMachineSettings']
    subMachineStrings = ['gearCnv', 'gearCvx', 'pinCnv', 'pinCvx']

    SystemData = {
        'HAND': 0,
        'shaft_angle': 0,
        'ratio': 0,
        'hypoidOffset': 0,
        'E': [],
        'P': [],
        'G': [],
        'alpha': [],
        }
    
    Common = [1,2]
    Machine = [1,2,3,4]
    Cutter = [1,2,3,4]

    for ii in range(0, 2):
        Common[ii] =  {
            f'{subCommonStrings[ii]}GenType' : 0,
            f'{subCommonStrings[ii]}NTEETH' : 0,
            f'{subCommonStrings[ii]}SPIRALANGLE' : 0,
            f'{subCommonStrings[ii]}OUTERCONEDIST' : 0,
            f'{subCommonStrings[ii]}FACEWIDTH' : 0,
            f'{subCommonStrings[ii]}FACEANGLE' : 0,
            f'{subCommonStrings[ii]}BACKANGLE' : 0,
            f'{subCommonStrings[ii]}FRONTANGLE' : 0,
            f'{subCommonStrings[ii]}PITCHANGLE' : 0,
            f'{subCommonStrings[ii]}PITCHAPEX' : 0,
            f'{subCommonStrings[ii]}FACEAPEX' : 0,
            f'{subCommonStrings[ii]}ROOTAPEX' : 0,
        }

    for ii in range(0, 4):
        Machine[ii] = {
            f'{subMachineStrings[ii]}RADIALSETTING': 0,
            f'{subMachineStrings[ii]}CRADLEANGLE': 0,
            f'{subMachineStrings[ii]}TILTANGLE': 0,
            f'{subMachineStrings[ii]}SWIVELANGLE': 0,
            f'{subMachineStrings[ii]}BLANKOFFSET': 0,
            f'{subMachineStrings[ii]}ROOTANGLE': 0,
            f'{subMachineStrings[ii]}MACHCTRBACK': 0,
            f'{subMachineStrings[ii]}SLIDINGBASE': 0,
            f'{subMachineStrings[ii]}RATIOROLL': 0,
            f'{subMachineStrings[ii]}2C': 0,
            f'{subMachineStrings[ii]}6D': 0,
            f'{subMachineStrings[ii]}24E': 0,
            f'{subMachineStrings[ii]}120F': 0,
            f'{subMachineStrings[ii]}720G': 0,
            f'{subMachineStrings[ii]}5040H': 0,
            f'{subMachineStrings[ii]}H1': 0,
            f'{subMachineStrings[ii]}H2': 0,
            f'{subMachineStrings[ii]}H3': 0,
            f'{subMachineStrings[ii]}H4': 0,
            f'{subMachineStrings[ii]}H5': 0,
            f'{subMachineStrings[ii]}H6': 0,
            f'{subMachineStrings[ii]}V1': 0,
            f'{subMachineStrings[ii]}V2': 0,
            f'{subMachineStrings[ii]}V3': 0,
            f'{subMachineStrings[ii]}V4': 0,
            f'{subMachineStrings[ii]}V5': 0,
            f'{subMachineStrings[ii]}V6': 0,
            f'{subMachineStrings[ii]}R1': 0,
            f'{subMachineStrings[ii]}R2': 0,
            f'{subMachineStrings[ii]}R3': 0,
            f'{subMachineStrings[ii]}R4': 0,
            f'{subMachineStrings[ii]}R5': 0,
            f'{subMachineStrings[ii]}R6': 0,
        }

        Cutter[ii] =  {
            f'{subCutterStrings[ii]}POINTRADIUS' : 0,
            f'{subCutterStrings[ii]}BLADEANGLE' : 0,
            f'{subCutterStrings[ii]}EDGERADIUS' : 0,
            f'{subCutterStrings[ii]}TYPE' : "STRAIGHT",
            f'{subCutterStrings[ii]}TopremTYPE' : "NONE",
            f'{subCutterStrings[ii]}FlankremTYPE' : "NONE",
            f'{subCutterStrings[ii]}RHO' : 0,
            f'{subCutterStrings[ii]}TopremDEPTH' : 0,
            f'{subCutterStrings[ii]}TopremRADIUS' : 0,
            f'{subCutterStrings[ii]}TopremANGLE' : 0,
            f'{subCutterStrings[ii]}FlankremDEPTH' : 0,
            f'{subCutterStrings[ii]}FlankremRADIUS' : 0,
            f'{subCutterStrings[ii]}FlankremANGLE' : 0
        }

    designData = {'SystemData' : SystemData,
                commonStrings[0] : Common[0],
                commonStrings[1] : Common[1],
                machineStrings[0]: Machine[0],
                machineStrings[1]: Machine[1],
                machineStrings[2]: Machine[2],
                machineStrings[3]: Machine[3],
                cutterStrings[0]: Cutter[0],
                cutterStrings[1]: Cutter[1],
                cutterStrings[2]: Cutter[2],
                cutterStrings[3]: Cutter[3]}

    return designData


def main():
    # designData = initialize_design_data()
    # dictprint(designData)
    # filename = r'C:\Users\egrab\Desktop\designData.txt'
    # print(filename)
    # dict_to_file(designData, filename)
    import subprocess
    import time 

    for ii in range(0,4):
        subprocess.Popen(["notepad.exe"], shell=True) 

if __name__ == '__main__':
    main()