from casadi.casadi import exp
import numpy as np
import casadi as ca
import screwCalculus as sc
from hypoid_functions import *
from hypoid_utils import *
from hypoid_sampling import *

class Hypoid:

    def __init__(self, designData, toothData, coneData, nProf = 11, nFace = 16, nFillet = 12):
        #tooth sampling size
        self.nProf = nProf
        self.nFace = nFace
        self.nFillet = nFillet

        # info
        self.pinGenerationProcess = " "
        self.gearGenerationProcess = " "
        self.initialToothData = {}
        self.initialConeData = {}

        # main design data (machine-tool settings)
        self.designData = {}

        # sampling data
        template_dictionary = {'pinion' : {'concave':[], 'convex':[]}, 'gear': {'concave':[], 'convex':[]}}
        self.surfPoints = template_dictionary
        self.surfNormals = template_dictionary
        self.surfTriplets = template_dictionary
        self.filletPoints = template_dictionary
        self.interpTriplets = template_dictionary
        self.surfcurvature = template_dictionary
        self.eqMeshing = template_dictionary
        self.Point = template_dictionary   # casadi function
        self.Normal = template_dictionary # casadi function
        self.pointsFullBounds = template_dictionary
        self.normalsFullBounds = template_dictionary

        # nurbs output
        template_dictionary = {'pinion' : {'concave':[], 'convex':[], 'both': []}, 'gear': {'concave':[], 'convex':[], 'both': []}}
        self.nurbsFit = template_dictionary

        # zR tooth data boundaries
        template_dictionary = {'pinion' : {'concave':[], 'convex':[]}, 'gear': {'concave':[], 'convex':[]}}
        self.zRfillet = template_dictionary
        self.zRfillet = template_dictionary                   # flank-fillet transition line in axial plane
        self.zRfullvec = template_dictionary                   # z - R coordinates in array form derived from sampling points (nProf + nFillet)xnFace
        self.zRbounds = template_dictionary                   # bounds on flank fillet transition
        self.zRwithRoot = {'pinion':[], 'gear':[]}                  # bounds with the fillet
        self.zRrootTriplets = template_dictionary              # triplets for the rootcone sampling
        self.zRfullBounds = template_dictionary
        self.zRPCA = template_dictionary
        self.zRinOther = template_dictionary
        self.zRinOtherCorners = template_dictionary
        self.zRtipOther = template_dictionary
        self.rootLineStruct = template_dictionary 

        # rigid TCA
        self.pathCurve = []
        self.TCAfun = []
        self.pinTCA = []
        self.gearTCA = []
        self.pinPhiRange  = []   
        self.gearPhiRange = []

        # rotor info
        self.EPGalpha = []

        # data structs for mach-tool identification 
        self.currentEaseOff = []
        self.identificationProblemConjugate = []      # conjugate pinion identification problem struct
        self.identificationProblemEaseOff = []        # easeOff identification problem struct
        self.identificationProblemOptimization = []   # embedded identification problem for the automatic optimization
        self.identificationProblemSpreadBlade = []    # spread blade identification
        self.identificationProblemTopography = []     # generic topography identification

        # data for Calyx interface
        self.LTCA = {}  # LTCA data structure

        # call constructor
        input = 'designData'
        if toothData is not None:
            input = 'coneData'
        self.constructHypoid(designData = designData, toothData = toothData, coneData = coneData, inputData = input)

    ## end of class constructor

    @staticmethod
    def conesIntersection(cone1, cone2):
        Ar = cone1[0]
        Az = cone1[1]
        B = cone1[2]

        Ar2 = cone2[0]
        Az2 = cone2[1]
        B2 = cone2[2]

        R = (Az2/Az*B - B2)/(Ar2 - Az2/Az*Ar)
        z = -(B+Ar*R)/Az

        zR = [z, R]
        return zR
    
    @staticmethod
    def EPGalphaToFrames(EPGalpha, data):
        """
        EPGalpha misalignments converted to frame displacements and z axis orientation
        """
        handPin = data['SystemData']['HAND']
        shaft_angle = data['SystemData']['shaft_angle']
        signOffset = -(int(handPin.lower() == 'right') - int(handPin.lower() == 'left'))
        offset = data['SystemData']['hypoidOffset']
        pin_dict = {}
        gear_dict = {}
        pin_dict['originXYZ'] = [EPGalpha[1], signOffset*(offset + EPGalpha[0]), 0]
        pin_dict['Zdir'] = [sin(shaft_angle*pi/180 + EPGalpha[3]), 0, cos(shaft_angle*pi/180 + EPGalpha[3])]
        gear_dict['originXYZ'] = [0, 0, EPGalpha[2]]
        gear_dict['Zdir'] = [0, 0, 1]
        return pin_dict, gear_dict
    
    @staticmethod
    def sideFromMemberAndFlank(member, flank):
        side = 'coast'
        if (member.lower() == 'pinion' and flank.lower() == 'concave') or (member.lower() == 'gear' and flank.lower() == 'convex'):
            side = 'drive'
        return side
    
    @staticmethod
    def flankFromMemberAndSide(member, side):
        flank = 'convex'
        if (side.lower() == 'drive' and member.lower() == 'pinion') or (side.lower() == 'coast' and member.lower() == 'gear'):
            flank = 'concave'
        return flank
    
    @staticmethod
    def getIndexArray():
        """
        sequential order of the machine-tool settings
        """
        indexes = [
                1, 10, 19, 28, 37, 46, 55, 64, # 1
                2, 11, 20, 29, 38, 47, 56, 65, # 9
                3, 12, 21, 30, 39, 48, 57, 66,# 17
                4, 13, 22, 31, 40, 49, 58, 67,# 25
                5, 14, 23, 32, 41, 50, 59, 68,# 33
                6, 15, 24, 33, 42, 51, 60, 69,# 41
                16, 25, 34, 43, 52, 61, 70,# 48
                8, 17, 26, 35, 44, 53, 62, 71,# 56
                9, 18, 27, 36, 45, 54, 63, # 64
                72, 73, 74, 75, 76, 77, 78, 79, 80, 81, # concave tool
                82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92 # convex tool
                ]
        return indexes
    
    def constructHypoid(self, designData = None, toothData = None, coneData = None, gearGenType = 'Generated', EPGalpha = np.zeros((4,1)), inputData = "designData"):
        self.initialCone = coneData
        self.initialToothData = toothData

        if inputData.lower() != "designData".lower():
            systemData = designData

            method = 1
            if systemData["hypoidOffset"] == 0:
                method = 0

            self.designData = AGMAcomputationHypoid(systemData["HAND"], 
                                                    systemData["taper"],
                                                    coneData, toothData,
                                                    rc0 = coneData["rc0"],
                                                    GearGenType = gearGenType,
                                                    Method = method)
            
            self.designData = shaftSegmentComputation(self.designData)
            self.designData, triplets_pin_CNV, triplets_pin_CVX = approxToolIdentification_casadi(self.designData, 'Pinion', RHO = 50000)
            self.designData, triplets_gear_CNV, triplets_gear_CVX = approxToolIdentification_casadi(self.designData, 'Gear'  , RHO = 500)
            setattr(self, 'init_triplet', {
                'pinion':
                                            {
                                                'concave': triplets_pin_CNV,
                                                'convex' : triplets_pin_CVX
                                            },
                                            'gear':
                                            {
                                                'concave':triplets_gear_CNV,
                                                'convex':triplets_gear_CVX
                                            }
                                          }
                    )
        
        return 
    
    def sampleSurface(self, member, flank, sampling_size = None, extend_tip = False, updateData = True):
        """
        sampling_size = [n_face, n_profile, n_fillet]
        extend_tip = extend the tip of the blank's face cone boundary (useful for boolean subtractions)
        updateData = updates Hypoid properties (surface points, enveloping triplets, etc. ). Such data is often used as initial guess for other methods
        """
        side = self.sideFromMemberAndFlank(member, flank)
        data = self.designData

        # extract the sampling size (face, profile and fillet points)
        nF = self.nFace; nP = self.nProf; nfil = self.nFillet
        if sampling_size is not None:
            nF = sampling_size[0];  nP = sampling_size[1];  nfil = sampling_size[2]
        
        if extend_tip:
            common, subcommon = get_data_field_names(member, flank, fields = 'common')
            data[common][f'{subcommon}FACEAPEX'] = data[common][f'{subcommon}FACEAPEX'] + 1
            data[common][f'{subcommon}OUTERCONEDIST'] = data[common][f'{subcommon}OUTERCONEDIST'] + 0.3
            data[common][f'{subcommon}FACEWIDTH'] = data[common][f'{subcommon}FACEWIDTH'] + 0.6

        p, n, p_tool, n_tool, csi_theta_phi, z_tool, p_fillet, p_root, n_root, root_variables, p_bounds, nbounds =\
              surface_sampling_casadi(data, member, flank, [nF, nP, nfil], triplet_guess = None, spreadblade = False)
        
        if updateData == True:
            self.surfPoints[member][flank] = p
            self.surfNormals[member][flank] = n
            self.filletPoints[member][flank] = p_fillet
            self.surfTriplets[member][flank] = csi_theta_phi
            z = p_fillet[2,:]
            R = np.sqrt(p_fillet[0,:]**2 + p_fillet[1,:]**2)
            self.zRfillet[member][flank] = np.r_[z, R]
            csi_theta_phi = reduce_2d(csi_theta_phi)

        
        return 

    def computeParameters(self):
        return
    
    def buildCasadiDerivatives(self):
        return
    
    def getIndexArray(self):
        return
    
    def samplezR(self):
        return
    
    # def fun(self):
    #     value = 5
    #     setattr(self, 'newAtrtibute', value)


# 371211
class MyClass:
    
    
    i = 12345
    
    
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    def fun(self):
        setattr(self, 'newAttribute', 3)
        return self.newAttribute
    
    def print(self):
        for attr, value in self.__dict__.items():
            print(attr, value)
    
    
    
    
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
    H = Hypoid(SystemData, toothData, coneData)

    H.sampleSurface('gear', 'concave')
    H.sampleSurface('gear', 'convex')
    
    H.sampleSurface('pinion', 'concave')
    H.sampleSurface('pinion', 'convex')
    
    
    return
    
if __name__ == "__main__":
    main()
    