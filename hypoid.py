from casadi.casadi import exp
import numpy as np
import casadi as ca
import screwCalculus as sc
from hypoid_functions import *

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
        self.surfPoints = []
        self.surfNormals = []
        self.surfTriplets = []
        self.filletPoints = []
        self.interpTriplets = []
        self.surfcurvature = []
        self.eqMeshing = []
        self.Point = []   # casadi function
        self.Normal = [] # casadi function
        self.pointsFullBounds = []
        self.normalsFullBounds = []

        # nurbs output
        self.nurbsFit = []

        # zR tooth data boundaries
        self.zRfillet = []
        self.zRfillet = []                    # flank-fillet transition line in axial plane
        self.zRfullvec = []                   # z - R coordinates in array form derived from sampling points (nProf + nFillet)xnFace
        self.zRbounds = []                    # bounds on flank fillet transition
        self.zRwithRoot = []                  # bounds with the fillet
        self.zRrootTriplets = []              # triplets for the rootcone sampling
        self.zRfullBounds = []
        self.zRPCA = []
        self.zRinOther = []
        self.zRinOtherCorners = []
        self.zRtipOther = []
        self.rootLineStruct = [] 

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
        self.constructHypoid()

    ## end of class constructor

    @staticmethod
    def conesIntersection(cone1, cone2):
        return
    @staticmethod
    def EPGalphaToFrames(EPGalpha, data):
        return
    @staticmethod
    def sideFromMemberAndFlank(member, flank):
        return
    @staticmethod
    def getIndexArray():
        return
    
    def constructHypoid(self, designData = None, toothData = None, coneData = None, gearGenType = 'Generated', EPGalpha = np.zeros((4,1)), inputData = "designData"):
        self.initialCone = coneData
        self.initialToothData = toothData

        if inputData.lower() != "designData".lower():
            systemData = designData

            if systemData["hypoidOffset"] == 0:
                method = 0
            else:
                method = 1
            
            self.designData = AGMAcomputationHypoid(systemData["HAND"], 
                                                    systemData["taper"],
                                                    coneData, toothData,
                                                    rc0 = coneData["rc0"],
                                                    GearGenType = gearGenType,
                                                    Method = method)
            
            self.designData = shaftSegmentComputation(self.designData)
            self.designData, tiplets_pin_CNV, triplets_pin_CVX = approxToolIdentification_casadi(self.designData, 'Pinion', RHO = 90000)
            self.designData, triplets_gear_CNV, triplets_gear_CVX = approxToolIdentification_casadi(self.designData, 'Gear'  , RHO = 750)
        
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
    # class1 = MyClass(1,2,3)
    # class1.fun()
    # class1.print()
    designData = {'systemData':{'a':1,'tau': 2}}
    print(designData)
    
    
if __name__ == "__main__":
    main()
    