import casadi as ca
import screwCalculus as sc
import numpy as np

def basis_function(u, p, U):
    """
    This basis function evaluation allows only numerical evaluation
    u: coordinate value
    p: degree
    U: knot vector
    i: knot interval
    """
    if not isinstance(U, np.ndarray):
        U = np.array(U)

    i = np.argwhere(u>U)
    i = i[-1][0]
    if u >= U[-1]:
        i = max(np.shape) - p - 2
    if u <= U[0]:
        i = p - 1

    # arrays initialization
    N = np.zeros((p+1)) # basis vector with non vanishing elements
    left = np.zeros((p+1))
    right = np.zeros((p+1))
    
    N[0] = 1
    for j in range(1, p+1):
        left[j] = u - U[i+1-j]
        right[j] = U[i+j] - u
        res = 0
        for r in range(0, j):
            temp = N[r]/(right[r+1]+left[j-r])
            N[r] = res + right[r+1]*temp
            res = left[j-r]*temp
        N[j] = res

    # need to fill the basis vector with vanishing elements to easily perform matrix operations
    lenU = max(U.shape)
    n = lenU - p - 1
    endL = n - i - 1
    z1 = np.zeros((endL))
    z2 = np.zeros((i-p))
    N = np.r_[z2, N, z1]

    return N

def fit_nurbs_surface(target_points, degreeU, degreeV, control_points_shape):

    return

class nurbs:
    def __init__(self, knotsU, knotsV, degU, degV, control_points) -> None:
        self.controlPoints = control_points
        self.knotsU = knotsU
        self.knotsV = knotsV
        self.degreeU = degU
        self.degreeV = degV
        self.orderU = []
        self.orderV = []
        self.fitResiduals = []
        self.casadi_function = []
        self.casadi_Ni = []
        self.casadi_Nj = []
        self.curvatureGaussian = []
        self.curvatureMean = []
        self.curvatureMax = []
        self.curvatureMin = []

        return
    
    def fit(self, target_points, degU, degV, control_points_shape):

        if isinstance(target_points, dict):
            grid_size = target_points['X'].shape
            points = np.zeros((3,grid_size[0], grid_size[1]))
            points[0,:,:] = target_points['X']
            points[1,:,:] = target_points['Y']
            points[2,:,:] = target_points['Z']
            target_points = points
        
        self.degreeU = degU
        self.degreeV = degV
        self.orderU = degU + 1
        self.orderV = degV + 1

        self.controlPoints, self.knotsU, self.knotsV, uk, vk = fit_nurbs_surface(target_points, self.degreeU, self.degreeV, control_points_shape)

def main():

    print(basis_function(0.3, 2, [0,0,0,0.3,0.5,0.7,1,1,1]))
    return
if __name__ == "__main__":
    main()