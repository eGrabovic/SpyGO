import casadi as ca
import screwCalculus as sc
import numpy as np
import matplotlib.pyplot as plt

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
    if i.shape[0] != 0:
        i = i[-1][0]
    else:
        i = 0
    if u >= U[-1]:
        i = max(U.shape) - p - 2
    if u <= U[0]:
        i = p

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
    z1 = np.array([])
    z2 = np.array([])
    if endL > 0:
        z1 = np.zeros((endL))
    if i-p > 0:
        z2 = np.zeros((i-p))
    N = np.r_[z2, N, z1]

    return N

def der_basis_fun_i(u, i, p, U, derOrder):
    """
    This basis function evaluation allows only numerical evaluation
    u: coordinate value
    i: knot interval
    p: degree
    U: knot vector
    derOrder: order of derivative to compute
    """
    if not isinstance(U, np.ndarray):
        U = np.array(U)

    # arrays initialization
    ndu = np.zeros((p + 1, p + 1)) # basis matrix with non vanishing elements
    left = np.zeros((p+1))
    right = np.zeros((p+1))
    if isinstance(u, ca.SX) or isinstance(u, ca.MX):
        ndu = ca.SX(p + 1, p + 1) # basis matrix with non vanishing elements
        left = ca.SX(1, p+1)
        right = ca.SX(1, p+1)
    ndu[0, 0] = 1
    
    for j in range(1, p+1):
        left[j] = u - U[i+1-j]
        right[j] = U[i+j] - u
        res = 0
        for r in range(0, j):
            ndu[j, r] = right[r+1]+left[j-r]
            temp = ndu[r, j-1]/(ndu[j, r])
            ndu[r, j] = res + right[r+1]*temp
            res = left[j-r]*temp
        ndu[j, j] = res

    # need to fill the basis vector with vanishing elements to easily perform matrix operations
    ders = np.zeros((derOrder+1, p+1))
    a = np.zeros((2, p+1))
    if isinstance(u, ca.SX) or isinstance(u, ca.MX):
        ders = ca.SX(derOrder+1, p+1) # basis matrix with non vanishing elements
        a = ca.SX(2, p+1)
    ders[0, :] = ndu[:, p]
    
    for r in range(0,p+1):
        s1 = 0
        s2 = 1
        a[0,0] = 1

        for k in range(1, derOrder+1):
            d = 0
            rk = r - k
            pk = p - k
            
            if r >= k:
                a[s2, 0] = a[s1, 0]/ndu[pk+1,rk]
                d = a[s2, 0]*ndu[rk, pk]
            
            if rk >= -1:
                j1 = 1
            else:
                j1 = -rk
            if r-1 <= pk:
                j2 = k-1
            else: 
                j2 = p-r

            for j in range(j1,j2+1):
                a[s2, j] = (a[s1, j] - a[s1,j-1])/ndu[pk + 1, rk + j]
                d += a[s2, j]*ndu[rk + j, pk]

            if r <= pk:
                a[s2, k] = -a[s1, k-1]/ndu[pk + 1, r]
                d += a[s2, k]*ndu[r, pk]

            ders[k, r] = d
            j = s1
            s1 = s2
            s2 = j

    r1 = p
    for k in range (1, derOrder+1):
        ders[k, :] = ders[k, :]*r1
        r1 *= (p-k)

    # need to add zeros to null bases
    lenU = max(U.shape)
    numCtrlPts = lenU-p-1
    endLen = numCtrlPts - i -1
    zers1 = np.array([[]])
    zers2 = np.array([[]])
    if endLen > 0:
        zers1 = np.zeros((derOrder+1, endLen))

    if i-p > 0:
        zers2 = np.zeros((derOrder+1, i-p))

    if isinstance(u, ca.SX) or isinstance(u, ca.MX):
        return ca.horzcat(zers2, ders, zers1).T
    return np.hstack([zers2, ders, zers1]).T

def basis_fun_casadi(p, U, derOrder):

    if not isinstance(U, np.ndarray):
        U = np.array(U)

    u = ca.SX.sym('u')
    i_start = np.argwhere(U == 0)
    i_start = i_start[-1][0]
    i_end = np.argwhere(U == 1)
    i_end = i_end[0][0]
    Z = ca.SX(max(U.shape)-(p+1), 1)

    for i in range(i_start,i_end):
        if i == i_end:
            bool = ca.logic_and(u>=U[i], u<=U[i+1])
        else:
            bool = ca.logic_and(u>=U[i], u<U[i+1])
        Z += bool*der_basis_fun_i(u, i, p, U, derOrder)
    return ca.Function('N', [u], [Z])

def chord_length_nodes(pU, pV, Q):
    """
    pU
    pV
    Q
    """
    shp = Q.shape
    n = shp[1]
    m = shp[2]

    u = np.full((m, n), np.nan)
    v = np.full((n, m), np.nan)
    X = np.reshape(Q[0,:,:], (n, m), order = 'F')
    Y = np.reshape(Q[1,:,:], (n, m), order = 'F')
    Z = np.reshape(Q[2,:,:], (n, m), order = 'F')


    for ii in range(0, m):
        u[ii, 0:n-1] = 0
        u[ii, -1] = 1

        cds = np.sqrt( np.diff(X[:, ii])**2 + np.diff(Y[:,ii])**2 + np.diff(Z[:,ii])**2 )
        total = np.sum(cds)
        d = 0; 
        for kk in range(0, max(cds.shape)):
            u[ii, kk] = d/total
            d += cds[kk]
    u = np.sum(u, axis=0)/m

    for ii in range(0, n):
        v[ii, 0:m-1] = 0
        v[ii, -1] = 1

        cds = np.sqrt( np.diff(X[ii, :])**2 + np.diff(Y[ii, :])**2 + np.diff(Z[ii, :])**2 )
        total = np.sum(cds)
        d = 0; 
        for kk in range(0, max(cds.shape)):
            v[ii, kk] = d/total
            d += cds[kk]
    v = np.sum(v, axis=0)/m

    knotsU = np.full((n + pU + 1), np.nan)
    knotsU[0:pU] = 0
    for j in range(0, n-pU):
        knotsU[j+pU] = 1/pU*sum(u[j: j+pU])

    knotsU[-pU:] = 1

    knotsV = np.full((m + pV + 1), np.nan)
    knotsV[0:pV] = 0
    for j in range(0, m-pV):
        knotsV[j+pV] = 1/pU*sum(u[j: j+pV])

    knotsV[-pV:] = 1

    return u, v, knotsU, knotsV

def fit_knots(uk, num_control_points, degree):
    n = max(uk.shape)
    knotsU = np.empty((num_control_points + degree + 1))
    knotsU[:] = np.nan
    knotsU[0:degree+1] = 0
    knotsU[-(degree+1):] = 1
    d = n/(num_control_points - degree)
    for jj in range(0, num_control_points-1-degree):
        i = int((jj+1)*d)
        alpha = (jj+1)*d - i
        knotsU[degree+jj+1] = (1-alpha)*uk[i-1] + alpha*uk[i]
    return knotsU

def fit_nurbs_surface(target_points, degreeU, degreeV, control_points_shape):
    """

    """
    Q = target_points
    if isinstance(target_points, dict):
        shp = Q['X'].shape
        R = np.zeros((3, shp[0], shp[1]))
        R[1,:,:] = Q['X']
        R[2,:,:] = Q['Y']
        R[3,:,:] = Q['Z']
        Q = R

    shp = Q.shape
    Xint = np.reshape(Q[0,:,:], (shp[1], shp[2]), order='F')
    Yint = np.reshape(Q[1,:,:], (shp[1], shp[2]), order='F')
    Zint = np.reshape(Q[2,:,:], (shp[1], shp[2]), order='F')

    pU = degreeU
    pV = degreeV

    uk, vk, _, _ = chord_length_nodes(pU, pV, Q)
    knotsU = fit_knots(uk, control_points_shape[0], degreeU)
    knotsV = fit_knots(vk, control_points_shape[1], degreeV)
    lenU = knotsU.shape[0]
    lenV = knotsV.shape[0]

    # basis functions calculation
    Ni = np.full((uk.shape[0], lenU-pU-1), np.nan)
    for kk in range(0, uk.shape[0]):
        Ni[kk, :] = basis_function(uk[kk], pU, knotsU)
        
    Nj = np.full((vk.shape[0], lenV-pV-1), np.nan)
    for kk in range(0, vk.shape[0]):
        Nj[kk, :] = basis_function(vk[kk], pV, knotsV)

    Wi = np.eye(Ni.shape[0])
    Wj = np.eye(Nj.T.shape[0])

    pseudoNi = np.linalg.pinv(Ni)
    pseudoNj2 = np.linalg.pinv(Nj.T)

    Px = pseudoNi@Xint@pseudoNj2
    Py = pseudoNi@Yint@pseudoNj2
    Pz = pseudoNi@Zint@pseudoNj2

    Px = np.reshape(Px, (1, control_points_shape[0], control_points_shape[1]), order='F')
    Py = np.reshape(Py, (1, control_points_shape[0], control_points_shape[1]), order='F')
    Pz = np.reshape(Pz, (1, control_points_shape[0], control_points_shape[1]), order='F')
    control_points = np.concatenate((Px, Py, Pz), axis = 0)


    return control_points, knotsU, knotsV, uk, vk

class Nurbs:
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
    
    def eval(self, u_values, v_values):
        
        if not isinstance(u_values, list):
            u_values = [u_values]
        if not isinstance(u_values, np.ndarray):
            u_values = np.array(u_values)
        if not isinstance(v_values, list):
            v_values = [v_values]
        if not isinstance(v_values, np.ndarray):
            v_values = np.array(v_values)

        shp = u_values.shape; n = shp[0]; 
        try: 
            m = shp[1]
        except: 
            m = 1

        # if u_values.shape != v_values.shape:
        #     raise TypeError("u and v arryas must have the same size (shape)")
        P = self.controlPoints
        U = self.knotsU
        V = self.knotsV
        pU = self.degreeU
        pV = self.degreeV

        u = u_values.flatten() # order = 'F' to match matlab
        v = v_values.flatten() # order = 'F' to match matlab

        if u.size == 1:
            u = u*np.ones(v.shape)
        if v.size == 1:
            v = v*np.ones(u.shape)

        Ni = np.nan*np.zeros((u.size, U.size - pU - 1))
        Nj = np.nan*np.zeros((v.size, V.size - pV - 1))
        for ii in range(0, u.size):
            Ni[ii, :] = basis_function(u[ii], pU, U)
            Nj[ii, :] = basis_function(v[ii], pV, V)

        X = P[0,:,:]
        Y = P[1,:,:]
        Z = P[2,:,:]

        S = {'x': np.reshape(np.sum(Ni.T*(X @ Nj.T), axis = 0), (n, m)).squeeze(),
             'y': np.reshape(np.sum(Ni.T*(Y @ Nj.T), axis = 0), (n, m)).squeeze(),
             'z': np.reshape(np.sum(Ni.T*(Z @ Nj.T), axis = 0), (n, m)).squeeze()}

        return S
    
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

        # compute residuals
        points = np.full((3, uk.shape[0], vk.shape[0]), np.nan)
        normals = np.full((3, uk.shape[0], vk.shape[0]), np.nan)
        return
    
    def plot(self, nU = 150, nV = 150, show_normals = False):
        u_values = np.linspace(0, 1, 100)
        v_values = np.linspace(0, 1, 100)
        X = np.nan*np.zeros(nU, nV)
        Y = np.nan*np.zeros(nU, nV)
        Z = np.nan*np.zeros(nU, nV)

        return

    

def main():

    
    file_path = 'Data/Q.dat'
    
    data = np.loadtxt('Data/Q.dat')
    
    Q = {
        'X': data[:, 0].reshape((120, 179), order='F'),
        'Y': data[:, 1].reshape((120, 179), order='F'),
        'Z': data[:, 2].reshape((120, 179), order='F')
    }
    nurbs = Nurbs([],[],[],[],[])
    nurbs.fit(Q, 3, 3, (50,80))
    print(nurbs.eval([0.1,0.2,0.3,0.5], [0.1,0.2,0.3,0.5]))

    return

if __name__ == "__main__":
    main()
    # print(basis_function(0.3, 3, [0,0,0,0,0.5,1,1,1,1]))
