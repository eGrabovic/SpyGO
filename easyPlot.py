# -*- coding: utf-8 -*-
"""
Plotting package for matlab-like synthax within the pyqtgraph library

@author: Eugeniu Grabovic
"""

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from pyqtgraph.graphicsItems.DateAxisItem import DateAxisItem
import pyqtgraph.opengl as gl
import numpy as np
import sys
import screwCalculus as sc
import time

# Figure container
class Figure:
    
    app = []
    figure = []
    children = []
    nChildrens = 0
    parent = []
    animFun = None
    animIter = 0
    animData = None

    def __init__(self, title = 'Title', cameraDist = 50, loop = True) -> None:
        self.app = QtWidgets.QApplication.instance()
        if self.app is None: # check if an application already exists which doesn't allow a creation of a new one  pyqtgraph.mkColor(*args)
            self.app = QtWidgets.QApplication([])
        self.figure = gl.GLViewWidget()
        self.figure.setBackgroundColor(pg.mkColor('w'))
        self.figure.show()
        self.figure.setWindowTitle(title)
        self.figure.setCameraPosition(distance = cameraDist)
        xgrid = gl.GLGridItem(color = pg.mkColor('k'))
        self.figure.addItem(xgrid)
        self.timer = QtCore.QTimer()
        # ygrid = gl.GLGridItem(color = pg.mkColor('k'))
        # zgrid = gl.GLGridItem(color = pg.mkColor('k'))
        # self.figure.addItem(ygrid)
        # self.figure.addItem(zgrid)
        # rotate x and y grids to face the correct direction
        # xgrid.rotate(90, 0, 1, 0)
        # ygrid.rotate(90, 1, 0, 0)

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            self.app.exec_()

    def updateImage(self):
        self.app.processEvents()

    def addPlotObject(self, object):
        self.children.append(object)
        self.figure.addItem(object.GraphicsItem)
        self.nChildrens += 1

    def removeGraphics(self, object):
        self.figure.removeItem(object.GraphicsItem)
        self.children.remove(object)
        self.nChildrens -= 1

    def setAnimFun(self, fcn, animData = None):
        self.animFun = fcn
        if not animData is None:
            self.animData = animData

    def animation(self, iter = 0, msUpdate = 30):
        self.animIter = iter
        self.timer.timeout.connect(self.animationFunction)
        self.timer.start(msUpdate)

    def animationFunction(self):
        self.animFun(self.children, self.animIter, self.animData)
        self.animIter += 1

class go():  # graphical object container

    def __init__(self, fig, parent = None):
        self.parent = parent
        self.children = []
        self.currentTransform = np.eye(4)
        self.parentTransform = np.eye(4)
        self.figure = fig
        if parent is not None:
            self.parent.addChildren(self)
        return

    def updateData(self, X, Y, Z):
        """
        manually update x, y, z data of a graphical object
        """
        self.XData = X
        self.YData = Y
        self.ZData = Z
        self.GraphicsItem.setData(x=X, y=Y, z=Z)

    def getData(self):
        return self.GraphicsItem._x, self.GraphicsItem._y, self.GraphicsItem._z

    def applyTransform(self, T):
        """
        T = homogeneous matrix transform
        """
        self.GraphicsItem.setTransform(self.parentTransform@T)
        self.currentTransform = T # keep track of the current transform
        if len(self.children)>0:
            for i in range(0, len(self.children)):
                self.children[i].parentTransform = T

    def addChildren(self, object):
        self.children.append(object)
        object.parent = self

    def addParent(self, object):
        self.parent = object
        object.children.append(self)


class surface(go):
    """
    surface graphical object.
    Can be either created with a X Y Z meshgrid or with vertices and faces data
    """
    def __init__(self, fig, X = None, Y = None, Z = None, vert = None, faces = None, parent = None, color = 'b') -> None:
        super().__init__(fig, parent)
        self.XData = X
        self.YData = Y
        self.ZData = Z

        if vert is None:
            self.GraphicsItem = gl.GLSurfacePlotItem(x=X, y=Y, z=Z, shader = 'normalColor', computeNormals=True, smooth=True, antialias = True)
        else:
            self.GraphicsItem = gl.GLMeshItem(meshdata = gl.MeshData(vertexes = vert, faces=faces), edgeColor = np.tile(np.array([1,1,1,1]), (vert.size, 1)))
            self.GraphicsItem.setColor(pg.mkColor(color))
            self.GraphicsItem.setShader('shaded')
            self.GraphicsItem.edgeColors = pg.mkColor('k')
         # shaders: 'balloon', 'normalColor', 'viewNormalColor', 'shaded', edgeHilight', 'heightColor'
        self.figure.addPlotObject(self)
        self.figure.figure.show()        

class line(go):

    def __init__(self, fig, x, y, z, color = 'k', lineWidth = 5, parent = None) -> None:
        super().__init__(fig, parent)
        self.XData = x
        self.YData = y
        self.ZData = z
        sz = np.size(x)
        points = np.concatenate(
            (x.reshape((sz, 1)),
             y.reshape((sz, 1)),
             z.reshape((sz, 1))), axis = 1
        )
        color = pg.mkColor(color)
        self.GraphicsItem = gl.GLLinePlotItem(pos = points, width = lineWidth, color = color, mode = 'line_strip', antialias = True)
        self.figure.addPlotObject(self)
        self.figure.figure.show()

def createCylinder(r, ax, lenTop, lenBot, axialOff, N=15):
    """
    r = radius
    ax = axis direction (3x1 array)
    lenTop = axial length along positive ax direction
    lenBot = axial length along negative ax direction (note it is a POSITIVE value)
    axialOff = offset along axis (lenTop and lenBot refer to this new origin translated along axis)
    """

    theta = np.linspace(0, 2*np.pi, N).reshape(1, N)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    zTop = lenTop*np.ones((1, N))
    zBot = -lenBot*np.ones((1, N))
    ptTop = np.concatenate((x, y, zTop), axis = 0)
    ptBot = np.concatenate((x, y, zBot), axis = 0)

    theta = np.arccos(ax[2])
    phi = np.arctan2(ax[0], ax[1])
    R = sc.rotZ(-phi)@sc.rotX(-theta)
    ptTop = R@ptTop
    ptBot = R@ptBot
    ptX = np.concatenate((ptTop[0, :].reshape(1, N), ptBot[0, :].reshape(1, N)), axis = 0)
    ptY = np.concatenate((ptTop[1, :].reshape(1, N), ptBot[1, :].reshape(1, N)), axis = 0)
    ptZ = np.concatenate((ptTop[2, :].reshape(1, N), ptBot[2, :].reshape(1, N)), axis = 0) - axialOff

    # vertices and fasec for the lateral surface
    X = np.expand_dims(
        np.array([ptBot[0, 0:N-1], ptBot[0, 1:N], ptTop[0, 0:N-1], ptTop[0, 1:N]]).T.flatten(),
        axis = 1)
    Y = np.expand_dims(
        np.array([ptBot[1, 0:N-1], ptBot[1, 1:N], ptTop[1, 0:N-1], ptTop[1, 1:N]]).T.flatten(),
        axis = 1)
    Z = np.expand_dims(
        np.array([ptBot[2, 0:N-1], ptBot[2, 1:N], ptTop[2, 0:N-1], ptTop[2, 1:N]]).T.flatten(),
        axis = 1) - axialOff

    
    vertices = np.concatenate((X,Y,Z), axis = 1)
    faces = np.empty(((N-1)*2, 3), dtype=int)

    for k in range(0, N-1):
        j = k*4
        ii = k*2
        faces[ii  , :] = np.array([j, j+1, j+2], dtype=int)
        faces[ii+1, :] = np.array([j+1, j+3, j+2], dtype=int)

    return (ptX, ptY, ptZ, vertices, faces) #, verticesTop, verticesBot, faces, faces





def main():

    pX, pY, pZ, vertices, faces = createCylinder(0.5, np.array([0,0,1]), 1, 1, 0)
    x = np.array([0, 1, 2, 5])
    y = np.array([0, 1, 3, 8])
    z = np.array([0, 1, 1, 0])

    F = Figure() # init figure
    line1 = line(F, x, y, z, color = 'red')
    x = np.linspace(-4,4,100)
    y = np.linspace(-4,4,100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X)  + np.cos(Y)
    # surf = surface(F, vert=vertices, faces=faces, parent = line1)
    surf = surface(F, X = x, Y = y, Z = Z)
    

    def animationFun(objs, iter, userData):
        q = userData(iter)
        objs[0].applyTransform(sc.TrotY(q))
        objs[1].applyTransform(sc.TrotX(iter*0.1))

    userData =  lambda x: x/10

    F.setAnimFun(animationFun, userData) # set custom animation  function
    # F.animation()              # start animation

    # for i in range(0,200):
    #     surf.applyTransform(sc.TrotY(i*0.1))
    #     line1.applyTransform(np.eye(4))
    #     F.updateImage()
    
    F.start()                  # start widget
    

# def main4():
#     import pandas as pd
#     import plotly.graph_objects as go

#     z_data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv').values
#     print(z_data)
#     z_data2 = z_data * 1.1
#     z_data3 = z_data * 1.2
#     z_data4 = z_data * 0.5

#     z_data_list = []
#     z_data_list.append(z_data)
#     z_data_list.append(z_data2)
#     z_data_list.append(z_data3)
#     z_data_list.append(z_data4)
#     z_data_list.append(z_data)
#     z_data_list.append(z_data2)
#     z_data_list.append(z_data3)
#     z_data_list.append(z_data4)

#     fig = go.Figure(
#         data=[go.Surface(z=z_data_list[0])],
#         layout=go.Layout(updatemenus=[dict(type="buttons", buttons=[dict(label="Play", method="animate", args=[None])])]),
#         frames=[go.Frame(data=[go.Surface(z=k)], name=str(i)) for i, k in enumerate(z_data_list)]
#     )

#     fig.update_traces(contours_z=dict(show=True, usecolormap=True, highlightcolor="tomato", project_z=True), colorscale='portland')

#     fig.update_layout(title='data HEATPILES', autosize=False, width=650, height=500, margin=dict(l=0, r=0, b=0, t=0))

#     def frame_args(duration):
#         return {
#                 "frame": {"duration": duration},
#                 "mode": "immediate",
#                 "fromcurrent": True,
#                 "transition": {"duration": duration, "easing": "linear"},
#             }

#     sliders = [
#                 {
#                     "pad": {"b": 10, "t": 60},
#                     "len": 0.9,
#                     "x": 0.1,
#                     "y": 0,
#                     "steps": [
#                         {
#                             "args": [[f.name], frame_args(0)],
#                             "label": str(k),
#                             "method": "animate",
#                         }
#                         for k, f in enumerate(fig.frames)
#                     ],
#                 }
#             ]
        
#     fig.update_layout(sliders=sliders)
    
#     import plotly.io as pio

#     ii = 1
#     pio.write_html(fig, file="Live3D_"+str(ii)+".html", auto_open=True)
#     # plotly.offline.plot(fig)

if __name__ == "__main__":
    main()