import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from scipy.optimize import Bounds as minBounds

# From https://github.com/marmakoide/inside-3d-mesh
import stlparser
from is_inside_mesh import is_inside_turbo as is_inside
class sphereOptimization():
    def __init__(self,triangles,minR,n):
        self.numAtoms = n
        self.triangles = triangles
        self.minR = minR
        self.inF = None
        self.P = None
        self.atoms = np.zeros([n,4])
        self.Bounds = None
        self.BoundVolume = 0.0

    def getRandomPoints(self,numTries):
        min_corner = np.array([self.Bounds[0::2]])
        max_corner = np.array([self.Bounds[1::2]])
        P = (max_corner - min_corner) * np.random.random((numTries, 3)) + min_corner
        # P = P.astype('longdouble')
        return P

    def getVolume(self, numTries):

        # self.getBounds()

        # self.updateVolume()
        if self.P is None:
            self.P = self.getRandomPoints(numTries)

        if self.inF is None:
            # inF = np.full(numTries,False)
            self.inF = is_inside(self.triangles,self.P)
        
        inS = np.full([numTries, self.numAtoms],False)
        for k in range(0,self.numAtoms):
            inS[:,k] = is_inside_sphere(self.atoms[k],self.P)
        
        return self.inF, inS
        

    def updateVolume(self):
        self.BoundVolume = (
            (self.Bounds[1]-self.Bounds[0])*
            (self.Bounds[3]-self.Bounds[2])*
            (self.Bounds[5]-self.Bounds[4])
        )        

    def getBounds(self):
        stlMin = np.amin(np.amin(self.triangles, axis=0), axis=0)
        stlMax = np.amax(np.amax(self.triangles, axis=0), axis=0)
        dX = stlMax - stlMin
        stlMin -= 0.5*dX
        stlMax += 0.5*dX
        # for atom in self.atoms:
        #     for k in range(3):
        #         stlMin[k] = min([stlMin[k], atom[k]-atom[3]])
        #         stlMax[k] = max([stlMax[k], atom[k]+atom[3]])
        
        self.Bounds = [
            stlMin[0], stlMax[0],
            stlMin[1], stlMax[1],
            stlMin[2], stlMax[2]
        ]

    def getInitAtoms(self):
        dx = 0.5*(self.Bounds[1]-self.Bounds[0])
        dy = 0.5*(self.Bounds[3]-self.Bounds[2])
        dz = 0.5*(self.Bounds[5]-self.Bounds[4])

        for k in range(self.numAtoms):
            self.atoms[k,0] = self.Bounds[0] + np.random.rand(1)*dx
            self.atoms[k,1] = self.Bounds[2] + np.random.rand(1)*dy
            self.atoms[k,2] = self.Bounds[4] + np.random.rand(1)*dz
            self.atoms[k,3] = np.random.rand(1)*(np.sqrt(dx*dx+dy*dy+dz*dz)-self.minR) + self.minR
            
    def sphereFun(self,x):
        for k in range(self.numAtoms):
            self.atoms[k,0] = x[0*self.numAtoms+k]
            self.atoms[k,1] = x[1*self.numAtoms+k]
            self.atoms[k,2] = x[2*self.numAtoms+k]
            self.atoms[k,3] = x[3*self.numAtoms+k]
        # self.getBounds()
        inF, inS = self.getVolume(10000)

        # # Make sure each sphere is goes inside the file
        # for k in range(self.numAtoms):
        #     if np.sum(np.bitwise_not(np.copy(inS[:,k]),np.copy(inF))) == 0:
        #         return 10.0*len(inF)

        # Make sure each sphere intersects with the edge of the file
        for k in range(self.numAtoms):
            if np.sum(np.bitwise_xor(np.copy(inF),np.copy(inS[:,k]))) == 0:
                return 10.0*len(inF)
        
        # Distance between each sphere
        distBetweenSpheres = 0.0
        for k1 in range(self.numAtoms):
            atom_1 = self.atoms[k1]
            for k2 in range(k1+1,self.numAtoms):
                atom_2 = self.atoms[k2]
                dX = atom_1[:3] - atom_2[:3]
                norm2 = np.sqrt(np.sum(dX*dX))
                if atom_1[3]+atom_2[3] < norm2:
                    return 10.0*len(inF)
                distBetweenSpheres += norm2
                

        inSs = np.bitwise_or.reduce(np.copy(inS),axis=1)
        inFxOrInS = np.bitwise_xor(np.copy(inF), np.copy(inSs))
        inFandNotInS = np.bitwise_not(np.copy(inF),np.copy(inSs))
        FunVal = (
            np.sum(inFxOrInS) + 
            0.0*np.sum(inFandNotInS) - 
            1.0*np.sum(self.atoms[:,3]) +
            1.0*np.abs(np.sum(inF)-np.sum(inSs)) -
            distBetweenSpheres )
        # print(FunVal)
        return float(FunVal)

    def doOpt(self):
        
        self.getBounds()
        self.getInitAtoms()
        
        mBL = np.zeros([self.numAtoms,4])
        mBH = np.zeros([self.numAtoms,4])
        mBL[:,0] = self.Bounds[0]+self.minR
        mBL[:,1] = self.Bounds[2]+self.minR
        mBL[:,2] = self.Bounds[4]+self.minR
        mBL[:,3] = self.minR
        mBH[:,0] = self.Bounds[1]-self.minR
        mBH[:,1] = self.Bounds[3]-self.minR
        mBH[:,2] = self.Bounds[5]-self.minR
        mBH[:,3] = 0.25*np.sqrt(
            (self.Bounds[1]-self.Bounds[0])*(self.Bounds[1]-self.Bounds[0]) + 
            (self.Bounds[3]-self.Bounds[2])*(self.Bounds[3]-self.Bounds[2]) + 
            (self.Bounds[5]-self.Bounds[4])*(self.Bounds[5]-self.Bounds[4]))
        mB = minBounds(np.reshape(mBL,4*self.numAtoms,1),np.reshape(mBH,4*self.numAtoms,1))
        x0 = np.reshape(self.atoms,self.numAtoms*4)
        res = differential_evolution(
            self.sphereFun,
            bounds=mB,
            init='latinhypercube',
            disp=True,
            workers=1,
            popsize=35)
        for k in range(self.numAtoms):
            self.atoms[k,0] = res.x[0*self.numAtoms+k]
            self.atoms[k,1] = res.x[1*self.numAtoms+k]
            self.atoms[k,2] = res.x[2*self.numAtoms+k]
            self.atoms[k,3] = res.x[3*self.numAtoms+k]
        print(self.atoms)
        self.plotResult()

    def plotResult(self):
        fig = plt.figure()
        ax = Axes3D(fig)
        N = len(self.triangles)
        for k in range(N):
            curTri = self.triangles[k]
            x = np.append(curTri[:,0],curTri[0,0])
            y = np.append(curTri[:,1],curTri[0,1])
            z = np.append(curTri[:,2],curTri[0,2])
            ax.plot(x,y,z)

        u = np.linspace(0.0,2.0*np.pi,10)
        v = np.linspace(0.0,np.pi,10)
        cosu = np.cos(u)
        sinu = np.sin(u)
        cosv = np.cos(v)
        sinv = np.sin(v)
        oneSizeU = np.ones(np.size(u))
        
        dx = self.Bounds[1]-self.Bounds[0]
        dy = self.Bounds[3]-self.Bounds[2]
        dz = self.Bounds[5]-self.Bounds[4]

        maxD = np.max(np.array([dx,dy,dz]))


        for k in range(self.numAtoms):
            atom = self.atoms[k]
            x = atom[3]*np.outer(cosu, sinv) + atom[0]
            y = atom[3]*np.outer(sinu, sinv) + atom[1]
            z = atom[3]*np.outer(oneSizeU, cosv) + atom[2]
            ax.plot_surface(x,y,z,color='yellow')
        
        ax.set_xlim3d(self.Bounds[0], self.Bounds[0]+maxD)
        ax.set_ylim3d(self.Bounds[2], self.Bounds[2]+maxD)
        ax.set_zlim3d(self.Bounds[4], self.Bounds[4]+maxD)
        plt.show()

def is_inside_sphere(atom, P):
    ax = atom[0]
    ay = atom[1]
    az = atom[2]
    r2 = atom[3]*atom[3]

    R2 = (P[:,0]-ax)*(P[:,0]-ax) + (P[:,1]-ay)*(P[:,1]-ay) + (P[:,2]-az)*(P[:,2]-az)

    return r2 > R2

def doSphereOpt(stlFile, stlScale, minR, numAtoms):
    with open(stlFile,'r') as f:
        triangles = np.array([X for X, N in stlparser.load(f)])
    stlTri = stlScale*triangles
    opt = sphereOptimization(stlTri, minR, numAtoms)
    opt.doOpt()
    return 0