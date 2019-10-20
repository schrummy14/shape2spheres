import os
import sys
from tkinter import Tk
from tkinter import filedialog as fd
from shapeOpti import doSphereOpt as opt

Tk().withdraw()

def getSTL():
    return fd.askopenfilename()

if __name__ == "__main__":
    numArgs = len(sys.argv)
    if numArgs < 5:
        stlFile = getSTL()
        if len(stlFile) == 0:
            print("No File Selected...")
            exit()
        stlScale = float(input("Scale for file"))
        minR = float(input("Min radius to use: "))
        numAtoms = int(input("Number of spheres to use in optimization: "))

    else:
        minR = float(sys.argv[1])
        numAtoms = int(sys.argv[2])
        stlFile = sys.argv[3]
        stlScale = float(sys.argv[4])

    if os.path.isfile(stlFile):
        opt(stlFile, stlScale, minR, numAtoms)
    else:
        print("File does not exist...")

# python3 main.py 0.001 4 ../stlFiles/triangle.ast 0.001