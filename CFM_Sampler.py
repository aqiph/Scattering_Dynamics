#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 14:59:23 2019

@author: Aqiph
"""

import numpy as np
import math
import linecache
from scipy.stats import boltzmann
from scipy.stats import maxwell
from scipy import interpolate
import random



class Sampler():

### Constructor ###    
    def __init__(self):
        """
        Define data attributes
        """
        self.cell = []
        self.atomTypes = []
        self.atomNumbers = []
        self.positions = []
        self.flags = []
        self.v = []
        self.title = 'POSCAR'
        self.masses = {'Pt':195.080, 'H':1.000, 'C':12.011, 'Cu':63.546,
                       'Al':26.981, 'O':16.000, 'Sr':87.620}

    
### Inputs ###     
    def readPositions(self, inFileName, form, lcell):
        """
        Read a structure from an existing file,
        :param inFileName: str, the name of the input file, i.e. POSCAR
        :param form: str, the form of the input file, 'vasp' or 'ase'
        :param lcell: boolean, whether or not overwrite cell
        :return:
        atomTypes: list of strs, element types
        atomNumbers: list of ints, the number of each element type
        positions: list of lists, [[x1, y1, z1], [x2, y2, z2], ...]
        flags: list of strs
        velocities: list of lists, velocities, [[x1, y1, z1], [x2, y2, z2], ...]
        """
        cell = []
        atomTypes = []
        atomNumbers = []
        positions = []
        flags = []
        v = []
        
        with open(inFileName, 'r') as data:
            
            # Read atom types for 'ase' format input file
            if form == 'ase':
                atomTypes = data.readline().strip().split()
            elif form == 'vasp':
                data.readline()
            else:
                raise Exception('File type error')
                
            # Read lattice constant and lattice vectors
            latticeConstant = float(data.readline().strip())
            
            for _ in range(3):
                vector = data.readline().strip().split()
                vector = [latticeConstant * float(i) for i in vector]
                assert len(vector) == 3
                cell.append(vector)
                
            # Read atom types for 'vasp' format input file and the numbers of atoms
            if form == 'vasp':
                atomTypes = data.readline().strip().split()
                
            atomNumbers = data.readline().strip().split()
            atomNumbers = [int(i) for i in atomNumbers]           
            assert len(atomTypes) == len(atomNumbers)            
            numAtoms = sum(atomNumbers)
            
            # Read positions and flags
            if form == 'vasp':
                data.readline() 
                
            coordType = data.readline().strip()
            
            for n in range(numAtoms):
                line = data.readline().strip().split()
                
                coordAtom = [float(i) for i in line[:3]]
                
                if len(line) == 3:
                    flag = '      T   T   T'
                elif len(line) == 6:
                    flag = '      ' + '   '.join(line[3:])
                else:
                    print('Read position error')
                    
                positions.append(coordAtom)
                flags.append(flag)
                
            # Read velocities
            data.readline()
            
            for n in range(numAtoms):
                vAtom = data.readline().strip().split()
                vAtom = [float(i) for i in vAtom]
                assert len(vAtom) == 3
                v.append(vAtom)  
                                              
        # Change to Cartesian coordinate if coordType == 'Direct'
        if coordType == 'Direct':
            positionsCart = []
            
            for atom in positions:
                atomCart = sum((np.array(cell).transpose() * np.array(atom)).transpose())
                atomCart = atomCart.tolist()
                positionsCart.append(atomCart)
                
            assert len(positionsCart) == numAtoms
            positions = positionsCart.copy()
        
        # Update self.cell
        if lcell:
            self.cell = cell      
        
        assert len(positions) == numAtoms
        assert len(flags) == numAtoms
        assert len(v) == numAtoms

        return atomTypes, atomNumbers, positions, flags, v
    
    
    def readModes(self, inFileName, numModes, numAtoms):
        """ 
        Read normal modes 
        :param inFileName: str, the name of normal modes
        :param numModes: int, the number of normal modes need to read
        :param numAtoms: int, the number of atoms
        :return:
        modes: list of list, modes, [[[dx1, dy1, dz1], [dx2, dy2, dz2], ...], ...]
        freqList: list of frequencies in eV
        """
        freqList = []
        modes = []
        
        with open(inFileName, 'r') as data:
            
            # find the first mode
            line = data.readline()
            
            while 'meV' not in line:
                line = data.readline()
            
            # read frequencies and normal modes                      
            for m in range(numModes):
                
                # Read one frequency
                line = line.strip().split()
                assert line[-1] == 'meV', 'Frequency not read correctly'
                freq = float(line[-2])/1000.0                      # freq in eV
                freqList.append(freq)
                
                data.readline()
                
                # Read one normal mode
                mode = []
                for atom in range(numAtoms):
                    line = data.readline().strip().split()
                    line = [float(i) for i in line[3:6]]
                    mode.append(line)
                assert len(mode) == numAtoms
                modes.append(mode)
                
                line = data.readline()
                line = data.readline()
        
        assert len(freqList) == numModes
        assert len(modes) == numModes
        
        return modes, freqList
    
    
    def readCoord(self, inFileName, lcell, geoNum):
        """
        Read structures from an existing .coor file (trajectory file),
        return atomTypes and atomNumbers, positions
        :param inFileName: str, the name of .coor file
        :param lcell: boolean, whether or not overwrite cell
        :param geoNum: int, geometry number, start from 1
        :return:
        atomTypes: list of strs, element types
        atomNumbers: list of ints, the number of each element type
        positions: list of lists, [[x1, y1, z1], [x2, y2, z2], ...]
        flags: list of strs
        """
        cell = []
        atomTypes = []
        atomNumbers = []
        positions = []
        flags = []
        
        # read positions
        with open(inFileName, 'r') as data:
            data.readline()
            
            # Read lattice constant and lattice vectors
            latticeConstant = float(data.readline().strip())
            
            for n in range(3):
                vector = data.readline().strip().split()
                vector = [latticeConstant * float(i) for i in vector]
                assert len(vector) == 3
                cell.append(vector)
            
            # Read atom types and atom numbers
            atomTypes = data.readline().strip().split()
            
            atomNumbers = data.readline().strip().split()
            atomNumbers = [int(i) for i in atomNumbers]
            
            assert len(atomTypes) == len(atomNumbers)
            numAtoms = sum(atomNumbers)
            
        # Read positions
        start = (geoNum - 1) * (numAtoms + 1) + 8
        
        line = linecache.getline(inFileName, start)
        line = line.strip().split()
        assert line[0] == 'Time'
        print('The structure is for time = ', float(line[-1]))
        
        for n in range(numAtoms):
            line = linecache.getline(inFileName, start + 1 + n)
            line = line.strip().split()
            
            coordAtom = [float(i) for i in line]
            flag = '   T   T   T'     
            positions.append(coordAtom)         
            flags.append(flag)
            
        # Update self.cell
        if lcell:
            self.cell = cell       
        
        assert len(positions) == numAtoms
        assert len(flags) == numAtoms

        return atomTypes, atomNumbers, positions, flags
    
    
    def readVelocity(self, inFileName, numAtoms, geoNum):
        """ 
        Read structures from an existing .velocity file
        :param inFileName: str, the name of .velocity file
        :param numAtoms: int, number of atoms
        :param geoNum: int, geometry number, start from 1
        :return: v, list of lists, [[x1, y1, z1], [x2, y2, z2], ...]
        """
        v = []
        
        # Read velocity
        start = (geoNum - 1) * (numAtoms + 1) + 1
        
        line = linecache.getline(inFileName, start)
        line = line.strip().split()        
        assert line[0] == 'Time'
        print('The velocity is for time = ', float(line[-1]))
        
        for n in range(numAtoms):
            line = linecache.getline(inFileName, start + 1 + n)
            line = line.strip().split()
            
            vAtom = [float(i) for i in line]  
            v.append(vAtom)                    
        
        return v
    
    
    def readEnergy(self, inFileName, geoNum):
        """
        Read energy from an existing .energy file, update self.title
        :param inFileName: str, the name of .energy file
        :param geoNum: int, geometry number, start from 1
        """
        # Read energy
        start = geoNum + 1
        
        line = linecache.getline(inFileName, start)
        line = line.strip().split()

        temp = float(line[2])
        Etot = float(line[4])
        V = float(line[8])
        Ek = float(line[19])
        
        energy_str = ' temperature '+"{:10.2f}".format(temp)+' K;'
        energy_str += ' Etot '+"{:10.6f}".format(Etot)+' eV;'
        energy_str += ' V '+"{:10.6f}".format(V)+' eV;'
        energy_str += ' Ek '+"{:10.6f}".format(Ek)+' eV;'
        
        self.title += energy_str


### Getters ###
    def getCell(self):
        return self.cell
    
    
    def getPositions(self):
        return self.positions
    
    
    def getAtomPosition(self, positions, atomNum):
        return positions[atomNum - 1]
    
    
### Main methods ###
    def genPositions(self, inFileName, form, impactPos, clusterCOM, distanceToCluster = 5.0):
        """
        Take in an impact atom postions, impactPos, on which the molecule collides with,
        and the center of mass of the cluster, clusterCOM,
        generate the position for molecule, and the direction for v_trans
        :param inFileName: str, the name of position file, i.e. POSCAR
        :param form: str, the form of position file, 'vasp', 'ase'
        :param impactPos: list of floats, the position of the impact cluster atom
        :param clusterCOM: list of floats, the position of the cluster center of mass
        :param distanceToCluster: float, the distance between molecule to impact site
        :return:
        vectorNormed: list of float, the vector from cluster COM to impact cluster atom
        positionsNew: list of lists, new positions, [[x1, y1, z1], [x2, y2, z2], ...]
        """
        # check input files
        assert len(impactPos) == 3
        assert len(clusterCOM) == 3

        # read initial position
        atomTypes, atomNumbers, positions, flags, v = self.readPositions(inFileName, form, lcell = False)
        
        # Compute the vector from cluster COM to impact cluster atom
        vector = np.array(clusterCOM) - np.array(impactPos)
        vectorNormed = vector/np.linalg.norm(vector)
        vectorNormed = vectorNormed.tolist()
        
        # Compute the position of the molecule COM
        comNew = np.array(impactPos) - np.array(vectorNormed) * distanceToCluster
        comOld = self.com(atomTypes, atomNumbers, positions)
        
        disp = comNew - comOld
        positionsNew = np.array(positions) + disp
        positionsNew = positionsNew.tolist()
        
        assert len(positionsNew) == sum(atomNumbers)
        return positionsNew, vectorNormed
    
    
    def genV_vib(self, inFilePositions, form, inFileModes, numModes, lcell, lrotation, method, temp, state=None, randomPhase=True, anharmonicity=None):
        """
        Generate vibrational velocity for a structure
        :param inFilePositions: str, the name of position file, i.e. POSCAR
        :param form: str, the form of position file, 'vasp', 'ase'
        :param inFileModes: str, the name of normal mode file
        :param numModes: int, the number of modes
        :param lcell: boolean, whether or not overwrite cell
        :param lrotation: boolean, whether or not rotate molecule
        :param method: str, the name of method
        :param temp: float, temperature
        :param state: list, a list of quantum numbers
        :param randomPhase: boolean or integer, if it is a boolean, perform random phase or not; if it is an integer, randomPhase * phi is the input phase
        :param anharmonicity: None or string, do not consider anharmonicity if None; file name indicates whether or not consider anharmonicity
        :return:
        atomTypes: list of strs, element types
        atomNumbers: list of ints, the number of each element type
        positions: list of lists, positions, [[x1, y1, z1], [x2, y2, z2], ...]
        flags: list of strs
        vVib: list of lists, vibrational velocities, [[x1, y1, z1], [x2, y2, z2], ...]
        """
        # constant
        kb = 8.61733035e-05       #Boltzmann constant in eV/K
        
        # Read positions and normal modes
        atomTypes, atomNumbers, positions, flags, vOld = self.readPositions(inFilePositions, form, lcell)
        numAtoms = sum(atomNumbers)
        modes, freqList = self.readModes(inFileModes, numModes, numAtoms)
        
        assert len(freqList) == numModes
        assert len(modes) == numModes
        assert len(modes[0]) == numAtoms
        
        # Rotate the molecule, update both positions and modes
        if lrotation:
            positions, modes = self.rotation(atomTypes, atomNumbers, positions, modes)   
                             
        # Compute energy for normal mode
        ### Equipartition ###
        if method == 'EP':
            energyList = np.array([kb * temp for i in range(numModes)])     # In eV  
        
        ### Equipartition for TS ###
        elif method == 'EP_TS':
            state = np.array([1.0 for _ in range(numModes - 1)] + [0.5])
            energyList = kb * temp * state     # In eV
        
        ### Quasi-classical trajectory ###
        elif method == 'QCT':
            if state == None:
                state = self.genState(freqList, numModes, temp)
                
            assert len(state) == numModes
            print('The state is ', state)
            
            stateStr = [str(i) for i in state]
            stateStr = ' states ' + ' '.join(stateStr)+' ;'
            self.title += stateStr
            
            energyList = np.array(freqList) * (np.array(state) + 0.5)     # In eV
        
        ### Quasi-classical trajectory for TS ###
        elif method == 'QCT_TS':
            # vibrational modes
            if state == None:
                state = self.genState(freqList[: -1], numModes - 1, temp)
            
            assert len(state) == numModes - 1
            print('The state is ', state)
            
            stateStr = [str(i) for i in state]
            stateStr = ' states ' + ' '.join(stateStr)
            self.title += stateStr
            
            energyList = np.array(freqList[: -1]) * (np.array(state) + 0.5)
            
            # reaction coordinates
            energyReactionCoord = self.genClassicBoltzmann(temp)
            energyList = energyList.tolist()
            energyList.append(energyReactionCoord)
            energyList = np.array(energyList)     # In eV
            
            self.title += '  {:2.6f} eV;'.format(energyReactionCoord)            
        
        ### Classical Boltzmann (Exponetial) ###
        elif method == 'CB':
            energyList = [self.genClassicBoltzmann(temp) for i in range(numModes)]     # In eV
        
        ### QCT without ZPE ###
        elif method == 'QCT_noZPE':
            energyList = np.array(freqList) * np.array(state)     # In eV
        
        ### test ###
        elif method == 'test':
            print(kb, temp, state)
            energyList = kb * temp * np.array(state)
        
        ### Error ###
        else:
            raise Exception('Unidentified method')
        
        Ev = sum(energyList)
        print('The vibrational energies are', energyList)
        print('The total vibrational energy is', Ev)
        self.title += ' Ev '+"{:10.5f}".format(Ev)+' eV;'

            
        # Generate coordNorm and vNorm
        print('Random phase: ', randomPhase)
        
        coordNorm, vNorm = self.genNormalCoordAndV(energyList, freqList, randomPhase, anharmonicity)
        
        if method == 'QCT_TS':
            coordNormTS, vNormTS = self.genNormalCoordAndV(energyList[-1:], freqList[-1:], randomPhase=1.5, anharmonicity=False)
            coordNorm[-1] = coordNormTS[0]
            vNorm[-1] = vNormTS[0]
        
        print('The velocity for each normal mode is', vNorm)
        print('The normal coordinate for each normal mode is', coordNorm)
        
        # Generate mass list
        mass = []
        for n, atom in enumerate(atomTypes):
            number = atomNumbers[n]
            massAtom = [self.masses[atom] for i in range(number * 3)]
            mass.extend(massAtom)
        assert len(mass) == numAtoms * 3
#        print(mass)
        
        # flat modes to modesMatrix
        modesMatrix = []
        for mode in modes:
            modeFlat = []
            for m in mode:
                modeFlat.extend(m)
            assert len(modeFlat) == 3 * numAtoms
            modesMatrix.append(modeFlat)
        assert len(modesMatrix) == numModes
        
        # Compute velocity
        vMatrix = np.array(modesMatrix).transpose() * np.array(vNorm)
        vMatrix = vMatrix.transpose()                 # v * mass**0.5 in angstrom/10 fs
        vList = sum(vMatrix) / (np.array(mass)**0.5)  # v in angstrom/10 fs
        vList = vList/10.0                            # v in angstrom/fs
        vList = vList.tolist()
        vVib = []
        for n in range(numAtoms):
            vVib.append(vList[n * 3 : (n + 1) * 3])
        
        # Compute position
        disMatrix = np.array(modesMatrix).transpose() * np.array(coordNorm)
        disMatrix = disMatrix.transpose()                  # dis * mass**0.5 in angstrom
        disList = sum(disMatrix) / (np.array(mass)**0.5)   # dis in angstrom
        disList = disList.tolist()
        dis = []
        for n in range(numAtoms):
            dis.append(disList[n * 3 : (n + 1) * 3])
            
        newPositions = np.array(positions) + np.array(dis)
        positions = newPositions.tolist().copy()
        
        assert len(vVib) == numAtoms
#        print('The vibrational velocity is', vVib)
#        print('The position is', positions)
        
        return atomTypes, atomNumbers, positions, flags, vVib
        
    
    def genV_trans(self, inFilePositions, form, tempTrans, directTrans, lcell=False, methodTrans='EP'):
        """
        Generate translational velocity for a structure
        :param inFilePositions: str, the name of position file, i.e. POSCAR
        :param form: str, the form of position file, 'vasp', 'ase'
        :param tempTrans: float, temperature of translational energy
        :param directTrans: list[float], vector indicate the direction of the translational velocity
        :param lcell: boolean, whether or not overwrite cell
        :param methodTrans: str, the name of method
        :return: vTrans: list of float, translational velocities, [x, y, z]
        """
        # constant
        tutoev = 1.0364314        #change tu unit to eV
        kb = 8.61733035e-05       #Boltzmann constant in eV/K
        
        # Read positions
        atomTypes, atomNumbers, positions, flags, vOld = self.readPositions(inFilePositions, form, lcell)
        
        # Normalize direction vector
        directTrans = directTrans / np.linalg.norm(directTrans)
        
        # Compute mass of the molecule
        totalMass = 0.0
        for n, atom in enumerate(atomTypes):
            number = atomNumbers[n]
            totalMass += self.masses[atom] * number
            
        # Compute the velocity of the molecule
        if methodTrans == 'EP':
            rateTrans = (((3 * kb * tempTrans / tutoev) / totalMass)**0.5) / 10.0
            
        elif methodTrans == 'MAXWELL':
            a = ((kb * tempTrans / tutoev) / totalMass)**0.5
            x = maxwell.rvs(size = 1)[0]
            rateTrans = a * x / 10.0
        
        # Check translational energy
        Et = tutoev * totalMass * (10.0 * rateTrans)**2.0 / 2.0
        EtAve = 3.0 * kb * tempTrans / 2.0
        print('The incident translational energy is '+str(Et)+' eV; average is '+str(EtAve))
        print('The translational velocity is '+str(rateTrans)+' angstrom/fs')
        
        # update title
        EtStr = " Et   {:10.5f}".format(Et)+' eV;'
        self.title += EtStr
        
        vTrans = np.array(directTrans) * rateTrans
        vTrans = vTrans.tolist()
        
        assert len(vTrans) == 3  
        print('The translational velocity is ', vTrans)
        
        return vTrans
    
    
    def updateVariables(self, atomTypes, atomNumbers, positions, flags, v):
        """
        Update self.atomTypes, self.atomNumbers, self.positions, self.flags and self.v
        :param atomTypes: list of strs, element types
        :param atomNumbers: list of ints, the number of each element type
        :param positions: list of lists, [[x1, y1, z1], [x2, y2, z2], ...]
        :param flags: list of strs
        :param v: list of lists, velocities, [[x1, y1, z1], [x2, y2, z2], ...]
        """
        # check input files
        assert len(atomTypes) == len(atomNumbers)
        num = sum(atomNumbers)
        assert len(positions) == num
        assert len(flags) == num
        assert len(v) == num
        
        # Update self.atoms
        self.atomTypes.extend(atomTypes)
        self.atomNumbers.extend(atomNumbers)
        self.positions.extend(positions)
        self.flags.extend(flags)
        self.v.extend(v)
        
    
    def com(self, atomTypes, atomNumbers, positions):
        """
        Compute the center-of-mass of a structure
        :param atomTypes: list of strs, element types
        :param atomNumbers: list of ints, the number of each element type
        :param positions: list of lists, [[x1, y1, z1], [x2, y2, z2], ...]
        :return: COM: list of floats, center of mass
        """
        # check input files and initialization
        assert len(atomTypes) == len(atomNumbers)
        COM = np.array([0.0, 0.0, 0.0])
        totalMass = 0
        
        # compute center of mass
        numAtoms = 0
        for n, atom in enumerate(atomTypes):
            massAtom = self.masses[atom]
            number = atomNumbers[n]
            positionAtom = positions[numAtoms : (numAtoms + number)]
            
            COM = COM + sum(np.array(positionAtom)) * massAtom
            totalMass += massAtom * number            
            numAtoms = numAtoms + number
        
        assert numAtoms == sum(atomNumbers)
        
        COM = COM / totalMass
        COM = COM.tolist()
        
        return COM
    
    
    def translation(self, ro, dist):
        """ 
        Translate the molecule.      
        If dist is 'origin', translate molecule to origin;
        if dist is a list of floats, translate molecule by dist.
        :param ro: list of lists, initial positions, [[x1, y1, z1], [x2, y2, z2], ...]
        :param dist: lists of floats or str
        :return: rn: list of lists, positions after translation, [[x1, y1, z1], [x2, y2, z2], ...]
        """
        ro = np.array(ro)
        
        # translate ro by dist
        if type(dist) == list:
            dist = np.array(dist)
            rn = ro + dist
        
        # translate ro to origin
        elif dist == 'origin':
            center = sum(ro) / len(ro)
            rn = ro - center
        
        rn = rn.tolist()
        return rn
    
    
    def rotation(self, atomTypes, atomNumbers, ro, modeo = None):
        """
        Randomly rotate the molecule and its normal modes
        :param atomTypes: list of strs, element types
        :param tomNumbers: list of ints, the number of each element type
        :param ro: list of lists, positions, [[x1, y1, z1], [x2, y2, z2], ...]
        :param modeo: list of list, modes, [[[dx1, dy1, dz1], [dx2, dy2, dz2], ...], ...]
        :return:
        rn: list of lists, positions, [[x1, y1, z1], [x2, y2, z2], ...]
        moden: list of list, modes, [[[dx1, dy1, dz1], [dx2, dy2, dz2], ...], ...]
        """
        numAtoms = sum(atomNumbers)
        
        # if modeo is None, make fake modes
        if modeo == None:
            modeo = [[[0.0 for j in range(3)] for i in range(numAtoms)]]
        
        # find center of mass
        ro = np.array(ro)
        center = self.com(atomTypes, atomNumbers, ro)
        
        # transpose positions and modes, translate molecule, such that the center of mass is at origin
        dis = ro - np.array(center)
        disMatrix = np.matrix(dis).transpose()
        
        numModes = len(modeo)
        modeMatrix = []
        for m in modeo:
            mode = m.copy()
            mode = np.array(mode)
            mode = np.matrix(mode).transpose()
            modeMatrix.append(mode)
        
        # generate a random rotation
        degrees = np.random.uniform(0.0, 2.0 * np.pi, 3)
        
        # rotate molecule
        # rotate molecule about x axis
        degree = degrees[0]
        s, c = np.sin(degree), np.cos(degree)
        rotMatrix = np.matrix([[1, 0, 0], [0, c, -s], [0, s, c]])
        
        rnMatrix = np.dot(rotMatrix, disMatrix)
        disMatrix = rnMatrix.copy()
        
        newModeMatrix = []
        for mode in modeMatrix:
            newMode = np.dot(rotMatrix, mode)
            newModeMatrix.append(newMode)
        modeMatrix = newModeMatrix.copy()
        
        # rotate molecule about y axis
        degree = degrees[1]
        s, c = np.sin(degree), np.cos(degree)
        rotMatrix = np.matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        
        rnMatrix = np.dot(rotMatrix, disMatrix)
        disMatrix = rnMatrix.copy()
        
        newModeMatrix = []
        for mode in modeMatrix:
            newMode = np.dot(rotMatrix, mode)
            newModeMatrix.append(newMode)
        modeMatrix = newModeMatrix.copy()
        
        # rotate molecule about z axis
        degree = degrees[2]
        s, c = np.sin(degree), np.cos(degree)
        rotMatrix = np.matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        
        rnMatrix = np.dot(rotMatrix, disMatrix)
        disMatrix = rnMatrix.copy()
        
        newModeMatrix = []
        for mode in modeMatrix:
            newMode = np.dot(rotMatrix, mode)
            newMode = np.array(newMode).transpose()
            newMode = newMode.tolist()
            newModeMatrix.append(newMode)
        moden = newModeMatrix.copy()
        
        # translate molecule to original position
        dis = np.array(disMatrix).transpose()
        rn = dis + np.array(center)         
        rn = rn.tolist()
        
        # write new modes
        with open('newModes', 'w') as outputFile:
            for n in range(numModes):
                outputFile.write(str(n+1)+' f'+'   meV'+'\n')
                outputFile.write('       X         Y         Z           dx          dy          dz'+'\n')
                
                for i in range(numAtoms):
                    line = ''
                    for j in range(3):
                        line += '   '+"{:10.6f}".format(rn[i][j])
                    for j in range(3):
                        line += '   '+"{:10.6f}".format(moden[n][i][j])
                    line += '\n'
                    outputFile.write(line)
                    
                outputFile.write('\n')
        
        return rn, moden


    def genQuantumNumber(self, freq, temp, probThreshold = 1.0e-6):
        """ 
        Generate quantum number from a boltzmann distribution for one mode
        :param freq: float, frequency in eV
        :param temp: float, temperature in K
        :param probThreshold: float, if P(state) < probThreshold, ignore it
        :return:
        quantum: int, quantum number
        """
        kb = 8.61733035e-05       #Boltzmann constant in eV/K
        lambda_ = freq / (kb * temp)
        
        # Compute bound for Boltzmann distribution
        N = math.ceil(-np.log(probThreshold)/lambda_)
        N = min(N, 20)
               
        # Generate a quantum number
        quantum = boltzmann.rvs(lambda_, N, size = 1)[0]
        
        return quantum
    
    
    def genState(self, freqList, numModes, temp):
        """
        Generate a state for a molecule from Boltzmann distribution
        :param freqList: list of floats, list of frequencies
        :param numModes: int, number of modes
        :param temp: float, temperature
        :return:
        state: list of ints, list of quantum numbers
        """
        assert len(freqList) == numModes
        
        state = []
        for freq in freqList:
            quantum = self.genQuantumNumber(freq, temp)
            state.append(quantum)
            
        assert len(state) == numModes
        
        return state
    
    
    def genClassicBoltzmann(self, temp):
        """
        Generate an energy from classical Boltzmann distribution
        :param temp: float, temperature
        :return:
        energy: float, energy
        """
        kb = 8.61733035e-05       #Boltzmann constant in eV/K
        
        beta = kb * temp
        energy = np.random.exponential(beta)
        
        return energy
    
    
    def genNormalCoordAndV(self, energyList, freqList, randomPhase = False, anharmonicity = None):
        """
        Input a list of energy (in eV) and a list of frequency (in eV),
        Return normal coordinates and normal velocities based on phase
        :param energyList: list of floats, list of energies
        :param freqList: list of floats, list of frequences
        :param randomPhase: boolean or integer, if it is a boolean, perform random phase or not; if it is an integer, randomPhase * phi is the input phase
        :param anharmonicity: None or string, do not consider anharmonicity if None; file name indicates whether or not consider anharmonicity
        :return:
        coordNorm: list of floats, normal coordinates
        vNorm: list of floats, normal mode velocities
        """
        # constants
        hbar = 0.06582119514      # eV * 10fs
        tutoev = 1.0364314        #change tu unit to eV
        
        # check inputs
        assert len(energyList) == len(freqList)
        numModes = len(freqList)
        
        # Compute amplitude
        energy = np.array(energyList) / tutoev    # energy unit: tu
        freq = np.array(freqList) / hbar          # Unit: (10fs)**(-1)  
        amplitude = np.sqrt(energy * 2.0) / freq  # list of amplitudes for all modes
        
        # Generate phase
        if type(randomPhase) == bool:
            if randomPhase:
                phase = np.random.uniform(0.0, 2.0 * np.pi, numModes)
                phase = np.array(phase)
            else:
                # start from equilibrium popsition
                phase = np.random.randint(0, 2, numModes)
                phase = np.array(phase) * np.pi + 0.5 * np.pi
        else:
            phase = np.array(randomPhase) * np.pi
        
        # Compute normal coordinates and normal velocities
        s, c = np.sin(phase), np.cos(phase)
        coordNorm = amplitude * c            # A * cos(phi)
        vNorm = -amplitude * freq * s        # -A * freq * sin(phi)
        
        # correct for anharmonicity
        if anharmonicity:
            inFile = open(anharmonicity, 'r')
            
            inFile.readline()
            
            for mode in range(1, numModes + 1):
                line = inFile.readline().strip().split()
                
                if line[-1] == 'False':
                    continue
                
                inFilePES = 'mode_' + str(mode)
                coordNormMode = coordNorm[mode - 1]
                freqMode = freq[mode - 1]
                coordNorm[mode - 1] = self.anharmonicityCorrection(inFilePES, coordNormMode, freqMode)
            
            inFile.close()
        
        return coordNorm, vNorm
    
    
    def anharmonicityCorrection(self, inFilePES, coordNorm, freq):
        """
        Correct normal coordinates, using real PES (including anharmonicity)
        :param inFilePES: string, the file name of the real PES
        :param coordNorm: float, the normal coordinate from harmonic oscillator model
        :param freq: float, the frequency of the mode, Unit: (10fs)**(-1)
        :return:
        coordNormCorrected: float, the normal coordinate corrected by real PES
        """
        # constants
        hbar = 0.06582119514      # eV * 10fs
        tutoev = 1.0364314        #change tu unit to eV
        
        # read potential energy
        pesSide1 = []
        pesSide2 = []
        q1 = []
        q2 = []
        
        inFile = open(inFilePES, 'r')
        
        # read side1
        inFile.readline()
        
        line = inFile.readline()
        while 'side2' not in line:
            data = line.strip().split()
            pesSide1.append(float(data[0]))
            q1.append(float(data[1]))
            line = inFile.readline()
        
        # read side2
        line = inFile.readline()
        while line:
            data = line.strip().split()
            pesSide2.append(float(data[0]))
            q2.append(float(data[1]))
            line = inFile.readline()
        
        #print(pesSide1, q1, pesSide2, q2)
        
        inFile.close()
        
        # fit the real PES from input data
        if coordNorm <= 0:
            pes = interpolate.interp1d(pesSide1, q1)
        else:
            pes = interpolate.interp1d(pesSide2, q2)
        
        # compute potential energy from input coordNorm
        V = 0.5 * (freq * coordNorm)**2.0 * tutoev    # in eV
        
        # compute the real normal coordinate by interpolation
        coordNormAHO = pes(V)
        
        print(inFilePES, 'frequency', freq * hbar, ', difference in normal coordinates:', coordNormAHO - coordNorm)
        
        return coordNormAHO


    def PES(self, in_file_positions, in_file_eigen, num_modes, num_atoms, mode, sign, out_file_name):
        """
        Generate POSCAR file for computing PES along one mode using AIMD
        """
        atom_types, atom_numbers, positions, flags, v_old = self.readPositions(in_file_positions, 'vasp', lcell=True)
        num_atoms = sum(atom_numbers)
        num = len(atom_types)
        eigens, freq_list = self.readModes(in_file_eigen, num_modes, num_atoms)
        assert len(freq_list) == num_modes
        assert len(eigens) == num_modes
        assert len(eigens[0]) == num_atoms        

        # Generate mass list
        mass = []
        for n in range(num):
            atom = atom_types[n]
            number = atom_numbers[n]
            mass_atom = [self.masses[atom] for i in range(number)]
            mass.extend(mass_atom)
        assert len(mass) == num_atoms
        
        # Generate velocity
        eigen = np.array(eigens[mode-1])
        v = eigen.transpose()/(np.array(mass)**0.5)
        v = v.transpose()*sign
        
        self.updateVariables(atom_types, atom_numbers, positions, flags, v)
        self.output(out_file_name)        
        

    def shift(self, in_file_name, atom_num, in_file_name_ref, atom_num_ref):
        """
        Shift the geometry in periodic system
        """
        atom_types_ref, atom_numbers_ref, positions_ref, flags_ref, v_ref = self.readPositions(in_file_name_ref, 'vasp', lcell=False)
        atom_types, atom_numbers, positions, flags, v = self.readPositions(in_file_name, 'vasp', lcell=True)
        direction = np.array(positions_ref[atom_num_ref-1])-np.array(positions[atom_num-1])
        pos = np.array(positions)

        pos_new = pos+direction
        pos_new = pos_new.tolist()
        for atom in pos_new:
            if atom[0]<0:
                atom[0]=atom[0]+self.cell[0][0]
            if atom[0]>self.cell[0][0]:
                atom[0]=atom[0]-self.cell[0][0]
            if atom[1]<0:
                atom[1]=atom[1]+self.cell[1][1]
            if atom[1]>self.cell[1][1]:
                atom[1]=atom[1]-self.cell[1][1]
        self.updateVariables(atom_types, atom_numbers, pos_new, flags, v)
        out_file_name = in_file_name+'shifted'
        self.output(out_file_name)

    
    def add_perp(self, in_file_name, Cu1_number, Cu2_number, c_number, O2_length, O2_dist):
        """
        Add O2 perpendicular to a given Cu-Cu bond
        :param Cu1_number: int, representing the position of Cu1
        :param Cu2_number: int, representing the position of Cu2
        :param c_number: list of 3 float or int, representing the position of a reference atom to define the direction that O2 comes in, if c_number is a list then use that as the position
        :param O2_length: float, representing O2 bondlength
        :param O2_dist: float, representing the distance from Cu-Cu center to O2
        """
        atom_types, atom_numbers, positions, flags, v = self.readPositions(in_file_name, 'vasp', lcell=True)
        self.updateVariables(atom_types, atom_numbers, positions, flags, v)
        
        Cu1 = self.get_atom_position(positions, Cu1_number)
        Cu2 = self.get_atom_position(positions, Cu2_number)
        if type(c_number) == int:
            c = self.get_atom_position(positions, c_number)
        elif type(c_number) == list:
            c = c_number.copy()
        Cu1 = np.array(Cu1)
        Cu2 = np.array(Cu2)
        c = np.array(c)
        
        Cu_center = (Cu1+Cu2)/2.0
        d1 = Cu1-c
        d2 = Cu2-c
        shift = np.cross(d1, d2)
        shift = shift/np.linalg.norm(shift)
        if np.dot(shift, np.array([0.0, 0.0, 1.0])) < 0:
            shift = -1.0*shift
    
        O2_center = Cu_center+shift*O2_dist
        e1 = Cu1-O2_center
        e2 = Cu2-O2_center
        direction = np.cross(e1, e2)
        direction = direction/np.linalg.norm(direction)
        O1 = O2_center+direction*O2_length*0.5
        O2 = O2_center-direction*O2_length*0.5
        O1 = O1.tolist()
        O2 = O2.tolist()
        
        O2_types = ['O']
        O2_numbers = [2]
        O2_positions = [O1, O2]
        O2_flags = ['   F   F   F', '   F   F   F']
        O2_v = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        self.updateVariables(O2_types, O2_numbers, O2_positions, O2_flags, O2_v)
        self.output(in_file_name+'_O2')
        print('Adding O2 done')


    def add_toTop(self, in_file_name, Cu_number, O1_number, O2_number, O2_length, O2_dist):
        """
        Add O2 to the top of a given Cu-Cu bond
        :param Cu_number: int, representing the position of Cu1
        :param c_number: list of 3 float or int, representing the position of a reference atom to define the direction that O2 comes in, if c_number is a list then use that as the position
        :param O2_length: float, representing O2 bondlength
        :param O2_dist: float, representing the distance from Cu-Cu center to O2
        """
        atom_types, atom_numbers, positions, flags, v = self.readPositions(in_file_name, 'vasp', lcell=True)
        self.updateVariables(atom_types, atom_numbers, positions, flags, v)
        
        Cu = np.array(self.get_atom_position(positions, Cu_number))
        O1 = np.array(self.get_atom_position(positions, O1_number))
        O2 = np.array(self.get_atom_position(positions, O2_number))
        direction = (O2+O1)/2.0-Cu
        direction = direction/np.linalg.norm(direction)
        O2Direction = O2-O1
        O2Direction = O2Direction/np.linalg.norm(O2Direction)
        
        center = Cu+direction*O2_dist
        O2New = center+O2Direction*O2_length/2.0
        O1New = center-O2Direction*O2_length/2.0
        O2New = O2New.tolist()
        O1New = O1New.tolist()
        
        positions[O2_number-1] = O2New
        positions[O1_number-1] = O1New
        print(O1New, O2New)
        self.positions = positions.copy()
        self.output(in_file_name+'_modified')
        print('Adding O2 done')
    
    
    def output(self, out_file_name):
        """
        Write the output file for the system
        """
        if len(self.cell) == 0:
            raise Exception('Unit cell not set')
        cell = self.cell.copy()
        atom_types = self.atomTypes.copy()
        atom_numbers = self.atomNumbers.copy()
        positions = self.positions.copy()
        flags = self.flags.copy()
        v = self.v.copy()        
        num_atoms = sum(atom_numbers)
        assert len(positions) == num_atoms
        assert len(flags) == num_atoms
        assert len(v) == num_atoms
        
        # Write POSCAR file
        with open(out_file_name, 'w') as output_file:
            output_file.write(self.title+'\n')
            output_file.write('   1.000'+'\n')
            
            cell_str = ''
            for i in range(3):
                for j in range(3):
                    cell_str += '   '+"{:10.8f}".format(float(cell[i][j]))
                cell_str += '\n'
            output_file.write(cell_str)
            
            atoms_type_str = '   '+'   '.join(atom_types)+'\n'
            output_file.write(atoms_type_str)
            atom_numbers = [str(n) for n in atom_numbers]
            atoms_num_str = '   '+'   '.join(atom_numbers)+'\n'
            output_file.write(atoms_num_str)
            output_file.write('Selective dynamics'+'\n'+'Cartesian'+'\n')
            
            positions_str = ''
            for i in range(num_atoms):
                for j in range(3):
                    positions_str += '   '+"{:10.8f}".format(positions[i][j])
                positions_str += flags[i]+'\n'
            output_file.write(positions_str)
            
            output_file.write('Cartesian'+'\n')
            v_str = ''
            for i in range(num_atoms):
                for j in range(3):
                    v_str += '   '+"{:10.8f}".format(v[i][j])
                v_str += '\n'
            output_file.write(v_str)
            

    def cluster_MD(self, clu_pos, clu_eigen, clu_modes, out_file_name, method, T, state=None):
        """
        Generate a POSCAR file for cluster MD
        """
        # Read positions and eigens files, generate velocity for cluster
        clu_atom_types, clu_atom_numbers, clu_positions, clu_flags, clu_v = self.genV_vib(clu_pos, 'vasp', clu_eigen, clu_modes, True, False, method, T, state)
        # Update data attributes
        self.updateVariables(clu_atom_types, clu_atom_numbers, clu_positions, clu_flags, clu_v)
        # Print POSCAR for cluster MD
        self.output(out_file_name)
        
        
    def molecule_MD(self, mo_pos, mo_eigen, mo_modes, lrotation, out_file_name, method, T, state=None):
        """
        Generate a POSCAR file for molecule MD
        """
        # Read positions and eigen files, generate velocity for molecule
        mo_atom_types, mo_atom_numbers, mo_positions, mo_flags, mo_v = self.genV_vib(mo_pos, 'vasp', mo_eigen, mo_modes, True, lrotation, method, T, state)
        # Update data attributes
        self.updateVariables(mo_atom_types, mo_atom_numbers, mo_positions, mo_flags, mo_v)
        # Print POSCAR for molecule MD
        self.output(out_file_name)
        
        
    def gas_phase_MD(self, clu_pos, clu_atom_num, mo_pos, trans_T, distance_to_clu, out_file_name, method_trans='EP'):
        """
        Read CONTCAR files from cluster MD and molecule MD,
        generate a POSCAR file for whole system MD in the gas phase
        """
        # Read positions and velocities for cluster
        clu_atom_types, clu_atom_numbers, clu_positions, clu_flags, clu_v = self.readPositions(clu_pos, 'vasp', lcell=True)
        # Update data attributes for cluster
        self.updateVariables(clu_atom_types, clu_atom_numbers, clu_positions, clu_flags, clu_v)
        # Read positions and velocities for molecule
        mo_atom_types, mo_atom_numbers, positions, mo_flags, mo_v_vib = self.readPositions(mo_pos, 'vasp', lcell=False)
        # Generate positions and moving direction for molecule, clu_atom_num is the cluter atom with which molecule collides
        clu_atom_pos = self.get_atom_position(clu_positions, clu_atom_num)
        clu_com = self.com(clu_atom_types, clu_atom_numbers, clu_positions)
        mo_positions, direct_trans = self.genPositions(mo_pos, 'vasp', clu_atom_pos, clu_com, distance_to_clu)
        # Generate velocity for molecule
        mo_v_trans = self.gen_v_trans(mo_pos, 'vasp', trans_T, direct_trans, False, method_trans)
        mo_v = np.array(mo_v_vib)+np.array(mo_v_trans)
        mo_v = mo_v.tolist()
        # Update data attribute for molecule
        self.updateVariables(mo_atom_types, mo_atom_numbers, mo_positions, mo_flags, mo_v)
        # Print POSCAR for molecule MD
        self.output(out_file_name)    


    def surface_MD(self, clu_pos, clu_atom_num, mo_pos, trans_T, distance_to_clu, out_file_name, method_trans='EP'):
        """
        Read CONTCAR files from cluster MD and molecule MD,
        generate a POSCAR file for whole system MD in the gas phase
        """
        # Read positions and velocities for cluster
        clu_atom_types, clu_atom_numbers, clu_positions, clu_flags, clu_v = self.readPositions(clu_pos, 'vasp', lcell=True)
        # Update data attributes for cluster
        self.updateVariables(clu_atom_types, clu_atom_numbers, clu_positions, clu_flags, clu_v)
        # Read positions and velocities for molecule
        mo_atom_types, mo_atom_numbers, positions, mo_flags, mo_v_vib = self.readPositions(mo_pos, 'vasp', lcell=False)
        # Generate positions and moving direction for molecule, clu_atom_num is the cluter atom with which molecule collides
        # if it is a int, O2 is at the top of Cu; if it is a list or tuple, O2 is the over the bridge site
        if type(clu_atom_num) == int:
            clu_atom_pos = self.get_atom_position(clu_positions, clu_atom_num)
        elif type(clu_atom_num) == tuple or type(clu_atom_num) == list:
            clu_atom_pos3 = self.get_atom_position(clu_positions, clu_atom_num[0])
            clu_atom_pos4 = self.get_atom_position(clu_positions, clu_atom_num[1])
            clu_atom_pos = (np.array(clu_atom_pos3)+np.array(clu_atom_pos4))/2.0
        clu_atom_pos2 = np.array(clu_atom_pos)-np.array([0.0, 0.0, 1.0])
        clu_atom_pos2 = clu_atom_pos2.tolist()
        mo_positions, direct_trans = self.genPositions(mo_pos, 'vasp', clu_atom_pos, clu_atom_pos2, distance_to_clu)
        # Generate velocity for molecule
        mo_v_trans = self.gen_v_trans(mo_pos, 'vasp', trans_T, direct_trans, False, method_trans)
        mo_v = np.array(mo_v_vib)+np.array(mo_v_trans)
        mo_v = mo_v.tolist()
        # Update data attribute for molecule
        self.updateVariables(mo_atom_types, mo_atom_numbers, mo_positions, mo_flags, mo_v)
        # Print POSCAR for molecule MD
        self.output(out_file_name)
        

    def surface_MD_random(self, clu_pos, impactRange, mo_pos, trans_T, distance_to_clu, out_file_name, method_trans='EP'):
        """
        Read CONTCAR files from cluster MD and molecule MD,
        generate a POSCAR file for whole system MD in the gas phase
        The impact site is randomly determined within impactRange = [[xmin, xmax], [ymin, ymax]]
        """
        # Read positions and velocities for cluster
        clu_atom_types, clu_atom_numbers, clu_positions, clu_flags, clu_v = self.readPositions(clu_pos, 'vasp', lcell=True)
        # Update data attributes for cluster
        self.updateVariables(clu_atom_types, clu_atom_numbers, clu_positions, clu_flags, clu_v)
        # Read positions and velocities for molecule
        mo_atom_types, mo_atom_numbers, positions, mo_flags, mo_v_vib = self.readPositions(mo_pos, 'vasp', lcell=False)
        # Generate positions and moving direction for molecule, clu_atom_num is the cluter atom with which molecule collides
        # if it is a int, O2 is at the top of Cu; if it is a list or tuple, O2 is the over the bridge site
        clu_atom_pos = []
        [[xmin, xmax], [ymin, ymax], z] = impactRange
        x = random.uniform(0, 1)*(xmax-xmin)+xmin
        clu_atom_pos.append(x)
        y = random.uniform(0, 1)*(ymax-ymin)+ymin
        clu_atom_pos.append(y)
        clu_atom_pos.append(z)

        clu_atom_pos2 = np.array(clu_atom_pos)-np.array([0.0, 0.0, 1.0])
        clu_atom_pos2 = clu_atom_pos2.tolist()
        mo_positions, direct_trans = self.genPositions(mo_pos, 'vasp', clu_atom_pos, clu_atom_pos2, distance_to_clu)
        # Generate velocity for molecule
        mo_v_trans = self.gen_v_trans(mo_pos, 'vasp', trans_T, direct_trans, False, method_trans)
        mo_v = np.array(mo_v_vib)+np.array(mo_v_trans)
        mo_v = mo_v.tolist()
        # Update data attribute for molecule
        self.updateVariables(mo_atom_types, mo_atom_numbers, mo_positions, mo_flags, mo_v)
        # Print POSCAR for molecule MD
        self.output(out_file_name)

        
    def select_cluster(self, clu_coor, clu_pos_equ, clu_v, clu_energy, geoNum, out_file_name):
        """
        Read positions and velocities from equilibration MD of cluster,
        return a POSCAR file
        """
        clu_atom_types, clu_atom_numbers, clu_positions, flags = self.readCoord(clu_coor, True, geoNum)
        atom_types, atom_numbers, positions, clu_flags, v = self.readPositions(clu_pos_equ, 'vasp', False)
        num_atoms = sum(clu_atom_numbers)
        clu_v = self.readVelocity(clu_v, num_atoms, geoNum)
        self.readEnergy(clu_energy, geoNum)
        self.updateVariables(clu_atom_types, clu_atom_numbers, clu_positions, clu_flags, clu_v)
        self.output(out_file_name)
        
     
        
        
        
        
