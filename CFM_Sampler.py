#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 14:59:23 2019

@author: Aqiph
"""
import os

import numpy as np
import math
import linecache
from scipy.stats import boltzmann
from scipy.stats import maxwell
from scipy import interpolate
import random

from utils.read_file import read_coordinates, read_normal_modes, read_coordinates_from_trajectory,\
    read_velocities_from_trajectory, read_energy_from_trajectory
from utils.tools import get_COM, rotate



class Sampler():

### Constructor ###    
    def __init__(self):
        """
        Define data attributes
        """
        self.cell = []
        self.atom_types = []
        self.atom_counts = []
        self.coordinates = []
        self.flags = []
        self.velocities = []
        self.title = ''
        self.masses = {'Pt':195.080, 'H':1.000, 'C':12.011, 'Cu':63.546,
                       'Al':26.981, 'O':16.000, 'Sr':87.620}


### Getters ###
    def get_cell(self):
        return self.cell
    
    
    def get_coordinates(self):
        return self.coordinates
    
    
    def get_atom_coordinates(self, coordinates, atomNum):
        return coordinates[atomNum - 1]


### Setters ###
    def set_cell(self, cell):
        """
        Set self.cell
        :param cell: list of lists, cell, [[x1, y1, z1], [x2, y2, z2], ...]
        """
        # check input variables
        assert len(cell) == 3 and len(cell[0]) == 3
        # set variables
        self.cell = cell


    def set_state(self, atom_types, atom_counts, coordinates, flags, velocities):
        """
        Set self.atom_types, self.atom_counts, self.coordinates, self.flags and self.velocities
        :param atom_types: list of strs, element types
        :param atom_counts: list of ints, number of atoms for each element type
        :param coordinates: list of lists, [[x1, y1, z1], [x2, y2, z2], ...]
        :param flags: list of strs, flags
        :param velocities: list of lists, velocities, [[x1, y1, z1], [x2, y2, z2], ...]
        """
        # check input variables
        num_atoms = sum(atom_counts)
        assert len(atom_types) == len(atom_counts)
        assert len(coordinates) == num_atoms
        assert len(flags) == num_atoms
        assert len(velocities) == num_atoms
        # set variables
        self.atom_types.extend(atom_types)
        self.atom_counts.extend(atom_counts)
        self.coordinates.extend(coordinates)
        self.flags.extend(flags)
        self.velocities.extend(velocities)


    def set_title(self, title):
        """
        Set self.title
        :param title: str, title
        """
        # check input variables
        assert isinstance(title, str)
        # set variables
        self.title += title


### Helper functions ###
    @staticmethod
    def gen_coordinates(input_file, file_format, coordinates_impact_site, coordinates_ref_atom, distance_to_cluster = 5.0):
        """
        Take in an impact atom position, coordinates_impact_site, on which the molecule collides with,
        and the coordinates of the reference atom, e.g. the COM of the cluster, coordinates_ref_atom,
        generate the coordinates for molecule, and the impact direction
        :param input_file: str, path of the input file, i.e. POSCAR
        :param file_format: str, format of the input coordinates and velocities file. Allowed values include 'vasp' or 'ase'
        :param coordinates_impact_site: list of floats, coordinates of the impact site
        :param coordinates_ref_atom: list of floats, coordinates of the reference atom
        :param distance_to_cluster: float, distance between the molecule's COM and the impact site
        :return:
        coordinates_new: list of lists, new coordinates, [[x1, y1, z1], [x2, y2, z2], ...]
        impact_vector_normed: list of float, normed vector of the impact direction
        """
        # check input files
        assert len(coordinates_impact_site) == 3
        assert len(coordinates_ref_atom) == 3

        # read initial coordinates
        _, atom_types, atom_counts, coordinates, _, _, _ = read_coordinates(input_file, file_format)
        
        # compute the vector from the impact site to the reference atom
        impact_vector = np.array(coordinates_ref_atom) - np.array(coordinates_impact_site)
        impact_vector_normed = impact_vector/np.linalg.norm(impact_vector)
        impact_vector_normed = impact_vector_normed.tolist()
        
        # compute the coordinates of the molecule COM
        COM_new = np.array(coordinates_impact_site) - np.array(impact_vector_normed) * distance_to_cluster
        COM_old = get_COM(atom_types, atom_counts, coordinates)
        coordinates_new = np.array(coordinates) + (COM_new - COM_old)
        coordinates_new = coordinates_new.tolist()
        
        assert len(coordinates_new) == sum(atom_counts)
        return coordinates_new, impact_vector_normed
    
    
    def gen_velocities_vib(self, input_file_coordinates, file_format, input_file_mode, num_modes, lcell, lrotation, method, temp, state=None, randomPhase=True, anharmonicity=None):
        """
        Generate vibrational velocity for a structure
        :param input_file_coordinates: str, path of atomic coordinates file, i.e. POSCAR
        :param file_format: str, format of the input coordinates and velocities file. Allowed values include 'vasp' or 'ase'
        :param input_file_mode: str, path of normal modes file
        :param num_modes: int, the number of modes
        :param lcell: boolean, whether overwrite cell
        :param lrotation: boolean, whether rotate molecule or not
        :param method: str, method for generating vibrational velocity. 'EP' for 'Equipartition', 'EP_TS' for 'Equipartition for TS',
        'QCT' for 'Quasi-classical trajectory', 'QCT_TS' for 'Quasi-classical trajectory for TS', 'CB' for 'Classical Boltzmann (Exponetial)',
        'QCT_noZPE' for 'QCT without ZPE'.
        :param temp: float, temperature
        :param state: list, a list of quantum numbers
        :param randomPhase: boolean or integer, if it is a boolean, perform random phase or not; if it is an integer, randomPhase * phi is the input phase
        :param anharmonicity: None or string, do not consider anharmonicity if None; file name indicates whether consider anharmonicity
        :return:
        atom_types: list of strs, element types
        atom_counts: list of ints, the number of each element type
        coordinates: list of lists, coordinates: [[x1, y1, z1], [x2, y2, z2], ...]
        flags: list of strs
        velocities_vib: list of lists, vibrational velocities, [[x1, y1, z1], [x2, y2, z2], ...]
        """
        # constant
        kb = 8.61733035e-05       #Boltzmann constant in eV/K
        
        # Read coordinates and normal modes
        cell, atom_types, atom_counts, coordinates, flags, _, _ = read_coordinates(input_file_coordinates, file_format)
        if lcell:
            self.set_cell(cell)
        num_atoms = sum(atom_counts)
        freq_list, modes = read_normal_modes(input_file_mode, num_modes, num_atoms)
        
        assert len(freq_list) == num_modes
        assert len(modes) == num_modes
        assert len(modes[0]) == num_atoms
        
        # Rotate the molecule, update both coordinates and modes
        if lrotation:
            coordinates, modes = rotate(atom_types, atom_counts, coordinates, modes)
                             
        # Compute energy for normal mode
        ### Equipartition ###
        if method == 'EP':
            energyList = np.array([kb * temp for i in range(num_modes)])     # In eV
        
        ### Equipartition for TS ###
        elif method == 'EP_TS':
            state = np.array([1.0 for _ in range(num_modes - 1)] + [0.5])
            energyList = kb * temp * state     # In eV
        
        ### Quasi-classical trajectory ###
        elif method == 'QCT':
            if state == None:
                state = self.genState(freq_list, num_modes, temp)
            assert len(state) == num_modes
            print('The state is ', state)
            stateStr = [str(i) for i in state]
            stateStr = ' states ' + ' '.join(stateStr)+' ;'
            self.title += stateStr
            energyList = np.array(freq_list) * (np.array(state) + 0.5)     # In eV
        
        ### Quasi-classical trajectory for TS ###
        elif method == 'QCT_TS':
            # vibrational modes
            if state == None:
                state = self.genState(freq_list[: -1], num_modes - 1, temp)
            assert len(state) == num_modes - 1
            print('The state is ', state)
            stateStr = [str(i) for i in state]
            stateStr = ' states ' + ' '.join(stateStr)
            self.title += stateStr
            energyList = np.array(freq_list[: -1]) * (np.array(state) + 0.5)
            # reaction coordinates
            energyReactionCoord = self.genClassicBoltzmann(temp)
            energyList = energyList.tolist()
            energyList.append(energyReactionCoord)
            energyList = np.array(energyList)     # In eV
            self.title += '  {:2.6f} eV;'.format(energyReactionCoord)            
        
        ### Classical Boltzmann (Exponetial) ###
        elif method == 'CB':
            energyList = [self.genClassicBoltzmann(temp) for i in range(num_modes)]     # In eV
        
        ### QCT without ZPE ###
        elif method == 'QCT_noZPE':
            energyList = np.array(freq_list) * np.array(state)     # In eV
        
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
        
        coordNorm, vNorm = self.genNormalCoordAndV(energyList, freq_list, randomPhase, anharmonicity)
        
        if method == 'QCT_TS':
            coordNormTS, vNormTS = self.genNormalCoordAndV(energyList[-1:], freq_list[-1:], randomPhase=1.5, anharmonicity=False)
            coordNorm[-1] = coordNormTS[0]
            vNorm[-1] = vNormTS[0]
        
        print('The velocity for each normal mode is', vNorm)
        print('The normal coordinate for each normal mode is', coordNorm)
        
        # Generate mass list
        mass = []
        for n, atom in enumerate(atom_types):
            number = atom_counts[n]
            massAtom = [self.masses[atom] for i in range(number * 3)]
            mass.extend(massAtom)
        assert len(mass) == num_atoms * 3
#        print(mass)
        
        # flat modes to modesMatrix
        modesMatrix = []
        for mode in modes:
            modeFlat = []
            for m in mode:
                modeFlat.extend(m)
            assert len(modeFlat) == 3 * num_atoms
            modesMatrix.append(modeFlat)
        assert len(modesMatrix) == num_modes
        
        # Compute velocity
        vMatrix = np.array(modesMatrix).transpose() * np.array(vNorm)
        vMatrix = vMatrix.transpose()                 # velocities * mass**0.5 in angstrom/10 fs
        vList = sum(vMatrix) / (np.array(mass)**0.5)  # velocities in angstrom/10 fs
        vList = vList/10.0                            # velocities in angstrom/fs
        vList = vList.tolist()
        velocities_vib = []
        for n in range(num_atoms):
            velocities_vib.append(vList[n * 3 : (n + 1) * 3])
        
        # Compute coordinates
        disMatrix = np.array(modesMatrix).transpose() * np.array(coordNorm)
        disMatrix = disMatrix.transpose()                  # dis * mass**0.5 in angstrom
        disList = sum(disMatrix) / (np.array(mass)**0.5)   # dis in angstrom
        disList = disList.tolist()
        dis = []
        for n in range(num_atoms):
            dis.append(disList[n * 3 : (n + 1) * 3])
            
        new_coordinates = np.array(coordinates) + np.array(dis)
        coordinates = new_coordinates.tolist().copy()
        
        assert len(velocities_vib) == num_atoms
#        print('The vibrational velocity is', velocities_vib)
#        print('The coordinates is', coordinates)
        
        return atom_types, atom_counts, coordinates, flags, velocities_vib
        
    
    def gen_velocities_trans(self, input_file, file_format, tempTrans, directTrans, lcell=False, methodTrans='EP'):
        """
        Generate translational velocity for a structure
        :param input_file: str, the name of coordinates file, i.e. POSCAR
        :param file_format: str, format of the input coordinates and velocities file. Allowed values include 'vasp' or 'ase'
        :param tempTrans: float, temperature of translational energy
        :param directTrans: list[float], vector indicate the direction of the translational velocity
        :param lcell: boolean, whether overwrite cell
        :param methodTrans: str, the name of method
        :return: vTrans: list of float, translational velocities, [x, y, z]
        """
        # constant
        tutoev = 1.0364314        #change tu unit to eV
        kb = 8.61733035e-05       #Boltzmann constant in eV/K
        
        # Read coordinates
        cell, atom_types, atom_counts, coordinates, flags, _, _ = read_coordinates(input_file, file_format)
        if lcell:
            self.set_cell(cell)
        
        # Normalize direction vector
        directTrans = directTrans / np.linalg.norm(directTrans)
        
        # Compute mass of the molecule
        totalMass = 0.0
        for n, atom in enumerate(atom_types):
            number = atom_counts[n]
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
    
    
    def genState(self, freq_list, num_modes, temp):
        """
        Generate a state for a molecule from Boltzmann distribution
        :param freq_list: list of floats, list of frequencies
        :param num_modes: int, number of modes
        :param temp: float, temperature
        :return:
        state: list of ints, list of quantum numbers
        """
        assert len(freq_list) == num_modes
        
        state = []
        for freq in freq_list:
            quantum = self.genQuantumNumber(freq, temp)
            state.append(quantum)
            
        assert len(state) == num_modes
        
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
    
    
    def genNormalCoordAndV(self, energyList, freq_list, randomPhase = False, anharmonicity = None):
        """
        Input a list of energy (in eV) and a list of frequency (in eV),
        Return normal coordinates and normal velocities based on phase
        :param energyList: list of floats, list of energies
        :param freq_list: list of floats, list of frequences
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
        assert len(energyList) == len(freq_list)
        num_modes = len(freq_list)
        
        # Compute amplitude
        energy = np.array(energyList) / tutoev    # energy unit: tu
        freq = np.array(freq_list) / hbar          # Unit: (10fs)**(-1)
        amplitude = np.sqrt(energy * 2.0) / freq  # list of amplitudes for all modes
        
        # Generate phase
        if isinstance(randomPhase, bool):
            if randomPhase:
                phase = np.random.uniform(0.0, 2.0 * np.pi, num_modes)
                phase = np.array(phase)
            else:
                # start from equilibrium popsition
                phase = np.random.randint(0, 2, num_modes)
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
            
            for mode in range(1, num_modes + 1):
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


    def PES(self, in_file_coordinates, in_file_eigen, num_modes, num_atoms, mode, sign, out_file_name):
        """
        Generate POSCAR file for computing PES along one mode using AIMD
        """
        cell, atom_types, atom_counts, coordinates, flags, velocities_old, _ = read_coordinates(in_file_coordinates, 'vasp')
        self.set_cell(cell)
        num_atoms = sum(atom_counts)
        num = len(atom_types)
        freq_list, modes = read_normal_modes(in_file_eigen, num_modes, num_atoms)
        assert len(freq_list) == num_modes
        assert len(modes) == num_modes
        assert len(modes[0]) == num_atoms

        # Generate mass list
        mass = []
        for n in range(num):
            atom = atom_types[n]
            number = atom_counts[n]
            mass_atom = [self.masses[atom] for i in range(number)]
            mass.extend(mass_atom)
        assert len(mass) == num_atoms
        
        # Generate velocity
        eigen = np.array(modes[mode-1])
        velocities = eigen.transpose()/(np.array(mass)**0.5)
        velocities = velocities.transpose()*sign
        
        self.set_state(atom_types, atom_counts, coordinates, flags, velocities)
        self.output(out_file_name)        
        

    def shift(self, in_file_name, atom_num, in_file_name_ref, atom_num_ref):
        """
        Shift the geometry in periodic system
        """
        _, atom_types_ref, atom_counts_ref, coordinates_ref, flags_ref, velocities_ref, _ = read_coordinates(in_file_name_ref, 'vasp')
        cell, atom_types, atom_counts, coordinates, flags, velocities, _ = read_coordinates(in_file_name, 'vasp')
        self.set_cell(cell)
        direction = np.array(coordinates_ref[atom_num_ref-1])-np.array(coordinates[atom_num-1])
        pos = np.array(coordinates)

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
        self.set_state(atom_types, atom_counts, pos_new, flags, velocities)
        out_file_name = in_file_name+'shifted'
        self.output(out_file_name)

    
    def add_perp(self, in_file_name, Cu1_number, Cu2_number, c_number, O2_length, O2_dist):
        """
        Add O2 perpendicular to a given Cu-Cu bond
        :param Cu1_number: int, representing the coordinates of Cu1
        :param Cu2_number: int, representing the coordinates of Cu2
        :param c_number: list of 3 float or int, representing the coordinates of a reference atom to define the direction that O2 comes in, if c_number is a list then use that as the coordinates
        :param O2_length: float, representing O2 bondlength
        :param O2_dist: float, representing the distance from Cu-Cu center to O2
        """
        cell, atom_types, atom_counts, coordinates, flags, velocities, _ = read_coordinates(in_file_name, 'vasp')
        self.set_cell(cell)
        self.set_state(atom_types, atom_counts, coordinates, flags, velocities)
        
        Cu1 = self.get_atom_coordinates(coordinates, Cu1_number)
        Cu2 = self.get_atom_coordinates(coordinates, Cu2_number)
        if isinstance(c_number, int):
            c = self.get_atom_coordinates(coordinates, c_number)
        elif isinstance(c_number, list):
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
        O2_coordinates = [O1, O2]
        O2_flags = ['   F   F   F', '   F   F   F']
        O2_velocities = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        self.set_state(O2_types, O2_numbers, O2_coordinates, O2_flags, O2_velocities)
        self.output(in_file_name+'_O2')
        print('Adding O2 done')


    def add_toTop(self, in_file_name, Cu_number, O1_number, O2_number, O2_length, O2_dist):
        """
        Add O2 to the top of a given Cu-Cu bond
        :param Cu_number: int, representing the coordinates of Cu1
        :param c_number: list of 3 float or int, representing the coordinates of a reference atom to define the direction that O2 comes in, if c_number is a list then use that as the coordinates
        :param O2_length: float, representing O2 bondlength
        :param O2_dist: float, representing the distance from Cu-Cu center to O2
        """
        cell, atom_types, atom_counts, coordinates, flags, velocities, _ = read_coordinates(in_file_name, 'vasp')
        self.set_cell(cell)
        self.set_state(atom_types, atom_counts, coordinates, flags, velocities)
        
        Cu = np.array(self.get_atom_coordinates(coordinates, Cu_number))
        O1 = np.array(self.get_atom_coordinates(coordinates, O1_number))
        O2 = np.array(self.get_atom_coordinates(coordinates, O2_number))
        direction = (O2+O1)/2.0-Cu
        direction = direction/np.linalg.norm(direction)
        O2Direction = O2-O1
        O2Direction = O2Direction/np.linalg.norm(O2Direction)
        
        center = Cu+direction*O2_dist
        O2New = center+O2Direction*O2_length/2.0
        O1New = center-O2Direction*O2_length/2.0
        O2New = O2New.tolist()
        O1New = O1New.tolist()
        
        coordinates[O2_number-1] = O2New
        coordinates[O1_number-1] = O1New
        print(O1New, O2New)
        self.coordinates = coordinates.copy()
        self.output(in_file_name+'_modified')
        print('Adding O2 done')
    
    
    def output(self, output_file):
        """
        Write the output file for the system
        """
        if len(self.cell) == 0:
            raise Exception('Unit cell not set')
        cell = self.cell.copy()
        atom_types = self.atom_types.copy()
        atom_counts = self.atom_counts.copy()
        coordinates = self.coordinates.copy()
        flags = self.flags.copy()
        velocities = self.velocities.copy()
        num_atoms = sum(atom_counts)
        assert len(coordinates) == num_atoms
        assert len(flags) == num_atoms
        assert len(velocities) == num_atoms
        
        # Write POSCAR file
        with open(output_file, 'w') as out:
            out.write(self.title+'\n')
            out.write('   1.000'+'\n')
            
            cell_str = ''
            for i in range(3):
                for j in range(3):
                    cell_str += '   '+"{:10.8f}".format(float(cell[i][j]))
                cell_str += '\n'
            out.write(cell_str)
            
            atoms_type_str = '   '+'   '.join(atom_types)+'\n'
            out.write(atoms_type_str)
            atom_counts = [str(n) for n in atom_counts]
            atoms_num_str = '   '+'   '.join(atom_counts)+'\n'
            out.write(atoms_num_str)
            out.write('Selective dynamics'+'\n'+'Cartesian'+'\n')
            
            coordinates_str = ''
            for i in range(num_atoms):
                for j in range(3):
                    coordinates_str += '   '+"{:10.8f}".format(coordinates[i][j])
                coordinates_str += flags[i]+'\n'
            out.write(coordinates_str)
            
            out.write('Cartesian'+'\n')
            velocities_str = ''
            for i in range(num_atoms):
                for j in range(3):
                    velocities_str += '   '+"{:10.8f}".format(velocities[i][j])
                velocities_str += '\n'
            out.write(velocities_str)


### Main functions ###
    def state_sampling(self, input_file_coordinates, input_file_mode, num_modes, lrotation, method, T, state=None, output_file=None):
        """
        Generate a state including coordinates and velocities, i.e. POSCAR file, for AIMD simulation MD
        :param input_file_coordinates: str, file path of the local minimum structure (POSCAR file)
        :param lrotation: boolean, whether rotate molecule or not
        :param method: str, method for generating vibrational velocity. 'EP' for 'Equipartition', 'EP_TS' for 'Equipartition for TS',
        'QCT' for 'Quasi-classical trajectory', 'QCT_TS' for 'Quasi-classical trajectory for TS', 'CB' for 'Classical Boltzmann (Exponetial)',
        'QCT_noZPE' for 'QCT without ZPE'.
        :param state: list of ints, quantum states
        """
        # Read coordinates and modes files, generate velocities
        atom_types, atom_counts, coordinates, flags, velocities_vib = self.gen_velocities_vib(input_file_coordinates, 'vasp', input_file_mode, num_modes,
                                                                                               lcell=True, lrotation=lrotation, method=method, temp=T, state=state)
        # Update data attributes
        self.set_state(atom_types, atom_counts, coordinates, flags, velocities_vib)
        # Output initial state, i.e. POSCAR, for MD
        if output_file is None:
            output_file = os.path.join(os.getcwd(), 'state_sampling.POSCAR')
        self.output(output_file)


    def state_sampling_from_trajectory(self, input_file_coordinates, input_file_traj_coord, input_file_velocities, input_file_energy,
                                       trajectory_range, output_file=None):
        """
        Read coordinates and velocities of the trajectory, generate a POSCAR file for given snapshot
        :param input_file_coordinates: str, file path of the local minimum structure (POSCAR file)
        :param trajectory_range: list of two ints, the range within which to sample a state
        """
        # Get atom_types, atom_counts and flags from the local minimum structure
        _, atom_types, atom_counts, _, flags, _, _ = read_coordinates(input_file_coordinates, 'vasp')
        num_atoms = sum(atom_counts)
        # Randomly generate a snapshot index
        snapshot_idx = np.random.randint(trajectory_range[0], trajectory_range[1])
        # Get cell, coordinates, velocities and energy from the trajectory
        cell, _, _, coordinates, _ = read_coordinates_from_trajectory(input_file_traj_coord, snapshot_idx)
        velocities = read_velocities_from_trajectory(input_file_velocities, num_atoms, snapshot_idx)
        energy_str = read_energy_from_trajectory(input_file_energy, snapshot_idx)
        # Update the state
        self.set_cell(cell)
        self.set_state(atom_types, atom_counts, coordinates, flags, velocities)
        self.set_title(energy_str)
        # Output initial state, i.e. POSCAR, for MD
        if output_file is None:
            output_file = os.path.join(os.getcwd(), 'state_sampling.POSCAR')
        self.output(output_file)


    def MD_initialization(self, cluster_infile_coordinates, molecule_infile_coordinates, impact_method, impact_site,
                          distance_to_cluster, trans_T, method_trans='EP', output_file=None):
        """

        :param cluster_infile_coordinates: str,
        :param molecule_infile_coordinates: str,
        :param impact_method: str, impact method. Allowed values include 'cluster_center', 'vertical_targeted' and 'vertical_nontargeted'
        :param impact_site: molecule impact site
        """
        ## Cluster/surface initialization
        cell, cluster_atom_types, cluster_atom_counts, cluster_coordinates, cluster_flags, cluster_velocities, _ = read_coordinates(cluster_infile_coordinates,'vasp')
        self.set_cell(cell)
        self.set_state(cluster_atom_types, cluster_atom_counts, cluster_coordinates, cluster_flags, cluster_velocities)

        ## Molecule initialization
        _, molecule_atom_types, molecule_atom_counts, _, molecule_flags, molecule_velocities_vib, _ = read_coordinates(molecule_infile_coordinates, 'vasp')
        # Calculate impact site coordinates and reference site coordinates
        if impact_method == 'cluster_center': # molecule collides with the impact site in the direction towards the center of mass of the cluster
            coordinates_impact_site = self.get_atom_coordinates(cluster_coordinates, impact_site)   # impact site coordinates
            coordinates_ref_atom = get_COM(cluster_atom_types, cluster_atom_counts, cluster_coordinates)   # cluster center of mass coordinates
        elif impact_method == 'vertical_targeted': # molecule collides with a cluster atom or a bond between two cluster atoms perpendicular to the xy plane.
            if isinstance(impact_site, int):
                coordinates_impact_site = self.get_atom_coordinates(cluster_coordinates, impact_site)
            elif (isinstance(impact_site, tuple) or isinstance(impact_site, list)) and len(impact_site) == 2:
                coordinates_impact_site_1 = self.get_atom_coordinates(cluster_coordinates, impact_site[0])
                coordinates_impact_site_2 = self.get_atom_coordinates(cluster_coordinates, impact_site[1])
                coordinates_impact_site = (np.array(coordinates_impact_site_1) + np.array(coordinates_impact_site_2)) / 2.0
            else:
                print('Error: invalid impact site.')
                return
            coordinates_ref_atom = (np.array(coordinates_impact_site) - np.array([0.0, 0.0, 1.0])).tolist()
        elif impact_method == 'vertical_nontargeted': # molecule collides with a cluster randomly within a range definde by impact_site
            assert isinstance(impact_site, list) and len(impact_site)==3 and len(impact_site[0])==2 and len(impact_site[1])==2, 'Error: invalid impact site.'
            [[xmin, xmax], [ymin, ymax], z] = impact_site
            coordinates_impact_site = []
            coordinates_impact_site.append(random.uniform(0, 1) * (xmax - xmin) + xmin)
            coordinates_impact_site.append(random.uniform(0, 1) * (ymax - ymin) + ymin)
            coordinates_impact_site.append(z)
            coordinates_ref_atom = (np.array(coordinates_impact_site) - np.array([0.0, 0.0, 1.0])).tolist()
        else:
            print('Error: invalid impact method.')
            return
        molecule_coordinates, direct_trans = Sampler.gen_coordinates(molecule_infile_coordinates, 'vasp', coordinates_impact_site, coordinates_ref_atom, distance_to_cluster)

        # Molecule initial velocity
        molecule_velocities_trans = self.gen_velocities_trans(molecule_infile_coordinates, 'vasp', trans_T, direct_trans, False, method_trans)
        molecule_velocities = (np.array(molecule_velocities_vib) + np.array(molecule_velocities_trans)).tolist()
        # Update data attribute for molecule
        self.set_state(molecule_atom_types, molecule_atom_counts, molecule_coordinates, molecule_flags, molecule_velocities)

        # Output initial state, i.e. POSCAR, for MD
        if output_file is None:
            output_file = os.path.join(os.getcwd(), 'MD_molecule_cluster.POSCAR')
        self.output(output_file)


if __name__ == '__main__':

    sampler = Sampler()

