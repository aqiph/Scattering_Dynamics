#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 14:59:23 2019

@author: Aqiph
"""
import os

import numpy as np
from scipy.stats import maxwell
from scipy import interpolate
import random

from utils.read_file import read_coordinates, read_normal_modes, read_coordinates_from_trajectory,\
    read_velocities_from_trajectory, read_energy_from_trajectory
from utils.tools import get_COM, rotate, gen_state_from_Boltzmann, gen_energy_from_Classical_Boltzmann



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


    def get_state(self):
        return self.atom_types, self.atom_counts, self.coordinates, self.flags, self.velocities


    def get_title(self):
        return self.title


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
        :param input_file: str, path of the coordinates input file, i.e. POSCAR
        :param file_format: str, format of the coordinates and velocities input file. Allowed values include 'vasp' or 'ase'
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
    
    
    def gen_velocities_vib(self, input_file_coordinates, file_format, input_file_modes, num_modes, lcell, lrotation, method_vib, temp_vib,
                           state=None, input_phase='random', input_file_anharmonicity=None):
        """
        Generate vibrational velocity for a structure
        :param input_file_coordinates: str, path of the coordinates input file, i.e. POSCAR
        :param file_format: str, format of the coordinates and velocities input file. Allowed values include 'vasp' or 'ase'
        :param input_file_modes: str, path of the normal modes input file
        :param num_modes: int, number of normal modes
        :param lcell: boolean, whether to overwrite the cell or not
        :param lrotation: boolean, whether to rotate molecule or not
        :param method_vib: str, method for generating vibrational velocities. 'EP' for 'Equipartition', 'EP_TS' for 'Equipartition for TS',
        'QCT' for 'Quasi-classical trajectory', 'QCT_TS' for 'Quasi-classical trajectory for TS', 'CB' for 'Classical Boltzmann (Exponetial)',
        'QCT_noZPE' for 'QCT without ZPE'.
        :param temp_vib: float, temperature of vibrational energy
        :param state: list, list of quantum numbers
        :param input_phase: str or list of floats; if it is a str, allowed values include 'random' and 'equilibrium';
        if it is a list of floats, input phases are defined by input_phase * pi
        :param input_file_anharmonicity: None or str, if it is None, do not consider anharmonicity;
        if it is a str, it is the path of the anharmonicity input file
        :return:
        atom_types: list of strs, element types
        atom_counts: list of ints, number of atoms for each element type
        coordinates: list of lists, [[x1, y1, z1], [x2, y2, z2], ...]
        flags: list of strs, flags
        velocities_vib: list of lists, vibrational velocities, [[x1, y1, z1], [x2, y2, z2], ...]
        """
        # constant
        kb = 8.61733035e-05       #Boltzmann constant in eV/K
        
        # read coordinates and normal modes
        cell, atom_types, atom_counts, coordinates, flags, _, _ = read_coordinates(input_file_coordinates, file_format)
        if lcell:
            self.set_cell(cell)
        num_atoms = sum(atom_counts)
        freq_list, modes = read_normal_modes(input_file_modes, num_modes, num_atoms)
        assert len(freq_list) == num_modes
        assert len(modes) == num_modes
        assert len(modes[0]) == num_atoms
        
        # rotate the molecule, update both coordinates and modes
        if lrotation:
            coordinates, modes = rotate(atom_types, atom_counts, coordinates, modes, lprint=False)
                             
        # compute energy for each normal mode
        ### Equipartition ###
        if method_vib == 'EP':
            state = np.array([1.0 for _ in range(num_modes)])
            energy_list = kb * temp_vib * state     # In eV
            state_str = ' states ' + ' '.join([str(i) for i in state]) + ' ;'

        ### Equipartition for TS ###
        elif method_vib == 'EP_TS':
            state = np.array([1.0 for _ in range(num_modes - 1)] + [0.5])
            energy_list = kb * temp_vib * state     # In eV
            state_str = ' states ' + ' '.join([str(i) for i in state]) + ' ;'

        ### Quasi-classical trajectory ###
        elif method_vib == 'QCT':
            if state == None:
                state = gen_state_from_Boltzmann(freq_list, num_modes, temp_vib)
            assert len(state) == num_modes
            energy_list = np.array(freq_list) * (np.array(state) + 0.5)  # In eV
            state_str = ' states ' + ' '.join([str(i) for i in state])+' ;'

        ### QCT without ZPE ###
        elif method_vib == 'QCT_noZPE':
            if state == None:
                state = gen_state_from_Boltzmann(freq_list, num_modes, temp_vib)
            assert len(state) == num_modes
            energy_list = np.array(freq_list) * np.array(state)     # In eV
            state_str = ' states ' + ' '.join([str(i) for i in state]) + ' ;'

        ### Quasi-classical trajectory for TS ###
        elif method_vib == 'QCT_TS':
            # vibrational modes except the imaginary mode
            if state == None:
                state = gen_state_from_Boltzmann(freq_list[: -1], num_modes - 1, temp_vib)
            assert len(state) == num_modes - 1
            energy_list = np.array(freq_list[: -1]) * (np.array(state) + 0.5)  # In eV
            # reaction coordinates
            energy_reaction_coordinate = gen_energy_from_Classical_Boltzmann(temp_vib)
            energy_list = np.append(energy_list, energy_reaction_coordinate)  # In eV
            state_str = ' states ' + ' '.join([str(i) for i in state]) + ' {:2.6f} eV;'.format(energy_reaction_coordinate)
        
        ### Classical Boltzmann (Exponetial) ###
        elif method_vib == 'CB':
            state = []
            energy_list = np.array([gen_energy_from_Classical_Boltzmann(temp_vib) for _ in range(num_modes)])     # In eV
            state_str = ' states ' + ' '.join([str(i) for i in state]) + ' ;'
        
        ### test ###
        elif method_vib == 'test':
            if state == None:
                state = np.array([1.0 for _ in range(num_modes)])
            assert len(state) == num_modes
            energy_list = kb * temp_vib * np.array(state)
            state_str = ' states ' + ' '.join([str(i) for i in state]) + ' ;'
        
        ### error ###
        else:
            raise Exception('Error: invalid method for vibrational velocities.')

        print('The state is ', state)
        self.set_title(state_str)

        # compute total vibrational energy
        Ev = np.sum(energy_list)
        print('The vibrational energies are', energy_list)
        print('The total vibrational energy is', Ev)
        self.set_title(' Ev '+"{:10.5f}".format(Ev)+' eV;')

        # compute normal coordinates and normal velocities
        print('Input phase: ', input_phase)
        coordinates_Norm, velocities_Norm = Sampler.gen_normal_coordinates_velocities(energy_list, freq_list, input_phase, input_file_anharmonicity)
        if method_vib == 'QCT_TS':
            coordinates_Norm_TS, velocities_Norm_TS = Sampler.gen_normal_coordinates_velocities(energy_list[-1:], freq_list[-1:], input_phase=[1.5],
                                                           input_file_anharmonicity=input_file_anharmonicity) # input_phase: 0.5 or 1.5 for TS
            coordinates_Norm[-1] = coordinates_Norm_TS[0]
            velocities_Norm[-1] = velocities_Norm_TS[0]
        print('The velocity for each normal mode is', velocities_Norm)
        print('The normal coordinate for each normal mode is', coordinates_Norm)
        
        # get mass list
        mass = []
        for n, atom in enumerate(atom_types):
            count = atom_counts[n]
            massAtom = [self.masses[atom] for _ in range(count * 3)]
            mass.extend(massAtom)
        assert len(mass) == num_atoms * 3
        
        # flat modes to modes_flat
        modes_flat = []
        for mode in modes:
            mode_flat = []
            for m in mode:
                mode_flat.extend(m)
            assert len(mode_flat) == 3 * num_atoms
            modes_flat.append(mode_flat)
        assert len(modes_flat) == num_modes
        
        # compute velocities
        velocities_matrix = (np.array(modes_flat).transpose() * np.array(velocities_Norm)).transpose() # velocities * mass**0.5 in angstrom/10 fs
        velocities_flat = sum(velocities_matrix) / (np.array(mass)**0.5)  # velocities in angstrom/10 fs
        velocities_flat = (velocities_flat/10.0).tolist()                 # velocities in angstrom/fs
        velocities_vib = []
        for n in range(num_atoms):
            velocities_vib.append(velocities_flat[n * 3 : (n + 1) * 3])
        assert len(velocities_vib) == num_atoms
        
        # compute coordinates
        displacement_matrix = (np.array(modes_flat).transpose() * np.array(coordinates_Norm)).transpose() # dis * mass**0.5 in angstrom
        displacement_flat = sum(displacement_matrix) / (np.array(mass)**0.5)  # dis in angstrom
        displacement_flat = displacement_flat.tolist()
        displacement = []
        for n in range(num_atoms):
            displacement.append(displacement_flat[n * 3 : (n + 1) * 3])
        coordinates = (np.array(coordinates) + np.array(displacement)).tolist()
        assert len(coordinates) == num_atoms
        
        return atom_types, atom_counts, coordinates, flags, velocities_vib
        
    
    def gen_velocities_trans(self, input_file, file_format, lcell, method_trans, temp_trans, translation_vector):
        """
        Generate translational velocities for a structure
        :param input_file: str, path of the coordinates input file, i.e. POSCAR
        :param file_format: str, format of the coordinates and velocities input file. Allowed values include 'vasp' or 'ase'
        :param lcell: boolean, whether to overwrite the cell or not
        :param method_trans: str, method for generating translational velocities. 'EP' for 'Equipartition', 'MAXWELL' for 'Maxwell'.
        :param temp_trans: float, temperature of translational energy
        :param translation_vector: list of floats, vector that describes the direction of the translational velocity
        :return: velocities_trans: list of lists, translational velocities, [[x1, y1, z1], [x2, y2, z2], ...]
        """
        # constant
        tutoev = 1.0364314        #change tu unit to eV
        kb = 8.61733035e-05       #Boltzmann constant in eV/K
        
        # read coordinates
        cell, atom_types, atom_counts, coordinates, flags, _, _ = read_coordinates(input_file, file_format, lenergy=False)
        if lcell:
            self.set_cell(cell)
        
        # normalize translation vector
        translation_vector_normed = translation_vector / np.linalg.norm(translation_vector)
        
        # compute mass of the molecule
        total_mass = 0.0
        for n, atom in enumerate(atom_types):
            number = atom_counts[n]
            total_mass += self.masses[atom] * number
            
        # compute velocities of the molecule
        if method_trans == 'EP':
            rate_trans = (((3 * kb * temp_trans / tutoev) / total_mass)**0.5) / 10.0
        elif method_trans == 'MAXWELL':
            a = ((kb * temp_trans / tutoev) / total_mass)**0.5
            x = maxwell.rvs(size = 1)[0]
            rate_trans = a * x / 10.0
        else:
            raise Exception('Error: invalid method for translational velocities.')
        velocities_trans = (np.array(translation_vector_normed) * rate_trans).tolist()
        assert len(velocities_trans) == 3

        # compute translational energy
        Et = tutoev * total_mass * (10.0 * rate_trans) ** 2.0 / 2.0
        Et_avg = 3.0 * kb * temp_trans / 2.0
        print('The incident translational energy is {:4.4f} eV; the average is {:4.4f} eV.'.format(Et, Et_avg))
        print('The translational rate is {:4.4f} angstrom/fs.'.format(rate_trans))
        print(f'The translational velocity is {velocities_trans}.')
        self.set_title(" Et   {:6.4f} eV;".format(Et))
        
        return velocities_trans


    @staticmethod
    def gen_normal_coordinates_velocities(energy_list, freq_list, input_phase='random', input_file_anharmonicity=None):
        """
        Given energy (in eV) and frequency (in eV) for each mode, compute normal coordinates and normal velocities.
        :param energy_list: list of floats, energy (in eV) for each mode
        :param freq_list: list of floats, frequency (in eV) for each mode
        :param input_phase: str or list of floats; if it is a str, allowed values include 'random' and 'equilibrium';
        if it is a list of floats, input phases are defined by input_phase * pi
        :param input_file_anharmonicity: None or str, if it is None, do not consider anharmonicity;
        if it is a str, it is the path of the anharmonicity input file
        :return:
        coordinates_normal: list of lists, normal coordinates, [[x1, y1, z1], [x2, y2, z2], ...]
        velocities_normal: list of lists, normal velocities, [[x1, y1, z1], [x2, y2, z2], ...]
        """
        # constants
        hbar = 0.06582119514  # eV * 10fs
        tutoev = 1.0364314  # change tu unit to eV

        # check inputs
        num_modes = len(freq_list)
        assert len(energy_list) == num_modes

        # compute amplitudes
        energies_tu = np.array(energy_list) / tutoev  # energy unit: tu
        freqs = np.array(freq_list) / hbar  # Unit: (10fs)**(-1)
        amplitudes = np.sqrt(energies_tu * 2.0) / freqs  # list of amplitudes for all modes

        # generate phase
        if isinstance(input_phase, str):
            if input_phase == 'random':
                # randomly generate a phase
                phase = np.random.uniform(0.0, 2.0 * np.pi, num_modes)
                phase = np.array(phase)
            elif input_phase == 'equilibrium':
                # start from equilibrium position
                phase = np.random.randint(0, 2, num_modes)
                phase = np.array(phase) * np.pi + 0.5 * np.pi
            else:
                raise Exception('Error: invalid input phase.')
        elif isinstance(input_phase, list):
            # generate phase according to input_phase
            phase = np.array(input_phase) * np.pi
        else:
            raise Exception('Error: invalid input phase.')

        # compute normal coordinates and normal velocities
        s, c = np.sin(phase), np.cos(phase)
        coordinates_normal = amplitudes * c  # A * cos(phi)
        velocities_normal = -amplitudes * freqs * s  # -A * freq * sin(phi)

        # correct for anharmonicity
        if input_file_anharmonicity is not None:
            with open(input_file_anharmonicity, 'r') as data:
                data.readline()
                for mode in range(1, num_modes + 1):
                    line = data.readline().strip().split()
                    if line[-1] == 'False':
                        continue
                    input_file_PES = 'mode_' + str(mode)
                    coord_normal_mode = coordinates_normal[mode - 1]
                    freq_mode = freqs[mode - 1]
                    coordinates_normal[mode - 1] = Sampler.anharmonicity_correction(input_file_PES, coord_normal_mode, freq_mode)

        return coordinates_normal, velocities_normal


    @staticmethod
    def anharmonicity_correction(input_file_PES, coord_normal_mode, freq_mode):
        """
        Correct normal coordinates, using real PES (containing anharmonicity)
        :param input_file_PES: string, path of the real PES input file
        :param coord_normal_mode: float, normal coordinate obtained from the harmonic oscillator model
        :param freq_mode: float, frequency of the mode, unit: (10fs)**(-1)
        :return:
        coord_normal_mode_AH: float, normal coordinate corrected by real PES
        """
        # constants
        hbar = 0.06582119514  # eV * 10fs
        tutoev = 1.0364314  # change tu unit to eV

        # read potential energy
        PES_negative = []
        PES_positive = []
        q1 = []
        q2 = []

        with open(input_file_PES, 'r') as data:
            # read negative side
            data.readline()
            line = data.readline()
            while 'side2' not in line:
                line = line.strip().split()
                PES_negative.append(float(line[0]))
                q1.append(float(line[1]))
                line = data.readline()
            # read positive side
            line = data.readline()
            while line:
                line = line.strip().split()
                PES_positive.append(float(line[0]))
                q2.append(float(line[1]))
                line = data.readline()

        # fit the real PES from input data
        if coord_normal_mode <= 0:
            PES = interpolate.interp1d(PES_negative, q1)
        else:
            PES = interpolate.interp1d(PES_positive, q2)

        # compute potential energy from input coord_normal_mode
        V = 0.5 * (freq_mode * coord_normal_mode) ** 2.0 * tutoev  # in eV

        # compute the real normal coordinate by interpolation
        coord_normal_mode_AH = PES(V)
        print(input_file_PES, 'frequency', freq_mode * hbar, ', difference in normal coordinates:',
              coord_normal_mode_AH - coord_normal_mode)

        return coord_normal_mode_AH

    
    def output(self, output_file):
        """
        Write the output file for the system
        :param output_file: str, path of the output file, i.e. POSCAR
        """
        cell = self.get_cell()
        if len(cell) == 0:
            raise Exception('Error: undefined unit cell.')
        atom_types, atom_counts, coordinates, flags, velocities = self.get_state()
        assert len(atom_types) == len(atom_counts)
        num_atoms = sum(atom_counts)
        assert len(coordinates) == num_atoms
        assert len(flags) == num_atoms
        assert len(velocities) == num_atoms
        title = self.get_title()
        
        # write POSCAR file
        with open(output_file, 'w') as out:
            # title
            out.write(title + '\n')
            out.write('   1.000'+'\n')
            # cell
            cell_str = ''
            for i in range(3):
                for j in range(3):
                    cell_str += '   '+"{:10.8f}".format(float(cell[i][j]))
                cell_str += '\n'
            out.write(cell_str)
            # atom types and atom numbers
            atom_types_str = '   ' + '   '.join(atom_types) + '\n'
            out.write(atom_types_str)
            atom_counts = [str(n) for n in atom_counts]
            atom_counts_str = '   ' + '   '.join(atom_counts) + '\n'
            out.write(atom_counts_str)
            out.write('Selective dynamics\nCartesian\n')
            # coordinates
            coordinates_str = ''
            for i in range(num_atoms):
                for j in range(3):
                    coordinates_str += '   {:10.8f}'.format(coordinates[i][j])
                coordinates_str += flags[i] + '\n'
            out.write(coordinates_str)
            # velocities
            out.write('Cartesian'+'\n')
            velocities_str = ''
            for i in range(num_atoms):
                for j in range(3):
                    velocities_str += '   {:10.8f}'.format(velocities[i][j])
                velocities_str += '\n'
            out.write(velocities_str)


### Main functions ###
    def sample_state_from_distribution(self, input_file_coordinates, input_file_modes, num_modes, lrotation,
                                       method_vib, temp_vib, state=None, output_file=None):
        """
        Generate a state including coordinates and velocities, i.e. POSCAR file, for AIMD simulation MD
        :param input_file_coordinates: str, path of the local minimum structure input file, i.e. POSCAR
        :param input_file_modes: str, path of the normal modes input file
        :param num_modes: int, number of normal modes need to input
        :param lrotation: boolean, whether to rotate molecule or not
        :param method_vib: str, method for generating vibrational velocities. 'EP' for 'Equipartition', 'EP_TS' for 'Equipartition for TS',
        'QCT' for 'Quasi-classical trajectory', 'QCT_TS' for 'Quasi-classical trajectory for TS', 'CB' for 'Classical Boltzmann (Exponetial)',
        'QCT_noZPE' for 'QCT without ZPE'.
        :param temp_vib: float, temperature of vibrational energy
        :param state: list of ints, quantum states
        :param output_file: str, path of the output file, i.e. POSCAR
        """
        # read coordinates and modes files, generate velocities
        atom_types, atom_counts, coordinates, flags, velocities_vib = self.gen_velocities_vib(input_file_coordinates, 'vasp', input_file_modes, num_modes,
                                                                                              lcell=True, lrotation=lrotation, method_vib=method_vib, temp_vib=temp_vib, state=state)
        # update data attributes
        self.set_state(atom_types, atom_counts, coordinates, flags, velocities_vib)
        # output initial state, i.e. POSCAR, for MD
        if output_file is None:
            output_file = os.path.join(os.getcwd(), 'state_sampling_from_distribution.POSCAR')
        self.output(output_file)


    def sample_state_from_trajectory(self, input_file_coordinates, input_file_traj_coord, input_file_velocities, input_file_energy,
                                     trajectory_range, output_file=None):
        """
        Read coordinates and velocities of the trajectory, generate a POSCAR file for given snapshot
        :param input_file_coordinates: str, path of the local minimum structure input file, i.e. POSCAR
        :param input_file_traj_coord: str, path of the trajectory coordinates input file, i.e. .coor file
        :param input_file_velocities: str, path of the trajectory velocities input file, i.e. .velocity file
        :param input_file_energy: str, path of the trajectory energy input file, i.e. .energy file
        :param trajectory_range: list of two ints, range within which to sample a state
        :param output_file: str, path of the output file, i.e. POSCAR
        """
        # get atom_types, atom_counts and flags from the local minimum structure
        _, atom_types, atom_counts, _, flags, _, _ = read_coordinates(input_file_coordinates, 'vasp')
        num_atoms = sum(atom_counts)
        # randomly generate a snapshot index
        snapshot_idx = np.random.randint(trajectory_range[0], trajectory_range[1])
        # get cell, coordinates, velocities and energy from the trajectory
        cell, _, _, coordinates, _ = read_coordinates_from_trajectory(input_file_traj_coord, snapshot_idx)
        velocities = read_velocities_from_trajectory(input_file_velocities, num_atoms, snapshot_idx)
        energy_str = read_energy_from_trajectory(input_file_energy, snapshot_idx)
        # update the state
        self.set_cell(cell)
        self.set_state(atom_types, atom_counts, coordinates, flags, velocities)
        self.set_title(energy_str)
        # output initial state, i.e. POSCAR, for MD
        if output_file is None:
            output_file = os.path.join(os.getcwd(), 'state_sampling_from_trajectory.POSCAR')
        self.output(output_file)


    def MD_initialization(self, cluster_input_file_coord, molecule_input_file_coord, impact_method, impact_site,
                          distance_to_cluster, method_trans, temp_trans, output_file=None):
        """
        Initialize the system, including cluster/surface and impact molecule, generate the input file, i.e. POSCAR, for AIMD simulation
        :param cluster_input_file_coord: str, path of the cluster coordinates input file, i.e. POSCAR
        :param molecule_input_file_coord: str, path of the molecular coordinates input file, i.e. POSCAR
        :param impact_method: str, impact method. Allowed values include 'cluster_center', 'vertical_targeted' and 'vertical_nontargeted'
        :param impact_site: int, molecule impact site
        :param distance_to_cluster: float, distance between the molecule's COM and the impact site
        :param method_trans: str, method for generating translational velocities. 'EP' for 'Equipartition', 'MAXWELL' for 'Maxwell'.
        :param temp_trans: float, temperature of translational energy
        :param output_file: str, path of the output file, i.e. POSCAR
        """
        ## Cluster/surface initialization
        cell, cluster_atom_types, cluster_atom_counts, cluster_coordinates, cluster_flags, cluster_velocities, _ = read_coordinates(cluster_input_file_coord,'vasp')
        self.set_cell(cell)
        self.set_state(cluster_atom_types, cluster_atom_counts, cluster_coordinates, cluster_flags, cluster_velocities)

        ## Molecule initialization
        _, molecule_atom_types, molecule_atom_counts, _, molecule_flags, molecule_velocities_vib, _ = read_coordinates(molecule_input_file_coord, 'vasp')
        # calculate impact site coordinates and reference site coordinates
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
        molecule_coordinates, translation_vector = Sampler.gen_coordinates(molecule_input_file_coord, 'vasp', coordinates_impact_site, coordinates_ref_atom, distance_to_cluster)

        # molecule initial velocity
        molecule_velocities_trans = self.gen_velocities_trans(molecule_input_file_coord, 'vasp', lcell=False,
                                                              method_trans=method_trans, temp_trans=temp_trans, translation_vector=translation_vector)
        molecule_velocities = (np.array(molecule_velocities_vib) + np.array(molecule_velocities_trans)).tolist()
        # update data attribute for molecule
        self.set_state(molecule_atom_types, molecule_atom_counts, molecule_coordinates, molecule_flags, molecule_velocities)

        # output initial state, i.e. POSCAR, for MD
        if output_file is None:
            output_file = os.path.join(os.getcwd(), 'MD_molecule_cluster.POSCAR')
        self.output(output_file)


if __name__ == '__main__':

    sampler = Sampler()

