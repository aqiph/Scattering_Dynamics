#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 09:44:35 2019

@author: Aqiph
"""

import numpy as np
from matplotlib.font_manager import FontProperties
font = FontProperties()
font.set_family('serif')
font.set_name('Calibri')
font.set_size(11)
import matplotlib.pyplot as plt
import math


class Trajectory(object):

    
### Contributor ###    
    def __init__(self):
        self.timestep = 0.0
        self.cell = []
        self.atom_types = []        
        self.atom_counts = []
        self.traj = []
        self.length = 0
        self.velocities = []
        
        self.timestep_ref = 0.0
        self.cell_ref = []
        self.atom_types_ref = []
        self.atom_counts_ref = []
        self.traj_ref = []
        self.length_ref = 0
        self.velocities_ref = []
        
        self.filename = 'trajectory'
        
        self.masses = {'Pt':195.080, 'H':1.000, 'C':12.011, 'Cu':63.546,
                       'Al':26.981, 'O':16.000, 'Sr':87.620}
        
        self.CH_reach_threshold = 3.0     # reach the cluster, if C-Pt bond length < self.CH_reach_threshold
        self.CH_threshold_1 = 1.6         # reach the TS, if C-H bond length > self.CH_bond_threshold
        self.CH_threshold_2 = 3.0         # dissociation ends, if if C-H bond length > self.CH_bond_threshold
        self.CPt_threshold_1 = 4.0        # scattering, if C-Pt bond length > self.CPt_bond_threshold_1
        self.CPt_threshold_2 = 1.0        # scattering, if C-Pt bond length > self.CPt_bond_threshold_2+bond_min
        
        self.O2_reach_threshold = 3.0          # O2 reaches the cluster, if Cu-O bond length < self.O2_reach_threshold
        self.O2_adsorption_threshold = 2.2     # O2 adsorption, if Cu-O bond length < self.O2_adsorption_threshold
        self.O2_reaction_TS_threshold = 1.95     # O2 reaches TS, if O-O bond length > self.O2_reaction_TS_threshold
        self.O2_reaction_end_threshold = 3.0    # O2 dissociates, if O-O bond length > self.O2_reaction_end_threshold
        self.CuO_scattering_threshold_1 = 3.00   # O2 leaves, if Cu-O > self.CuO_scattering_threshold_1
        self.CuO_scattering_threshold_2 = 0.6   # O2 leaves, if Cu-O > self.CuO_scattering_threshold_2+CuO_min

    
### Inputs ###    
    def read_trajectory(self, input_file, timestep, lref=False):
        """ Read trajectory from .XDATCAR file """
        XDATCAR = open(input_file, 'r')
        self.filename = input_file[:-8]
        
        cell = []
        atom_types = []
        atom_counts = []
        
        # Read unit cell
        XDATCAR.readline()
        lattice_cons = float(XDATCAR.readline().strip())
        for n in range(3):
            vector = XDATCAR.readline().strip().split()
            vector = [float(i)*lattice_cons for i in vector]     # Including lattice constant
            assert len(vector) == 3
            cell.append(vector)
        assert len(cell) == 3
        
        # Read atom_types and atom_counts
        atom_types = XDATCAR.readline().strip().split()
        atom_counts = XDATCAR.readline().strip().split()
        atom_counts = [int(i) for i in atom_counts]
        assert len(atom_types) == len(atom_counts)
        num_atoms = sum(atom_counts)
        
        if lref:
            self.timestep_ref = timestep
            self.atom_types_ref = atom_types.copy()
            self.atom_counts_ref = atom_counts.copy()
            self.cell_ref = cell.copy()
        else:
            self.timestep = timestep
            self.atom_types = atom_types.copy()
            self.atom_counts = atom_counts.copy()
            self.cell = cell.copy()            
        
        # Read trajectory
        tot = 0
        line = XDATCAR.readline().strip().split()
        while line:
            tot += 1
            assert line[0] == 'Direct'
            coord = []
            for atom in range(num_atoms):
                atom_coord = XDATCAR.readline().strip().split()
                atom_coord = [float(i) for i in atom_coord]
                assert len(atom_coord) == 3
                coord.append(atom_coord)
            # Change to Catesian coordinates
            coord_cart = []
            for atom in coord:
                atom_cart = sum((np.array(cell).transpose()*np.array(atom)).transpose())
                atom_cart = atom_cart.tolist()
                coord_cart.append(atom_cart)
            assert len(coord_cart) == num_atoms
            coord = coord_cart.copy()
            
            if lref:
                self.traj_ref.append(coord)
            else:
                self.traj.append(coord)
                
            line = XDATCAR.readline().strip().split()
        
        if lref:
            self.length_ref = tot
        else:
            self.length = tot
        print(str(tot)+' snapshots on this trajectory')

        XDATCAR.close()


    def read_coordinates(self, input_file, form='vasp', lenergy=False):
        """
        Read a structure from an existing file,
        return atom_types and atom_counts, coordinates, flags, velocities
        """
        cell = []
        atom_types = []
        atom_counts = []
        coordinates = []
        flags = []
        velocities = []
        energy = 0.0
        
        with open(input_file, 'r') as input_file:
            # Read atom types for 'ase' format input file
            if form == 'ase':
                atom_types = input_file.readline().strip().split()
            elif form == 'vasp':
                if lenergy:
                    energy = float(input_file.readline().strip())
                else:
                    input_file.readline()
            else:
                raise Exception('File type error')
            # Read lattice constant and lattice vectors
            lattice_cons = float(input_file.readline().strip())
            for n in range(3):
                vector = input_file.readline().strip().split()
                vector = [lattice_cons*float(i) for i in vector].copy()
                assert len(vector) == 3
                cell.append(vector)
            # Read atom types for 'vasp' format input file and the numbers of atoms
            if form == 'vasp':
                atom_types = input_file.readline().strip().split()                
            atom_counts = input_file.readline().strip().split()
            atom_counts = [int(i) for i in atom_counts].copy()
            assert len(atom_types) == len(atom_counts)
            num_atoms = sum(atom_counts)
            # Read coordinates
            if form == 'vasp':
                input_file.readline()            
            coord_type = input_file.readline().strip()
            for n in range(num_atoms):
                new_line = input_file.readline().strip().split()
                new_atom = new_line[:3].copy()
                new_atom = [float(i) for i in new_atom].copy()
                if len(new_line) == 3:
                    new_flag = '   T   T   T'
                elif len(new_line) == 6:
                    new_flag = '   '+'   '.join(new_line[3:])
                else:
                    print('Read coordinates error')
                coordinates.append(new_atom)
                flags.append(new_flag)
            # Read velocity
            input_file.readline()
            for n in range(num_atoms):
                new_atom_velocities = input_file.readline().strip().split()
                new_atom_velocities = [float(i) for i in new_atom_velocities].copy()
                assert len(new_atom_velocities) == 3
                velocities.append(new_atom_velocities)
        # Change to Cartesian coordinate if coord_type == 'Direct'
        if coord_type == 'Direct':
            coordinates_Cart = []
            for atom in coordinates:
                atom_cart = sum((np.array(cell).transpose()*np.array(atom)).transpose())
                atom_cart = atom_cart.tolist()
                coordinates_Cart.append(atom_cart)
            assert len(coordinates) == len(coordinates_Cart)
            coordinates = coordinates_Cart.copy()
        
        assert len(coordinates) == num_atoms
        assert len(flags) == num_atoms
        assert len(velocities) == num_atoms

        return atom_types, atom_counts, coordinates, flags, velocities, energy


    def read_normal_modes(self, input_file, num_modes, num_atoms):
        """
        Read normal modes
        :param input_file: str, the name of normal modes
        :param num_modes: int, the number of normal modes need to read
        :param num_atoms: int, the number of atoms
        :return:
        modes: list of list, modes, [[[dx1, dy1, dz1], [dx2, dy2, dz2], ...], ...]
        freq_list: list of frequencies in eV
        """
        freq_list = []
        modes = []
        
        with open(input_file, 'r') as data:
            new_line = data.readline()
            m = 0
            while new_line != '' and m < num_modes:
                new_line = new_line.strip()
                if 'meV' in new_line:
                    m += 1  
                    # Read frequencies
                    new_line = new_line.split()
                    freq = float(new_line[-2])/1000.0
                    freq_list.append(freq)
                    data.readline()
                    # Read normal modes
                    mode = []
                    for atom in range(num_atoms):
                        new_line = data.readline().strip().split()
                        new_line = [float(i) for i in new_line].copy()
                        mode.extend(new_line[3:6])
                    assert len(mode) == num_atoms*3
                    modes.append(mode)
                new_line = data.readline()
        
        assert len(freq_list) == num_modes
        assert len(modes) == num_modes
        
        return modes, freq_list

    
### Getters ###
    def get_timestep(self, lref=False):
        """ Return the timestep of the trajectory """
        return self.timestep_ref if lref else self.timestep
        
        
    def get_cell(self, lref=False):
        """ Return cell """
        return self.cell_ref.copy() if lref else self.cell.copy()        


    def get_atom_types(self, lref=False):
        """ Return atom types """
        return self.atom_types_ref.copy() if lref else self.atom_types.copy()
    
    
    def get_atom_counts(self, lref=False):
        """ Return atom numbers """
        return self.atom_counts_ref.copy() if lref else self.atom_counts.copy()
    
    
    def get_pos(self, n, lref=False):
        """ Return the nth coordinates, starting from 1 """
        if lref:
            if self.traj_ref:
                return self.traj_ref[n-1].copy()
            else:
                return None
        else:
            if self.traj:
                return self.traj[n-1].copy()
            else:
                return None
            
            
    def get_velocities(self, n, lref=False):
        """ Return the nth velocity """
        if lref:
            if self.velocities_ref:
                return self.velocities_ref[n-1].copy()
            else:
                return None
        else:
            if self.velocities:
                return self.velocities[n-1].copy()
            else:
                return None
    
   
    def get_length(self, lref=False):
        """ Return the length of the trajectory """
        return self.length_ref if lref else self.length


### Helpers ###     
    def com(self, compute_list, coordinates, lref=False):
        """ Compute the center of mass of the atoms from the compute_list """
        if lref:
            atom_types = self.atom_types_ref.copy()
            atom_counts = self.atom_counts_ref.copy()
        else:
            atom_types = self.atom_types.copy()
            atom_counts = self.atom_counts.copy()
        
        atom_list = []
        for c, num in enumerate(atom_counts):
            atom_list.extend([atom_types[c]]*num)
            
        tot_mass = 0.0
        COM = np.array([0.0, 0.0, 0.0])
        for atom in compute_list:
            pos = np.array(coordinates[atom-1])
            mass = self.masses[atom_list[atom-1]]
            COM += pos*mass
            tot_mass += mass
        COM = COM/tot_mass
        COM = COM.tolist()
        
        return COM
    
    
    def bond(self, atom1, atom2):
        """ Compute the bond length atom1-atom2 in angstrom """
        assert len(atom1) == 3
        assert len(atom2) == 3
        
        bondlength2 = np.sum((np.array(atom2)-np.array(atom1))**2.0)
        bondlength = np.sqrt(bondlength2)
        
        return bondlength


    def angle(self, atom1, atom2, atom3):
        """ Compute the bond angle atom1-atom2-atom3 in degree """
        assert len(atom1) == 3
        assert len(atom2) == 3
        assert len(atom3) == 3
        
        vector1 = np.array(atom1)-np.array(atom2)
        vector2 = np.array(atom3)-np.array(atom2)
        
        cos = np.dot(vector1, vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))
        degree = math.acos(cos)*180.0000/math.pi
        
        return degree
    
    
    def vibQuanNumber(self, pos, velocities, posEqu, mass, eigens, freq_list):
        """ Compute the quantum of vibrational energy in each mode 
        Input:
        pos, current coordinates
        v, current velocity
        posEqu, equilibrium coordinates
        mass, a list of masses for atoms
        eigens, normal modes
        freq_list, a list of frequencies
        Return:
        a list of quantum numbers for all modes """
        hbar = 0.06582119514      # eV*10fs
        tutoev = 1.0364314        #change tu unit to eV

        num_atoms = len(pos)
        num_modes = len(freq_list)
        
        assert len(velocities) == num_atoms
        assert len(posEqu) == num_atoms
        assert len(mass) == num_atoms
        assert len(eigens) == num_modes
        
        # Change to mass-weighted coordinates
        dis = np.array(pos)-np.array(posEqu) # displacement 
        velocities = np.array(velocities)
        dis = dis.transpose()*np.array(mass)**0.5
        velocities = velocities.transpose()*np.array(mass)**0.5
        dis = dis.transpose().flatten()
        velocities = velocities.transpose().flatten()
        
        # Normalize normal modes
        eigensNorm = []
        for mode in eigens:
            mode = np.array(mode)/np.linalg.norm(np.array(mode))
            eigensNorm.append(mode.tolist())
        eigensNorm = np.array(eigensNorm)
        
        # Generate normal coordinates, coord_norm, and normal velocity, velocities_norm
        coord_norm = np.dot(dis, eigensNorm.transpose())
        velocities_norm = np.dot(velocities, eigensNorm.transpose())*10.0
        
        # Compute vibrational energy on each mode
        freqs = np.array(freq_list)/hbar      # Unit: (10fs)**(-1)
        Ev = 0.5*(freqs**2)*(coord_norm**2)*tutoev
        Ek = 0.5*(velocities_norm**2)*tutoev
        Etot = Ev+Ek
        nuv = Ev/np.array(freq_list)
        nuk = Ek/np.array(freq_list)
        nutot = Etot/np.array(freq_list)
        return nuv, nuk, nutot
    
        
### Analysis for all ###             
    def gen_xyz(self, lref=False):
        """ Generate .xyz file and coord file """
        out_file_movie = self.filename+'.xyz'
        out_file_coord = self.filename+'.coord'
        movie = open(out_file_movie, 'w')
        coord = open(out_file_coord, 'w')
        
        if lref:
            timestep = self.timestep_ref
            cell = self.cell_ref.copy()
            atom_types = self.atom_types_ref.copy()
            atom_counts = self.atom_counts_ref.copy()
            traj = self.traj_ref.copy()
            length = self.length_ref
        else:
            timestep = self.timestep
            cell = self.cell.copy()
            atom_types = self.atom_types.copy()
            atom_counts = self.atom_counts.copy()
            traj = self.traj.copy()
            length = self.length
        
        coord_str = 'POSCAR for MD'+'\n'+'   1.0'+'\n'
        for i in range(3):
            for j in range(3):
                coord_str += '   '+"{:10.8f}".format(cell[i][j])
            coord_str += '\n'
        coord_str += '   '.join(atom_types)
        coord_str += '\n'
        for num in atom_counts:
            coord_str += '   '+str(num)
        coord_str += '\n'
        coord.write(coord_str)
        
        num_atoms = sum(atom_counts)
        atom_list = []
        for c, num in enumerate(atom_counts):
            atom_list.extend([atom_types[c]]*num)
            
        for n in range(length):
            coordinates = traj[n].copy()
            movie_str = str(num_atoms)+'\n'+'Coordinate'+str(n+1)+'\n'
            coord_str = '   Time ='+"{:10.2f}".format((n+1)*timestep)+'\n'
            for atom in range(num_atoms):
                movie_str += atom_list[atom]
                for i in range(3):
                    s = '   '+"{:10.8f}".format(coordinates[atom][i])
                    movie_str += s                    
                    coord_str += s
                movie_str += '\n'
                coord_str += '\n'
            movie.write(movie_str)
            coord.write(coord_str)
        movie.close()
        coord.close()


    def gen_poscar(self, geo_n, equ_poscar, out_file_name, lref=False):
        """ Write the output POSCAR file for the system """
        if lref:
            cell = self.cell_ref.copy()
            atom_types = self.atom_types_ref.copy()
            atom_counts = self.atom_counts_ref.copy()
        else:
            cell = self.cell.copy()
            atom_types = self.atom_types.copy()
            atom_counts = self.atom_counts.copy()

        num_atoms = sum(atom_counts)
        coordinates = self.get_pos(geo_n, lref)
        flags = ['   T   T   T']*num_atoms
        velocities = [[0.00, 0.00, 0.00]]*num_atoms
        
        atom_typesTemp, atom_countsTemp, coordinatesTemp, flags, vTemp, energy = self.read_coordinates(equ_poscar, 'vasp', lenergy=False)
        if len(flags) < num_atoms:
            flags = flags + ['   T   T   T']*(num_atoms-len(flags))
        elif len(flags) > num_atoms:
            flags = flags[:num_atoms]

        assert len(coordinates) == num_atoms
        assert len(flags) == num_atoms
        assert len(velocities) == num_atoms
        
        # Write POSCAR file
        with open(out_file_name, 'w') as output_file:
            output_file.write('POSCAR'+'\n')
            output_file.write('   1.000'+'\n')
            
            cell_str = ''
            for i in range(3):
                for j in range(3):
                    cell_str += '   '+"{:10.8f}".format(float(cell[i][j]))
                cell_str += '\n'
            output_file.write(cell_str)
            
            atoms_type_str = '   '+'   '.join(atom_types)+'\n'
            output_file.write(atoms_type_str)
            atom_counts = [str(n) for n in atom_counts]
            atoms_num_str = '   '+'   '.join(atom_counts)+'\n'
            output_file.write(atoms_num_str)
            output_file.write('Selective dynamics'+'\n'+'Cartesian'+'\n')
            
            coordinates_str = ''
            for i in range(num_atoms):
                for j in range(3):
                    coordinates_str += '   '+"{:10.8f}".format(coordinates[i][j])
                coordinates_str += flags[i]+'\n'
            output_file.write(coordinates_str)
            
            output_file.write('Cartesian'+'\n')
            velocities_str = ''
            for i in range(num_atoms):
                for j in range(3):
                    velocities_str += '   '+"{:10.8f}".format(velocities[i][j])
                velocities_str += '\n'
            output_file.write(velocities_str)
            

    def gen_rmsd(self, compute_list, lmin=False, min_name=None, lref=False):
        """ Compute root-mean-square displacement relative to the first image,
        if lmin = False; relative to the equilibrium configuration, if lmin = True,
        min_name is the input file name for equilibrium structure """
        out_file_rmsd = self.filename+'.rmsd'
        rmsd = open(out_file_rmsd, 'w')
        
        if lref:
            timestep = self.timestep_ref
            atom_types = self.atom_types_ref.copy()
            atom_counts = self.atom_counts_ref.copy()
            traj = self.traj_ref.copy()
            length = self.length_ref
        else:
            timestep = self.timestep
            atom_types = self.atom_types.copy()
            atom_counts = self.atom_counts.copy()
            traj = self.traj.copy()
            length = self.length
        
        if lmin:
            assert min_name != None
            atom_types_min, atom_counts_min, coordinates_min, flags_min, velocities_min, energy = self.read_coordinates(min_name, 'vasp')
            assert atom_types_min == atom_types
            assert atom_counts_min == atom_counts
        
        atom_list = []
        for c, num in enumerate(atom_counts):
            atom_list.extend([atom_types[c]]*num)
        rmsd_str = '   Time(fs)'
        for atom in compute_list:
            rmsd_str += '   '+atom_list[atom-1]+str(atom)
        rmsd_str += '   average'+'\n'
        rmsd.write(rmsd_str)
        
        if lmin:
            pos_ref = np.array(coordinates_min.copy())
        else:
            pos_ref = np.array(traj[0].copy())
        RMSD = []
        for n in range(length):
            diff_image = []
            ave = 0.0
            pos = np.array(traj[n].copy())
            diff = pos-pos_ref
            for atom in compute_list:
                diff_atom2 = np.sum(diff[atom-1]**2.0)
                ave += diff_atom2
                diff_atom = np.sqrt(diff_atom2)
                diff_image.append(diff_atom)
            ave = np.sqrt(ave/len(compute_list))
            diff_image.append(ave)
            RMSD.append(ave)
            
            rmsd_str = '   '+"{:10.2f}".format((n+1)*timestep)
            for num in diff_image:
                rmsd_str += '   '+"{:10.4f}".format(num)
            rmsd_str += '\n'
            
            rmsd.write(rmsd_str)
        
        rmsd.close()
        return RMSD


    def geo_difference(self, compute_list, compute_list_ref, time=200.0):
        """ Compute the geometry difference between two structures """
        if not self.traj:
            print('Trajectory is not found!')
            return None
        if not self.traj_ref:
            print('Reference trajectory is not found!')
            return None
        assert len(compute_list) == len(compute_list_ref)
        num_atoms = len(compute_list)
        
        out_file_difference = self.filename+'.difference'
        difference = open(out_file_difference, 'w')
        
        atom_types = self.atom_types.copy()
        atom_counts = self.atom_counts.copy()
        length = min(self.length, self.length_ref)

        atom_list = []
        for c, num in enumerate(atom_counts):
            atom_list.extend([atom_types[c]]*num)
        
        difference_str = '   Time(fs)'
        for atom in compute_list:
            difference_str += '   '+atom_list[atom-1]+str(atom)
        difference_str += '   rmsd'+'\n'
        difference.write(difference_str)
        
        rmsd_time = 0.0
        
        for n in range(length):
            pos = self.traj[n].copy()
            pos_compute = np.array([pos[i-1] for i in compute_list])
            pos_ref = self.traj_ref[n].copy()
            pos_compute_ref = np.array([pos_ref[i-1] for i in compute_list_ref])
            diff_time = pos_compute-pos_compute_ref
            displace_atom = sum(np.transpose(diff_time**2.0))
            rmsd = np.sqrt(sum(displace_atom)/num_atoms)
            displace_atom = np.sqrt(displace_atom)
            displace_atom = displace_atom.tolist()
            displace_atom.append(rmsd)
            
            if round(time/self.timestep) == n+1:
                rmsd_time = rmsd

            difference_str = '   '+"{:10.2f}".format((n+1)*self.timestep)
            for num in displace_atom:
                difference_str += '   '+"{:10.8f}".format(num)
            difference_str += '\n'
            
            difference.write(difference_str)                
          
        difference.close()
        return time, rmsd_time
        

    def gen_bonds(self, bond_list, lmin=False, min_name=None, lref=False):
        """ Compute bond lengths for tuples in bond_list, [(n1, n2), (n3, n4), ...]
        and the change in bond length relative to the first image, if lmin = False;
        relative to the equilibrium configuration, if lmin = True,
        min_name is the input file name for equilibrium structure """
        out_file_bonds = self.filename+'.bonds'
        out_file_bonds_chg = self.filename+'.bonds_chg'
        bonds = open(out_file_bonds, 'w')
        bonds_chg = open(out_file_bonds_chg, 'w')
        
        if lref:
            timestep = self.timestep_ref
            atom_types = self.atom_types_ref.copy()
            atom_counts = self.atom_counts_ref.copy()
            traj = self.traj_ref.copy()
            length = self.length_ref
        else:
            timestep = self.timestep
            atom_types = self.atom_types.copy()
            atom_counts = self.atom_counts.copy()
            traj = self.traj.copy()
            length = self.length
        
        if lmin:
            assert min_name != None
            atom_types_min, atom_counts_min, coordinates_min, flags_min, velocities_min, energy = self.read_coordinates(min_name, 'vasp')
            assert atom_types_min == atom_types
            assert atom_counts_min == atom_counts
        
        atom_list = []
        for c, num in enumerate(atom_counts):
            atom_list.extend([atom_types[c]]*num)
        
        bonds_str = '   Time(fs)'
        for bond in bond_list:
            bonds_str += '   '+atom_list[bond[0]-1]+str(bond[0])+'_'+atom_list[bond[1]-1]+str(bond[1])
        bonds_str += '\n'
        bonds.write(bonds_str)
        bonds_chg.write(bonds_str)
        
        bonds0 = []
        if lmin:
            pos = coordinates_min.copy()
            for bond in bond_list:
                bondlength = np.linalg.norm(np.array(pos[bond[1]-1])-np.array(pos[bond[0]-1]))
                bonds0.append(bondlength)
                    
        for n in range(length):
            pos = traj[n].copy()
            bond_image = []
            bond_chg_image = []
            for c, bond in enumerate(bond_list):
                bondlength = np.linalg.norm(np.array(pos[bond[1]-1])-np.array(pos[bond[0]-1]))
                if n == 0 and lmin == False:
                    bonds0.append(bondlength)
                bond_image.append("{:10.4f}".format(bondlength))
                bond_chg_image.append("{:10.4f}".format(bondlength-bonds0[c]))
            bonds_str = '   '+"{:10.2f}".format((n+1)*timestep)+'   '
            bonds_chg_str = '   '+"{:10.2f}".format((n+1)*timestep)+'   '
            bonds_str += '   '.join(bond_image)
            bonds_chg_str += '   '.join(bond_chg_image)
            bonds_str += '\n'
            bonds_chg_str += '\n'
            
            bonds.write(bonds_str)
            bonds_chg.write(bonds_chg_str)
                
        bonds.close()
        bonds_chg.close()
        
        
    def geo_angles(self, angle_list, lref=False):
        """ Compute the angles atom1-atom2-atom3 from the list,
        [(atom1_n, atom2_n, atom3_n)] """

        out_file_angles = self.filename+'.angles'
        angles = open(out_file_angles, 'w')
        
        if lref:
            timestep = self.timestep_ref
            atom_types = self.atom_types_ref.copy()
            atom_counts = self.atom_counts_ref.copy()
            traj = self.traj_ref.copy()
            length = self.length_ref
        else:
            timestep = self.timestep
            atom_types = self.atom_types.copy()
            atom_counts = self.atom_counts.copy()
            traj = self.traj.copy()
            length = self.length
        
        atom_list = []
        for c, num in enumerate(atom_counts):
            atom_list.extend([atom_types[c]]*num)
        
        angles_str = '   Time(fs)'
        for angle in angle_list:
            atom1 = atom_list[angle[0]-1]
            atom2 = atom_list[angle[1]-1]
            atom3 = atom_list[angle[2]-1]
            angles_str += '   '+atom1+str(angle[0])+'-'+atom2+str(angle[1])+'-'+atom3+str(angle[2])
        angles_str += '\n'
        angles.write(angles_str)
        
        for n in range(length):
            pos = traj[n].copy()
            angle_image = []
            for angle in angle_list:
                atom1 = pos[angle[0]-1]
                atom2 = pos[angle[1]-1]
                atom3 = pos[angle[2]-1]
                degree = self.angle(atom1, atom2, atom3)
                angle_image.append(degree)
                
            angles_str = '   '+"{:10.2f}".format((n+1)*timestep)+'   '
            angles_str += '   '.join("{:10.3f}".format(degree) for degree in angle_image)
            angles_str += '\n'
            
            angles.write(angles_str)
                
        angles.close()


    def gen_velocities(self, lref=False):
        """ Generate velocity using forward finite difference method """
        out_file_velocity = self.filename+'.velocity'
        velocity = open(out_file_velocity, 'w')
        
        if lref:
            timestep = self.timestep_ref
            atom_counts = self.atom_counts_ref.copy()
            traj = self.traj_ref.copy()
            length = self.length_ref
        else:
            timestep = self.timestep
            atom_counts = self.atom_counts.copy()
            traj = self.traj.copy()
            length = self.length
        
        num_atoms = sum(atom_counts)
        
        coord_current = np.array(traj[0].copy())
        for n in range(1, length):
            coord_next = np.array(traj[n].copy())
            velocities_n = (coord_next - coord_current)/timestep
            velocities_n = velocities_n.tolist()
            coord_current = coord_next.copy()
            
            if lref:
                self.velocities_ref.append(velocities_n)
            else:
                self.velocities.append(velocities_n)
            
            velocities_str = '   Time ='+"{:10.2f}".format(n*timestep)+'\n'
            for atom in range(num_atoms):
                for i in range(3):
                    velocities_str += '   '+"{:10.8f}".format(velocities_n[atom][i])
                velocities_str += '\n'
            velocity.write(velocities_str)
                
        velocity.close() 
        
    
    def gen_vaf(self, compute_list, lref=False):
        """ Compute velocity autocorrelation function """
        out_file_vaf = self.filename+'.vaf'
        vaf = open(out_file_vaf, 'w')
        
        if lref:
            timestep = self.timestep_ref
            atom_types = self.atom_types_ref.copy()
            atom_counts = self.atom_counts_ref.copy()
            velocities = self.velocities_ref.copy()
            length = self.length_ref
        else:
            timestep = self.timestep
            atom_types = self.atom_types.copy()
            atom_counts = self.atom_counts.copy()
            velocities = self.velocities.copy()
            length = self.length        
        
        atom_list = []
        for c, num in enumerate(atom_counts):
            atom_list.extend([atom_types[c]]*num) 
        
        vaf_str = '   Time(fs)   Average over'
        for atom in compute_list:
            vaf_str += ' '+atom_list[atom-1]+str(atom)
        vaf_str += '\n'
        vaf.write(vaf_str)
        
        velocities_0 = velocities[0].copy()
        for n in range(length-1):
            result = 0.0
            velocities_n = velocities[n].copy()
            for atom in compute_list:
                result += np.dot(np.array(velocities_0[atom-1]), np.array(velocities_n[atom-1]))
            result = result/len(compute_list)

            vaf_str = '   '+"{:10.2f}".format((n+1)*timestep)+'   '+"{:10.8f}".format(result)
            vaf_str += '\n'
            
            vaf.write(vaf_str)        
        
        vaf.close()
        
        
    def gen_Ek(self, compute_list, n_collide, lref=False):
        """ Compute kinetic energy """
        tutoev = 1.0364314        #change tu unit to eV
        
        out_file_kinetic = self.filename+'.kinetic'
        kinetic = open(out_file_kinetic, 'w')
        
        if lref:
            timestep = self.timestep_ref
            atom_types = self.atom_types_ref.copy()
            atom_counts = self.atom_counts_ref.copy()
            velocities = self.velocities_ref.copy()
            length = self.length_ref
        else:
            timestep = self.timestep
            atom_types = self.atom_types.copy()
            atom_counts = self.atom_counts.copy()
            velocities = self.velocities.copy()
            length = self.length        

        atom_list = []
        for c, num in enumerate(atom_counts):
            atom_list.extend([atom_types[c]]*num)
        
        mass_list = [self.masses[atom] for atom in atom_list]
#        print(mass_list)

        kinetic_str = '   Time(fs)'
        for atom in compute_list:
            kinetic_str += '   '+atom_list[atom-1]+str(atom)
        kinetic_str += '   total'+'\n'
        kinetic.write(kinetic_str)
        
        times = []
        ek_collide = []
        ek_tot_list = []
        for n in range(length-1):
            velocities_n = velocities[n].copy()
            time = (n+1)*timestep
            ek_tot = 0.0
            ek = []
            for atom in compute_list:
                mass = mass_list[atom-1]
                velocities_atom = np.array(velocities_n[atom-1])
                ek_atom = np.sum(0.5*mass*(10.0*velocities_atom)**2.0)*tutoev
                if atom == n_collide:
                    ek_collide.append(ek_atom)
                ek_tot += ek_atom
                ek.append(ek_atom)
            ek.append(ek_tot)
            
            times.append(time)
            ek_tot_list.append(ek_tot)

            kinetic_str = '   '+"{:10.2f}".format(time)
            for num in ek:
                kinetic_str += '   '+"{:10.8f}".format(num)
            kinetic_str += '\n'
            
            kinetic.write(kinetic_str)            
        
        kinetic.close()
        return times, ek_collide, ek_tot_list


    def gen_vibQuanNumber(self, in_file_posEqu, input_file_mode, num_modes, lref=False):
        """ Compute vibrational quantum number along the trajectory
        Input:
        in_file_posEqu, the file name for equilibrium structure 
        input_file_mode, the file name for frequency OUTCAR
        lref, whether or not to use reference structure """
        
        out_file_vibQuanNumber = self.filename+'.quantum'
        quantum = open(out_file_vibQuanNumber, 'w')
        quantum.write('   Time(fs)   nu_V (mode1, mode2, ...)   nu_Ek (mode1, mode2, ...)   nu_Etot (mode1, mode2, ...)'+'\n')
        
        if lref:
            timestep = self.timestep_ref
            atom_types = self.atom_types_ref.copy()
            atom_counts = self.atom_counts_ref.copy()
            traj = self.traj_ref.copy()
            velocities = self.velocities_ref.copy()
            length = self.length_ref
        else:
            timestep = self.timestep
            atom_types = self.atom_types.copy()
            atom_counts = self.atom_counts.copy()
            traj = self.traj.copy()
            velocities = self.velocities.copy()
            length = self.length
        
        # Get equilibrium coordinates
        atom_typesEqu, atom_countsEqu, posEqu, f, vEqu, e = self.read_coordinates(in_file_posEqu, form='vasp', lenergy=False)
        assert atom_typesEqu == atom_types
        assert atom_countsEqu == atom_counts
        num_atoms = sum(atom_counts)
        
        # Get normal modes
        modes, freq_list = self.read_normal_modes(input_file_mode, num_modes, num_atoms)
        
        # Generate mass list
        mass = []
        for c, atom in enumerate(atom_types):
            mass = mass + [self.masses[atom]]*atom_counts[c]
        
        # Iterate over the trajectory
        for n in range(length-1):
            posImg = traj[n]
            vImg = velocities[n]
            time = timestep*(n+1)                    
            nuv, nuk, nutot = self.vibQuanNumber(posImg, vImg, posEqu, mass, modes, freq_list)
            quan_str = '   '+str(time)+'   '
            nuv_str = ' '.join(['{:2.4f}'.format(i) for i in nuv])
            nuk_str = ' '.join(['{:2.4f}'.format(i) for i in nuk])
            nutot_str = ' '.join(['{:2.4f}'.format(i) for i in nutot])
            quan_str += nuv_str+'   '+nuk_str+'   '+nutot_str+'\n'
            quantum.write(quan_str)
        
        quantum.close()
        

### Analysis for CH4@Pt13 ###                     
    def id_traj_type_Pt(self, n_PtCH):
        """ Input the atom numbers for C, H1, H2, H3, H4 and Pt as n_PtCH = [nC, nH1, nH2, nH3, nH4, nPt]
        Identify the type of the trajectory: reaction, trapping or scattering """
        [C_n, H1_n, H2_n, H3_n, H4_n, Pt_n] = n_PtCH
        bond_list = [(C_n, H1_n), (C_n, H2_n), (C_n, H3_n), (C_n, H4_n), (C_n, Pt_n)]
        self.gen_bonds(bond_list)
        
        in_file_bonds = self.filename+'.bonds'
        bonds = open(in_file_bonds, 'r')
        
        CH_reach_threshold = self.CH_reach_threshold     # reach the cluster, if C-Pt bond length < self.CH_reach_threshold, default = 3.0
        CH_threshold_1 = self.CH_threshold_1     # reach the TS, if C-H bond length > self.CH_bond_threshold, default = 1.6
        CH_threshold_2 = self.CH_threshold_2     # dissociation ends, if C-H bond length > self.CH_bond_threshold, default = 3.0
        CPt_threshold_1 = self.CPt_threshold_1   # scattering, if C-Pt bond length > self.CPt_bond_threshold_1, default = 4.0
        CPt_threshold_2 = self.CPt_threshold_2   # scattering, if C-Pt bond length > self.CPt_bond_threshold_2+bond_min, default = 1.0   
        
        CPt_reach = ['NA', (n_PtCH[0], n_PtCH[5]), 0.0]
        CH_reaction_TS = ['NA', 'NA', 0.0]
        CH_reaction_end = ['NA', 'NA', 0.0]
        CPt_min = ['NA', (n_PtCH[0], n_PtCH[5]), 5.0]
        CPt_scattering = None
        
        bonds.readline()
        length = min(self.length, 2000)
        for n in range(length):
            line = bonds.readline().strip().split()
            line = [float(i) for i in line]
            time = line[0]
            
            # Reach the cluster
            CPt_bond = line[5]     # Pt-C bond length
            if CPt_reach[0] == 'NA' and CPt_bond <= CH_reach_threshold:   # C-Pt first reaches CH_reach_threshold
                CPt_reach[0] = time
                CPt_reach[2] = CPt_bond
                
            # Update minimum Pt-C bond length
            if CPt_bond < CPt_min[2]:
                CPt_min[2] = CPt_bond
                CPt_min[0] = time
            
            # Identify reaction trajectory
            CH_bonds = line[1:5].copy()    # C-H bond lengths
            for i in range(4):
                CH_bond = CH_bonds[i]
                if CH_bond >= CH_threshold_1 and CH_bond < 5.0:
                    if CH_reaction_TS[0] == 'NA':      # C-H first reaches CH_threshold_1
                        CH_reaction_TS[0] = time
                        CH_reaction_TS[1] = (n_PtCH[0], n_PtCH[i+1])
                        CH_reaction_TS[2] = CH_bond
                if CH_bond >= CH_threshold_2 and CH_bond < 5.0:
                    if CH_reaction_end[0] == 'NA':
                        CH_reaction_end[0] = time
                        CH_reaction_end[1] = (n_PtCH[0], n_PtCH[i+1])
                        CH_reaction_end[2] = CH_bond
                    
                    CH_bond_name = 'C'+str(n_PtCH[0])+'-H'+str(n_PtCH[i+1])
                    time_str = "{:5.2f}".format(time)+' fs'
                    CH_bond_str = "{:5.2f}".format(CH_bond)+' angstrom'
                    print('CH4 is dissociated at '+time_str+', the '+CH_bond_name+' is '+CH_bond_str)
                    
                    bonds.close()                                       
                    return 'reaction', CPt_reach, CH_reaction_TS, CH_reaction_end
            
            # Identify scattering trajectory
            if CPt_bond > max((CPt_min[2]+CPt_threshold_2), CPt_threshold_1) :
                CPt_scattering = [time]
                CPt_scattering.append((n_PtCH[0], n_PtCH[5]))
                CPt_scattering.append(CPt_bond)
                
                time_str = "{:5.2f}".format(CPt_min[0])+' fs'
                CPt_bond_str = "{:5.2f}".format(CPt_min[2])+' angstrom'
                print('CH4 is scattered, the minimum C-Pt bond length is '+CPt_bond_str+' at '+time_str)
                
                bonds.close()
                return 'scattering', CPt_reach, CPt_min, CPt_scattering
        
        # Identify trapping trajectory
        else:
            if CH_reaction_TS[0] != 'NA':
                print('It is dissociating!')                
            time_str = "{:5.2f}".format(CPt_min[0])+' fs'
            CPt_bond_str = "{:5.2f}".format(CPt_min[2])+' angstrom'
            print('CH4 is trapped, the minimum C-Pt bond length is '+CPt_bond_str+' at '+time_str)
            
            bonds.close()
            return 'trapping', CPt_reach, CPt_min, [time, (n_PtCH[0], n_PtCH[5]), CPt_bond]


    def find_reaction_start_Pt(self, time_TS, duration, C_n, H_n, CH_threshold=1.3):
        """ Find the starting point of the reaction,
        backtrack from the TS in the during time,
        when C-H bond length starts to be less than CH_threshold
        it is considered as starting point """
        timestep = self.timestep
        
        time_threshold = time_TS-duration
        step_threshold = int(time_threshold/timestep)
        step_TS = int(time_TS/timestep)
    
        for n in range(step_TS, step_threshold, -1):
            coordinates = self.get_pos(n)
            H = coordinates[H_n-1].copy()
            C = coordinates[C_n-1].copy()
            CH = self.bond(C, H)
            if CH < CH_threshold:
                time = n*timestep
                bond_str = str(C_n)+'-'+str(H_n)
                return time, bond_str, CH
        else:
            print('Please increase during time')
            bond_str = str(C_n)+'-'+str(H_n)
            return time_threshold, bond_str, CH


    def find_interaction_region_Pt(self, time1, time2, C_n, Pt_n, CPt_threshold1, CPt_threshold2, CPt_min):
        """ Find the time, time_enter, that CH4 enters the interaction region,
        and find the time, time_leave, that CH4 leaves the interaction region.
        Input: 
        time1 and time2: float, indicating the region considered
        C_n, Pt_n: int, the number of C and Pt atoms
        CPt_threshold1 and CPt_threshold2: float, the region in terms of C-Pt bond length,
        if CPt_min = None, directly use CPt_threshold1 and CPt_threshold2,
        otherwise use CPt_threshold1+CPt_min and CPt_threshold2+CPt_min
        Return: 
        time_enter, time_leave, time_diff: float, time in the region, if CH4 does not leave return time2
        bond_str: str, the bond name
        enter, leave: boolean, whether CH4 leaves the region """
        timestep = self.timestep
        
        step1 = int(time1/timestep)
        step2 = int(time2/timestep)
        time_enter = time2
        time_leave = time2
        time_diff = 0.0
        bond_str = str(C_n)+'-'+str(Pt_n)
        enter = True
        leave = True
        
        if CPt_min:
            CPt_threshold1 = CPt_min+CPt_threshold1
            CPt_threshold2 = CPt_min+CPt_threshold2                
        
        # Compute time for entering the region
        for n in range(step1, step2+1):
            coordinates = self.get_pos(n)
            C = coordinates[C_n-1].copy()
            Pt = coordinates[Pt_n-1].copy()
            CPt = self.bond(C, Pt)
            if CPt < CPt_threshold1:
                time_enter = n*timestep
                break
        else:
            enter = False
            leave = False
            return enter, leave, bond_str, time_enter, time_leave, time_diff
        
        # Compute time for leaving the region
        coordinates = self.get_pos(step2)
        C = coordinates[C_n-1].copy()
        Pt = coordinates[Pt_n-1].copy()
        CPt = self.bond(C, Pt)
        if CPt < CPt_threshold2:
            leave = False
            time_diff = time_leave-time_enter
            return enter, leave, bond_str, time_enter, time_leave, time_diff
            
        for n in range(step2+1, step1, -1):
            coordinates = self.get_pos(n)
            C = coordinates[C_n-1].copy()
            Pt = coordinates[Pt_n-1].copy()
            CPt = self.bond(C, Pt)
            if CPt < CPt_threshold2:
                time_leave = n*timestep
                time_diff = time_leave-time_enter
                return enter, leave, bond_str, time_enter, time_leave, time_diff


    def geo_projection(self, n_collide, n_Pt, n_CH, time=None):
        """ Compute the projection of coordinates and velocity of the metal atom
        over which the molecule dissociates onto the direction along which the
        molecule comes in 
        n_Pt: a list of Pt atoms 
        n_CH: a list of CH4 atoms """

        out_file_projection = self.filename+'.projection'
        projection = open(out_file_projection, 'w')
        
        timestep = self.timestep
        traj = self.traj.copy()
        velocities = self.velocities.copy()
        length = self.length
        pos_proj_z_time = 0.0
        pos_proj_xy_time = 0.0
        velocities_proj_z_time = 0.0
        velocities_proj_xy_time = 0.0
        
        coordinates = traj[0].copy()
        Pt = coordinates[n_collide-1]
        com_Pt = self.com(n_Pt, coordinates)
        com_CH = self.com(n_CH, coordinates)
        ref_direct = np.array(com_CH)-np.array(com_Pt)    # initial direction of CH4
        ref_direct = ref_direct/np.linalg.norm(ref_direct)
        
        projection_str = '   Time(fs)   Pt dis_projection (z, xy)   Pt_velocities_projection (z, xy)'+'\n'
        projection.write(projection_str)

        for n in range(length-1):
#            Pt_dis = np.array(traj[n][n_collide-1])-np.array(com_Pt)
            Pt_dis = np.array(traj[n][n_collide-1])-np.array(Pt)
            pos_proj_z = np.dot(Pt_dis, ref_direct)
            pos_proj_xy = np.linalg.norm(Pt_dis-pos_proj_z*ref_direct)
            
            Pt_velocities = np.array(velocities[n][n_collide-1])
            velocities_proj_z = np.dot(Pt_velocities, ref_direct)
            velocities_proj_xy = np.linalg.norm(Pt_velocities-velocities_proj_z*ref_direct)
            
            if time and time == (n+1)*timestep:
                pos_proj_z_time = pos_proj_z
                pos_proj_xy_time = pos_proj_xy
                velocities_proj_z_time = velocities_proj_z
                velocities_proj_xy_time = velocities_proj_xy
            
            projection_str = '   '+"{:10.2f}".format((n+1)*timestep)
            projection_str += '   '+"{:10.4f}".format(pos_proj_z)+'   '+"{:10.8f}".format(pos_proj_xy)
            projection_str += '   '+"{:10.4f}".format(velocities_proj_z)+'   '+"{:10.8f}".format(velocities_proj_xy)
            projection_str += '\n'
            
            projection.write(projection_str)        
        
        projection.close()
        return time, pos_proj_z_time, pos_proj_xy_time, velocities_proj_z_time, velocities_proj_xy_time
        

    def diff_projection(self, n_collide, n_Pt, n_CH):
        """ Compute the projection of coordinates difference and velocity difference
        of the metal atom over which the molecule dissociates onto the direction
        along which the molecule comes in 
        n_Pt: a list of Pt atoms 
        n_CH: a list of CH4 atoms """
        if not self.traj:
            print('Trajectory is not found!')
            return None
        if not self.traj_ref:
            print('Reference trajectory is not found!')
            return None
        
        if not self.velocities:
            print('Velocity is not found!')
            return None
        if not self.velocities_ref:
            print('Reference velocity is not found!')
            return None
        
        out_file_projection = self.filename+'.Diffprojection'
        projection = open(out_file_projection, 'w')        
        
        # Compute referece direction which is the initial direction CH4 comes
        coordinates = self.traj[0].copy()
        com_Pt = self.com(n_Pt, coordinates)
        com_CH = self.com(n_CH, coordinates)
        timestep = self.timestep
        ref_direct = np.array(com_CH)-np.array(com_Pt)    # initial direction of CH4
        ref_direct = ref_direct/np.linalg.norm(ref_direct)
        
        length = min(self.length, self.length_ref)-1
        
        pos_proj_z_tot = []
        pos_proj_xy_tot = []
        velocities_proj_z_tot = []
        velocities_proj_xy_tot = []
        time = []
        
        projection_str = '   Time(fs)   Pt dis_projection (z, xy)   Pt_velocities_projection (z, xy)'+'\n'
        projection.write(projection_str)        
        
        for n in range(length):
            time.append((n+1)*timestep)
            posPt = np.array(self.traj[n][n_collide-1].copy())
            posPt_ref = np.array(self.traj_ref[n][n_collide-1].copy())
            vPt = np.array(self.velocities[n][n_collide-1].copy())
            vPt_ref = np.array(self.velocities_ref[n][n_collide-1].copy())
            
            posDiff = posPt-posPt_ref
            vDiff = vPt-vPt_ref
            
            pos_proj_z = np.dot(posDiff, ref_direct)
            pos_proj_xy = np.linalg.norm(posDiff-pos_proj_z*ref_direct)
            pos_proj_z_tot.append(pos_proj_z)
            pos_proj_xy_tot.append(pos_proj_xy)
            
            velocities_proj_z = np.dot(vDiff, ref_direct)
            velocities_proj_xy = np.linalg.norm(vDiff-velocities_proj_z*ref_direct)
            velocities_proj_z_tot.append(velocities_proj_z)
            velocities_proj_xy_tot.append(velocities_proj_xy)

            projection_str = '   '+"{:10.2f}".format((n+1)*timestep)
            projection_str += '   '+"{:10.8f}".format(pos_proj_z)+'   '+"{:10.8f}".format(pos_proj_xy)
            projection_str += '   '+"{:10.8f}".format(velocities_proj_z)+'   '+"{:10.8f}".format(velocities_proj_xy)
            projection_str += '\n'
            
            projection.write(projection_str)        
        
        projection.close()
        return time, pos_proj_z_tot, pos_proj_xy_tot, velocities_proj_z_tot, velocities_proj_xy_tot


    def remove_CH4(self, geo_n, out_file_name):
        """ 
        Take one image on the trajectory, remove CH4, generate a POSCAR file
        Input:
            geo_n, int, the number of geometry in one trajectory
            out_file_name, string, the name of the output POSCAR name
        """
        cell = self.cell.copy()
        atom_types = self.atom_types[:1]
        atom_counts = self.atom_counts[:1]

        num_atoms = sum(atom_counts)
        coordinates = self.get_pos(geo_n, lref=False)
        coordinates = coordinates[:num_atoms]
        flags = ['   T   T   T']*num_atoms
        velocities = [[0.00, 0.00, 0.00]]*num_atoms

        assert len(coordinates) == num_atoms
        assert len(flags) == num_atoms
        assert len(velocities) == num_atoms
        
        # Write POSCAR file
        with open(out_file_name, 'w') as output_file:
            output_file.write('POSCAR'+'\n')
            output_file.write('   1.000'+'\n')
            
            cell_str = ''
            for i in range(3):
                for j in range(3):
                    cell_str += '   '+"{:10.8f}".format(float(cell[i][j]))
                cell_str += '\n'
            output_file.write(cell_str)
            
            atoms_type_str = '   '+'   '.join(atom_types)+'\n'
            output_file.write(atoms_type_str)
            atom_counts = [str(n) for n in atom_counts]
            atoms_num_str = '   '+'   '.join(atom_counts)+'\n'
            output_file.write(atoms_num_str)
            output_file.write('Selective dynamics'+'\n'+'Cartesian'+'\n')
            
            coordinates_str = ''
            for i in range(num_atoms):
                for j in range(3):
                    coordinates_str += '   '+"{:10.8f}".format(coordinates[i][j])
                coordinates_str += flags[i]+'\n'
            output_file.write(coordinates_str)
            
            output_file.write('Cartesian'+'\n')
            velocities_str = ''
            for i in range(num_atoms):
                for j in range(3):
                    velocities_str += '   '+"{:10.8f}".format(velocities[i][j])
                velocities_str += '\n'
            output_file.write(velocities_str)


### Analysis for O2@Cu4O2 ### 
    def id_traj_type_cu(self, n_CuO):
        """ Input the atom numbers for O1, O2, Cu1, Cu2, Cu3 and Cu4 as 
        n_PtCH = [nO1, nO2, nCu1, nCu2, nCu3, nCu4]
        Identify the type of the trajectory: reaction, adsorption or scattering """        
        [nO1, nO2, nCu1, nCu2, nCu3, nCu4] = n_CuO
        bond_list = [(nO1, nO2), (nO1, nCu1), (nO1, nCu2), (nO1, nCu3), (nO1, nCu4),
                     (nO2, nCu1), (nO2, nCu2), (nO2, nCu3), (nO2, nCu4)]
        self.gen_bonds(bond_list)
        
        in_file_bonds = self.filename+'.bonds'
        bonds = open(in_file_bonds, 'r')
        name = bonds.readline().strip().split()
        CuO_name = name[2:]
        O2_name = name[1]

        O2_reach_threshold = self.O2_reach_threshold         # O2 reaches the cluster, if Cu-O bond length < self.O2_reach_threshold
        O2_adsorption_threshold = self.O2_adsorption_threshold     # O2 adsorption, if Cu-O bond length < self.O2_adsorption_threshold
        O2_reaction_TS_threshold = self.O2_reaction_TS_threshold     # O2 reaches TS, if O-O bond length > self.O2_reaction_TS_threshold
        O2_reaction_end_threshold = self.O2_reaction_end_threshold   # O2 dissociates, if O-O bond length > self.O2_reaction_end_threshold
        CuO_scattering_threshold_1 = self.CuO_scattering_threshold_1      # O2 leaves, if Cu-O > self.CuO_scattering_threshold_1
        CuO_scattering_threshold_2 = self.CuO_scattering_threshold_2      # O2 leaves, if Cu-O > self.CuO_scattering_threshold_2+CuO_min
        
        O2_reach = ['NA', 'NA', 0.0]            # O2 reach one Cu, [time, bond_name, bondlength]
        O2_adsorption = ['NA', 'NA', 0.0]       # O2 adsorbs to one Cu [time, bond_name, bondlength]
        O2_reaction_TS = ['NA', O2_name, 0.0]      # O2 dissociate, reach TS, [time, bond_name, bondlength]
        O2_reaction_end = ['NA', O2_name, 0.0]     # O2 dissociate, reaction end, [time, bond_name, bondlength]
        O2_scattering = ['NA', 'NA', 0.0]            # O2 leave, [time, bond_name, bondlength]
        CuO_min = ['NA', 'NA', 10.0]            # Minimum distance between Cu and O
        O2_max = ['NA', O2_name, 0.0]             # Maximum distance between O and O
        
        length = min(self.length, 2000)
        for n in range(length):
            line = bonds.readline().strip().split()
            line = [float(i) for i in line]
            time = line[0]
            O2 = line[1]
            CuO = line[2:]   # all Cu-O bond lengths
            CuO_tmp_min = np.min(np.array(CuO))     # minimum Cu-O bond length at time t
            CuO_tmp_min_index = np.argmin(np.array(CuO))   # the index of the minimum Cu-O bond length at time t
            
            # Reach the cluster
            if O2_reach[0] == 'NA' and CuO_tmp_min <= O2_reach_threshold:    # Cu-O first reaches CuO_adsorption_threshold
                O2_reach[0] = time
                O2_reach[1] = CuO_name[CuO_tmp_min_index]
                O2_reach[2] = CuO_tmp_min
            
            # O2 adsorption (first adsorption)
            if O2_adsorption[0] == 'NA' and CuO_tmp_min <= O2_adsorption_threshold:    # Cu-O first reaches CuO_adsorption_threshold
                O2_adsorption[0] = time
                O2_adsorption[1] = CuO_name[CuO_tmp_min_index]
                O2_adsorption[2] = CuO_tmp_min
            
            # Update minimum O-Cu bond length
            if CuO_tmp_min < CuO_min[2]:
                CuO_min[0] = time
                CuO_min[1] = CuO_name[CuO_tmp_min_index]
                CuO_min[2] = CuO_tmp_min
            
            # Update maximum O-O bond length
            if O2 > O2_max[2]:
                O2_max[0] = time
                O2_max[2] = O2
                
            # Update scattering time: the time when Cu-O < CuO_scattering_threshold_1 for the last time
            if CuO_tmp_min < CuO_scattering_threshold_1:
                O2_scattering[0] = time
                O2_scattering[1] = CuO_name[CuO_tmp_min_index]
                O2_scattering[2] = CuO_tmp_min
            
            # Identify reaction trajectory
            if O2_reaction_TS[0] == 'NA' and (O2 >= O2_reaction_TS_threshold and O2 < 5.0):
                O2_reaction_TS[0] = time
                O2_reaction_TS[2] = O2
            if O2_reaction_end[0] == 'NA' and (O2 >= O2_reaction_end_threshold and O2 < 5.0):
                O2_reaction_end[0] = time
                O2_reaction_end[2] = O2
                time_str = "{:5.2f}".format(time)+' fs'
                O2_str = "{:5.2f}".format(O2)+' angstrom'
                print('O2 is dissociated at '+time_str+', the O-O is '+O2_str)
                bonds.close()                                       
                return 'reaction', O2_reach, O2_adsorption, O2_reaction_TS, O2_max, CuO_min
            
        # Identify scattering trajectory (Cu-O reaches 4 angstrom)
        if CuO_tmp_min > max(CuO_scattering_threshold_1, CuO_scattering_threshold_2+CuO_min[2]):
            CuO_str = "{:5.2f}".format(CuO_min[2])+' angstrom'
            # if O2_reach[0] == 'NA':
            #     print('O2 is not arrived, the minimum Cu-O '+CuO_min[1]+' is '+CuO_str)
            #     bonds.close()
            #     return 'notArrived', O2_reach, O2_adsorption, O2_scattering, O2_max, CuO_min
            print('O2 is scattered, the minimum Cu-O '+CuO_min[1]+' is '+CuO_str)
            bonds.close()
            return 'scattering', O2_reach, O2_adsorption, O2_scattering, O2_max, CuO_min
        
        # Identify adsorption trajectory
        else:
            if O2_reaction_TS[0] != 'NA':
                print('It is dissociating!')
            CuO_str = "{:5.2f}".format(CuO_min[2])+' angstrom'
            # if O2_adsorption[0] == 'NA':
            #     print('O2 is not arrived and not leave, the minimum Cu-O '+CuO_min[1]+' is '+CuO_str)
            #     bonds.close()
            #     return 'notArrived', O2_reach, O2_adsorption, O2_scattering, O2_max, CuO_min
            print('O2 is adsorbed, the minimum Cu-O '+CuO_min[1]+' is '+CuO_str)
            bonds.close()
            return 'adsorption', O2_reach, O2_adsorption, O2_scattering, O2_max, CuO_min

            
    def analysis_cu_bonds(self, n_CuO):
        """ Input the atom numbers for O1, O2, Cu1, Cu2, Cu3 and Cu4 as n_CuO = [nO1, nO2, nCu1, nCu2, nCu3, nCu4]
        return the largest O2 bond length and shortest Cu-O bond length """
        [nO1, nO2, nCu1, nCu2, nCu3, nCu4] = n_CuO
        bond_list = [(nO1, nO2), (nO1, nCu1), (nO1, nCu2), (nO1, nCu3), (nO1, nCu4),
                     (nO2, nCu1), (nO2, nCu2), (nO2, nCu3), (nO2, nCu4)]
        self.gen_bonds(bond_list)
        
        in_file_bonds = self.filename+'.bonds'
        bonds = open(in_file_bonds, 'r')
        
        O2_max = 0.0
        O2_max_list = [0.0, '', 0.0]
        CuO_min = 100.0
        CuO_min_list = [0.0, '', 0.0]
        
        title = bonds.readline().strip().split()
        length = min(self.length, 2000)
        for n in range(length):
            line = bonds.readline().strip().split()
            line = [float(i) for i in line]
            time = line[0]
            O2 = line[1]
            CuO = line[2:]
            if O2 > O2_max:
                O2_max = O2
                O2_max_list[0] = time
                O2_max_list[1] = title[1]
                O2_max_list[2] = O2
            for c, j in enumerate(CuO):
                if j < CuO_min:
                    CuO_min = j
                    CuO_min_list[0] = time
                    CuO_min_list[1] = title[c+2]
                    CuO_min_list[2] = j
                    
        bonds.close()
        return O2_max_list, CuO_min_list
    
    
    def analyzeCuOBond(self, inputContcar, n_Cu4, n_O2):
        """ Analyze the Cu-O bond, return energy, Cu-O name and bond length for one structure
        Input:
            inputContcar: the name of the input CONTCAR file
            n_Cu4: list, the atom numbers for Cu4
            n_O2: list, the atom numbers for O2
        Return:
            energy
            CuO1Name, CuO1Bonding, CuO1Min: the name, whether it is considered as bonding, the lowest distance between O1 and Cu
            CuO2Name, CuO2Bonding, CuO2Min: the name, whether it is considered as bonding, the lowest distance between O2 and Cu
        """
        CuOThreshold1 = 2.2
#        CuOThreshold2 = 2.6
        
        atom_types, atom_counts, coordinates, flags, v, energy = self.read_coordinates(inputContcar, form='vasp', lenergy=True)
        
        CuO1Min = 20.0
        CuO1Name = ''
        CuO2Min = 20.0
        CuO2Name = ''
        O1 = np.array(coordinates[n_O2[0]-1])
        O2 = np.array(coordinates[n_O2[1]-1])
        for nCu in n_Cu4:
            Cu = np.array(coordinates[nCu-1])
            CuO1 = np.linalg.norm(Cu-O1)
            CuO2 = np.linalg.norm(Cu-O2)
            if CuO1 < CuO1Min:
                CuO1Min = CuO1
                CuO1Name = 'Cu'+str(nCu)+'_'+'O'+str(n_O2[0])
            if CuO2 < CuO2Min:
                CuO2Min = CuO2
                CuO2Name = 'Cu'+str(nCu)+'_'+'O'+str(n_O2[1])
        isCuO1Bonding = True if CuO1Min <= CuOThreshold1 else False
        isCuO2Bonding = True if CuO2Min <= CuOThreshold1 else False    
        
        return energy, CuO1Name, isCuO1Bonding, CuO1Min, CuO2Name, isCuO2Bonding, CuO2Min
    
    
    def CuOWhenlargeDifference(self, input_file, input_file_ref, timestep, diff_threshold, compute_list, compute_list_ref, timeRange = 1000):
        """
        Return the time and the shortest Cu-O distance 
        when the difference between production and reference trajectores are
        larger than diff_threshold
        Input:
            input_file: XDATCAR file for production trajectory
            input_file_ref: XDATCAR file for reference trajectory
            diff_threshold: geometry difference threshold
            compute list: atoms to be calculate geometry difference for production trajectory
            compute_list_ref: atoms to be calculate geometry difference for reference trajectory
            timeRange: the time range to be considered
        """
        if not compute_list:
            compute_list = [28, 29, 30, 31, 32, 33]
        if not compute_list_ref:
            compute_list_ref = [28, 29, 30, 31, 32, 33]
        
        # read production and reference trajectories
        self.read_trajectory(input_file, timestep, lref=False)
        self.read_trajectory(input_file_ref, timestep, lref=True)
        
        # Find the point that the geometry difference is larger than threshold
        length = min(self.length, self.length_ref, int(timeRange/timestep))
        num_atoms = len(compute_list)
        
        for n in range(length):            
            posConsidered = np.array([self.traj[n][i-1] for i in compute_list])
            posConsidered_ref = np.array([self.traj_ref[n][i-1] for i in compute_list_ref])
            diff_time = posConsidered-posConsidered_ref
            displace_atom = np.sum((diff_time**2.0), axis = 1)
            rmsd = np.sqrt(sum(displace_atom)/num_atoms)
            if rmsd >= diff_threshold:
                break
        
        # Compute Cu-O distance
        n_CuO = [59, 60, 28, 29, 30, 31]
        [nO1, nO2, nCu1, nCu2, nCu3, nCu4] = n_CuO
        bond_list = [(nO1, nCu1), (nO1, nCu2), (nO1, nCu3), (nO1, nCu4),
                     (nO2, nCu1), (nO2, nCu2), (nO2, nCu3), (nO2, nCu4)]
        
        pos = self.traj[n]
        minBond = None
        minBondLength = float('inf')
        for bond in bond_list:
            bondlength = np.linalg.norm(np.array(pos[bond[1]-1])-np.array(pos[bond[0]-1]))
            if bondlength < minBondLength:
                minBondLength = bondlength
                minBond = bond
        
        minIndexCu = minBond[1]
        O1Cu = np.linalg.norm(np.array(pos[nO1-1])-np.array(pos[minIndexCu-1]))
        O2Cu = np.linalg.norm(np.array(pos[nO2-1])-np.array(pos[minIndexCu-1]))
        
        return (n+1)*timestep, rmsd, 'Cu'+str(minIndexCu), O1Cu, O2Cu
    
    
    def geo_difference_vs_CuO(self, compute_list, compute_list_ref, Cu_n, O_n, timeRange=500.0):
        """ 
        Compute the geometry difference between two structures as a function of Cu-O difference
        Input:
            Cu_n: the Cu atom number to compute Cu-O distance
            O_n: the O atom number to compute Cu-O distance
        """
        if not self.traj:
            print('Trajectory is not found!')
            return None
        if not self.traj_ref:
            print('Reference trajectory is not found!')
            return None
        assert len(compute_list) == len(compute_list_ref)
        num_atoms = len(compute_list)
        
        out_file_difference = self.filename+'.difference'
        difference = open(out_file_difference, 'w')
        
        # build title for difference file 
        atom_types = self.atom_types.copy()
        atom_counts = self.atom_counts.copy()

        atom_list = []
        for c, num in enumerate(atom_counts):
            atom_list.extend([atom_types[c]]*num)
        
        difference_str = '   Time(fs)   Cu'+str(Cu_n)+'-O'+str(O_n)
        for atom in compute_list:
            difference_str += '   '+atom_list[atom-1]+str(atom)
        difference_str += '   rmsd'+'\n'
        difference.write(difference_str)
        
        # compute structural difference and Cu-O bond length
        length = min(self.length, self.length_ref, int(timeRange/self.timestep))
        
        for n in range(length):
            
            # compute structural divergence for each atom and the average divergence
            pos = self.traj[n].copy()
            pos_compute = np.array([pos[i-1] for i in compute_list])
            pos_ref = self.traj_ref[n].copy()
            pos_compute_ref = np.array([pos_ref[i-1] for i in compute_list_ref])
            diff_time = pos_compute-pos_compute_ref
            
            displace_atom = sum(np.transpose(diff_time**2.0))
            rmsd = np.sqrt(sum(displace_atom)/num_atoms)
            displace_atom = np.sqrt(displace_atom)
            displace_atom = displace_atom.tolist()
            displace_atom.append(rmsd)
            
            # compute Cu-O bond length from production trajectory
            CuObond = np.linalg.norm(np.array(pos[O_n - 1]) - np.array(pos[Cu_n - 1]))

            difference_str = '   '+"{:10.2f}".format((n+1)*self.timestep)
            difference_str += '   {:10.4f}'.format(CuObond)
            for num in displace_atom:
                difference_str += '   '+"{:10.6f}".format(num)
            difference_str += '\n'
            
            difference.write(difference_str)                
          
        difference.close()
        return 
    
    
    def atomDisplacement(self, atomList, Cu_n, O_n, CuOList, lEqu, posEqu):
        """
        Compute the displacement of given atom when Cu-O reaches certain distances
        Input:
            atomList: the atoms to consider structural change
            Cu_n: the Cu atom number to compute Cu-O distance
            O_n: the O atom number to compute Cu-O distance
            CuOList: a list, return the structural divergence vector when Cu-O distance is in CuO_list
            lEqu: if True, structural change is relative to equilibrium structure, otherwise relative to reference structure
            posEqu: the name of equilibrium POSCAR
        """
        if not self.traj:
            print('Trajectory is not found!')
            return None
        if lEqu:
            if posEqu is None:
                print('Equilibrium structure is not found!')
                return
        else:
            if not self.traj_ref:
                print('Reference trajectory is not found!')
                return None
        
        displacementOut = self.filename+'.displacement'
        displacement = open(displacementOut, 'w')
        
        # build title for displacement file 
        CuOName = 'Cu'+str(Cu_n)+'-O'+str(O_n)
        displacementStr = '   Time(fs)   expect ' + CuOName + ' real ' + CuOName + ' (x, y, z)\n'
        displacement.write(displacementStr)
        
        # compute displacement and bond length at given Cu-O
        length = min(self.length, 2000)
        if not lEqu:
            length = min(length, self.length_ref)
        numCuO = len(CuOList)
        CuO_n = 0
        
        displacementVector = None
        
        for n in range(length):
            pos = self.traj[n]
            if lEqu:
                _, _, posRef, _, _, _ = self.read_coordinates(posEqu)
            else:
                posRef = self.traj_ref[n]
            
            # compute Cu-O bond length from production trajectory
            CuObond = np.linalg.norm(np.array(pos[O_n - 1]) - np.array(pos[Cu_n - 1]))
            
            # compute atom displacement at given Cu-O
            if CuObond < CuOList[CuO_n]:   
                
                posVector = []
                posRefVector = []
                for atom_n in atomList:
                    posVector.extend(pos[atom_n - 1])
                    posRefVector.extend(posRef[atom_n - 1])
                degree = len(posVector)
                displacementVector = np.array(posVector) - np.array(posRefVector)
                
                displacementStr = '   {:10.2f}'.format((n + 1) * self.timestep)
                displacementStr += '   {:10.4f}'.format(CuOList[CuO_n])
                displacementStr += '   {:10.4f}'.format(CuObond)
                for num in displacementVector:
                    displacementStr += '   {:10.6f}'.format(num)
                displacementStr += '\n'
                
                displacement.write(displacementStr)
                
                CuO_n += 1
                if CuO_n == numCuO:
                    break
        
        # put 'NA' if Cu-O cannot reach the target Cu-O distance
        while CuO_n < numCuO:
            displacementStr = '   NA   {:10.4f}'.format(CuOList[CuO_n])
            displacementStr += '   NA' * (degree + 1)
            displacementStr += '\n'
            
            displacement.write(displacementStr)
                
            CuO_n += 1
        
        displacement.close()
        
        return
    
    
    def atomDisplacementMEP(self, atomList, Cu_n, O_n, posEqu, lEqu=False):
        """
        Compute the displacement of given atom when Cu-O reaches certain distances on MEP
        Input:
            atomList: the atoms to consider structural change
            Cu_n: the Cu atom number to compute Cu-O distance
            O_n: the O atom number to compute Cu-O distance
            posEqu: the name of equilibrium POSCAR
        """
        if not self.traj:
            print('Trajectory is not found!')
            return None
        if posEqu is None:
            print('Equilibrium structure is not found!')
            return
        
        displacementOut = self.filename+'.displacement'
        displacement = open(displacementOut, 'w')
        
        # build title for displacement file 
        CuOName = 'Cu'+str(Cu_n)+'-O'+str(O_n)
        displacementStr = '   expect ' + CuOName + ' (x, y, z)\n'
        displacement.write(displacementStr)
        
        # compute displacement and bond length
        length = self.length
        
        displacementVector = None
        
        for n in range(length):
            pos = self.traj[n]
            if lEqu:
                _, _, posRef, _, _, _ = self.read_coordinates(posEqu)
            else:
                posRef = self.traj[0]
            
            # compute Cu-O bond length, ignore images when Cu-O < 2.2 angstrom
            CuObond = np.linalg.norm(np.array(pos[O_n - 1]) - np.array(pos[Cu_n - 1]))
            if CuObond < 2.2:
                break
            
            # compute atom displacement
            posVector = []
            posRefVector = []
            for atom_n in atomList:
                posVector.extend(pos[atom_n - 1])
                posRefVector.extend(posRef[atom_n - 1])
            degree = len(posVector)
            displacementVector = np.array(posVector) - np.array(posRefVector)
                
            displacementStr = '   {:10.4f}'.format(CuObond)
            for num in displacementVector:
                displacementStr += '   {:10.6f}'.format(num)
            displacementStr += '\n'
                
            displacement.write(displacementStr)
        
        displacement.close()
        
        return
    
    
    def genO2velocity(self):
        """
        generate O2 velocity for production trajectory
        """
        numO = [59, 60]
        numCu = [28, 29, 30, 31]
        
        outFileVelocity = self.filename+'.O2Velocity'
        velocity = open(outFileVelocity, 'w')
        
        timestep = self.timestep
        traj = self.traj.copy()
        length = self.length
        
        outStr = "Time   Cu_n   O_n   Cu-O   O2 velocity\n"
        velocity.write(outStr)
        
        # compute O2 velocity at each image
        currCoord = (np.array(traj[0][numO[1] - 1]) + np.array(traj[0][numO[0] - 1])) / 2.0
        
        for n in range(1, length):
            # compute O2 velocity
            nextCoord = (np.array(traj[n][numO[1] - 1]) + np.array(traj[n][numO[0] - 1])) / 2.0
            
            velocities_n = (nextCoord - currCoord) / timestep
            velocities_n = velocities_n.tolist()
            
            currCoord = nextCoord
            
            # compute Cu-O distance
            minCu = None
            minO = None
            minCuO = float('inf')
            
            for O_n in numO:
                for Cu_n in numCu:
                    CuO = np.linalg.norm(np.array(traj[n][Cu_n - 1]) - np.array(traj[n][O_n - 1]))
                    
                    if CuO < minCuO:
                        minCuO = CuO
                        minCu = Cu_n
                        minO = O_n
            
            # compute time
            time = timestep * n        
            
            # write to output
            outStr = '    {:5.2f}'.format(time)
            outStr += '   ' + str(minCu) + '   ' + str(minO) + '   {:4.4f}'.format(minCuO)
            
            for vx_n in velocities_n:
                outStr += '   {:4.6f}'.format(vx_n)
            
            outStr += '\n'
            
            velocity.write(outStr)
            
        velocity.close()

        
    def remove_O2(self, geo_n, equ_poscar, out_file_name):
        """
        Take one image on the trajectory, remove O2, generate a POSCAR file
        Input:
            geo_n, int, the number of geometry in one trajectory
            equ_poscar, string, the name of equilibrium structure
            out_file_name, string, the name of the output POSCAR name
        """
        cell = self.cell.copy()
        atom_types = self.atom_types[:-1]
        atom_counts = self.atom_counts[:-1]

        num_atoms = sum(atom_counts)
        coordinates = self.get_pos(geo_n, lref=False)
        coordinates = coordinates[:num_atoms]
        velocities = [[0.00, 0.00, 0.00]]*num_atoms
        
        atom_typesTemp, atom_countsTemp, coordinatesTemp, flags, vTemp, energy = self.read_coordinates(equ_poscar, 'vasp', lenergy=False)
        flags = flags[:num_atoms]

        assert len(coordinates) == num_atoms
        assert len(flags) == num_atoms
        assert len(velocities) == num_atoms
        
        # Write POSCAR file
        with open(out_file_name, 'w') as output_file:
            output_file.write('POSCAR'+'\n')
            output_file.write('   1.000'+'\n')
            
            cell_str = ''
            for i in range(3):
                for j in range(3):
                    cell_str += '   '+"{:10.8f}".format(float(cell[i][j]))
                cell_str += '\n'
            output_file.write(cell_str)
            
            atoms_type_str = '   '+'   '.join(atom_types)+'\n'
            output_file.write(atoms_type_str)
            atom_counts = [str(n) for n in atom_counts]
            atoms_num_str = '   '+'   '.join(atom_counts)+'\n'
            output_file.write(atoms_num_str)
            output_file.write('Selective dynamics'+'\n'+'Cartesian'+'\n')
            
            coordinates_str = ''
            for i in range(num_atoms):
                for j in range(3):
                    coordinates_str += '   '+"{:10.8f}".format(coordinates[i][j])
                coordinates_str += flags[i]+'\n'
            output_file.write(coordinates_str)
            
            output_file.write('Cartesian'+'\n')
            velocities_str = ''
            for i in range(num_atoms):
                for j in range(3):
                    velocities_str += '   '+"{:10.8f}".format(velocities[i][j])
                velocities_str += '\n'
            output_file.write(velocities_str)
        
                
        
        