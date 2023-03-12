#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 16:17:00 2023

@author: Aqiph
"""

import numpy as np
import linecache



# Read coordinates and velocities #
def read_coordinates(input_file, file_format, lenergy=False):
    """
    Read coordinats and velocities from an existing file, e.g. POSCAR file
    :param input_file: str, path of the coordinates input file, i.e. POSCAR
    :param file_format: str, format of the coordinates and velocities input file. Allowed values include 'vasp' or 'ase'
    :param lenergy: boolean, whether read energy or not
    :return:
    cell: list of lists, cell, [[x1, y1, z1], [x2, y2, z2], ...]
    atom_types: list of strs, element types
    atom_counts: list of ints, number of atoms for each element type
    coordinates: list of lists, [[x1, y1, z1], [x2, y2, z2], ...]
    flags: list of strs, flags
    velocities: list of lists, velocities, [[x1, y1, z1], [x2, y2, z2], ...]
    energy: float, energy if lenergy == True, else 0.0
    """
    cell = []
    atom_types = []
    atom_counts = []
    coordinates = []
    flags = []
    velocities = []
    energy = 0.0

    with open(input_file, 'r') as data:
        # read atom types for 'ase' file
        if file_format == 'ase':
            atom_types = data.readline().strip().split()
        elif file_format == 'vasp':
            if lenergy:
                energy = float(data.readline().strip())
            else:
                data.readline()
        else:
            raise Exception('Error: invalid file format.')

        # read lattice constant and lattice vectors
        lattice_constant = float(data.readline().strip())
        for _ in range(3):
            vector = data.readline().strip().split()
            vector = [lattice_constant * float(i) for i in vector]
            assert len(vector) == 3
            cell.append(vector)

        # read atom types for 'vasp' file and atom counts
        if file_format == 'vasp':
            atom_types = data.readline().strip().split()
        atom_counts = data.readline().strip().split()
        atom_counts = [int(i) for i in atom_counts]
        assert len(atom_types) == len(atom_counts)
        num_atoms = sum(atom_counts)

        # read coordinates and flags
        if file_format == 'vasp':
            data.readline()

        coord_type = data.readline().strip()
        for n in range(num_atoms):
            line = data.readline().strip().split()
            coord_atom = [float(i) for i in line[:3]]
            if len(line) == 3:
                flag = '      T   T   T'
            elif len(line) == 6:
                flag = '      ' + '   '.join(line[3:])
            else:
                print('Error: invalid input file.')
            coordinates.append(coord_atom)
            flags.append(flag)

        # read velocities
        data.readline()
        for n in range(num_atoms):
            velocities_atom = data.readline().strip().split()
            velocities_atom = [float(i) for i in velocities_atom]
            assert len(velocities_atom) == 3
            velocities.append(velocities_atom)

    # change coordinates to Cartesian coordinate if coord_type == 'Direct'
    if coord_type == 'Direct':
        coordinates_Cart = []
        for atom in coordinates:
            atom_Cart = sum((np.array(cell).transpose() * np.array(atom)).transpose())
            atom_Cart = atom_Cart.tolist()
            coordinates_Cart.append(atom_Cart)
        assert len(coordinates_Cart) == num_atoms
        coordinates = coordinates_Cart.copy()

    # check variables
    assert len(coordinates) == num_atoms
    assert len(flags) == num_atoms
    assert len(velocities) == num_atoms

    return cell, atom_types, atom_counts, coordinates, flags, velocities, energy


# Read normal modes #
def read_normal_modes(input_file, num_modes, num_atoms):
    """
    Read frequencies and normal modes from an existing file
    :param input_file: str, path of the normal modes input file
    :param num_modes: int, number of normal modes need to input
    :param num_atoms: int, number of atoms
    :return:
    freq_list: list of frequencies in eV
    modes: list of list, modes, [[[dx1, dy1, dz1], [dx2, dy2, dz2], ...], ...]
    """
    freq_list = []
    modes = []

    with open(input_file, 'r') as data:
        # find the first mode
        line = data.readline()
        while 'meV' not in line:
            line = data.readline()

        # read frequencies and normal modes
        for m in range(num_modes):
            # read frequency
            line = line.strip().split()
            assert line[-1] == 'meV', 'Error: frequency is not read correctly.'
            freq = float(line[-2]) / 1000.0  # freq in eV
            freq_list.append(freq)
            data.readline()

            # read normal mode
            mode = []
            for atom in range(num_atoms):
                line = data.readline().strip().split()
                line = [float(i) for i in line[3:6]]
                mode.append(line)
            assert len(mode) == num_atoms
            modes.append(mode)
            data.readline()
            line = data.readline()

    # check variables
    assert len(freq_list) == num_modes
    assert len(modes) == num_modes

    return freq_list, modes


# Read coordinates from a trajectory #
def read_coordinates_from_trajectory(input_file, snapshot_idx):
    """
    Read coordinates from an existing .coord file (trajectory coordinates file)
    :param input_file: str, path of the trajectory coordinates input file, i.e. .coor file
    :param snapshot_idx: int, snapshot index, start from 1
    :return:
    cell: list of lists, cell, [[x1, y1, z1], [x2, y2, z2], ...]
    atom_types: list of strs, element types
    atom_counts: list of ints, number of atoms for each element type
    coordinates: list of lists, [[x1, y1, z1], [x2, y2, z2], ...]
    flags: list of strs, flags
    """
    cell = []
    coordinates = []
    flags = []

    with open(input_file, 'r') as data:
        data.readline()

        # read lattice constant and lattice vectors
        lattice_constant = float(data.readline().strip())
        for _ in range(3):
            vector = data.readline().strip().split()
            vector = [lattice_constant * float(i) for i in vector]
            assert len(vector) == 3
            cell.append(vector)

        # read atom types and atom counts
        atom_types = data.readline().strip().split()
        atom_counts = data.readline().strip().split()
        atom_counts = [int(i) for i in atom_counts]
        assert len(atom_types) == len(atom_counts)
        num_atoms = sum(atom_counts)

    # read coordinates and flags
    start = (snapshot_idx - 1) * (num_atoms + 1) + 8
    line = linecache.getline(input_file, start)
    line = line.strip().split()
    assert line[0] == 'Time'
    print('The structure is for time = ', float(line[-1]))
    for n in range(num_atoms):
        line = linecache.getline(input_file, start + 1 + n)
        line = line.strip().split()
        coord_atom = [float(i) for i in line]
        flag = '      T   T   T'
        coordinates.append(coord_atom)
        flags.append(flag)

    # check variables
    assert len(coordinates) == num_atoms
    assert len(flags) == num_atoms

    return cell, atom_types, atom_counts, coordinates, flags


# Read velocities from a trajectory #
def read_velocities_from_trajectory(input_file, num_atoms, snapshot_idx):
    """
    Read velocities from an existing .velocity file (trajectory velocities file)
    :param input_file: str, path of the trajectory velocities input file, i.e. .velocity file
    :param num_atoms: int, number of atoms
    :param snapshot_idx: int, snapshot index, start from 1
    :return:
    velocities: list of lists, [[x1, y1, z1], [x2, y2, z2], ...]
    """
    velocities = []

    # read velocities
    start = (snapshot_idx - 1) * (num_atoms + 1) + 1
    line = linecache.getline(input_file, start)
    line = line.strip().split()
    assert line[0] == 'Time'
    print('The velocity is for time = ', float(line[-1]))

    for n in range(num_atoms):
        line = linecache.getline(input_file, start + 1 + n)
        line = line.strip().split()
        velocities_atom = [float(i) for i in line]
        velocities.append(velocities_atom)

    # check variables
    assert len(velocities) == num_atoms

    return velocities


# Read energy from a trajectory #
def read_energy_from_trajectory(input_file, snapshot_idx):
    """
    Read energy from an existing .energy file (trajectory energy file)
    :param input_file: str, path of the trajectory energy input file, i.e. .energy file
    :param snapshot_idx: int, snapshot index, start from 1
    :return:
    energy: str, energy string
    """
    # read energy
    start = snapshot_idx + 1
    line = linecache.getline(input_file, start)
    line = line.strip().split()
    assert line[1] == 'T='
    temp = float(line[2])
    Etot = float(line[4])
    V = float(line[8])
    Ek = float(line[10])

    # write energy string
    energy_str = ' temperature ' + "{:6.2f}".format(temp) + ' K;'
    energy_str += ' Etot ' + "{:6.6f}".format(Etot) + ' eV;'
    energy_str += ' V ' + "{:6.6f}".format(V) + ' eV;'
    energy_str += ' Ek ' + "{:6.6f}".format(Ek) + ' eV;'

    return energy_str



if __name__ == '__main__':

    # input_file = 'tests/test_read_coordinates.POSCAR'
    # cell, atom_types, atom_counts, coordinates, flags, velocities, energy = \
    #     read_coordinates(input_file, file_format='vasp', lenergy=True)
    # print(cell, atom_types, atom_counts, coordinates, flags, velocities, energy)

    # input_file = 'tests/test_read_normal_modes'
    # freq_list, modes = read_normal_modes(input_file, num_modes=6, num_atoms=2)
    # print(freq_list, modes)

    # input_file = 'tests/test_read_coordinates_from_trajectory.coord'
    # cell, atom_types, atom_counts, coordinates, flags = \
    #     read_coordinates_from_trajectory(input_file, snapshot_idx=1)
    # print(cell, atom_types, atom_counts, coordinates, flags)

    # input_file = 'tests/test_read_velocities_from_trajectory.velocity'
    # velocities = read_velocities_from_trajectory(input_file, num_atoms=58, snapshot_idx=1)
    # print(velocities)

    input_file = 'tests/test_read_energy_from_trajectory.energy'
    energy = read_energy_from_trajectory(input_file, snapshot_idx=1)
    print(energy)




