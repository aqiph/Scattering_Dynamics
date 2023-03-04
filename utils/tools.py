#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 22:14:00 2023

@author: Aqiph
"""

import numpy as np


# Calculate the center of mass #
def get_COM(atom_types, atom_counts, coordinates):
    """
    Compute the center of mass of a structure
    :param atom_types: list of strs, element types
    :param atom_counts: list of ints, number of atoms for each element type
    :param coordinates: list of lists, [[x1, y1, z1], [x2, y2, z2], ...]
    :return:
    COM: list of floats, center of mass
    """
    MASSES = {'Pt': 195.080, 'H': 1.000, 'C': 12.011, 'Cu': 63.546,
              'Al': 26.981, 'O': 16.000, 'Sr': 87.620}

    # check input files and initialization
    assert len(atom_types) == len(atom_counts)
    COM = np.array([0.0, 0.0, 0.0])
    totalMass = 0

    # compute center of mass
    num_atoms = 0
    for n, atom in enumerate(atom_types):
        mass_atom = MASSES[atom]
        number = atom_counts[n]
        coord_atom = coordinates[num_atoms: (num_atoms + number)]
        COM = COM + sum(np.array(coord_atom)) * mass_atom
        totalMass += mass_atom * number
        num_atoms = num_atoms + number

    assert num_atoms == sum(atom_counts)
    COM = COM / totalMass
    COM = COM.tolist()

    return COM


# Translate a molecule #
def translate(coordinates_before, translation_vector):
    """
    Translate the molecule:
    if translation_vector is 'origin', translate molecule to origin;
    if translation_vector is a list of floats, translate molecule by translation_vector.
    :param coordinates_before: list of lists, coordinates before translation, [[x1, y1, z1], [x2, y2, z2], ...]
    :param translation_vector: lists of floats or str, translation vector
    :return:
    coordinates_after: list of lists, coordinates after translation, [[x1, y1, z1], [x2, y2, z2], ...]
    """
    coordinates_before = np.array(coordinates_before)

    # translate coordinates_before by translation_vector
    if isinstance(translation_vector, list):
        translation_vector = np.array(translation_vector)
        coordinates_after = coordinates_before + translation_vector

    # translate coordinates_before to origin
    elif translation_vector == 'origin':
        center = sum(coordinates_before) / len(coordinates_before)
        coordinates_after = coordinates_before - center

    else:
        print('Error: invalid translation vector.')
        return

    return coordinates_after.tolist()


# Rotate a molecule and its normal modes #
def rotate(atom_types, atom_counts, coordinates_before, modes_before=None):
    """
    Randomly rotate the molecule and its normal modes
    :param atom_types: list of strs, element types
    :param atom_counts: list of ints, number of atoms for each element type
    :param coordinates_before: list of lists, coordinates before rotation, [[x1, y1, z1], [x2, y2, z2], ...]
    :param modes_before: list of lists, modes before rotation, [[[dx1, dy1, dz1], [dx2, dy2, dz2], ...], ...]
    :return:
    coordinates_after: list of lists, coordinates after rotation, [[x1, y1, z1], [x2, y2, z2], ...]
    modes_after: list of lists, modes after rotation, [[[dx1, dy1, dz1], [dx2, dy2, dz2], ...], ...]
    """
    num_atoms = sum(atom_counts)

    # if modes_before is None, make fake modes
    if modes_before == None:
        modes_before = [[[0.0 for j in range(3)] for i in range(num_atoms)]]

    # find the center of mass
    coordinates_before = np.array(coordinates_before)
    COM = get_COM(atom_types, atom_counts, coordinates_before)

    # translate molecule, such that the center of mass is at origin; transpose coordinates and modes
    coordinates_centralized = coordinates_before - np.array(COM)
    coordinates_T = np.matrix(coordinates_centralized).transpose()

    modes_T = []
    for mode in modes_before:
        mode = np.matrix(mode).transpose()
        modes_T.append(mode)

    # generate a random rotation
    degrees = np.random.uniform(0.0, 2.0 * np.pi, 3)

    # rotate coordinates and modes about x axis
    degree = degrees[0]
    s, c = np.sin(degree), np.cos(degree)
    rotation_matrix = np.matrix([[1, 0, 0], [0, c, -s], [0, s, c]])

    coordinates_T = np.dot(rotation_matrix, coordinates_T)

    modes_T_new = []
    for mode_T in modes_T:
        mode_T_new = np.dot(rotation_matrix, mode_T)
        modes_T_new.append(mode_T_new)
    modes_T = modes_T_new.copy()

    # rotate coordinates and modes about y axis
    degree = degrees[1]
    s, c = np.sin(degree), np.cos(degree)
    rotation_matrix = np.matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    coordinates_T = np.dot(rotation_matrix, coordinates_T)

    modes_T_new = []
    for mode_T in modes_T:
        mode_T_new = np.dot(rotation_matrix, mode_T)
        modes_T_new.append(mode_T_new)
    modes_T = modes_T_new.copy()

    # rotate coordinates and modes about z axis
    degree = degrees[2]
    s, c = np.sin(degree), np.cos(degree)
    rotation_matrix = np.matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    coordinates_T = np.dot(rotation_matrix, coordinates_T)

    modes_T_new = []
    for mode_T in modes_T:
        mode_T_new = np.dot(rotation_matrix, mode_T).transpose().tolist()
        modes_T_new.append(mode_T_new)
    modes_after = modes_T_new.copy()

    # translate molecule to original coordinates
    coordinates_centralized = coordinates_T.transpose()
    coordinates_after = coordinates_centralized + np.array(COM)
    coordinates_after = coordinates_after.tolist()

    # write new modes
    if False:
        num_modes = len(modes_before)
        with open('new_modes', 'w') as output:
            for n in range(num_modes):
                output.write(str(n + 1) + ' f' + '   meV' + '\n')
                output.write('       X         Y         Z           dx          dy          dz' + '\n')

                for i in range(num_atoms):
                    line = ''
                    for j in range(3):
                        line += "   {:10.6f}".format(coordinates_after[i][j])
                    for j in range(3):
                        line += "   {:10.6f}".format(modes_after[n][i][j])
                    line += '\n'
                    output.write(line)

                output.write('\n')

    return coordinates_after, modes_after
