#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 12:51:51 2019

@author: Aqiph
"""

import numpy as np


def get_diatomic_perpendicular_to_bond(atom1, atom2, ref, diatomic_bond_length, diatomic_height):
    """
    Generate a diatomic molecule perpendicular to a given bond defined by atom1 and atom2
    :para atom1: list of 3 float, representing the position of atom1
    :para atom2: list of 3 float, representing the position of atom2
    :para ref: list of 3 float, representing the position of a reference atom
    :para diatomic_bond_length: float, representing diatomic molecule bond length
    :para diatomic_height: float, representing the distance from the bond center to the diatomic molecule
    """
    atom1 = np.array(atom1)
    atom2 = np.array(atom2)
    ref = np.array(ref)
    
    bond_center = (atom1+atom2)/2.0
    d1 = atom1-ref
    d2 = atom2-ref
    shift = np.cross(d1, d2)
    shift = shift/np.linalg.norm(shift)
    if np.dot(shift, np.array([0.0, 0.0, 1.0])) < 0:
        shift = -1.0*shift
    
    diatomic_center = bond_center+shift*diatomic_height
    e1 = atom1-diatomic_center
    e2 = atom2-diatomic_center
    direction = np.cross(e1, e2)
    direction/np.linalg.norm(direction)
    D1 = diatomic_center+direction*diatomic_bond_length*0.5
    D2 = diatomic_center-direction*diatomic_bond_length*0.5
    
    print(D1, D2)
    return D1, D2


def get_diatomic_parallel_to_bond(atom1, atom2, ref, diatomic_bond_length, diatomic_height):
    """
    Generate a diatomic molecule parallel to a given bond
    :para atom1: list of 3 float, representing the position of atom1
    :para atom2: list of 3 float, representing the position of atom2
    :para ref: list of 3 float, representing the position of a reference atom
    :para diatomic_bond_length: float, representing diatomic molecule bond length
    :para diatomic_height: float, representing the distance from the bond center to the diatomic molecule
    """
    atom1 = np.array(atom1)
    atom2 = np.array(atom2)
    ref = np.array(ref)

    direction = (atom2-atom1)/np.linalg.norm(atom2-atom1)
    d1 = atom1-ref
    d2 = atom2-ref
    v_shift = np.cross(d1, d2)
    v_shift = v_shift/np.linalg.norm(v_shift)

    center = (atom1+atom2)/2.0
    center = center+v_shift*diatomic_height

    D1 = center-direction*diatomic_bond_length*0.5
    D2 = center+direction*diatomic_bond_length*0.5

    print(D1, D2)
    return D1, D2

    
    

if __name__ == '__main__':
    
    atom1 = [8.9436752352639619, 12.3159534052442687, 11.4805223285458915]
    atom2 = [9.1141859775081429,  9.3094011707446249, 11.4010011139942549]
    ref = [10.6149446159026120, 10.8981273189749750, 12.6174679638619640]
    diatomic_bond_length = 1.4
    diatomic_height = 3.0
    
    get_diatomic_perpendicular_to_bond(atom1, atom2, ref, diatomic_bond_length, diatomic_height)
    
    get_diatomic_parallel_to_bond(atom1, atom2, ref, diatomic_bond_length, diatomic_height)


