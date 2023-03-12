#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 23:19:00 2019

@author: Aqiph
"""
import sys
import numpy as np
import os

from MD_preparation import Sampler
from trajectory_analysis import Trajectory


###############################################################
### Step 1: Generate initial structure for metal cluster MD ###
###############################################################

def cluster_MD_preparation():
    """
    generate a metal cluster initial state, i.e. POSCAR, for AIMD simulation
    """
    # optimized cluster structure
    cluster_input_file_coord = 'tests/step1/cluster.POSCAR'
    cluster_input_file_modes = 'tests/step1/cluster_eigen'
    cluster_num_modes = 54
    cluster_lrotation = False
    cluster_method_vib = 'CB'   # use Classical Boltzmann (Exponetial) method
    cluster_temp_vib = 700.0
    cluster_state = None
    cluster_output_file = 'tests/step1/cluster_MD.POSCAR'

    cluster = Sampler()
    cluster.sample_state_from_distribution(input_file_coordinates=cluster_input_file_coord, input_file_modes=cluster_input_file_modes,
                                           num_modes=cluster_num_modes, lrotation=cluster_lrotation, method_vib=cluster_method_vib,
                                           temp_vib=cluster_temp_vib, state=cluster_state, output_file=cluster_output_file)

# Metal cluster equilibration:
# run AIMD simulation for metal clusters to get trajectory file (XDATCAR) and energy file for cluster MD


########################################################################################
### Step 2: Generate initial structure for scattering dynamics (molecule on cluster) ###
########################################################################################

def MD_preparation(impact_method):
    """
    generate a POSCAR file in which molecule is over the cluster for AIMD simulation
    :param impact_method: str, impact method. Allowed values include 'cluster_center', 'vertical_targeted' and 'vertical_nontargeted'
    """
    # Generate .coord and .velocity files for equilibrated cluster trajectory
    print('Start: generate coordinates and velocities for equilibrated cluster trajectory ...')
    cluster_input_file = 'tests/step2/cluster.XDATCAR'
    cluster_timestep = 0.5
    cluster_trajectory = Trajectory()
    cluster_trajectory.read_trajectory(input_file=cluster_input_file, timestep=cluster_timestep, lref=False)
    cluster_trajectory.gen_xyz(lref=False)
    cluster_trajectory.gen_velocities(lref=False)
    print('Done.')

    # Randomly select a snapshot from equilibrated cluster trajectory
    print('Start: select a snapshot from equilibrated cluster trajectory ...')
    cluster_input_file_coord = 'tests/step2/cluster.POSCAR'
    cluster_input_file_traj_coord = 'tests/step2/cluster.coord'
    cluster_input_file_velocities = 'tests/step2/cluster.velocity'
    cluster_input_file_energy = 'tests/step2/cluster.energy'
    cluster_trajectory_range = [1000, 2000]
    cluster_output_file = 'tests/step2/selected_cluster.POSCAR'
    cluster = Sampler()
    cluster.sample_state_from_trajectory(input_file_coordinates=cluster_input_file_coord, input_file_traj_coord=cluster_input_file_traj_coord,
                                         input_file_velocities=cluster_input_file_velocities, input_file_energy=cluster_input_file_energy,
                                         trajectory_range=cluster_trajectory_range, output_file=cluster_output_file)
    print('Done.')

    # Generate a molecule's initial vibrational state
    print('Start: initialize molecular vibrational state ...')
    molecule_input_file_coord = 'tests/step2/molecule.POSCAR'
    molecule_input_file_modes = 'tests/step2/molecule_eigen'
    molecule_num_modes = 1
    molecule_lrotation = True
    molecule_method_vib = 'QCT'
    molecule_temp_vib = 700.0
    molecule_state = [0]
    molecule_output_file = 'tests/step2/molecule_MD.POSCAR'
    molecule = Sampler()
    molecule.sample_state_from_distribution(input_file_coordinates=molecule_input_file_coord, input_file_modes=molecule_input_file_modes,
                                            num_modes=molecule_num_modes, lrotation=molecule_lrotation, method_vib=molecule_method_vib,
                                            temp_vib=molecule_temp_vib, state=molecule_state, output_file=molecule_output_file)
    print('Done.')

    # Combine cluster and molecule POSCAR files and generate the initial state, i.e. POSCAR file, for AIMD simulation
    print('Start: initialize the system ...')
    cluster_input_file_coord = 'tests/step2/selected_cluster.POSCAR'
    molecule_input_file_coord = 'tests/step2/molecule_MD.POSCAR'
    if impact_method == 'cluster_center':   # molecule collides with the impact site in the direction towards the center of mass of the cluster
        impact_site = 28
    elif impact_method == 'vertical_targeted':   # molecule collides with a cluster atom or a bond between two cluster atoms perpendicular to the xy plane.
        impact_site = 28   # molecule is over a cluster atom
        impact_site = [28, 29]   # molecle is over a cluster bond
    elif impact_method == 'vertical_nontargeted':   # molecule collides with a cluster randomly within a range definde by impact_site
        impact_site = [[8.94236 - 1.5, 11.72384 + 1.5], [9.11124 - 1.5, 13.13515 + 1.5], 12.74983]
    else:
        print('Error: invalid impact method.')
        return
    distance_to_cluster = 5.0
    method_trans = 'EP'
    temp_trans = 700
    output_file = 'tests/step2/MD_molecule_cluster.POSCAR'
    system = Sampler()
    system.MD_initialization(cluster_input_file_coord=cluster_input_file_coord, molecule_input_file_coord=molecule_input_file_coord,
                             impact_method=impact_method, impact_site=impact_site, distance_to_cluster=distance_to_cluster,
                             method_trans=method_trans, temp_trans=temp_trans, output_file=output_file)
    print('Done.')



if __name__ == '__main__':

    # cluster_MD_preparation()

    impact_method = 'vertical_nontargeted'
    MD_preparation(impact_method)

