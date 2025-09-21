#!/usr/bin/env python3
import numpy as np

TRAJECTORY_FILE_MAP = {
    1: 'home2bin1_verified.pkl',
    2: 'home2bin2_verified.pkl',
    3: 'home2bin3_verified.pkl',
    4: 'home2assembly_verified.pkl',
    5: 'bin12home_verified.pkl',
    6: 'bin12assembly_verified.pkl',
    7: 'bin22home_verified.pkl',
    8: 'bin22assembly_verified.pkl',
    9: 'bin32home_verified.pkl',
    10: 'bin32assembly_verified.pkl',
    11: 'assembly2home_verified.pkl',
    12: 'assembly2bin1_verified.pkl',
    13: 'assembly2bin2_verified.pkl',
    14: 'assembly2bin3_verified.pkl',
    15: 'assembly2bin4_verified.pkl',
    16: 'assembly2bin5_verified.pkl',
    17: 'assembly2bin6_verified.pkl',
    18: 'home2bin4_verified.pkl',
    19: 'home2bin5_verified.pkl',
    20: 'home2bin6_verified.pkl',
    21: 'bin42home_verified.pkl',
    22: 'bin42assembly_verified.pkl',
    23: 'bin52home_verified.pkl',
    24: 'bin52assembly_verified.pkl',
    25: 'bin62home_verified.pkl',
    26: 'bin62assembly_verified.pkl',
}

TRAJECTORY_ID_MAP = {
    'home': {'bin1': 1, 'bin2': 2, 'bin3': 3, 'assembly': 4, 'bin4': 18, 'bin5': 19, 'bin6':20},
    'bin1': {'home': 5, 'assembly': 6},
    'bin2': {'home': 7, 'assembly': 8},
    'bin3': {'home': 9, 'assembly': 10},
    'bin4': {'home': 21, 'assembly': 22},
    'bin5': {'home': 23, 'assembly': 24},
    'bin6': {'home': 25, 'assembly': 26},
    'assembly': {'home': 11, 'bin1': 12, 'bin2': 13, 'bin3': 14, 'bin4': 15, 'bin5': 16, 'bin6': 17}
}

KIOSK_COLLISION_BOXES = np.array([
    [0.25, 0.55, 0.5, 0, 0, 0, 1.1, 0.01, 1.1],
    [0.25, -0.55, 0.5, 0, 0, 0, 1.1, 0.01, 1.1],
    [-0.41, 0, 0.5, 0, 0, 0, 0.01, 1, 1.1], 
    [0.77, 0, 0.5, 0, 0, 0, 0.01, 1, 1.1],
    [0.25, 0, 1, 0, 0, 0, 1.2, 1, 0.01],
    [0.25, 0, -0.05, 0, 0, 0, 1.2, 1, 0.01],
    [0.5, 0.25, 0.125, 0, 0, 0, 0.68, 0.695, 0.26],
    [0.43, -0.3615, 0.0, 0, 0, 0, 0.68, 0.375, 0.001],
    [0.14, -0.3615, 0.125, 0, 0, 0, 0.08, 0.375, 0.26],
    [0.344, -0.3615, 0.125, 0, 0, 0, 0.05, 0.375, 0.26],
    [0.542, -0.3615, 0.125, 0, 0, 0, 0.05, 0.375, 0.26],
    [0.75, -0.3615, 0.125, 0, 0, 0, 0.08, 0.375, 0.26],
    [0.43, -0.215, 0.125, 0, 0, 0, 0.68, 0.07, 0.26],
    [0.43, -0.52, 0.125, 0, 0, 0, 0.68, 0.07, 0.26]
])


BIN_OFFSETS = {1: [0.64, -0.37, 0.27], 
               2: [0.45, -0.37, 0.27],
               3: [0.27, -0.37, 0.27],
               4: [0.27, 0.34, 0.27],
               5: [0.45, 0.34, 0.27],
               6: [0.64, 0.34, 0.27]              
               }
# meters