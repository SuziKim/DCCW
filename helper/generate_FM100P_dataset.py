import os
import random
import time
import sys
import csv
sys.path.append(".")
sys.path.append("..")

from experiments.experiments_enums import *
from experiments.experiments_helper_functions import *

FM100P_full_colors = read_farnsworth_munsell_100()
palette_count_per_dataset = 100
neighbor_threshold = 5

gd_base_path = os.path.join('experiments', 'DCCW-dataset', 'FM100P')

def batch_palette_with_length(palette_length):
    random.seed(time.time())
    palette_indices = [random.randint(0, len(FM100P_full_colors)-1)]

    while len(palette_indices) < palette_length:
        # print('finding %d-th color' % len(palette_indices))
        random_color_index = random.randint(0, len(FM100P_full_colors)-1)

        if random_color_index in palette_indices:
            # print('\tfail to found %d-th color' % len(palette_indices))
            continue
        
        for index in palette_indices:
            if abs(index - random_color_index) <= neighbor_threshold:
                # print('\tfound %d-th color' % len(palette_indices))
                palette_indices.append(random_color_index)
                break

    return palette_indices


dir_name_base = 'FM100P-k%d'
file_name_base = 'FM100P-k%d-p%d.csv'

for FM100P_type in FM100PType:
    palettes_indices = []
    palette_length = FM100P_type.value

    dir_name = dir_name_base % palette_length
    os.makedirs(os.path.join(gd_base_path, dir_name, '%s-csv' % dir_name), exist_ok=True)

    while len(palettes_indices) < palette_count_per_dataset:
        print('finding %d-th palette' % len(palettes_indices))
        palette_indices = batch_palette_with_length(palette_length)
        palette_indices.sort()
        
        if palette_indices in palettes_indices:
            print('\tfail to found %d-th palette' % len(palettes_indices))
            continue

        print('\tfound %d-th palette' % len(palettes_indices))
        palettes_indices.append(palette_indices)

        file_name = file_name_base % (palette_length, len(palettes_indices)-1)
        with open(os.path.join(gd_base_path, dir_name, '%s-csv' % dir_name, file_name), 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            shuffle_indices = random.sample(palette_indices, len(palette_indices))
            for index in shuffle_indices:
                writer.writerow([index, FM100P_full_colors[index]])
    