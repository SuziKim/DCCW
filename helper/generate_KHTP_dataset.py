import os
import random
import time
import sys
import csv
import numpy as np

sys.path.append(".")
sys.path.append("..")

from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color

from dccw.color_palette import *

from experiments.experiments_enums import *
from experiments.experiments_helper_functions import *


def add_interpolation_and_jitter_save_palette(generated_palette_count, input_palette, target_palette, interpolation_count, jitter_offset, palette_index):
	palette1_hex, palette2_hex = add_interpolation_to_palette(input_palette, target_palette, interpolation_count)

	return jitter_and_save_palette(generated_palette_count, palette1_hex, palette2_hex, interpolation_count, jitter_offset, palette_index)


def jitter_and_save_palette(generated_palette_count, palette1_hex, palette2_hex, interpolation_count, jitter_offset, palette_index):
	palette1 = ColorPalette(auto_fetched=False, palette_length=len(palette1_hex), colors=palette1_hex)
	palette1.jitter_colors(jitter_offset)

	palette2 = ColorPalette(auto_fetched=False, palette_length=len(palette2_hex), colors=palette2_hex)
	palette2.jitter_colors(jitter_offset)

	return save_palette(generated_palette_count, interpolation_count, jitter_offset, palette1.to_hex_list(), palette2.to_hex_list(), palette_index)


def get_interpolated_color(lab1, lab2):
	lab1_value = lab1.get_value_tuple()
	lab2_value = lab2.get_value_tuple()

	interpolated_alpha = 0
	while True:
		interpolated_alpha = random.random()
		if interpolated_alpha > 0.1 and interpolated_alpha < 0.9:
			break
	
	new_lab_value = [a+(b-a)*interpolated_alpha for a, b in zip(lab1_value, lab2_value)]
	new_lab_color = LabColor(new_lab_value[0], new_lab_value[1], new_lab_value[2], observer=lab1.observer, illuminant=lab1.illuminant)
	
	return new_lab_color, convert_color(new_lab_color, sRGBColor).get_rgb_hex()


def add_interpolation_to_palette(palette1_hex, palette2_hex, interpolation_count):
	palette1 = ColorPalette(auto_fetched=False, palette_length=len(palette1_hex), colors=palette1_hex)
	palette1_color_objects = palette1.get_color_objects(color_space='lab')
	added_count = 0
	while added_count < interpolation_count:
		target_index = random.randint(0, len(palette1_color_objects)-2)
		new_color, new_color_hex = get_interpolated_color(palette1_color_objects[target_index], palette1_color_objects[target_index+1])

		if new_color_hex in [convert_color(c, sRGBColor).get_rgb_hex() for c in palette1_color_objects]:
			continue

		palette1_color_objects.insert(target_index+1, new_color)
		added_count += 1

	palette2 = ColorPalette(auto_fetched=False, palette_length=len(palette2_hex), colors=palette2_hex)
	palette2_color_objects = palette2.get_color_objects(color_space='lab')
	added_count = 0
	while added_count < interpolation_count:
		target_index = random.randint(0, len(palette2_color_objects)-2)
		new_color, new_color_hex = get_interpolated_color(palette2_color_objects[target_index], palette2_color_objects[target_index+1])

		if new_color_hex in [convert_color(c, sRGBColor).get_rgb_hex() for c in palette2_color_objects]:
			continue

		palette2_color_objects.insert(target_index+1, new_color)
		added_count += 1
	
	palette1_new_hex = [convert_color(c, sRGBColor).get_rgb_hex() for c in palette1_color_objects]
	palette2_new_hex = [convert_color(c, sRGBColor).get_rgb_hex() for c in palette2_color_objects]
	
	return palette1_new_hex, palette2_new_hex


def add_interpolation_and_save_palette(generated_palette_count, palette1_hex, palette2_hex, interpolation_count, jitter_offset, palette_index):
	palette1_new_hex, palette2_new_hex = add_interpolation_to_palette(palette1_hex, palette2_hex, interpolation_count)
	
	return save_palette(generated_palette_count, interpolation_count, jitter_offset, palette1_new_hex, palette2_new_hex, palette_index)


def save_palette(generated_palette_count, interpolation_count, jitter_offset, palette1, palette2, palette_index):	
	# palette1, palette2: list of hex [#]
	print('[KHTP] saving %d-th palette' % (palette_index))
	file_name = file_name_base % (interpolation_count, jitter_offset, palette_index)

	indexed_palette1 = [(i, c) for i, c in enumerate(palette1)]
	indexed_palette2 = [(i, c) for i, c in enumerate(palette2)]

	shuffled_palette1 = random.sample(indexed_palette1, len(indexed_palette1))
	shuffled_palette2 = random.sample(indexed_palette2, len(indexed_palette2))

	shuffled_palette1_index = [i for i, c in shuffled_palette1]
	shuffled_palette1_hex = [c for i, c in shuffled_palette1]
	shuffled_palette2_index = [i for i, c in shuffled_palette2]
	shuffled_palette2_hex = [c for i, c in shuffled_palette2]

	dir_path = os.path.join(gd_base_path, dir_name_base % (interpolation_count, jitter_offset), '%s-csv' % (dir_name_base % (interpolation_count, jitter_offset)))
	os.makedirs(dir_path, exist_ok=True)
	with open(os.path.join(dir_path, file_name), 'w') as f:
		writer = csv.writer(f, delimiter='\t')
		writer.writerow(shuffled_palette1_index)
		writer.writerow(shuffled_palette1_hex)
		writer.writerow(shuffled_palette2_index)
		writer.writerow(shuffled_palette2_hex)

	return generated_palette_count + 1


gd_base_path = os.path.join('experiments', 'DCCW-dataset', 'KHTP')

file_name_base = 'KHTP-interpolation%d-jitter%d-p%d.csv'
os.makedirs(gd_base_path, exist_ok=True)
dir_name_base = 'KHTP-interpolation%d-jitter%d'


input_palettes = np.array(read_KHTP()) # 2D list of hexes [['#aaaaaa', '#bbbbbb'], ['#aaaaaa', '#bbbbbb']...]

neighbor_threshold = 1
mixed_pair_counts = 100
mixed_pair_difference_threshold = 8

target_matrix = input_palettes
mixed_palettes_pairs_list = []
h, w = target_matrix.shape
generated_palette_count = 0
palette_index = 0

while len(mixed_palettes_pairs_list) < mixed_pair_counts:

	# fix first palette order
	initial_color_index = random.randint(0, h-1)
	first_palette_indices = [initial_color_index]
	while len(first_palette_indices) < w:
		random_color_index = first_palette_indices[-1] + random.randint(-neighbor_threshold, neighbor_threshold)

		if random_color_index < 0 or random_color_index > h-1:
			continue
		
		first_palette_indices.append(random_color_index)
	
	# find second palette order
	second_palette_indices = []
	for index in first_palette_indices:
		random_color_index = 0

		while True:
			random_color_index = index + random.randint(-neighbor_threshold, neighbor_threshold)
			if random_color_index >= 0 and random_color_index < h:
				break
			
		second_palette_indices.append(random_color_index)
		
	# check the pairs differences
	difference = sum([abs(c1-c2) for c1, c2 in zip(first_palette_indices, second_palette_indices)])
	if difference < mixed_pair_difference_threshold:
		continue

	# check the pairs are already generated
	sorted_palette_pair = sorted([first_palette_indices, second_palette_indices])
	if sorted_palette_pair in mixed_palettes_pairs_list:
		continue

	mixed_palettes_pairs_list.append(sorted_palette_pair)
	
	first_palette_hexes = [target_matrix[i][j] for i, j in zip(first_palette_indices, range(w))]
	second_palette_hexes = [target_matrix[i][j] for i, j in zip(second_palette_indices, range(w))]
	

	generated_palette_count = save_palette(generated_palette_count, 0, 0, first_palette_hexes, second_palette_hexes, palette_index)
	
	for jitter_offset_type in KHTPJitterOffset:
		generated_palette_count = jitter_and_save_palette(generated_palette_count, first_palette_hexes, second_palette_hexes, 0, jitter_offset_type.value, palette_index)

	for interpolation_count in KHTPInterpolationCount:
		generated_palette_count = add_interpolation_and_save_palette(generated_palette_count, first_palette_hexes, second_palette_hexes, interpolation_count.value, 0, palette_index)

	for jitter_offset_type in KHTPJitterOffset:
		for interpolation_count in KHTPInterpolationCount:
			generated_palette_count = add_interpolation_and_jitter_save_palette(generated_palette_count, first_palette_hexes, second_palette_hexes, interpolation_count.value, jitter_offset_type.value, palette_index)

	palette_index += 1