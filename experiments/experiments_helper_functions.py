import os
import enum
import csv
import random

from dccw.single_palette_sorter import * 
from dccw.color_palettes import *
from dccw.color import *

from experiments.similarity_dataset_processor import *
from experiments.experiments_enums import *

# ==========================================
# Functions
# ==========================================

def read_farnsworth_munsell_100():
	#return list of hexes ['#aaaaaa', '#bbbbbb', ...]
	path = os.path.join('experiments', 'original_dataset', 'FM100P', 'farnsworth_munsell_100.txt')
	colors = [line.rstrip() for line in open(path, 'r')]

	return colors

def read_coolors_gradient():
	#return 2D list of hexes [['#aaaaaa', '#bbbbbb'], ['#aaaaaa', '#bbbbbb']...]
	palettes = []
	path = os.path.join('experiments', 'original_dataset', 'coolors', 'coolors_gradient.csv')
	with open(path, 'r') as f:
		reader = csv.reader(f, delimiter=',')
		
		for row in reader:
			palette = [hex_color.rstrip() for hex_color in row]
			palettes.append(palette)

	return palettes

def read_KHTP():
	#return 2D list of hexes [['#aaaaaa', '#bbbbbb'], ['#aaaaaa', '#bbbbbb']...]
	path = os.path.join('experiments', 'original_dataset', 'KHTP', 'color_image_scale.csv')

	hue_names = ['R', 'YR', 'Y', 'GY', 'G', 'BG', 'B', 'PB', 'P', 'RP']
	tone_names = ['V', 'S', 'B', 'P', 'VP', 'LGR', 'L', 'GR', 'DL', 'DP', 'DK', 'DGR']

	# initialize palettes
	palettes = {}
	for tone_name in tone_names:
		palette = {}
		for hue_name in hue_names:
			palette[hue_name] = ''

		palettes[tone_name] = palette

	# read files
	with open(path, 'r') as f:
		reader = csv.DictReader(f, dialect=csv.excel_tab)
		for row in reader:
			color = Color([int(row['r']), int(row['g']), int(row['b'])])
			palettes[row['tone']][row['hue']] = color.HEX_value()

	palette_list = []
	for tone_name in tone_names:
		palette = []
		for hue_name in hue_names:
			palette.append(palettes[tone_name][hue_name])
		palette_list.append(palette)
	print(palette_list)
	return palette_list


def random_batch_single_palette_from_FM100P(palette_length):
	gd_base_path = os.path.join('experiments', 'DCCW-dataset', 'FM100P')
	dir_name = 'FM100P-k%d' % palette_length
	dataset_dir = os.path.join(gd_base_path, dir_name, '%s-csv' % dir_name)

	sorted_dirs = os.listdir(dataset_dir)
	randm_file = random.choice(sorted_dirs)
	random_file_path = os.path.join(dataset_dir, randm_file)
	return read_single_file_of_FM100P(random_file_path)

def read_single_file_of_FM100P(file_path):
	indices = []
	hex_colors = []

	with open(file_path) as f:
		reader = csv.reader(f, delimiter='\t')
		for row in reader:
			indices.append(int(row[0]))
			hex_colors.append(row[1])
	
	return indices, hex_colors


def random_batch_palette_pair_from_KHTP(khtp_category):
	# khtp_category: enum of AllowedKHTPCategory

	gd_base_path = os.path.join('experiments', 'DCCW-dataset', 'KHTP')	
	khtp_type = random.choice(khtp_category.value)
	dataset_dir = os.path.join(gd_base_path, khtp_type.value, '%s-csv' % khtp_type.value)

	sorted_dirs = os.listdir(dataset_dir)
	random_file = random.choice(sorted_dirs)
	random_file_path = os.path.join(dataset_dir, random_file)
	return read_single_file_of_KHTP(random_file_path)

def read_single_file_of_KHTP(file_path):
	indices = [] # 2D array of indices
	hex_colors = [] # 2D array of hex colors 

	with open(file_path) as f:
		reader = csv.reader(f, delimiter='\t')
		for index, row in enumerate(reader):
			if index % 2 == 0:
				indices.append([int(i) for i in row])
			else:
				hex_colors.append(row)
	
	return indices, hex_colors

def random_batch_palette_pair_from_LHSP(lhsp_category):
	# lhsp_category: enum of AllowedLHSPCategory

	gd_base_path = os.path.join('experiments', 'DCCW-dataset', 'LHSP')
	lhsp_type = random.choice(lhsp_category.value)

	dataset_processor = SimilarityDatasetProcessor(lhsp_type, True)
	dataset = dataset_processor.get_dataset()
	
	query_image_name, _, query_palette = random.choice(dataset.get_query_palettes())

	maximum_retrieval_palettes_count = 10
	maximum_correct_answer_count = 1
	correct_answer_count = 0
	retrieval_palettes = []
	retrieval_image_names = []

	while len(retrieval_palettes) < maximum_retrieval_palettes_count:
		retrieval_image_name, _, retrieval_palette = random.choice(dataset.get_retrieval_palettes())
		
		if retrieval_palette in retrieval_palettes:
			continue

		if (query_image_name == retrieval_image_name and correct_answer_count < maximum_correct_answer_count):
			retrieval_palettes.append(retrieval_palette)
			retrieval_image_names.append(retrieval_image_name)
			correct_answer_count += 1
			continue
		else:
			if (len(retrieval_palettes) - correct_answer_count < maximum_retrieval_palettes_count - maximum_correct_answer_count):
				retrieval_palettes.append(retrieval_palette)
				retrieval_image_names.append(retrieval_image_name)

	return query_palette, query_image_name, retrieval_palettes, retrieval_image_names

	


def levenshtein_distance(a, b):
	# http://rosettacode.org/wiki/Levenshtein_distance#Python
	costs = []
	for j in range(len(b) + 1):
		costs.append(j)
	for i in range(1, len(a) + 1):
		costs[0] = i
		nw = i - 1
		for j in range(1, len(b) + 1):
			cj = min(1 + min(costs[j], costs[j - 1]),
					 nw if a[i - 1] == b[j - 1] else nw + 1)
			nw = costs[j]
			costs[j] = cj
 
	return costs[len(b)] / max(len(a), len(b))


def get_longest_increasing_subsequence_length(indices):
	LIS = []
	def insert(target):
		left, right = 0, len(LIS) - 1
		# Find the first index "left" which satisfies LIS[left] >= target
		while left <= right:
			mid = left + (right - left) // 2
			if LIS[mid] >= target:
				right = mid - 1
			else:
				left = mid + 1
		# If not found, append the target.
		if left == len(LIS):
			LIS.append(target);
		else:
			LIS[left] = target

	for num in indices:
		insert(num)
	
	return len(LIS) / len(indices)


def get_longest_common_subsequence_length(x, y):
	m = len(x)
	n = len(y)
	L = [[0] * (n + 1) for _ in range(m + 1)]
	for i in range(1, m+1):
		for j in range(1, n+1):
			if x[i-1] == y[j-1]:
				L[i][j] = L[i-1][j-1] + 1
			else:
				L[i][j] = max(L[i][j-1], L[i-1][j])
				
	return L[m][n] / min(len(x), len(y))

def get_average_color_difference(sorted_hex_colors):
	diffs = 0
	for sorted_hex_color in sorted_hex_colors:
		palette = ColorPalette(auto_fetched=False, palette_length=len(sorted_hex_colors), colors=sorted_hex_color)
		diffs += palette.get_graph_length_in_order([i for i in range(len(sorted_hex_color))]) / (len(sorted_hex_color)-1)
	
	return diffs / len(sorted_hex_colors)


def get_max_color_difference(sorted_hex_colors):
	max_diffs = []
	for sorted_hex_color in sorted_hex_colors:
		palette = ColorPalette(auto_fetched=False, palette_length=len(sorted_hex_colors), colors=sorted_hex_color)
		length_list = palette.get_graph_length_list_in_order([i for i in range(len(sorted_hex_color))])
		max_diffs.append(max(length_list))
		
	return sum(max_diffs) / len(max_diffs)
