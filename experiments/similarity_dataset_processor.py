import os
import csv
import json
import random
import sys
import re
sys.path.append(".")
sys.path.append("..")

from colormath.color_objects import sRGBColor

from dccw.color_palette import *

from experiments.experiments_enums import *
from experiments.similarity_dataset import *


class SimilarityDatasetProcessor:
	def __init__(self, target_dataset, use_saved_database):
		self.dataset = None
		self._preprocess(target_dataset, use_saved_database)

	def get_dataset(self):
		return self.dataset

	def _preprocess(self, target_dataset, use_saved_database):
		if use_saved_database:
			query_palettes, retrieval_palettes, swatches = self._import(target_dataset)
			self.dataset = SimilarityDataset(target_dataset.name, query_palettes, retrieval_palettes, swatches, save_images=False)
		else:
			query_palettes, retrieval_palettes, swatches = self._parse(target_dataset)
			self.dataset = SimilarityDataset(target_dataset.name, query_palettes, retrieval_palettes, swatches, save_images=True)
			self._export(target_dataset.name.split('-')[0], target_dataset.name, query_palettes, retrieval_palettes, swatches)
		
# ====================================================================
# ====================================================================

	def _import(self, target_dataset):
		query_palettes = None
		retrieval_palettes = None
		swatches = None

		if target_dataset in LHSPType:
			query_palettes, retrieval_palettes, swatches = self._import_lin_and_hanrahan( target_dataset.name)

		else:
			assert False, '[_import] No such target dataset'

		return query_palettes, retrieval_palettes, swatches
	

	def _parse(self, target_dataset):
		query_palettes = None
		retrieval_palettes = None
		swatches = None

		if target_dataset in LHSPType:
			palette_length, random_count, jitter_offset = self._get_palette_length_and_replacement_count_of_dataset(target_dataset)

			if (random_count > 0 or jitter_offset > 0) and self._does_saved_dataset_exist(target_dataset.name.split('-')[0], target_dataset.name):
				# use saved base dataset
				query_palettes, retrieval_palettes, swatches = self._reuse_lin_and_hanrahan(target_dataset, palette_length, random_count, jitter_offset)
			else:
				query_palettes, retrieval_palettes, swatches = self._parse_lin_and_hanrahan(palette_length, random_count, jitter_offset)

		else:
			assert False, '[_parse] No such target dataset'

		return query_palettes, retrieval_palettes, swatches
	
	def _export(self, target_dataset_type, target_dataset_name, query_palettes, retrieval_palettes, swatches):
		self._export_list_to_csv(target_dataset_type, target_dataset_name, query_palettes, LHSPPaletteType.QueryPalettes)
		self._export_list_to_csv(target_dataset_type, target_dataset_name, retrieval_palettes, LHSPPaletteType.TargetPalettes)
		self._export_list_to_csv(target_dataset_type, target_dataset_name, swatches, LHSPPaletteType.Swatches)
		
# ====================================================================
# ====================================================================

	def _import_lin_and_hanrahan(self, target_dataset_name):
		# csv file structure
		# source, numcolors, palette_hex_string
		target_dataset_type = 'LHSP'

		query_palettes_path = self._get_csv_path_of_data(target_dataset_type, target_dataset_name, LHSPPaletteType.QueryPalettes)
		retrieval_palettes_path = self._get_csv_path_of_data(target_dataset_type, target_dataset_name, LHSPPaletteType.TargetPalettes)
		swatches_path = self._get_csv_path_of_data(target_dataset_type, target_dataset_name, LHSPPaletteType.Swatches)

		query_palettes = self._import_csv_info_to_list(query_palettes_path)
		retrieval_palettes = self._import_csv_info_to_list(retrieval_palettes_path)
		swatches = self._import_csv_info_to_list(swatches_path)

		return query_palettes, retrieval_palettes, swatches


	def _import_csv_info_to_list(self, csv_path):
		csv_contents = []
		
		if os.path.exists(csv_path):
			with open(csv_path, 'r') as csv_file:
				reader = csv.reader(csv_file, delimiter='\t')
				for row in reader:
					source = row[0]
					numcolors = int(row[1])
					palette_hex_colors = row[2:]
					palette = ColorPalette(auto_fetched=False, colors=palette_hex_colors)
					csv_contents.append([source, numcolors, palette])

		else:
			assert False, '[_import_csv_info_to_list] csv does not exist'
		
		return csv_contents

	def _export_list_to_csv(self, target_dataset_type, target_dataset_name, data_list, data_list_type):
		csv_base_path = os.path.join('experiments', 'DCCW-dataset', target_dataset_type, target_dataset_name)
		os.makedirs(csv_base_path, exist_ok=True)

		csv_path = self._get_csv_path_of_data(target_dataset_type, target_dataset_name, data_list_type)
		try:
			with open(csv_path, 'w') as csv_file:
				writer = csv.writer(csv_file, delimiter='\t')
				for data in data_list:
					source = data[0]
					numcolors = data[1]
					palette_hex_string = data[2].to_hex_list()
					writer.writerow([source, numcolors] + palette_hex_string)
		except IOError:
			print("I/O error")

	def _get_csv_path_of_data(self, dataset_type, dataset_name, data_list_type):
		path = None
		dataset_name = dataset_name.replace('_', '-')
		
		if data_list_type == LHSPPaletteType.QueryPalettes:
			data_list_type_name = 'query-palettes'
			path = os.path.join('experiments', 'DCCW-dataset', dataset_type, dataset_name, '%s.csv' % (data_list_type_name))

		elif data_list_type == LHSPPaletteType.TargetPalettes:
			data_list_type_name = 'retrieval-palettes'
			path = os.path.join('experiments', 'DCCW-dataset', dataset_type, dataset_name, '%s.csv' % (data_list_type_name))

		elif data_list_type == LHSPPaletteType.Swatches:
			path = os.path.join('experiments', 'DCCW-dataset', dataset_type, 'swatches', 'swatches.csv')
		else:
			assert False, '[_get_csv_path_of_data] No such data list type'

		return path

	def _does_saved_dataset_exist(self, dataset_type, dataset_name):
		path = os.path.join('experiments', 'DCCW-dataset', dataset_type, dataset_name.replace('_', '-'))
		return os.path.isdir(path)
		

# ====================================================================
# ====================================================================

	def _reuse_lin_and_hanrahan(self, target_dataset, palette_length, random_count, jitter_offset):
		query_palettes, retrieval_palettes, swatches = self._import_lin_and_hanrahan(target_dataset.name)

		query_palettes = self._generate_LHSP_replacement_and_jitter(query_palettes, random_count, jitter_offset)
		retrieval_palettes = self._generate_LHSP_replacement_and_jitter(retrieval_palettes, random_count, jitter_offset)
		
		return query_palettes, retrieval_palettes, swatches


	def _parse_lin_and_hanrahan(self, palette_length, random_count, jitter_offset):
		query_palettes = None # list of [source, numcolors, palette]
		retrieval_palettes = None # list of [source, numcolors, palette]

		# 1. read swatches
		swatches = self._read_LHSP_swatches_json()

		# 2. read query and targer palettes
		if palette_length == 5:
			query_palettes = self._generate_LHSP_5_query_palettes(random_count, jitter_offset)
			retrieval_palettes = self._generate_LHSP_5_retrieval_palettes(random_count, jitter_offset)

		elif palette_length > 5:
			query_palettes = self._generate_LHSP_query_palettes(palette_length, random_count, jitter_offset)
			retrieval_palettes = self._generate_LHSP_retrieval_palettes(swatches, palette_length, random_count, jitter_offset)
		else:
			assert False, '[_parse_lin_and_hanrahan] Unsupported palette length'
		
		return query_palettes, retrieval_palettes, swatches
		

	def _generate_LHSP_5_query_palettes(self, random_count, jitter_offset):
		# source: optimized
		query_palettes = self._read_LHSP_single_tsv('optimized.tsv')
		query_palettes =  self._generate_LHSP_replacement_and_jitter(query_palettes, random_count, jitter_offset)

		return query_palettes

	def _generate_LHSP_5_retrieval_palettes(self, random_count, jitter_offset):
		# source: turk
		retrieval_palettes = self._read_LHSP_single_tsv('turk.tsv')
		retrieval_palettes =  self._generate_LHSP_replacement_and_jitter(retrieval_palettes, random_count, jitter_offset)

		return retrieval_palettes


	def _generate_LHSP_query_palettes(self, palette_length, random_count, jitter_offset):
		# source: art(11), cmeans(1), kmeans(1), optimized(1), oracle-art(1), oracle-turk(1)
		# kmeans + art3 / cmeans + art3 / optimized + art3 / oracle-art + art3 / oracle-turk + art3

		tsv_file_names = ['art.tsv', 'cmeans.tsv', 'kmeans.tsv', 'optimized.tsv', 'oracle-art.tsv', 'oracle-turk.tsv', 'random.tsv']
		palettes_list = {}
		for tsv_file_name in tsv_file_names:
			tsv_category = tsv_file_name.split('.')[0]
			palettes_list[tsv_category] = self._read_LHSP_single_tsv(tsv_file_name)

		permitted_sources = ['chikanobu2.png', 'homer1.png', 'macke1.png', 'monet1.png', 'photo_joaquin_rosado.png', 'photo_krikit.png', 'photo_powi.png', 'photo_radiofreebarton.png', 'photo_ssdginteriors.png', 'seurat1.png']

		merged_palettes = []
		for permitted_source in permitted_sources:
			source_name = permitted_source.split('.')[0]

			random_hex_colors = self._get_hex_list_of_palettes_with_source_name(palettes_list['random'], source_name, False)
			cur_random_hex_color_index = 0
			
			# kmeans + random3
			merged_color_object, cur_random_hex_color_index = self._get_merged_palettes_for_LHSP_query_palettes(palettes_list['kmeans'], source_name, random_hex_colors, cur_random_hex_color_index, palette_length)
			merged_palettes.append([source_name, palette_length, merged_color_object])

			# optimized + random3
			merged_color_object, cur_random_hex_color_index = self._get_merged_palettes_for_LHSP_query_palettes(palettes_list['optimized'], source_name, random_hex_colors, cur_random_hex_color_index, palette_length)
			merged_palettes.append([source_name, palette_length, merged_color_object])

			# oracle-art + random3
			merged_color_object, cur_random_hex_color_index = self._get_merged_palettes_for_LHSP_query_palettes(palettes_list['oracle-art'], source_name, random_hex_colors, cur_random_hex_color_index, palette_length)
			merged_palettes.append([source_name, palette_length, merged_color_object])

			# oracle-turk + random3
			merged_color_object, cur_random_hex_color_index = self._get_merged_palettes_for_LHSP_query_palettes(palettes_list['oracle-turk'], source_name, random_hex_colors, cur_random_hex_color_index, palette_length)
			merged_palettes.append([source_name, palette_length, merged_color_object])

			# cmeans + random3
			merged_color_object, cur_random_hex_color_index = self._get_merged_palettes_for_LHSP_query_palettes(palettes_list['cmeans'], source_name, random_hex_colors, cur_random_hex_color_index, palette_length)
			merged_palettes.append([source_name, palette_length, merged_color_object])
		
		return self._generate_LHSP_replacement_and_jitter(merged_palettes, random_count, jitter_offset)

	def _get_merged_palettes_for_LHSP_query_palettes(self, palettes_list, source_name, random_hex_colors, cur_random_hex_color_index, palette_length):
		merged_hex_colors = self._get_hex_list_of_palettes_with_source_name(palettes_list, source_name)
		added_count = 0
		while added_count < (palette_length - 5):
			cur_random_hex_colors = random_hex_colors[cur_random_hex_color_index % len(random_hex_colors)]
			for cur_random_hex_color in cur_random_hex_colors:
				if cur_random_hex_color not in merged_hex_colors:
					added_count += 1
					merged_hex_colors += [cur_random_hex_color]

				if added_count >= (palette_length - 5):
					break

			cur_random_hex_color_index += 1

		return ColorPalette(auto_fetched=False, colors=merged_hex_colors), cur_random_hex_color_index

	def _generate_LHSP_retrieval_palettes(self, swatches, palette_length, random_count, jitter_offset):
		# source: swatches
		source_set = set([source for source, _, _ in swatches])

		merged_palettes = []
		for source in source_set:
			same_source_palettes = self._get_hex_list_of_palettes_with_source_name(swatches, source, True)

			for i in range(10):
				# pick up the randaom 20 colors
				random.shuffle(same_source_palettes)
				merged_color_object = ColorPalette(auto_fetched=False, colors=same_source_palettes[:palette_length])
				merged_palettes.append([source, palette_length, merged_color_object])
				
		return self._generate_LHSP_replacement_and_jitter(merged_palettes, random_count, jitter_offset)

	def _generate_LHSP_replacement_and_jitter(self, palettes, random_count, jitter_offset):
		palettes = self._add_jitter_palettes(palettes, jitter_offset)
		return self._add_replacement_color_into_palettes(palettes, random_count)

	def _add_replacement_color_into_palettes(self, palettes, random_count):
		# palettes: [[source, numcolors, paletteo_bject], ...]
		if random_count == 0:
			return palettes

		for source, numcolors, palette in palettes:
			palette.change_partial_color_with_random(random_count)

		return palettes

	def _add_jitter_palettes(self, palettes, jitter_offset):
		# palettes: [[source, numcolors, paletteo_bject], ...]
		if jitter_offset == 0:
			return palettes

		for source, numcolors, palette in palettes:
			palette.jitter_colors(jitter_offset)

		return palettes

# ====================================================================
# ====================================================================

	def _read_LHSP_single_tsv(self, tsv_file_name):
		palettes = []
		dir_name = self._get_LHSP_dir_name()
		with open(os.path.join(dir_name, tsv_file_name)) as fd:
			rd = csv.DictReader(fd, delimiter="\t", quotechar='"')
			tsv_data = [x for x in rd]

			for tsv_row in tsv_data:
				cur_data = []
				source_name, _ = os.path.splitext(tsv_row['image'])
				cur_data.append(source_name) # source
				cur_data.append(tsv_row['numColors']) # numcolors

				hex_colors = self._convert_rgb_color_strings_to_hex_list(tsv_row['colors'])
				cur_data.append(ColorPalette(auto_fetched=False, colors=hex_colors)) # color 
				palettes.append(cur_data)
		
		return palettes

	def _read_LHSP_swatches_json(self):
		swatches = []
		swatches_json_dir_name = 'swatches'
		dir_name = self._get_LHSP_dir_name()

		dir_path = os.path.join(dir_name, swatches_json_dir_name)
		for file in os.listdir(dir_path):
			if not file.endswith('.json'):
				continue

			with open(os.path.join(dir_path, file)) as fd:
				json_list = json.load(fd)
				cur_data = []

				source_name, _ = os.path.splitext(file)
				cur_data.append(source_name) # source
				cur_data.append(len(json_list)) # numcolors
				
				rgb_strings = ''
				for color_dict in json_list:
					rgb_strings += '%d,%d,%d ' % (color_dict['r'], color_dict['g'], color_dict['b'])
				
				hex_colors = self._convert_rgb_color_strings_to_hex_list(rgb_strings)
				cur_data.append(ColorPalette(auto_fetched=False, colors=hex_colors)) #color palette
				swatches.append(cur_data)

		return swatches

# ====================================================================
# ====================================================================

	def _get_LHSP_dir_name(self):
		return os.path.join('experiments', 'original_dataset', 'LHSP')

	def _get_hex_list_of_palettes_with_source_name(self, palettes_list, source_name, make_flatten=True):
		hex_2d_list = [p_list[2].to_hex_list() for p_list in palettes_list if p_list[0] == source_name]
		if make_flatten:
			return self._flatten_2d_list(hex_2d_list)
		else:
			return hex_2d_list

	def _flatten_2d_list(self, target_list):
		# [[], [], []] -> [, , ,]
		return sum(target_list, [])

	def _convert_rgb_color_strings_to_hex_list(self, rgb_color_strings):
		# ex) rgb_color_strings: 147,151,90 193,116,88 253,120,186 90,99,98 28,24,21
		hex_list = []
		for rgb_string in rgb_color_strings.strip().split(' '):
			r, g, b = [int(i.strip()) for i in rgb_string.split(',')]
			rgb_object = sRGBColor(r, g, b, is_upscaled=True)
			hex_list.append(rgb_object.get_rgb_hex())
		
		return hex_list

	def _get_palette_length_and_replacement_count_of_dataset(self, dataset_type):

		dataset_string = dataset_type.name
		pattern_palette_length = re.compile('-k\d+')
		pattern_replacement_count = re.compile('-replacement\d+')
		pattern_jitter_offset = re.compile('-jitter\d+')

		searched_palette_length = pattern_palette_length.search(dataset_string)
		searched_replacement_count = pattern_replacement_count.search(dataset_string)
		searched_jitter_offset = pattern_jitter_offset.search(dataset_string)
		
		palette_length = 0
		random_count = 0
		jitter_offset = 0
		
		if searched_palette_length:
			palette_length = int(searched_palette_length.group()[2:])

		if searched_replacement_count:
			random_count = int(searched_replacement_count.group()[7:])

		if searched_jitter_offset:
			jitter_offset = int(searched_jitter_offset.group()[7:])

		return palette_length, random_count, jitter_offset
