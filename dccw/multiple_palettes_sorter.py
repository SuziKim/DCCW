import numpy as np 
import math
import time

from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import LabColor

from scipy.optimize import linear_sum_assignment

from dccw.geo_sorter_helper import *
from dccw.single_palette_sorter import * 
from dccw.color_palette import *

class MultiplePalettesSorter:
	def __init__(self, palettes, palette_count, lab_distance_mode):
		self.palettes = palettes # ColorPalettes Object
		self.palette_count = palette_count
		self.lab_distance_mode = lab_distance_mode
		
#================================================================================
# Sort Functions
#================================================================================

	def standard_sort(self):
		multipe_palette_sort_mode = MultiplePalettesSortMode.Merge_LKH
		merge_cut_type = MergeCutType.With_Cutting
				
		return self.sort(multipe_palette_sort_mode, merge_cut_type)
		

	def sort(self, multiple_palette_sort_mode, merge_cut_type=MergeCutType.With_Cutting):
		# return: {'merge': [[1,2,3,4,5],[1,2,3,4,5]], 'matching': [[1,2,3,4,5],[1,2,3,4,5]]}
		sorted_result = {}
		start_time = time.time()
		single_palette_standard_sort_mode = self._get_single_palette_standard_sort_mode()
		merged_sorted_indices = None

		if multiple_palette_sort_mode == MultiplePalettesSortMode.Separate_Luminance:
			sorted_result = self._sort_separate(SinglePaletteSortMode.Luminance)

		elif multiple_palette_sort_mode == MultiplePalettesSortMode.Separate_HSV:
			sorted_result = self._sort_separate(SinglePaletteSortMode.HSV)

		elif multiple_palette_sort_mode == MultiplePalettesSortMode.Separate_LKH:
			sorted_result = self._sort_separate(single_palette_standard_sort_mode)

		elif multiple_palette_sort_mode == MultiplePalettesSortMode.Merge_Luminance:
			sorted_result, _, merged_sorted_indices = self._sort_merge(SinglePaletteSortMode.Luminance, merge_cut_type)

		elif multiple_palette_sort_mode == MultiplePalettesSortMode.Merge_HSV:
			sorted_result, _, merged_sorted_indices = self._sort_merge(SinglePaletteSortMode.HSV, merge_cut_type)

		elif multiple_palette_sort_mode == MultiplePalettesSortMode.Merge_LKH:
			sorted_result, _, merged_sorted_indices = self._sort_merge(single_palette_standard_sort_mode, merge_cut_type)

		elif multiple_palette_sort_mode == MultiplePalettesSortMode.BPS:
			sorted_result = self._sort_bps(is_improved=False)

		elif multiple_palette_sort_mode == MultiplePalettesSortMode.Improved_BPS:
			sorted_result = self._sort_bps(is_improved=True)

		# elif multiple_palette_sort_mode == MultiplePalettesSortMode.MBME:
			# sorted_result = self._sort_minimum_bipartite_matching_error()

		else:
			assert False, '[sort] No such multipe palettes sort mode'
		
		elapsed_time = time.time() - start_time

		return sorted_result, elapsed_time, merged_sorted_indices

	
#================================================================================
# Each Sorting Method
#================================================================================
	def _sort_separate(self, single_palette_sort_mode):
		single_palettes = self.palettes.get_single_palettes_list()
		sorted_result = []

		for single_palette in single_palettes:
			single_palette_sorter = SinglePaletteSorter(single_palette)
			sorted_indices, _ = single_palette_sorter.sort(single_palette_sort_mode)
			sorted_result.append(sorted_indices)
		
		return sorted_result


	def _sort_bps(self, is_improved):
		original_palettes = self.palettes.get_single_palettes_list()
		sorted_palettes = self._sort_bps_list(original_palettes, is_improved=is_improved)
		
		sorted_result = []
		for original_palette, sorted_palette in zip(original_palettes, sorted_palettes):
			original_palette_hex = original_palette.to_hex_list()
			sorted_palette_hex = sorted_palette.to_hex_list()

			sorted_result.append([original_palette_hex.index(c) for c in sorted_palette_hex])

		return sorted_result


	def _sort_merge(self, single_palette_sort_mode, merge_cut_type):
		sorted_result = []

		# 1. Merge to single palette
		merged_palette = self.palettes.merge_to_single_palette()
		
		# 2. geo sort
		single_palette_sorter = SinglePaletteSorter(merged_palette)
		merged_sorted_indices, _ = single_palette_sorter.sort(single_palette_sort_mode, merge_cut_type)

		# 3. Re-distribute the result
		color_start_index = 0
		color_end_index = 0
		for p_index in range(self.palette_count):
			color_end_index += self.palettes.get_single_palettes_list()[p_index].length()
			sorted_result.append([i - color_start_index for i in merged_sorted_indices if i >= color_start_index and i < color_end_index])
			color_start_index = color_end_index

		merged_length = merged_palette.get_graph_length_in_order(merged_sorted_indices)

		return sorted_result, merged_length, merged_sorted_indices


	def _sort_minimum_bipartite_matching_error(self):
		single_palettes_list = self.palettes.get_single_palettes_list()
		sorted_result = []

		# sort basis palette
		basis_palette = single_palettes_list[0]
		single_palette_sorter = SinglePaletteSorter(basis_palette)

		single_palette_sort_mode = self._get_single_palette_standard_sort_mode()
		sorted_basis_indices, _ = single_palette_sorter.sort(single_palette_sort_mode)

		sorted_basis_hex_list = basis_palette.get_values_in_order('hex', sorted_basis_indices)
		sorted_basis_palette = ColorPalette(auto_fetched=False, palette_length=len(sorted_basis_hex_list), colors=sorted_basis_hex_list)
		sorted_result.append(sorted_basis_indices)

		# sort remaining palette 
		for target_palette in single_palettes_list[1:]:
			distance_matrix = self._get_distance_matrix_between_two_palettes(target_palette, sorted_basis_palette)
			
			row_ind, col_ind = linear_sum_assignment(distance_matrix)
			sorted_result.append(col_ind.tolist())
		
		return sorted_result

	def _get_single_palette_standard_sort_mode(self):
		if self.lab_distance_mode == LabDistanceMode.CIEDE2000:
			return SinglePaletteSortMode.LKH_CIEDE2000
		elif self.lab_distance_mode == LabDistanceMode.Euclidean:
			return SinglePaletteSortMode.LKH_Euclidean
		else:
			assert False, '[_get_single_palette_standard_sort_mode] No such lab distance mode'
		

#================================================================================
# BPS Helper
#================================================================================
	def _sort_bps_list(self, palettes_list, is_improved, is_left=False):
		if len(palettes_list) <= 1:
			if is_improved and is_left:
				palette = palettes_list[0]
				single_palette_sorter = SinglePaletteSorter(palette)
				single_palette_sort_mode = self._get_single_palette_standard_sort_mode()
				sorted_indices, _ = single_palette_sorter.sort(single_palette_sort_mode)
				hex_list = palette.get_values_in_order('hex', sorted_indices)
				palette_list = ColorPalette(auto_fetched=False, palette_length=palette.length(), colors=hex_list)
				palettes_list = [palette_list]
				
			return palettes_list

		mid = len(palettes_list) // 2
		left_list = self._sort_bps_list(palettes_list[:mid], is_improved=is_improved, is_left=True)
		right_list = self._sort_bps_list(palettes_list[mid:], is_improved=is_improved, is_left=False)
		return self._bps_merge(left_list, right_list)


	def _bps_merge(self, left_list, right_list):
		K = left_list[0].length()
		cost = np.zeros((K, K))
		for i in range(K):
			left_colors = [palette.get_color_objects('lab')[i] for palette in left_list]
			for j in range(K):
				right_colors = [palette.get_color_objects('lab')[j] for palette in right_list]
				cost[i, j] = self._get_hausdorff_distance(left_colors, right_colors)

		row_ind, col_ind = linear_sum_assignment(cost)
		sorted_right_hex_list = [palette.get_values_in_order('hex', col_ind) for palette in right_list]
		
		sorted_right_color_palettes = []
		for palette_hex in sorted_right_hex_list:
			color_palette = ColorPalette(auto_fetched=False, palette_length=K, colors=palette_hex)
			sorted_right_color_palettes.append(color_palette)

		return left_list + sorted_right_color_palettes


	def _get_hausdorff_distance(self, A, B):
		# A, B: list of lab values [[L, A, B], [L, A, B], ...]
		return max(self._get_hausdorff_distance_d_A_B(A, B), self._get_hausdorff_distance_d_A_B(B, A))


	def _get_hausdorff_distance_d_A_B(self, A, B):
		sum_of_min = 0
		for a in A:
			min_distance = math.inf
			for b in B:
				distance = self._get_distance_between_two_labs(a, b)
				if min_distance > distance:
					min_distance = distance
			sum_of_min += min_distance

		return sum_of_min / len(A)


#================================================================================
# Helper
#================================================================================

	def _get_distance_matrix_between_two_palettes(self, palette_a, palette_b):
		distance_matrix = []

		for lab_b in palette_b.get_color_objects('lab'):
			sub_distance_matrix = []
			for lab_a in palette_a.get_color_objects('lab'):
				distance = self._get_distance_between_two_labs(lab_a, lab_b)
				sub_distance_matrix.append(distance)
			distance_matrix.append(sub_distance_matrix)

		return distance_matrix


	def _get_distance_between_two_labs(self, lab_a, lab_b):
		use_euclidean = False
		if use_euclidean:
			return self._get_Euclidean_distance_between_two_labs(lab_a, lab_b)
		else:
			return self._get_CIEDE2000_distance_between_two_labs(lab_a, lab_b)

	def _get_CIEDE2000_distance_between_two_labs(self, lab_a, lab_b):
		return delta_e_cie2000(lab_a, lab_b)

	def _get_Euclidean_distance_between_two_labs(self, lab_a, lab_b):
		a = lab_a.get_value_tuple()
		b = lab_b.get_value_tuple()
		return math.sqrt(sum([(x - y) ** 2 for x, y in zip (a, b)]))

