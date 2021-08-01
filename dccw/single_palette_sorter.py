import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import math
import numpy as np
import enum
import time
from operator import itemgetter
from random import randrange

import elkai
from sko.GA import GA_TSP
from sko.SA import SA_TSP
from sko.ACA import ACA_TSP

from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import LabColor

from PyNomaly import loop

from dccw.mlrose.fitness import *
from dccw.mlrose.algorithms import *
from dccw.mlrose.opt_probs import *
from dccw.geo_sorter_helper import *

class SinglePaletteSorter:
	def __init__(self, palette, color_space=None):
		self.palette = palette
		self.ccounts = palette.length()
		self.color_space = color_space if color_space else 'lab' 

	#================================================================================
	# Sort Functions
	#================================================================================
	def standard_sort(self, merge_cut_type=MergeCutType.Without_Cutting):
		sorted_result = {}

		ldm = LabDistanceMode.CIEDE2000
		tsm = TSPSolverMode.LKH

		sorted_indices = self._tsp_sort(ldm, tsm, merge_cut_type)

		return sorted_indices


	def sort(self, spsm, merge_cut_type=MergeCutType.Without_Cutting):
		sorted_indices = {}
		start_time = time.time()

		if spsm == SinglePaletteSortMode.Luminance:
			sorted_indices = self._lex_sort('lab')

		elif spsm == SinglePaletteSortMode.HSV:
			sorted_indices = self._lex_sort('hsv')

		elif spsm in [SinglePaletteSortMode.NN_Euclidean,
						SinglePaletteSortMode.NN_Euclidean , 
						SinglePaletteSortMode.NN_CIEDE2000 , 
						SinglePaletteSortMode.GA_Euclidean , 
						SinglePaletteSortMode.GA_CIEDE2000 , 
						SinglePaletteSortMode.FIA_Euclidean, 
						SinglePaletteSortMode.FIA_CIEDE2000, 
						SinglePaletteSortMode.SA_Euclidean , 
						SinglePaletteSortMode.SA_CIEDE2000 , 
						SinglePaletteSortMode.ACO_Euclidean, 
						SinglePaletteSortMode.ACO50_CIEDE2000, 
						SinglePaletteSortMode.ACO10_CIEDE2000, 
						SinglePaletteSortMode.ACO5_CIEDE2000, 
						SinglePaletteSortMode.ACO2_CIEDE2000, 
						SinglePaletteSortMode.LKH_Euclidean, 
						SinglePaletteSortMode.LKH_CIEDE2000]:
			sorted_indices = self._tsp_sort(spsm.value['lab_distance'], spsm.value['tsp_solver'], merge_cut_type)
		
		else:
			assert False, '[sort] No such single palette sort mode'
		
		elapsed_time = time.time() - start_time

		# print(spsm.name, sorted_indices)
		return sorted_indices, elapsed_time


	#================================================================================
	# Each Sorting Method
	#================================================================================
	def lex_sort(self):
		return self._lex_sort(self.color_space)
		
	def _lex_sort(self, color_space):
		color_values_orig = self.palette.get_values(color_space, is_geo=False)
		sorted_result = sorted(range(self.ccounts), key=lambda k: (color_values_orig[k][0], color_values_orig[k][1]))

		return sorted_result

	def _tsp_sort(self, lab_distance_mode, tsp_solver_mode, merge_cut_type=MergeCutType.Without_Cutting):
		# 1. Distance Matrix
		distance_matrix = self._get_distance_matrix(lab_distance_mode)

		# 2. TSP
		best_sate = []

		if tsp_solver_mode == TSPSolverMode.LKH:
			enlarged_distance_matrix = [[round(e*10000) for e in dm] for dm in distance_matrix]
			best_state = elkai.solve_int_matrix(enlarged_distance_matrix, 100)

		elif tsp_solver_mode == TSPSolverMode.NN:
			best_state = self._nearest_neighbor_tsp(distance_matrix)

		elif tsp_solver_mode == TSPSolverMode.GA:
			best_state = self._genetic_tsp(distance_matrix)

		elif tsp_solver_mode == TSPSolverMode.FIA:
			best_state = self._farthest_insertion_tsp(distance_matrix)

		elif tsp_solver_mode == TSPSolverMode.SA:
			best_state = self._simulated_annealing_tsp(distance_matrix)

		elif tsp_solver_mode == TSPSolverMode.ACO:
			best_state = self._ant_colony_tsp(distance_matrix)
		
		elif tsp_solver_mode == TSPSolverMode.ACO50:
			best_state = self._ant_colony_tsp(distance_matrix, 50)
		
		elif tsp_solver_mode == TSPSolverMode.ACO10:
			best_state = self._ant_colony_tsp(distance_matrix, 10)

		elif tsp_solver_mode == TSPSolverMode.ACO5:
			best_state = self._ant_colony_tsp(distance_matrix, 5)

		elif tsp_solver_mode == TSPSolverMode.ACO2:
			best_state = self._ant_colony_tsp(distance_matrix, 2)

		else:
			assert False, '[_tsp_sort] No such tsp solver mode'

		# 3. Select Initial Node
		reordered_best_state = None
		direction, initial_node_index = self._TSP_graph_cut(best_state, distance_matrix, merge_cut_type)

		if direction == GraphCutDirection.Forward:
			reordered_best_state = best_state

		elif direction == GraphCutDirection.Reverse:
			best_state.reverse()
			reordered_best_state = best_state

		else:
			assert False, '[_tsp_sort] No such graph cut direction'
		
		reordered_best_state = reordered_best_state[reordered_best_state.index(initial_node_index):] + reordered_best_state[:reordered_best_state.index(initial_node_index)]

		color_values = self.palette.get_values(self.color_space, is_geo=True)
		sorted_result = reordered_best_state

		return sorted_result


	#================================================================================
	# TSP Functions
	#================================================================================
	def _nearest_neighbor_tsp(self, distance_matrix):
		distances = []
		tours = []
		num_points = self.palette.length()

		for initial_color_index in range(self.palette.length()):
			tour = [initial_color_index]

			while len(tour) < self.palette.length():
				closest_neighbor_index = self._find_closest_neighbor_index(distance_matrix, tour)
				tour.append(closest_neighbor_index)
			
			distance = sum([distance_matrix[tour[i%num_points]][tour[(i+1)%num_points]] for i in range(num_points)])

			distances.append(distance)
			tours.append(tour)

		best_tour_index = distances.index(min(distances))
		return tours[best_tour_index]

	def _genetic_tsp(self, distance_matrix):
		def _genetic_total_distance(order):
			num_points = self.palette.length()
			return sum([distance_matrix[order[i%num_points]][order[(i+1)%num_points]] for i in range(num_points)])

		ga_tsp = GA_TSP(func=_genetic_total_distance, n_dim=self.palette.length(), size_pop=50, max_iter=100, prob_mut=1)
		best_state, best_distance = ga_tsp.run()
		return best_state.tolist()


	def _simulated_annealing_tsp(self, distance_matrix):
		def _simulated_annealing_total_distance(order):
			num_points = self.palette.length()
			return sum([distance_matrix[order[i%num_points]][order[(i+1)%num_points]] for i in range(num_points)])

		T_max = 500
		T_min = 80
		iteration_per_temperature = 100
		# cooling factor = 0.9 in SKO

		sa_tsp = SA_TSP(func=_simulated_annealing_total_distance, x0=range(self.palette.length()), T_max=T_max, T_min=T_min, L=iteration_per_temperature)

		best_state, best_distance = sa_tsp.run()
		return best_state.tolist()


	def _ant_colony_tsp(self, distance_matrix, ant_count = 50):
		def _ant_colony_total_distance(order):
			num_points = self.palette.length()
			return sum([distance_matrix[order[i%num_points]][order[(i+1)%num_points]] for i in range(num_points)])

		aca = ACA_TSP(func=_ant_colony_total_distance, n_dim=self.palette.length(), size_pop=ant_count, max_iter=100, distance_matrix=distance_matrix)

		best_state, best_y = aca.run()
		return best_state.tolist()

	def _farthest_insertion_tsp(self, distance_matrix):
		distances = []
		tours = []
		num_points = self.palette.length()

		for initial_color_index in range(self.palette.length()):
			tour = [initial_color_index]
			while len(tour) < self.palette.length():
				farthest_neighbor_index = self._find_farthest_neighbor_index(distance_matrix, tour)
				closest_edge_index  = self._find_closest_edge(distance_matrix, tour, farthest_neighbor_index)
				tour.insert(closest_edge_index + 1, farthest_neighbor_index)
			
			distance = sum([distance_matrix[tour[i%num_points]][tour[(i+1)%num_points]] for i in range(num_points)])

			distances.append(distance)
			tours.append(tour)

		best_tour_index = distances.index(min(distances))
		return tours[best_tour_index]

	def _find_farthest_neighbor_index(self, distance_matrix, tour):
		max_distance = -math.inf
		max_index = -1

		for color_index in tour:
			distances = [(index, distance) for index, distance in enumerate(distance_matrix[color_index]) if index not in tour]
			cur_max_index, cur_max_distance = max(distances, key=itemgetter(1))

			if cur_max_distance > max_distance:
				max_distance = cur_max_distance
				max_index = cur_max_index

		return max_index

	def _find_closest_neighbor_index(self, distance_matrix, tour):
		min_distance = math.inf
		min_index = -1

		for color_index in tour:
			distances = [(index, distance) for index, distance in enumerate(distance_matrix[color_index]) if index not in tour]
			cur_min_index, cur_min_distance = min(distances, key=itemgetter(1))

			if cur_min_distance < min_distance:
				min_distance = cur_min_distance
				min_index = cur_min_index

		return min_index

	def _find_closest_edge(self, distance_matrix, tour, farthest_neighbor_index):
		min_sub_tour_distance = math.inf
		min_sub_tour_index = -1

		for current_tour_index in range(len(tour)):
			next_tour_index = (current_tour_index+1)%len(tour)

			new_sub_tour_distance = distance_matrix[tour[current_tour_index]][farthest_neighbor_index] + distance_matrix[farthest_neighbor_index][tour[next_tour_index]]
			if min_sub_tour_distance > new_sub_tour_distance:
				min_sub_tour_distance = new_sub_tour_distance
				min_sub_tour_index = current_tour_index

		return min_sub_tour_index


	#================================================================================
	# TSP Helper Functions
	#================================================================================

	def _TSP_graph_cut(self, TSP_graph_indices, distance_matrix, merge_cut_type):
		# find farthest pair
		start_index = None
		end_index = None
		
		if merge_cut_type == MergeCutType.Without_Cutting:
			start_index, end_index = self._TSP_cut_whole_largest(TSP_graph_indices, distance_matrix)

		elif merge_cut_type == MergeCutType.With_Cutting:
			start_index, end_index = self._TSP_cut_palette_largest(TSP_graph_indices, distance_matrix, merge_cut_type)

		else:
			assert False, '[_TSP_graph_cut] No such merge cut type'

		# direction decision: compare two tips
		# Regardless of the direction, initial_node_index indicates the index of first element in the sorted result.
		direction = None
		initial_node_index = None

		lightness_costs = self._get_lightness_costs()
		if lightness_costs[start_index] >= lightness_costs[end_index]:
			initial_node_index = start_index
			direction = GraphCutDirection.Reverse
		else:
			initial_node_index = end_index
			direction = GraphCutDirection.Forward

		return direction, initial_node_index


	def _TSP_cut_whole_largest(self, TSP_graph_indices, distance_matrix):
		max_distance = - math.inf
		start_index = None
		end_index = None

		neighbor_pairs_indices = [((i), (i + 1) % len(TSP_graph_indices)) for i in range(len(TSP_graph_indices))] 
		neighbor_pairs = [(TSP_graph_indices[i], TSP_graph_indices[j]) for i, j in neighbor_pairs_indices]

		for neighbor_pair_start, neighbor_pair_end  in neighbor_pairs:
			distance = distance_matrix[neighbor_pair_start][neighbor_pair_end]
			if max_distance < distance:
				max_distance = distance
				start_index = neighbor_pair_start
				end_index = neighbor_pair_end

		return start_index, end_index


	def _TSP_cut_palette_largest(self, TSP_graph_indices, distance_matrix, merge_cut_type):
		# When we should cut between A and B located in forward direction, then A becomes start_idnex and B does end_index.
		start_index = None
		end_index = None
		
		largest_palette_start, largest_palette_end = self._find_largest_palette_indices(TSP_graph_indices, distance_matrix)
		# Decide the target: inner or outer colors
		smaller_boundary = min(TSP_graph_indices.index(largest_palette_start), TSP_graph_indices.index(largest_palette_end))
		larger_boundary = max(TSP_graph_indices.index(largest_palette_start), TSP_graph_indices.index(largest_palette_end))

		is_outer, target_colors = self._find_colors_inside_boundary(TSP_graph_indices, larger_boundary, smaller_boundary)

		if len(target_colors) == 0:
			start_index = largest_palette_start
			end_index = largest_palette_end

		else:
			if is_outer:
				start_index, end_index = self._find_outer_case_start_end_indices(TSP_graph_indices, target_colors, distance_matrix, larger_boundary, smaller_boundary, merge_cut_type)
			else:
				start_index, end_index = self._find_inner_case_start_end_indices(TSP_graph_indices, target_colors, distance_matrix, larger_boundary, smaller_boundary, merge_cut_type)
		
		return start_index, end_index


	def _find_largest_palette_indices(self, TSP_graph_indices, distance_matrix):
		max_distance = - math.inf
		original_palette_group = self.palette.get_original_palette_group()
		largest_palette_start = None
		largest_palette_end = None
		
		for group_no in range(max(original_palette_group)+1):
			mask_list = [original_palette_group[k] == group_no for k in TSP_graph_indices]
			filtered_list = [i for (i, v) in zip(TSP_graph_indices, mask_list) if v]

			neighbor_pairs_indices = [((i), (i + 1) % len(filtered_list)) for i in range(len(filtered_list))] 
			neighbor_pairs = [(filtered_list[i], filtered_list[j]) for i, j in neighbor_pairs_indices]

			for neighbor_pair_start, neighbor_pair_end  in neighbor_pairs:
				distance = distance_matrix[neighbor_pair_start][neighbor_pair_end]
				if max_distance < distance:
					max_distance = distance
					largest_palette_start = neighbor_pair_start
					largest_palette_end = neighbor_pair_end
					
		return largest_palette_start, largest_palette_end


	def _find_colors_inside_boundary(self, TSP_graph_indices, larger_boundary, smaller_boundary):
		is_outer = False
		target_colors = []
		if larger_boundary - smaller_boundary > self.palette.length() * 0.5:
			# outer colors: [, smaller_boundary-1] and [larger_boundary+1, ]
			is_outer = True

			for i in range(smaller_boundary):
				target_colors.append(TSP_graph_indices[i])

			for i in range(larger_boundary+1, self.palette.length()):
				target_colors.append(TSP_graph_indices[i])

		else:
			# inner colors: [smaller_boundary+1, larger_boundary-1]
			for i in range(smaller_boundary+1, larger_boundary):
				target_colors.append(TSP_graph_indices[i])

		return is_outer, target_colors

	def _find_nearest_in_the_same_palette_index(self, TSP_graph_indices, direction, target_index):
		# direction: 1 (forward), -1 (backward)
		group_count = max(self.palette.get_original_palette_group()) + 1
		palette_length = len(TSP_graph_indices) / group_count
		target_group_no = target_index // palette_length

		c = TSP_graph_indices.index(target_index)
		for i in range(1, len(TSP_graph_indices)):
			c_ = c + i*direction
			if c_ < 0:
				c_ += len(TSP_graph_indices)
			elif c_ >= len(TSP_graph_indices):
				c_ -= len(TSP_graph_indices)

			# check whether involved in same group
			if TSP_graph_indices[c_] // palette_length == target_group_no:
				return TSP_graph_indices[c_]
		
		assert False, '[SPS] there is no same group color'
		

	def _find_outer_case_start_end_indices(self, TSP_graph_indices, target_colors, distance_matrix, larger_boundary, smaller_boundary, merge_cut_type):
		larger_index = TSP_graph_indices[larger_boundary]
		smaller_index = TSP_graph_indices[smaller_boundary]
		start_index = target_colors[-1]
		end_index = smaller_index

		if merge_cut_type == MergeCutType.With_Cutting:
			distance_from_smaller_boundary = []
			distance_from_larger_boundary = []

			for i in range(len(target_colors)):
				cur_index = TSP_graph_indices[(larger_boundary + 1 + i) % len(TSP_graph_indices)]
				distance_from_smaller_boundary.append(distance_matrix[smaller_index][cur_index])
				distance_from_larger_boundary.append(distance_matrix[larger_index][cur_index])

			min_distance_sum = math.inf			
			for i in range(len(distance_from_smaller_boundary)+1):
				cur_distance_sum = sum(distance_from_larger_boundary[:i]) + sum(distance_from_smaller_boundary[i:])

				if min_distance_sum > cur_distance_sum:
					start_index = TSP_graph_indices[(larger_boundary + i) % len(TSP_graph_indices)]
					end_index = TSP_graph_indices[(larger_boundary + i + 1) % len(TSP_graph_indices)]
					min_distance_sum = cur_distance_sum

		else:
			assert False, '[SPS] Unsupported MergeCutType'
			
		return start_index, end_index


	def _find_inner_case_start_end_indices(self, TSP_graph_indices, target_colors, distance_matrix, larger_boundary, smaller_boundary, merge_cut_type):
		larger_index = TSP_graph_indices[larger_boundary]
		smaller_index = TSP_graph_indices[smaller_boundary]
		start_index = target_colors[-1]
		end_index = larger_index

		if merge_cut_type == MergeCutType.With_Cutting:
			distance_from_smaller_boundary = []
			distance_from_larger_boundary = []

			for i in range(smaller_boundary + 1, larger_boundary):
				cur_index = TSP_graph_indices[i]
				distance_from_smaller_boundary.append(distance_matrix[smaller_index][cur_index])
				distance_from_larger_boundary.append(distance_matrix[larger_index][cur_index])

			min_distance_sum = math.inf			
			for i in range(len(distance_from_smaller_boundary)+1):
				cur_distance_sum = sum(distance_from_smaller_boundary[:i]) + sum(distance_from_larger_boundary[i:])

				if min_distance_sum > cur_distance_sum:
					start_index = TSP_graph_indices[smaller_boundary + i]
					end_index = TSP_graph_indices[smaller_boundary + i + 1]
					min_distance_sum = cur_distance_sum

		else:
			assert False, '[SPS] Unsupported MergeCutType'

		return start_index, end_index


	#================================================================================
	# Helper
	#================================================================================

	def _get_tag(self, lab_distance_mode, merge_cut_type):
		tag = "ldm-%s_mct-%s" % (lab_distance_mode.name, merge_cut_type.name)
		return tag

	def _get_distance_matrix(self, lab_distance_mode, is_geo=False):
		color_list = None 
		if lab_distance_mode == LabDistanceMode.CIEDE2000:
			color_list = self.palette.get_color_objects('lab')
		elif lab_distance_mode == LabDistanceMode.Euclidean:
			color_list = self.palette.get_values(self.color_space, is_geo=is_geo)
		else:
			assert False, '[_get_distance_matrix] No such lab distance mode'

		distance_matrix = []
		for color_a in color_list:
			sub_distance_matrix = []

			for color_b in color_list:
				distance = self._get_distance_between_two_colors(lab_distance_mode, color_a, color_b)
				sub_distance_matrix.append(distance)

			distance_matrix.append(sub_distance_matrix)

		return distance_matrix


	def _get_distance_between_two_colors(self, distance_mode, a, b):
		if distance_mode == LabDistanceMode.Euclidean:
			return self._get_euclidean_distance_between_two_coords(a, b)
		elif distance_mode == LabDistanceMode.CIEDE2000:
			return self._get_CIEDE2000_distance_between_two_labs(a, b)
		else:
			assert False, '[_get_distance_between_two_colors] No such lab distance mode'			

	def _get_euclidean_distance_between_two_coords(self, a, b):
		return math.sqrt(sum([(x - y) ** 2 for x, y in zip (a, b)]))


	def _get_CIEDE2000_distance_between_two_labs(self, lab1, lab2):
		return delta_e_cie2000(lab1, lab2)


	def _get_lightness_costs(self):
		# White has the lowest value 0 and black has the highest value 1
		lightness_costs = []
		geo_coords = self.palette.get_values('lab', is_geo=True)
		lightness_costs = [0.5 - geo_coord[2] for geo_coord in geo_coords]

		return lightness_costs



















