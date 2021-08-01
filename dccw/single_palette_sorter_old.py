import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import math
import numpy as np
import enum

import elkai

from dccw.mlrose.fitness import *
from dccw.mlrose.algorithms import *
from dccw.mlrose.opt_probs import *


class InitialSeedMode(enum.Enum):
	WoLightness = 0
	WoIsolation = 1
	# Random = 2
	All = 3

class DistanceMode(enum.Enum):
	CIE2000 = 0
	Euclidean = 1

class LightnessMode(enum.Enum):
	UseLab = 0
	UseCustom = 1

class IsolationMode(enum.Enum):
	ClosestDistanceOnly = 0	# 
	ClosestHueDistanceOnly = 1
	TripleHueDistanceOnly = 2
	WeightedRank = 3
	HueOnlyWeightedRank = 4
	SquareWeightedRank = 5
	HueOnlySquareWeightedRank = 6

class BoundsMode(enum.Enum):
	AbsoluteBounds = 0
	GlobalRelativeBounds = 1
	NormalizedBounds = 2
	AverageBounds = 3

class SpiralMode(enum.Enum):
	NoPenalty = 0
	ReverseHuePenalty = 1 
	ReverseHueLightnessPenalty = 2

class AntiLightnessBonusMode(enum.Enum):
	NoBonus = 0
	AntiLightness = 1 # HSL, LCH, HSV only

class TSPMode(enum.Enum):
	SymmetricTSP = 0  # use elkai
	AsymmetricTSP = 1 # use mlrose

class ReverseMode(enum.Enum):
	FixedDirection = 0
	ClosestHueDirection = 1
	GlobalHueDirection = 2
	GlobalClosestHueDirection = 3
	ClosestDirection = 4

class SinglePaletteSorter:
	def __init__(self, palette, target_spaces):
		# palette: ColorPalette Object
		# target_space: ['rgb', 'hsl', 'hsv', 'lab', 'lch']

		self.palette = palette
		self.ccounts = palette.length()
		self.target_spaces = target_spaces

		self.colors = {}
		self.colors['hex'] = palette.get("hex")
		self.colors['lab'] = palette.get("lab")

		for ts in target_spaces:
			tag = 'lex_' + ts
			self.colors[tag] = palette.get(ts)

			tag = 'geo_' + ts
			self.colors[tag] = palette.get(ts, is_geo = True)
		

	def _lex_sort_of_colorspace(self, color_space):
		lex_color_space = 'lex_' + color_space
		geo_color_space = 'geo_' + color_space

		sorted_indices = sorted(range(self.ccounts), key=lambda k: (self.colors[lex_color_space][k][0], self.colors[lex_color_space][k][1]))
		
		sorted_result = {}
		sorted_result['hex'] = [self.colors['hex'][i] for i in sorted_indices]
		sorted_result['val_lex'] = [list(self.colors[lex_color_space][i]) for i in sorted_indices]
		sorted_result['val_geo'] = [list(self.colors[geo_color_space][i]) for i in sorted_indices]

		return sorted_result


	def lex_sort(self):
		"""
		Return Sorted palettes in lexicographic order.

		@rtype	:	[color_space]['hex', 'val'] ex) ['rgb']['hex', 'val']
					'hex': hex tuple, 'val': original value tuple (0-255)
		@return :	2D dictionary
		"""

		sorted_result = {}

		for cs in self.target_spaces:
			# print('[LexSort] ', cs)
			sorted_result[cs.upper()] = self._lex_sort_of_colorspace(cs)
		
		return sorted_result
	

	def geo_sort(self):
		"""
		Return Sorted palettes in geometric order.

		@rtype	:	[color_space]['hex', 'val'] ex) ['rgb']['hex', 'val']
					'hex': hex tuple, 'val': original value tuple (0-255)
		@return :	2D dictionary
		"""

		sorted_result = {}
		tsp_mode = TSPMode.AsymmetricTSP
		reverse_mode = ReverseMode.ClosestDirection
		antilightness_mode = AntiLightnessBonusMode.AntiLightness
		distance_mode = DistanceMode.CIE2000

		for initial_seed_mode in InitialSeedMode:
			distances = {}
			initial_node_indices = {}

			for cs in self.target_spaces:
				initial_node_indices[cs], distances[cs] = self._get_triples_of_colors(cs, initial_seed_mode, SpiralMode.ReverseHueLightnessPenalty, tsp_mode, reverse_mode, antilightness_mode, distance_mode)

			# distances (list of triples, default: None) – List giving the distances, d, between all pairs of nodes, u and v, for which travel is possible, with each list item in the form (u, v, d). Order of the nodes does not matter, so (u, v, d) and (v, u, d) are considered to be the same. If a pair is missing from the list, it is assumed that travel between the two nodes is not possible. 
			for k, d in distances.items():
				# print('[GeoSort] ', k, d)
				kIdx = k.upper()

				reordered_best_state = None
				best_state = None
				if tsp_mode is TSPMode.SymmetricTSP:
					fitness_dists = TravellingSales(distances = d)
					problem_fit = TSPOpt(length = self.ccounts, fitness_fn = fitness_dists, maximize=False)
					best_state, best_fitness = genetic_alg(problem_fit, mutation_prob = 0.2, max_attempts = 1, random_state = 2)

					best_state = best_state.tolist()

				elif tsp_mode is TSPMode.AsymmetricTSP:
					# best_state = elkai.solve_float_matrix(d)
					d_ = [[round(e*10000) for e in d_] for d_ in d]
					best_state = elkai.solve_int_matrix(d_, 100)

				else:
					error("[geo_sort] There isn't such TSPMode type!")
					return

				initial_index = initial_node_indices[k]
				reordered_best_state = best_state[best_state.index(initial_index):] + best_state[:best_state.index(initial_index)]

				# print('[GeoSort] Best state: ', reordered_best_state)
				# print('[GeoSort] Fitness: ', best_fitness)

				if kIdx not in sorted_result:
					sorted_result[kIdx] = {}

				sorted_result[kIdx][initial_seed_mode.name] = {}
				sorted_result[kIdx][initial_seed_mode.name]['hex'] = [self.colors['hex'][i] for i in reordered_best_state]
				sorted_result[kIdx][initial_seed_mode.name]['val'] = [list(self.colors['geo_' + k][i]) for i in reordered_best_state]

			# Alternatives of the randomized optimization and search algorithms: 
			# 1. hill_climb
			# 2. random_hill_climb
			# 3. simulated_annealing
			# 4. genetic_alg
			# 5. mimic

		return sorted_result


	def _get_lightness_cost(self, color_space, lightness_mode, distance_mode):
		"""
		Return lightness cost list of each color

		@rtype	:	list of float ranged in [0, 1]
		@return :	list
		"""

		if lightness_mode is LightnessMode.UseLab:
			bounds = 100
			lightness_costs = [1 - self.colors['lab'][i][0] / bounds for i in range(self.ccounts)]

		elif lightness_mode is LightnessMode.UseCustom:
			lightness_costs = []
			if color_space == 'geo_rgb':
				# distance between (-0.5, -0.5, -0.5)
				for u in range(self.ccounts):
					black_point = [-0.5, -0.5, -0.5]
					distance = self._get_distance_between_two_nodes(distance_mode, self.colors[color_space][u], black_point)
					bounds = math.sqrt(3)

					lightness_costs.append(1 - distance/bounds)

			else:
				# proportional to z value
				for u in range(self.ccounts):
					lightness_costs.append(0.5 - self.colors[color_space][u][2])

		else:
			error("[_get_lightness_cost] There isn't such LightnessMode type!")
			return

		return lightness_costs


	def _get_isolation_cost(self, color_space, isolation_mode, bounds_mode, distance_mode):
		"""
		Return isolation cost list of each color

		@rtype	:	list of float ranged in [0, 1]
		@return :	list
		"""

		isolation_costs = [] 
		absolute_bounds = 0
		if color_space == 'geo_rgb':
			# cube type
			absolute_bounds = math.sqrt(3)
		elif color_space == 'geo_hsl' or color_space == 'geo_hsv':
			# cylinder type
			absolute_bounds = math.sqrt(2)
		elif color_space == 'geo_lab':
			# sphere type
			absolute_bounds = 1
		elif color_space == 'geo_lch':
			# bi-cone type
			absolute_bounds = math.sqrt(2)
		elif color_space == 'geo_cielch':
			# cylinder type
			absolute_bounds = math.sqrt(2)
		else:
			error("[_get_initial_node] There isn't such color_space type!")
			return

		global_relative_bounds = 0
		for u in range(self.ccounts):
			# print(self.colors['lex_hsl'][u][0] / 360)

			for v in range(self.ccounts):
				cur = 0
				if (isolation_mode is IsolationMode.HueOnlyWeightedRank) or (isolation_mode is IsolationMode.ClosestHueDistanceOnly) or (isolation_mode is IsolationMode.TripleHueDistanceOnly):
					cur = self._get_hue_distance_between_two_nodes_index(u, v)
				else:
					cur = self._get_distance_between_two_nodes(distance_mode, self.colors[color_space][u], self.colors[color_space][v])

				if global_relative_bounds < cur:
					global_relative_bounds = cur

		average_bounds = []

		for u in range(self.ccounts):
			sub_isolation_costs = []
			for v in range(self.ccounts):
				if u == v:
					sub_isolation_costs.append(0)
				else:
					if (isolation_mode is IsolationMode.HueOnlyWeightedRank) or (isolation_mode is IsolationMode.ClosestHueDistanceOnly):
						distance = self._get_hue_distance_between_two_nodes_index(u, v)
						sub_isolation_costs.append(distance)
						average_bounds.append(distance)

					else:
						distance = self._get_distance_between_two_nodes(self.colors[color_space][u], self.colors[color_space][v])
						sub_isolation_costs.append(distance)
						average_bounds.append(distance)


			if (isolation_mode is IsolationMode.WeightedRank) or (isolation_mode is IsolationMode.HueOnlyWeightedRank):
				sorted_indices = sorted(range(self.ccounts), key=lambda k: sub_isolation_costs[k])
				
				weighted_costs_of_u = [(self.ccounts - i) * sub_isolation_costs[sorted_indices[i]] for i in range(0, self.ccounts)]

				bounds = 0
				whole_sum = self.ccounts * (self.ccounts + 1 ) / 2 # sum of k
				if bounds_mode is BoundsMode.AbsoluteBounds:
					bounds = absolute_bounds * whole_sum
					isolation_costs.append(sum(weighted_costs_of_u) / bounds)

				elif bounds_mode is BoundsMode.GlobalRelativeBounds:
					bounds = global_relative_bounds * whole_sum
					isolation_costs.append(sum(weighted_costs_of_u) / bounds)

				elif (bounds_mode is BoundsMode.NormalizedBounds) or (bounds_mode is BoundsMode.AverageBounds):
					isolation_costs.append(sum(weighted_costs_of_u))
				else:
					error("[_get_initial_node] There isn't such BoundsMode type!")
					return

			elif (isolation_mode is IsolationMode.SquareWeightedRank) or (isolation_mode is IsolationMode.HueOnlySquareWeightedRank):
				sorted_indices = sorted(range(self.ccounts), key=lambda k: sub_isolation_costs[k])
				
				weighted_costs_of_u = [(self.ccounts - i) * (self.ccounts - i) * sub_isolation_costs[sorted_indices[i]] for i in range(0, self.ccounts)]

				bounds = 0
				whole_sum = self.ccounts * (self.ccounts + 1) * (2 * self.ccounts + 1) / 6 # sum of k^2
				if bounds_mode is BoundsMode.AbsoluteBounds:
					bounds = absolute_bounds * whole_sum
					isolation_costs.append(sum(weighted_costs_of_u) / bounds)

				elif bounds_mode is BoundsMode.GlobalRelativeBounds:
					bounds = global_relative_bounds * whole_sum
					isolation_costs.append(sum(weighted_costs_of_u) / bounds)

				elif (bounds_mode is BoundsMode.NormalizedBounds) or (bounds_mode is BoundsMode.AverageBounds):
					isolation_costs.append(sum(weighted_costs_of_u))
				else:
					error("[_get_initial_node] There isn't such BoundsMode type!")
					return

			elif (isolation_mode is IsolationMode.ClosestDistanceOnly) or (isolation_mode is IsolationMode.ClosestHueDistanceOnly) or (isolation_mode is IsolationMode.TripleHueDistanceOnly):
				# purpose of partition: avoid to select u==v case
				second_smallest = np.partition(np.array(sub_isolation_costs), 2)[1]
				third_smallest = np.partition(np.array(sub_isolation_costs), 2)[2]

				cost = second_smallest
				if isolation_mode is IsolationMode.TripleHueDistanceOnly:
					cost = (second_smallest + third_smallest) * 0.5

				if bounds_mode is BoundsMode.AbsoluteBounds: 
					isolation_costs.append(cost / absolute_bounds)

				elif bounds_mode is BoundsMode.GlobalRelativeBounds:
					isolation_costs.append(cost / global_relative_bounds)

				elif (bounds_mode is BoundsMode.NormalizedBounds) or (bounds_mode is BoundsMode.AverageBounds):
					isolation_costs.append(cost)	

				else:
					error("[_get_initial_node] There isn't such BoundsMode type!")
					return

			else:
				error("[_get_initial_node] There isn't such IsolationMode type!")
				return

		average_bounds = sum(average_bounds) / len(average_bounds)
		if bounds_mode is BoundsMode.NormalizedBounds:
			isolation_costs = [c / max(isolation_costs) for c in isolation_costs]
		elif bounds_mode is BoundsMode.AverageBounds:
			isolation_costs = [c / average_bounds for c in isolation_costs]

		return isolation_costs


	def _get_initial_node(self, color_space, initial_node_mode, distance_mode):
		"""
		Return an index of initial node. Return the dark and lonely buddy.
		- Criteria:
			1) random choose
			2) lightness: https://stackoverflow.com/a/56678483/3923340
			3) isolation: argmax (min neighbor distance)
		
		@type	color_space:	Name of the color space. (ex) 'geo_hsv', 'lex_rgb'
		@param	color_space:	String
		@type	initial_node_mode:	initial node selection mode.
		@param	initial_node_mode:	item of InitialSeedMode

		@rtype	:	initial node index
		@return :	integer
		"""

		lightness_costs = self._get_lightness_cost(color_space, LightnessMode.UseCustom, distance_mode)
		isolation_costs = self._get_isolation_cost(color_space, IsolationMode.HueOnlySquareWeightedRank, BoundsMode.NormalizedBounds, distance_mode)
		
		print(color_space, 'lightness_costs: ', lightness_costs)
		print(color_space, 'isolation_costs: ', isolation_costs)

		# The initial node has a high cost sum
		cost_sum = 0

		if initial_node_mode is InitialSeedMode.WoLightness:
			cost_sum = isolation_costs

		elif initial_node_mode is InitialSeedMode.WoIsolation:
			cost_sum = lightness_costs

		# elif initial_node_mode is InitialSeedMode.Random:
			# cost_sum = np.arange(self.ccounts)
			# np.random.shuffle(cost_sum)

		elif initial_node_mode is InitialSeedMode.All:
			cost_sum = [x*0.5 + y for x, y in zip(lightness_costs, isolation_costs)]

		else:
			error("[_get_initial_node] There isn't such InitialSeedMode type!")
			return

		
		print('[Initial Point] ', color_space, ' cost_sum: ', cost_sum)
		print('[Initial Point] ', color_space, ' initialPoint: ', np.array(cost_sum).argmax())

		return np.array(cost_sum).argmax()


	def _is_reversed_lightness_between_two_nodes_index(self, a_index, b_index):
		a = self.colors['lex_hsl'][a_index][2]
		b = self.colors['lex_hsl'][b_index][2]

		return a > b


	def _is_reversed_color_between_two_nodes_index(self, color_space, base_index, a_index, b_index, reverse_mode):
		# Hue has a cycle (Tail and head colors are connected). So, we compare the reversity of hue based on the base_index color.
		# Return True if A precedes B.

		hue_direction = True # True: 0 to 1, False: 1 to 0

		k = self.colors['lex_hsl'][base_index][0] / 360
		a = self.colors['lex_hsl'][a_index][0] / 360
		b = self.colors['lex_hsl'][b_index][0] / 360

		a_ = a - k
		b_ = b - k

		if reverse_mode == ReverseMode.ClosestHueDirection:
			# TODO: this is hue only, it should be xy plane
			hues = [c[0] / 360 for c in self.colors['lex_hsl']]
			hues_duplicates_removed = list(dict.fromkeys(hues))
			hues_sorted = sorted(hues_duplicates_removed)
			
			k_index = hues_sorted.index(k)
			k_left_index = k_index - 1 if k_index > 0 else len(hues_sorted)-1
			k_right_index = k_index + 1 if k_index+1 < len(hues_sorted) else 0

			left_diff = abs(hues_sorted[k_index] - hues_sorted[k_left_index])
			right_diff = abs(hues_sorted[k_index] - hues_sorted[k_right_index])

			if left_diff < right_diff:
				hue_direction = False

		elif reverse_mode == ReverseMode.ClosestDirection:
			planes = [c for c in self.colors[color_space]]
			plane_distances = [self._get_distance_between_two_nodes(planes[base_index], p) for p in planes]

			second_smallest = np.partition(np.array(plane_distances), 2)[1]
			second_smallest_index = plane_distances.index(second_smallest)
			second_smallest_hue = self.colors['lex_hsl'][second_smallest_index][0]

			if second_smallest_hue < k:
				hue_direction = False

		elif reverse_mode == ReverseMode.GlobalHueDirection:
			hues = [c[0] / 360 for c in self.colors['lex_hsl']]
			hues_sorted = sorted(hues)

			k_index = hues_sorted.index(k)

			lefts = []
			rights = []

			if k_index < 0.5:
				# calculate counts of item in range [k ~ k + 0.5]
				rights = [e for e in hues_sorted if e > k and e <= k + 0.5]
				lefts = [e for e in hues_sorted if e not in rights and e != k]
			else:
				# calculate counts of item in range [hues_sorted[k_index] ~ 1 && 0 ~ 1 - hues_sorted[k_index]]
				lefts = [e for e in hues_sorted if e < k and e >= k - 0.5]
				rights = [e for e in hues_sorted if e not in lefts and e != k]
				
			if len(lefts) > len(rights):
				hue_direction = False


		if a_ * b_ < 0:
			# k exists inbetween a and b
			if a_ < b_:
				# order: a - k - b
				return True if hue_direction else False
			else:
				# order: b - k - a
				return False if hue_direction else True
		elif a_ * b_ > 0:
			if a_ < b_:
				# order: k - a - b / a - b - k
				return False if hue_direction else True

			else:
				# order: k - b - a / b - a - k
				return True if hue_direction else False
		elif a_ == 0:
			# a is base
			result = (-0.5 < b - a < 0) or (b -a > 0.5)
			return result if hue_direction else not result

		elif b_ == 0:
			# b is base
			result = (0 < a - b < 0.5) or (a - b < -0.5)
			return result if hue_direction else not result

		else:
			error("[_is_reversed_color_between_two_nodes_index] wrong a, b values")
			return


	def _get_hue_distance_between_two_nodes_index(self, a_index, b_index):
		# Hue is circulated, and we consider it.

		a = self.colors['lex_hsl'][a_index][0] / 360 # range [0, 1]
		b = self.colors['lex_hsl'][b_index][0] / 360 # range [0, 1]

		distance = abs(a - b)

		if distance > 0.5:
			return 1 - distance
		else:
			return distance


	def _get_distance_between_two_nodes(self, distance_mode, a, b):
		if distance_mode == DistanceMode.Euclidean:
			# The coordinates are already converted in the polar or cartesian way
			# So, this just returns euclidean distance between two nodes
			return math.sqrt(sum([(x - y) ** 2 for x, y in zip(a, b)]))
		# elif distance_mode == DistanceMode.CIE2000:
			# SUZI-TODO
			# return
		else:
			error("[_get_distance_between_two_nodes] wrong distance_mode")
			return


	def _get_triples_of_colors(self, color_space, initial_node_mode, spiral_mode, tsp_mode, reverse_mode, antilightness_mode, distance_mode):
		"""
		Return initial node index and distance matrix of colors.
		
		@type	color_space:	Name of the color space. (ex) 'hsv', 'rgb'
		@param	color_space:	String
		@type	initial_node_mode:	initial node selection mode.
		@param	initial_node_mode:	item of InitialSeedMode

		@rtype	:	initial_node_index, distances
		@return :	integer, 2D array
		"""

		distances_matrix = []
		distances_tuples = []
		tag = 'geo_' + color_space

		# index of initial node S
		initial_node_index = self._get_initial_node(tag, initial_node_mode, distance_mode)

		for u in range(self.ccounts):
			distances_matrix_row = []
			for v in range(self.ccounts):
				if u == v:
					distances_matrix_row.append(0)
					continue

				distance = sys.float_info.epsilon 

				if distance_mode == DistanceMode.Euclidean:
					lightness_distance = 0
					plane_distance = 0

					if v != initial_node_index:
						lightness_distance = self._get_distance_between_two_nodes(distance_mode, self.colors[tag][u][2:], self.colors[tag][v][2:])

						if antilightness_mode == AntiLightnessBonusMode.AntiLightness:
							plane_distance = self._get_distance_between_two_nodes(distance_mode, self.colors[tag][u][:2], self.colors[tag][v][:2])

						reflected_plane_distance = plane_distance * 2
						distance = lightness_distance + reflected_plane_distance

						if spiral_mode is SpiralMode.ReverseHuePenalty:
							is_reversed = self._is_reversed_color_between_two_nodes_index(tag, initial_node_index, u, v, reverse_mode)

							if is_reversed:
								distance *= 1.3

						elif spiral_mode is SpiralMode.ReverseHueLightnessPenalty:
							is_hue_reversed = self._is_reversed_color_between_two_nodes_index(tag, initial_node_index, u, v, reverse_mode)

							is_lightness_reversed = self._is_reversed_lightness_between_two_nodes_index(u, v)

							if is_hue_reversed:
								distance *= 1
							# else:
								# distance += distance

							# if is_lightness_reversed:
								# distance *= 1.05
							# else:
								# distance += distance

				elif distance_mode == DistanceMode.CIE2000:

				else:
					error("[_get_triples_of_colors] wrong distance mode")
					return

					else:
						error("[_get_triples_of_colors] There isn't such SpiralMode type!")
						return

					# distance = self._get_distance_between_two_nodes(self.colors[tag][u], self.colors[tag][v])

					# if spiral_mode is SpiralMode.ReverseHuePenalty:
					# 	is_reversed = self._is_reversed_color_between_two_nodes_index(initial_node_index, u, v, reverse_mode)

					# 	if is_reversed:
					# 		distance *= 1.3
					# else:
					# 	error("[_get_triples_of_colors] There isn't such SpiralMode type!")
					# 	return

					# if antilightness_mode == AntiLightnessBonusMode.AntiLightness:
					# 	bonus_ratio = 2
					# 	if color_space in ['hsl', 'hsv', 'lch']:
					# 		anti_distance = self._get_distance_between_two_nodes(self.colors[tag][u][:2], self.colors[tag][v][:2])
					# 		distance += bonus_ratio * anti_distance

				distances_tuples.append((u, v, distance))
				distances_matrix_row.append(distance)

			distances_matrix.append(distances_matrix_row)

		if tsp_mode is TSPMode.SymmetricTSP:
			return initial_node_index, distances_tuples

		elif tsp_mode is TSPMode.AsymmetricTSP:
			return initial_node_index, distances_matrix

		else:
			error("[_get_triples_of_colors] There isn't such TSPMode type!")
			return


