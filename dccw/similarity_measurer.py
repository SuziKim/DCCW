import numpy as np
import math
import time
from scipy.spatial.distance import *
from scipy import signal
from scipy.optimize import linear_sum_assignment
from fastdtw import fastdtw

from pyemd import emd_with_flow

from colormath.color_objects import LabColor

from dccw.geo_sorter_helper import *
from dccw.multiple_palettes_sorter import * 
from dccw.color_palettes import * 

class SimilarityMeasurer:
	def __init__(self, source_palette, target_palette, lab_distance_mode):
		self.source_palette = source_palette
		self.target_palette = target_palette

		self.color_palettes = ColorPalettes(auto_fetched=False, color_palettes_list=[source_palette, target_palette], is_hex_list=False)

		multiple_palettes_sorter = MultiplePalettesSorter(self.color_palettes, 2, lab_distance_mode)
		self.sorted_palette_indices, _, _ = multiple_palettes_sorter.standard_sort()

	def get_palette_sorted_indices(self):
		return self.sorted_palette_indices

	def get_palette_bipartite_matching_indices(self):
		distance_matrix = np.array(self._get_distance_map())
		_, col_ind = linear_sum_assignment(distance_matrix)
		return col_ind.argsort().tolist()

	def standard_measure(self):
		return self._measure_with_strategy(SimMeasureStrategy.DynamicClosestColorWarping)
		
	def measure(self, include_elapsed_time=False):
		similarities = {}
		comments = {}
		elapsed_times = {}

		# smss = [SimMeasureStrategy.MergedPaletteHistogramSimilarityMeasure]
		# for sms in smss:

		for sms in SimMeasureStrategy:
			similarities[sms.name], comments[sms.name], elapsed_times[sms.name] = self._measure_with_strategy(sms)
			print(similarities[sms.name], sms.name, elapsed_times[sms.name])
			
		if include_elapsed_time:
			return similarities, comments, elapsed_times
		
		return similarities, comments

	def closest_points_by_dccw(self):
		source_labs = self.source_palette.get_values_in_order('lab', self.sorted_palette_indices[0], is_geo=False)
		target_labs = self.target_palette.get_values_in_order('lab', self.sorted_palette_indices[1], is_geo=False)

		_, _, source_to_target_closest_points = self._dccw_from_A_to_B(source_labs, target_labs, False)
		_, _, target_to_source_closest_points = self._dccw_from_A_to_B(target_labs, source_labs, False)

		return source_to_target_closest_points, target_to_source_closest_points

	def _measure_with_strategy(self, similarity_measurement_strategy):
		similarity = 0
		comment = None

		if similarity_measurement_strategy == SimMeasureStrategy.ClassicHausdorffDistance:
			start_time = time.time()
			similarity, comment = self._measure_hausdorff_distance(
				m_func = self._hausdorff_distance_helper_m_max, 
				g_func = self._hausdorff_distance_helper_g_max,
				q_func = self._hausdorff_distance_helper_q_min,
				r_func = self._hausdorff_distance_helper_r_d)

		elif similarity_measurement_strategy == SimMeasureStrategy.ModifiedHausdorffDistance:
			start_time = time.time()
			similarity, comment = self._measure_hausdorff_distance(
				m_func = self._hausdorff_distance_helper_m_max, 
				g_func = self._hausdorff_distance_helper_g_avg,
				q_func = self._hausdorff_distance_helper_q_min,
				r_func = self._hausdorff_distance_helper_r_d)

		elif similarity_measurement_strategy == SimMeasureStrategy.LeastTrimmedSquareHausdorffDistance:
			start_time = time.time()
			similarity, comment = self._measure_hausdorff_distance(
				m_func = self._hausdorff_distance_helper_m_max, 
				g_func = self._hausdorff_distance_helper_g_partial_avg,
				q_func = self._hausdorff_distance_helper_q_min,
				r_func = self._hausdorff_distance_helper_r_d)

		elif similarity_measurement_strategy == SimMeasureStrategy.MinimumColorDifference:
			start_time = time.time()
			similarity, comment = self._measure_hausdorff_distance(
				m_func = self._hausdorff_distance_helper_m_avg, 
				g_func = self._hausdorff_distance_helper_g_avg,
				q_func = self._hausdorff_distance_helper_q_min,
				r_func = self._hausdorff_distance_helper_r_d)

		elif similarity_measurement_strategy == SimMeasureStrategy.DynamicTimeWarping:
			start_time = time.time()
			similarity, comment = self._measure_dynamic_time_warping()

		elif similarity_measurement_strategy == SimMeasureStrategy.DynamicClosestColorWarping:
			start_time = time.time()
			similarity, comment = self._measure_dynamic_closest_color_warping(reflect_cycle=False)

		elif similarity_measurement_strategy == SimMeasureStrategy.DynamicClosestColorWarpingConnected:
			start_time = time.time()
			similarity, comment = self._measure_dynamic_closest_color_warping(reflect_cycle=True)

		# elif similarity_measurement_strategy == SimMeasureStrategy.SumOfDTWAndDCCW:
			# similarity, comment = self._measure_sum_of_dtw_and_dccw()
		
		elif similarity_measurement_strategy == SimMeasureStrategy.PairwiseAverage:
			start_time = time.time()
			similarity, comment = self._measure_pairwise_average()

		elif similarity_measurement_strategy == SimMeasureStrategy.SignatureQuadraticFormDistance:
			start_time = time.time()
			similarity, comment = self._measure_signature_quadratic_form_distance()

		elif similarity_measurement_strategy == SimMeasureStrategy.MergedPaletteHistogramSimilarityMeasure:
			start_time = time.time()
			similarity, comment = self._measure_merged_palette_histogram_similarity_measure()

		elif similarity_measurement_strategy == SimMeasureStrategy.ColorBasedEarthMoversDistance:
			start_time = time.time()
			similarity, comment = self._measure_color_based_earth_movers_distance()

		elif similarity_measurement_strategy == SimMeasureStrategy.MinimumBipartiteMatchingError:
			start_time = time.time()
			similarity, comment = self._measure_minimum_bipartite_matching_error()

		else:
			assert False, '[_measure_with_strategy] No such similarity measurement strategy'
		elapsed_time = time.time() - start_time

		return similarity, comment, elapsed_time


	def _measure_hausdorff_distance(self, m_func, g_func, q_func, r_func, use_sorted_palettes=False):
		source_labs = target_labs = None

		if use_sorted_palettes:
			source_labs = self.source_palette.get_color_objects_in_order('lab', self.sorted_palette_indices[0])
			target_labs = self.target_palette.get_color_objects_in_order('lab', self.sorted_palette_indices[1])
		else:
			source_labs = self.source_palette.get_color_objects('lab')
			target_labs = self.target_palette.get_color_objects('lab')

		source_to_target = target_to_source = -1
		comment = ''
		
		source_to_target = g_func(q_func, r_func, source_labs, target_labs)
		target_to_source = g_func(q_func, r_func, target_labs, source_labs)

		similarity, comment = m_func(source_to_target, target_to_source)
		
		return round(similarity, 4), comment

# ====================================================================
# ====================================================================

	def _hausdorff_distance_helper_m_max(self, x, y):
		return max(x, y), ''

	def _hausdorff_distance_helper_m_avg(self, x, y):
		comment = 'm_x: %.4f, m_y: %.4f' % (x, y)
		return (x + y) * 0.5, comment

# ====================================================================
# ====================================================================

	def _hausdorff_distance_helper_g_max(self, q_func, r_func, A, B):
		max_distance = - math.inf

		for a in A:
			distance = q_func(r_func, a, B)
			if distance > max_distance:
				max_distance = distance

		return max_distance


	def _hausdorff_distance_helper_g_avg(self, q_func, r_func, A, B):
		distance = 0
		for a in A:
			distance += q_func(r_func, a, B)
			
		return distance / len(A)


	def _hausdorff_distance_helper_g_partial_avg(self, q_func, r_func, A, B):
		h = 0.6
		H = round(h * len(A))

		d_B_a_set = []
		for a in A:
			distance = q_func(r_func, a, B)
			d_B_a_set.append(distance)

		d_B_a_set.sort()
		return sum(d_B_a_set[:H])/ H


# ====================================================================
# ====================================================================

	def _hausdorff_distance_helper_q_min(self, r_func, a, B):
		min_distance = math.inf

		for b in B:
			distance = r_func(a, b, B)
			if distance < min_distance:
				min_distance = distance
		
		return min_distance


	def _hausdorff_distance_helper_q_avg(self, r_func, a, B):
		distance = 0
		for b in B:
			distance += r_func(a, b, B)
		
		return distance / len(B)


	def _hausdorff_distance_helper_r_d(self, a, b, B=None):
		return self._get_Euclidean_distance_between_two_labs(a, b)

# ====================================================================
# ====================================================================
	
	def _measure_dynamic_closest_color_warping(self, reflect_cycle):
		source_labs = self.source_palette.get_values_in_order('lab', self.sorted_palette_indices[0], is_geo=False)
		target_labs = self.target_palette.get_values_in_order('lab', self.sorted_palette_indices[1], is_geo=False)

		distance_s_t, count_s_t, _ = self._dccw_from_A_to_B(source_labs, target_labs, reflect_cycle)
		distance_t_s, count_t_s, _ = self._dccw_from_A_to_B(target_labs, source_labs, reflect_cycle)

		return (distance_s_t + distance_t_s) / (count_s_t + count_t_s), ''
		
	def _dccw_from_A_to_B(self, A_colors, B_colors, reflect_cycle):
		distance = 0
		closest_points = []
		for a in A_colors:
			d, closest_point = self._dccw_from_a_to_B(a, B_colors, reflect_cycle)
			distance += d
			closest_points.append(closest_point)

		return distance, len(A_colors), closest_points
	
	def _dccw_from_a_to_B(self, a_color, B_colors, reflect_cycle):
		min_distance = math.inf
		min_closest_point = None

		color_range = len(B_colors)-1
		if reflect_cycle:
			color_range = len(B_colors)

		for b_index in range(color_range):
			b_segment_start = np.array(B_colors[b_index])
			b_segment_end = np.array(B_colors[(b_index+1) % len(B_colors)])

			a = np.array(a_color)

			distance, closest_point = self._point_to_line_dist(a, b_segment_start, b_segment_end)
			if distance < min_distance:
				min_distance = distance
				min_closest_point = closest_point
		
		return min_distance, min_closest_point


	def _point_to_line_dist(self, p, a, b):
		# https://stackoverflow.com/a/44129897/3923340
		# project c onto line spanned by a,b but consider the end points should the projection fall "outside" of the segment    
		n, v = b - a, p - a

		# the projection q of c onto the infinite line defined by points a,b
		# can be parametrized as q = a + t*(b - a). In terms of dot-products,
		# the coefficient t is (c - a).(b - a)/( (b-a).(b-a) ). If we want
		# to restrict the "projected" point to belong to the finite segment
		# connecting points a and b, it's sufficient to "clip" it into
		# interval [0,1] - 0 corresponds to a, 1 corresponds to b.

		t = max(0, min(np.dot(v, n)/np.dot(n, n), 1))
		closest_point = (a + t*n)
		distance = np.linalg.norm(p - closest_point) #or np.linalg.norm(v - t*n)

		return distance, closest_point

	def _measure_sum_of_dtw_and_dccw(self):
		dtw, _ = self._measure_dynamic_time_warping()
		dccw, _ = self._measure_dynamic_closest_color_warping()

		return dtw+dccw, ''

	def _measure_dynamic_time_warping(self):
		source_labs = self.source_palette.get_values_in_order('lab', self.sorted_palette_indices[0], is_geo=False)
		target_labs = self.target_palette.get_values_in_order('lab', self.sorted_palette_indices[1], is_geo=False)

		distance, path = fastdtw(source_labs, target_labs, dist=self._get_distance_between_two_labs_values)
		return distance, path

	def _measure_pairwise_average(self):
		pairwise_distance_map = self._get_distance_map()
		similarity = sum(sum(x) for x in pairwise_distance_map) / (len(pairwise_distance_map) * len(pairwise_distance_map[0]))

		return similarity, ""

	def _measure_signature_quadratic_form_distance(self):
		source_labs = self.source_palette.get_color_objects('lab')
		target_labs = self.target_palette.get_color_objects('lab')

		distance_SS = distance_TT = distance_ST = 0
		for c_s_1 in source_labs:
			for c_s_2 in source_labs:
				L2 = self._get_distance_between_two_labs(c_s_1, c_s_2)
				distance_SS += 1 / (1 + L2)

		for c_t_1 in target_labs:
			for c_t_2 in target_labs:
				L2 = self._get_distance_between_two_labs(c_t_1, c_t_2)
				distance_TT += 1 / (1 + L2)

		for c_s in source_labs:
			for c_t in target_labs:
				L2 = self._get_distance_between_two_labs(c_s, c_t)
				distance_ST += 1 / (1 + L2)

		distance = 0
		if distance_SS + distance_TT - 2 * distance_ST > 0:
			distance = math.sqrt(distance_SS + distance_TT - 2 * distance_ST)
		comment = "sum(S,S)=%.2f / sum(T,T)=%.2f / sum(S,T)=%.2f" % (distance_SS, distance_TT, distance_ST)
		return distance, comment

	def _measure_merged_palette_histogram_similarity_measure(self):
		source_labs = self.source_palette.get_color_objects('lab')
		target_labs = self.target_palette.get_color_objects('lab')

		distance_map = np.array(self._get_distance_map(use_euclidean=True))
		Td = 15

		# 1. Generate common palette
		common_palette = []
		closest_s_index, closest_t_index = np.unravel_index(np.argmin(distance_map, axis=None), distance_map.shape)
		closest_a_lab = source_labs[closest_s_index]
		closest_b_lab = target_labs[closest_t_index]
		is_a_from_s = is_b_from_t = True
		remaining_s_indices = np.arange(len(source_labs))
		remaining_t_indices = np.arange(len(target_labs))

		while self._get_Euclidean_distance_between_two_labs(closest_a_lab, closest_b_lab) <= Td:
			# Indices of the minimum elements of a N-dimensional array
			# https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmin.html
			
			c_m = [(a+b)/2 for a,b in zip(closest_a_lab.get_value_tuple(), closest_b_lab.get_value_tuple())]
			c_m_lab = LabColor(c_m[0], c_m[1], c_m[2], observer=closest_a_lab.observer, illuminant=closest_a_lab.illuminant)
			common_palette.append(c_m_lab)

			if is_a_from_s:
				distance_map[closest_s_index,:] = np.iinfo(np.int64).max
				remaining_s_indices = np.delete(remaining_s_indices, np.argwhere(remaining_s_indices == closest_s_index))
			if is_b_from_t:
				distance_map[:,closest_t_index] = np.iinfo(np.int64).max
				remaining_t_indices = np.delete(remaining_t_indices, np.argwhere(remaining_t_indices == closest_t_index))

			if len(remaining_s_indices) == 0 and len(remaining_t_indices) == 0:
				break

			# (1) the closest between two descriptors
			st_closest_s_index, st_closest_t_index = np.unravel_index(np.argmin(distance_map, axis=None), distance_map.shape)
			st_closest_s_lab = source_labs[st_closest_s_index]
			st_closest_t_lab = target_labs[st_closest_t_index]
			st_distance = distance_map[st_closest_s_index, st_closest_t_index]

			# (2) the closest between one descriptor and common palette
			scp_distance = tcp_distance = math.inf
			if len(common_palette) > 0:
				if len(remaining_s_indices) > 0:
					remaining_s_labs = [source_labs[i] for i in remaining_s_indices]
					scp_distance_map = np.array(self._get_distance_map_between_two_lab_lists(remaining_s_labs, common_palette, use_euclidean=True))
					scp_closest_s_index, scp_closest_cp_index = np.unravel_index(np.argmin(scp_distance_map, axis=None), scp_distance_map.shape)
					scp_distance = scp_distance_map[scp_closest_s_index, scp_closest_cp_index]

				if len(remaining_t_indices) > 0:
					remaining_t_labs = [target_labs[i] for i in remaining_t_indices]
					tcp_distance_map = np.array(self._get_distance_map_between_two_lab_lists(remaining_t_labs, common_palette, use_euclidean=True))
					tcp_closest_t_index, tcp_closest_cp_index = np.unravel_index(np.argmin(tcp_distance_map, axis=None), tcp_distance_map.shape)
					tcp_distance = tcp_distance_map[tcp_closest_t_index, tcp_closest_cp_index]

			if st_distance == min(st_distance, scp_distance, tcp_distance):
				is_a_from_s = is_b_from_t = True
				closest_a_lab = st_closest_s_lab
				closest_s_index = st_closest_s_index
				closest_b_lab = st_closest_t_lab
				closest_t_index = st_closest_t_index

			elif scp_distance == min(st_distance, scp_distance, tcp_distance):
				is_a_from_s = True
				is_b_from_t = False
				closest_a_lab = remaining_s_labs[scp_closest_s_index]
				closest_s_index = remaining_s_indices[scp_closest_s_index]
				closest_b_lab = common_palette[scp_closest_cp_index]

			else:
				is_a_from_s = False
				is_b_from_t = True
				closest_a_lab = common_palette[tcp_closest_cp_index]
				closest_b_lab = remaining_t_labs[tcp_closest_t_index]
				closest_t_index = remaining_t_indices[tcp_closest_t_index]

		for remaining_s_index in remaining_s_indices:
			common_palette.append(source_labs[remaining_s_index])

		for remaining_t_index in remaining_t_indices:
			common_palette.append(target_labs[remaining_t_index])

		# 2. Refine Pa and Pb
		redefined_source_p = [0 for i in range(len(common_palette))]
		redefined_target_p = [0 for i in range(len(common_palette))]

		for c_idx, c_i in enumerate(common_palette):
			for source_lab in source_labs:
				if self._get_Euclidean_distance_between_two_labs(c_i, source_lab) <= Td:
					redefined_source_p[c_idx] += 1
			
			for target_lab in target_labs:
				if self._get_Euclidean_distance_between_two_labs(c_i, target_lab) <= Td:
					redefined_target_p[c_idx] += 1
			
		# 3. MPHSM
		MPHSM = sum([min(p_1mi, p_2mi) for p_1mi, p_2mi in zip(redefined_source_p, redefined_target_p)])
		comment = 'common palette length: %d' % len(common_palette)

		return MPHSM, comment


	def _measure_color_based_earth_movers_distance(self):
		source_labs = self.source_palette.get_color_objects('lab')
		target_labs = self.target_palette.get_color_objects('lab')

		threshold = 20
		N_s = len(source_labs)
		N_t = len(target_labs)

		first_histogram = np.concatenate((np.ones((N_s,)), np.zeros((N_t,)))).astype('float64')
		second_histogram = np.concatenate((np.zeros((N_s,)), np.ones((N_t,)))).astype('float64')
		distance_matrix = np.full((N_s + N_t, N_s + N_t), threshold)
		np.fill_diagonal(distance_matrix, 0)

		distance_map = np.array(self._get_distance_map())
		distance_matrix[:N_s, N_s:] = distance_map
		distance_matrix[N_s:, :N_s] = distance_map.transpose()

		cemd, flow = emd_with_flow(first_histogram, second_histogram, distance_matrix.astype('float64'))

		return cemd, ''

	def _measure_minimum_bipartite_matching_error(self):
		distance_matrix = np.array(self._get_distance_map())
		row_ind, col_ind = linear_sum_assignment(distance_matrix)
		mbme = distance_matrix[row_ind, col_ind].sum()
		# comment = 'matched source index: ', col_ind.argsort()

		return mbme, '' #comment
		

# ====================================================================
# ====================================================================

	def _get_distance_map(self, use_euclidean=False):
		# return [[d_s1_t1, d_s1_t2_, ..., d_s1_tm], [d_s2_t1, ..., d_s2_,tm], ..., [d_sk_t1, ..., d_sk_tm]]
		distance_map = []
		for source_idx, source_lab in enumerate(self.source_palette.get_color_objects('lab')):
			distance_array = []
			for target_idx, target_lab in enumerate(self.target_palette.get_color_objects('lab')):
				distance = self._get_distance_between_two_labs(source_lab, target_lab, use_euclidean)
				distance_array.append(distance)
			distance_map.append(distance_array)

		return distance_map

	def _get_distance_map_between_two_lab_lists(self, A_Labs, B_Labs, use_euclidean=False):
		# return [[d_s1_t1, d_s1_t2_, ..., d_s1_tm], [d_s2_t1, ..., d_s2_,tm], ..., [d_sk_t1, ..., d_sk_tm]]
		distance_map = []
		for source_idx, source_lab in enumerate(A_Labs):
			distance_array = []
			for target_idx, target_lab in enumerate(B_Labs):
				distance = self._get_distance_between_two_labs(source_lab, target_lab, use_euclidean)
				distance_array.append(distance)
			distance_map.append(distance_array)

		return distance_map

	def _get_Euclidean_distance_between_two_labs_values(self, a, b):
		return math.sqrt(sum([(x - y) ** 2 for x, y in zip (a, b)]))

	def _get_distance_between_two_labs(self, lab_a, lab_b, use_euclidean=False):
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

	def _get_distance_between_two_labs_values(self, lab_a, lab_b, use_euclidean=False):
		if use_euclidean:
			return self._get_Euclidean_distance_between_two_labs_values(lab_a, lab_b)
		else:
			a = LabColor(lab_a[0], lab_a[1], lab_a[2])
			b = LabColor(lab_b[0], lab_b[1], lab_b[2])
			return self._get_CIEDE2000_distance_between_two_labs(a, b)
