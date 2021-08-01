from django.shortcuts import render
from django.http import HttpResponse
from django.contrib import messages

from dccw.models import *
from dccw.color_palette import *
from dccw.color_palettes import *

from dccw.single_palette_sorter import * 
from dccw.multiple_palettes_sorter import * 
from dccw.similarity_measurer import *
from dccw.geo_sorter_helper import *

from dccw.views import view_helper

from experiments.experiments_enums import *
from experiments.experiments_helper_functions import *


def experiment1(request, palette_length):
	if palette_length not in [f.value for f in FM100PType]:
		return HttpResponse('Unsupported palette length in FM100P.')

	shuffled_color_indices, shuffled_hex_colors = random_batch_single_palette_from_FM100P(palette_length)
	
	correct_hex_colors = [h for _,h in sorted(zip(shuffled_color_indices, shuffled_hex_colors))]
	correct_color_indices = sorted(shuffled_color_indices)

	color_palette = ColorPalette(auto_fetched=False, palette_length=len(shuffled_hex_colors), colors=shuffled_hex_colors)
	color_sorter = SinglePaletteSorter(color_palette)

	sorted_indices = {}
	elapsed_times = {}
	successes = {}
	levenshtein_distances = {}
	LLISs = {}

	for spsm in allowed_SPS_modes:
		indices, elapsed_times[spsm.name] = color_sorter.sort(spsm)
		sorted_color_indices = [shuffled_color_indices[k] for k in indices]

		if sorted_color_indices.index(min(sorted_color_indices)) > palette_length * 0.5:
			sorted_color_indices = list(reversed(sorted_color_indices))
			indices = list(reversed(indices))

		sorted_indices[spsm.name] = indices
		successes[spsm.name] = sorted_color_indices == correct_color_indices
		levenshtein_distances[spsm.name] = levenshtein_distance(sorted_color_indices, correct_color_indices)
		LLISs[spsm.name] = get_longest_increasing_subsequence_length(sorted_color_indices)

	geo_coords = color_palette.get_values('lab', is_geo=True)
	geo_coords = [list(t) for t in geo_coords] # convert list of tuples to 2D list

	return render(request, 'exp1.html', {
	'palette_length': palette_length,
	'correct_hex_data': correct_hex_colors,
	'hex_data': color_palette.to_hex_list(),
	'elapsed_time': elapsed_times,
	'success' : successes,
	'levenshtein_distance' : levenshtein_distances,
	'llis' : LLISs,
	'geo_coords': geo_coords,
	'sorted_indices': sorted_indices, 
	})


# ==========================================
# ==========================================

def experiment2(request, khtp_type):
	if khtp_type not in [a.name for a in AllowedKHTPCategory]:
		return HttpResponse('Unsupported KHTP Type')

	shuffled_color_indices, shuffled_hex_colors = random_batch_palette_pair_from_KHTP(AllowedKHTPCategory[khtp_type])
	
	color_palettes = ColorPalettes(auto_fetched=False, color_palettes_list=shuffled_hex_colors, is_hex_list=True)
	multiple_palettes_sorter = MultiplePalettesSorter(color_palettes, len(shuffled_hex_colors), LabDistanceMode.CIEDE2000)

	sorted_indices = {}
	elapsed_times = {}
	naturalness_levenshtein_distances = {}
	naturalness_LLISs = {}
	concurrency_levenshtein_distances = {}
	concurrency_LLCSs = {}

	mpsm_modes = [MultiplePalettesSortMode.Merge_LKH, MultiplePalettesSortMode.BPS, MultiplePalettesSortMode.Improved_BPS]
	for mpsm in mpsm_modes:
		indices, elapsed_times[mpsm.name], _ = multiple_palettes_sorter.sort(mpsm)
		sorted_indices[mpsm.name] = []

		lv_distances = 0
		lliss = 0
		flattened_palettes = []
		for p_index in range(len(indices)):
			sorted_color_indices = [shuffled_color_indices[p_index][k] for k in indices[p_index]]
			initial_color_index = sorted_color_indices.index(0)
			flattened_color_indices = sorted_color_indices[initial_color_index:] + sorted_color_indices[:initial_color_index]
			flattened_palettes.append(flattened_color_indices)

			sorted_indices[mpsm.name].append(indices[p_index][initial_color_index:] + indices[p_index][:initial_color_index])

			# evaluate naturalness (LD) of each palettes
			lv_distances += levenshtein_distance(flattened_color_indices, sorted(sorted_color_indices))

			# evaluate naturalness (LLIS) of each palettes
			lliss += get_longest_increasing_subsequence_length(flattened_color_indices)

		naturalness_levenshtein_distances[mpsm.name] = lv_distances / len(indices)
		naturalness_LLISs[mpsm.name] = lliss / len(indices)

		# evaluate concurrency (LD) of each palettes
		concurrency_levenshtein_distances[mpsm.name] = levenshtein_distance(flattened_palettes[0], flattened_palettes[1])

		# evaluate concurrency (LLCS) of each palettes
		concurrency_LLCSs[mpsm.name] = get_longest_common_subsequence_length(flattened_palettes[0], flattened_palettes[1])


	geo_coords = color_palettes.get_geo_coords()
	
	return render(request, 'exp2.html', {
	'khtp_type': khtp_type,
	'hex_data': color_palettes.to_hex_list(),
	'elapsed_time': elapsed_times,
	'geo_coords': geo_coords,
	'sorted_indices': sorted_indices,
	'max_palette_length': max(map(len, color_palettes.to_hex_list())),
	'naturalness_LD': naturalness_levenshtein_distances,
	'naturalness_LLIS': naturalness_LLISs,
	'concurrency_LD': concurrency_levenshtein_distances,
	'concurrency_LLCS': concurrency_LLCSs,
	})


# ==========================================
# ==========================================
def experiment3(request, khtp_type):
	if khtp_type not in [a.name for a in AllowedKHTPCategory]:
		return HttpResponse('Unsupported KHTP Type')

	shuffled_color_indices, shuffled_hex_colors = random_batch_palette_pair_from_KHTP(AllowedKHTPCategory[khtp_type])

	print(shuffled_color_indices)
	print(shuffled_hex_colors)
	shuffled_color_indices = [[3, 2, 0, 9, 6, 8, 5, 4, 1, 7], [8, 2, 1, 7, 0, 3, 6, 9, 4, 5]]
	shuffled_hex_colors = [['#dbdc5d', '#f9efbd', '#ecd9ca', '#cd9a95', '#c2def2', '#b8bebd', '#d1ead3', '#dde8cf', '#f2b267', '#cbd7e8'], ['#b8bebd', '#e1ec4f', '#f1b066', '#a5b8c7', '#e9a390', '#dbdc5d', '#7fafa6', '#ceb9c5', '#9bc471', '#a6c9a3']]
	
	color_palettes = ColorPalettes(auto_fetched=False, color_palettes_list=shuffled_hex_colors, is_hex_list=True)
	multiple_palettes_sorter = MultiplePalettesSorter(color_palettes, len(shuffled_hex_colors), LabDistanceMode.CIEDE2000)

	sorted_indices = {}
	elapsed_times = {}
	max_traversal_lengths = {}
	avg_traversal_lengths = {}
	
	for merge_cut_type in MergeCutType:
		indices, elapsed_times[merge_cut_type.name], _ = multiple_palettes_sorter.sort(MultiplePalettesSortMode.Merge_LKH, merge_cut_type)
		sorted_indices[merge_cut_type.name] = indices
		sorted_hex_colors = []

		for p_index in range(len(indices)):
			sorted_hex_colors.append([shuffled_hex_colors[p_index][k] for k in indices[p_index]])
			
		max_traversal_lengths[merge_cut_type.name] = get_max_color_difference(sorted_hex_colors)
		avg_traversal_lengths[merge_cut_type.name] = get_average_color_difference(sorted_hex_colors)

	geo_coords = color_palettes.get_geo_coords()
	
	return render(request, 'exp3.html', {
	'khtp_type': khtp_type,
	'hex_data': color_palettes.to_hex_list(),
	'elapsed_time': elapsed_times,
	'geo_coords': geo_coords,
	'sorted_indices': sorted_indices,
	'max_palette_length': max(map(len, color_palettes.to_hex_list())),
	'max_traversal_lengths': max_traversal_lengths,
	'avg_traversal_lengths': avg_traversal_lengths
	})


# ==========================================
# ==========================================
def experiment4(request):
	palette_count = 2
	palette_lengths = [6, 10]

	if request.method == "POST":
		if request.POST.get("random"):
			color_palettes = view_helper.fetch_new_palettes(palette_count, palette_lengths, is_color_count_a_list=True)
			
		elif request.POST.get("load"):
			palettes_hex_str_list_strs = request.POST.get('hex_input') # "#5d5753#6b6460,#ffffff#000000"

			validation_result = view_helper.validate_input(palettes_hex_str_list_strs, -1)
			if validation_result['result'] is False:
				print("[Error]", validation_result["message"])
				messages.info(request, validation_result["message"])
				return render(request, 'exp4.html')

			palettes_hex_str_list = palettes_hex_str_list_strs.split(',') # ['#5d5753#6b6460', '#ffffff#000000']
			
			color_palettes_list = []
			for palette_index, palette_hex_str in enumerate(palettes_hex_str_list):
				palette_hex = palette_hex_str.split('#')[1:]
				palette_hex = ['#' + h for h in palette_hex]
				color_palettes_list.append(palette_hex)
			
			palette_count = len(color_palettes_list)
			color_palettes = ColorPalettes(auto_fetched=False, color_palettes_list=color_palettes_list, is_hex_list=True)
		else:
			print("[ERROR] No such post name..")

	else:
		color_palettes = view_helper.fetch_new_palettes(palette_count, palette_lengths, is_color_count_a_list=True)

	geo_coords = color_palettes.get_geo_coords()

	multiple_palettes_sorter = MultiplePalettesSorter(color_palettes, palette_count, LabDistanceMode.CIEDE2000)
	sorted_indices, elapsed_time, _ = multiple_palettes_sorter.sort(MultiplePalettesSortMode.Merge_LKH)

	
	return render(request, 'exp4.html', {
	'hex_data': color_palettes.to_hex_list(),
	'elapsed_time': elapsed_time,
	'geo_coords': geo_coords,
	'sorted_indices': sorted_indices
	})


# ==========================================
# ==========================================
def experiment5(request):
	palette_count = 3
	color_count = 10
	palette_lengths = [5, 8, 10]

	if request.method == "POST":
		if request.POST.get("random"):
			color_palettes = view_helper.fetch_new_palettes(palette_count, palette_lengths, is_color_count_a_list=True)
			
		elif request.POST.get("load"):
			palettes_hex_str_list_strs = request.POST.get('hex_input') # "#5d5753#6b6460,#ffffff#000000"
			
			validation_result = view_helper.validate_input(palettes_hex_str_list_strs, -1)
			if validation_result['result'] is False:
				print("[Error]", validation_result["message"])
				messages.info(request, validation_result["message"])
				return render(request, 'exp5.html')

			palettes_hex_str_list = palettes_hex_str_list_strs.split(',') # ['#5d5753#6b6460', '#ffffff#000000']
			
			color_palettes_list = []
			for palette_index, palette_hex_str in enumerate(palettes_hex_str_list):
				palette_hex = palette_hex_str.split('#')[1:]
				palette_hex = ['#' + h for h in palette_hex]
				color_palettes_list.append(palette_hex)
			
			palette_count = len(color_palettes_list)
			color_palettes = ColorPalettes(auto_fetched=False, color_palettes_list=color_palettes_list, is_hex_list=True)
		else:
			print("[ERROR] No such post name..")

	else:
		color_palettes = view_helper.fetch_new_palettes(palette_count, palette_lengths, is_color_count_a_list=True)

	geo_coords = color_palettes.get_geo_coords()

	multiple_palettes_sorter = MultiplePalettesSorter(color_palettes, palette_count, LabDistanceMode.CIEDE2000)
	sorted_indices, elapsed_time, _ = multiple_palettes_sorter.sort(MultiplePalettesSortMode.Merge_LKH)

	
	return render(request, 'exp5.html', {
	'hex_data': color_palettes.to_hex_list(),
	'elapsed_time': elapsed_time,
	'geo_coords': geo_coords,
	'sorted_indices': sorted_indices,
	'max_palette_length': max(map(len, color_palettes.to_hex_list()))
	})


# ==========================================
# ==========================================

def experiment6(request, lhsp_type):
	if lhsp_type not in [a.name for a in AllowedLHSPCategory]:
		return HttpResponse('Unsupported LHSP Type')
	
	print('[exp6] exp6 start')
	query_palette, query_image_name, retrieval_palettes, retrieval_image_names = random_batch_palette_pair_from_LHSP(AllowedLHSPCategory[lhsp_type])
	print('[exp6] batch is done')
	target_data_list = []
	ranks_data = {}
	for retrieval_index, retrieval_palette in enumerate(retrieval_palettes):
		print('\tmeasure: %d / %d' % (retrieval_index, len(retrieval_palettes)))
		target_data = {}

		similarity_measurer = SimilarityMeasurer(query_palette, retrieval_palette, LabDistanceMode.CIEDE2000)

		target_data['similarities'], target_data['comments'], target_data['elapsed_time'] = similarity_measurer.measure(include_elapsed_time=True)  # similarities:
		target_data['sorted_indices'] = similarity_measurer.get_palette_sorted_indices()
		target_data['matched_source_indices'] = similarity_measurer.get_palette_bipartite_matching_indices()
		target_data['is_correct'] = query_image_name == retrieval_image_names[retrieval_index]

		for tag, similarity in target_data['similarities'].items():
			if tag in ranks_data.keys():
				ranks_data[tag].append(similarity)
			else:
				ranks_data[tag] = [similarity]

		target_data_list.append(target_data)

	print('[exp6] measure is done')

	measurement_list = []
	ranks = {} # ranks[measurement_type][target_palette_index]
	for tag, similarities in ranks_data.items():
		measurement_list.append(tag)
		if tag in [SimMeasureStrategy.ClassicHausdorffDistance.name,
				SimMeasureStrategy.PairwiseAverage.name,
				SimMeasureStrategy.ModifiedHausdorffDistance.name,
				SimMeasureStrategy.LeastTrimmedSquareHausdorffDistance.name,
				SimMeasureStrategy.MinimumColorDifference.name,
				SimMeasureStrategy.ColorBasedEarthMoversDistance.name,
				SimMeasureStrategy.MinimumBipartiteMatchingError.name, 
				SimMeasureStrategy.DynamicTimeWarping.name,
				SimMeasureStrategy.DynamicClosestColorWarping.name,
				SimMeasureStrategy.DynamicClosestColorWarpingConnected.name,
				SimMeasureStrategy.SignatureQuadraticFormDistance.name]:
			# ascending order
			# ranks[tag] = [sorted(similarities).index(x)+1 for x in similarities]
			
			# ascending order to disallow ties
			ranks[tag] = []
			sorted_similarities = [(s,i+1) for i, s in enumerate(sorted(similarities))]
			for similarity in similarities:
				rank = [ss[1] for ss in sorted_similarities if ss[0] == similarity]
				ranks[tag].append(rank[0])
				sorted_similarities.remove((similarity, rank[0]))

		elif tag in [SimMeasureStrategy.MergedPaletteHistogramSimilarityMeasure.name]:
			# descending order
			# ranks[tag] = [sorted(similarities, reverse=True).index(x)+1 for x in similarities]

			# descending order to disallow ties
			ranks[tag] = []
			sorted_similarities = [(s,i+1) for i, s in enumerate(sorted(similarities, reverse=True))]
			for similarity in similarities:
				rank = [ss[1] for ss in sorted_similarities if ss[0] == similarity]
				ranks[tag].append(rank[0])
				sorted_similarities.remove((similarity, rank[0]))

		else:
			assert False, '[Error] There is no such SimMeasureStrategy name'
	

	sorted_data_list = {}
	for cur_measurement in measurement_list:
		rank_data = []
		for cur_rank in range(1, len(retrieval_palettes)+1):
			# print(cur_measurement, cur_rank, ranks)
			palette_index = ranks[cur_measurement].index(cur_rank)
			
			measurement_data = {}
			measurement_data['hex_index'] = palette_index 
			measurement_data['comments'] = target_data_list[palette_index]['comments'][cur_measurement]
			measurement_data['elapsed_time'] = target_data_list[palette_index]['elapsed_time'][cur_measurement]
			measurement_data['similarity'] = target_data_list[palette_index]['similarities'][cur_measurement]

			if cur_measurement in [SimMeasureStrategy.DynamicClosestColorWarping.name, SimMeasureStrategy.DynamicClosestColorWarpingConnected.name, SimMeasureStrategy.PairwiseAverage.name]:
				measurement_data['sorted_indices'] = target_data_list[palette_index]['sorted_indices']

			elif cur_measurement == SimMeasureStrategy.MinimumBipartiteMatchingError.name:
				measurement_data['matched_source_indices'] = target_data_list[palette_index]['matched_source_indices']

			rank_data.append(measurement_data)

		sorted_data_list[cur_measurement] = rank_data

	source_data = query_palette.to_hex_list()
	hex_data_list = [retrieval_palette.to_hex_list() for retrieval_palette in retrieval_palettes]
	correct_palette_indices = [i for i, x in enumerate([t['is_correct'] for t in target_data_list]) if x]

	print('Query image name: ', query_image_name)
	print('Query palette: ', query_palette.to_serialized_hex_string())
	print('Retrieval palettes: ')
	for retrieval_palette in retrieval_palettes:
		print(retrieval_palette.to_serialized_hex_string())
		
	# n: # of target palettes (except source palette)
	# m: # of measurement mode
	# sorted_data_list structure: [1st rank data, 2nd rank data, ..., nth rank data]
	#   rank data structure: [1st measurment data, 2nd measurement data, ..., m-th measurement data]
	#      measurement data: {hex_index: int, comments: string, similarity: number}	
	return render(request, 'exp6.html', {
	'lhsp_type': lhsp_type,
	'correct_palette_indices': correct_palette_indices,
	'hex_data_list': hex_data_list,
	'source_data': source_data,
	'sorted_data_list': sorted_data_list, # dependent to measurement_list order
	})
