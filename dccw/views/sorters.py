from django.http.response import HttpResponseRedirect
from django.shortcuts import render
from django.contrib import messages

import ast

from dccw.models import *
from dccw.color_palette import *
from dccw.color_palettes import *
from dccw.comprehensive_single_palette_sorter import * 
from dccw.multiple_palettes_sorter import * 
from dccw.similarity_measurer import *
from dccw.geo_sorter_helper import *

from dccw.views import view_helper

def sortSinglePalette(request):
	color_count = 10

	if request.method == "POST":
		if request.POST.get("random"):
			color_palette = ColorPalette(auto_fetched=True, palette_length=color_count)

		elif request.POST.get("load"):
			palette_hex_str = request.POST.get('hex_input') 
			validation_result = view_helper.validate_input(palette_hex_str, 1)
			if validation_result['result'] is False:
				print("[Error]", validation_result["message"])
				messages.info(request, validation_result["message"])
				return render(request, 'sortSinglePalette.html')

			palette_hex = palette_hex_str.split('#')[1:]
			palette_hex = ['#' + h for h in palette_hex]
			print(palette_hex)
			color_palette = ColorPalette(auto_fetched=False, colors=palette_hex)
		else:
			print("[ERROR] No such post name..")

	else:
		color_palette = ColorPalette(auto_fetched=True, palette_length=color_count)

	target_spaces = ['rgb', 'hsv', 'vhs', 'lab']
	color_sorter = ComprehensiveSinglePaletteSorter(color_palette, target_spaces)

	lex_sorted_colors = {}
	lex_sorted_indices = color_sorter.lex_sort()
	standard_sorted_indices = color_sorter.standard_sort()

	geo_coords = {}
	for target_space in target_spaces:
		geo_coords[target_space] = color_palette.get_values(target_space, is_geo=True)

	# =========================
	# For Figures=============
	# print(' '.join(color_palette.to_hex_list()))
	# sorted_print_hex = [color_palette.to_hex_list()[i] for i in standard_sorted_indices]
	# print(' '.join(sorted_print_hex))

	return render(request, 'sortSinglePalette.html', {
	'hex_data': color_palette.to_hex_list(),
	'geo_coords': geo_coords,
	'lex_sorted_indices': lex_sorted_indices, 
	'standard_sorted_indices': standard_sorted_indices, 
	})


def sortMultiplePalettes(request):
	color_count = 10
	palette_count = 2
	palette_lengths = [10, 10]

	if request.method == "POST":
		if request.POST.get("random"):
			color_palettes = view_helper.fetch_new_palettes(palette_count, palette_lengths, is_color_count_a_list=True)

		elif request.POST.get("load"):
			palettes_hex_str_list_strs = request.POST.get('hex_input') # "#5d5753#6b6460,#ffffff#000000"

			validation_result =view_helper.validate_input(palettes_hex_str_list_strs, -1)
			if validation_result['result'] is False:
				print("[Error]", validation_result["message"])
				messages.info(request, validation_result["message"])
				return render(request, 'sortMultiplePalettes.html')

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
		# For Replicability Stamp, initialize with palettes used in teaser image
		color_palettes_list = [["#063f75","#065e73","#44b991","#ead53d","#f26c69","#ebd7a9","#f1af43","#ba4830","#382728","#321d37"], ["#43434a","#6d9489","#e0dbc5","#e5b393","#c17d3e","#343333","#3d4d5b","#688b9b","#d4e6d7","#e0c5b2"]]
		color_palettes = ColorPalettes(auto_fetched=False, color_palettes_list=color_palettes_list, is_hex_list=True)

	geo_coords = color_palettes.get_geo_coords()

	multiple_palettes_sorter = MultiplePalettesSorter(color_palettes, palette_count, LabDistanceMode.CIEDE2000)
	sorted_indices = {}
	elapsed_times = {}
	merged_sorted_indices = {}
	
	# 1. Standard Sorting
	tag = "Ours (LKH)"
	sorted_indices[tag], elapsed_times[tag], merged_sorted_indices[tag] = multiple_palettes_sorter.standard_sort()

	# 2. Merge_LKH without mergecutting
	mpsm = MultiplePalettesSortMode.Merge_LKH
	tag = "Ours (LKH) w/o cutting strategy"
	sorted_indices[tag], elapsed_times[tag], merged_sorted_indices[tag]  = multiple_palettes_sorter.sort(mpsm, MergeCutType.Without_Cutting)

	# =========================
	# For Figures=============
	print('merged', ' '.join(color_palettes.to_hex_string_list()))
	cp_print_hex_list = color_palettes.to_hex_list()
	for i in range(len(cp_print_hex_list)):
		print('input-%d' % i, ' '.join(cp_print_hex_list[i]))

	merged_cp_print_hex_list = color_palettes.to_merged_hex_list()
	for key, print_indices in merged_sorted_indices.items():
		if print_indices:
			print('merged-sorted-%s' % key, ' '.join([merged_cp_print_hex_list[i] for i in print_indices]))

	for key, print_indices in sorted_indices.items():
		for i in range(len(print_indices)):
			sorted_print_hex = [color_palettes.to_hex_list()[i][j] for j in print_indices[i]]
			print('%s-%d' % (key, i), ' '.join(sorted_print_hex))

	# For Figures=============
	# =========================
	
	return render(request, 'sortMultiplePalettes.html', {
	'hex_data': color_palettes.to_hex_list(),
	'elapsed_time': elapsed_times,
	'geo_coords': geo_coords,
	'sorted_indices': sorted_indices,
	'max_palette_length': max(map(len, color_palettes.to_hex_list()))
	})


def measureSimilarity(request):
	color_count = 5
	target_palettes_count = 3
	palettes = None

	if request.method == "POST":
		if request.POST.get("random"):
			palettes = view_helper.fetch_new_palettes(target_palettes_count+1, color_count)

		elif request.POST.get("load"):
			palettes_hex_list_str = request.POST.get('hex_input') # ex) '#000000#000000#000000,#000000#000000#000000,#000000#000000#000000'

			validation_result = view_helper.validate_input(palettes_hex_list_str, -1, 2)
			if validation_result['result'] is False:
				print("[Error]", validation_result["message"])
				messages.info(request, validation_result["message"])
				return render(request, 'measureSimilarity.html')
			

			palettes_hex_list = palettes_hex_list_str.split(',') # ex) ['#000000#000000#000000', '#000000#000000#000000', '#000000#000000#000000']
			
			target_palettes_count = len(palettes_hex_list) - 1
			palettes_list = []
			for palette_hex_str in palettes_hex_list:
				palette_hex = palette_hex_str.split('#')[1:]
				palette_hex = ['#' + h for h in palette_hex]
				palettes_list.append(palette_hex)

			palettes = ColorPalettes(auto_fetched=False, color_palettes_list=palettes_list, is_hex_list=True)
		else:
			print("[ERROR] No such post name..")

	else:
		palettes = view_helper.fetch_new_palettes(target_palettes_count+1, color_count)

	source_palette = palettes.get_single_palettes_list()[0]
	source_data = source_palette.to_hex_list()

	target_data_list = []
	ranks_data = {}
	for target_palette in palettes.get_single_palettes_list()[1:]:
		target_data = {}
		
		similarity_measurer = SimilarityMeasurer(source_palette, target_palette, LabDistanceMode.CIEDE2000)
		target_data['similarities'], target_data['comments'] = similarity_measurer.measure()
		target_data['sorted_indices'] = similarity_measurer.get_palette_sorted_indices()
		target_data['matched_source_indices'] = similarity_measurer.get_palette_bipartite_matching_indices()

		for tag, similarity in target_data['similarities'].items():
			if tag in ranks_data.keys():
				ranks_data[tag].append(similarity)
			else:
				ranks_data[tag] = [similarity]

		target_data_list.append(target_data)

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
		for cur_rank in range(1, target_palettes_count+1):
			# print(cur_measurement, cur_rank, ranks)
			palette_index = ranks[cur_measurement].index(cur_rank)
			
			measurement_data = {}
			measurement_data['hex_index'] = palette_index
			measurement_data['comments'] = target_data_list[palette_index]['comments'][cur_measurement]
			measurement_data['similarity'] = target_data_list[palette_index]['similarities'][cur_measurement]

			if cur_measurement in [SimMeasureStrategy.DynamicClosestColorWarping.name, SimMeasureStrategy.DynamicClosestColorWarpingConnected.name, SimMeasureStrategy.PairwiseAverage.name]:
				measurement_data['sorted_indices'] = target_data_list[palette_index]['sorted_indices']

			elif cur_measurement == SimMeasureStrategy.MinimumBipartiteMatchingError.name:
				measurement_data['matched_source_indices'] = target_data_list[palette_index]['matched_source_indices']

			rank_data.append(measurement_data)

		sorted_data_list[cur_measurement] = rank_data


	hex_data_list = palettes.to_hex_list()[1:]

	# =========================
	# For Figures=============
	print('merged', ' '.join(palettes.to_hex_string_list()))
	cp_print_hex_list = palettes.to_hex_list()
	for i in range(len(cp_print_hex_list)):
		print('palette-%d' % i, ' '.join(cp_print_hex_list[i]))

	for i, target_data in enumerate(target_data_list):
		print_indices = target_data['sorted_indices']
		sorted_print_hex = [cp_print_hex_list[i+1][j] for j in print_indices[1]]
		print('%d' % i, ' '.join(sorted_print_hex))

		if i==0:
			sorted_print_hex = [cp_print_hex_list[0][j] for j in print_indices[0]]
			print('query', ' '.join(sorted_print_hex))


	# For Figures=============
	# =========================
	
	# n: # of target palettes (except source palette)
	# m: # of measurement mode
	# sorted_data_list structure: [1st rank data, 2nd rank data, ..., nth rank data]
	#   rank data structure: [1st measurment data, 2nd measurement data, ..., m-th measurement data]
	#      measurement data: {hex_index: int, comments: string, similarity: number}	
	return render(request, 'measureSimilarity.html', {
	'hex_data_list': hex_data_list,
	'source_data': source_data,
	'sorted_data_list': sorted_data_list, # dependent to measurement_list order
	})

