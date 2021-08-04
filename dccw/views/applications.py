from DCCWWebDemo.settings import MEDIA_ROOT
from django.shortcuts import render
from django.http import HttpResponse
from django.db.models import Q
from django.conf import settings
from django.contrib import messages
from django.utils.crypto import get_random_string

import os
import ast
import random
from PIL import Image
import numpy as np
import base64
import json
import glob
from json import JSONEncoder
from scipy import sparse
import enum

from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color

from dccw.models import *
from dccw.forms import *
from dccw.color_palette import *
from dccw.color_palettes import *

from dccw.single_palette_sorter import * 
from dccw.multiple_palettes_sorter import * 
from dccw.similarity_measurer import *
from dccw.geo_sorter_helper import *

from dccw.views import view_helper
from dccw.tan_decomposing import *

################################
# For Recoloring
USE_OPENCL = False
try:
    import pyopencl_example
    if len( pyopencl_example.cl.get_platforms() ) > 0:
        USE_OPENCL = True
except: pass
print( "Using OpenCL:", USE_OPENCL )
################################


def paletteInterpolation(request):
	color_count = random.randint(5, 10)

	if request.method == "POST":
		if request.POST.get("random"):
			color_palette = ColorPalette(auto_fetched=True, palette_length=color_count)

		elif request.POST.get("load"):
			palette_hex_str = request.POST.get('hex_input') 

			validation_result = view_helper.validate_input(palette_hex_str, 1)
			if not validation_result['result']:
				print("[Error]", validation_result["message"])
				messages.info(request, validation_result["message"])
				return render(request, 'paletteInterpolator.html')

			palette_hex = palette_hex_str.split('#')[1:]
			palette_hex = ['#' + h for h in palette_hex]
			print(palette_hex)
			color_palette = ColorPalette(auto_fetched=False, colors=palette_hex)
		else:
			print("[ERROR] No such post name..")

	else:
		color_palette = ColorPalette(auto_fetched=True, palette_length=color_count)

	color_sorter = SinglePaletteSorter(color_palette)
	sorted_indices_dccw = {}
	sorted_indices_hsv = {}
	sorted_indices_luminance = {}
	
	sorted_indices_dccw['orig'] = color_sorter.standard_sort()
	sorted_indices_hsv['orig'], _ = color_sorter.sort(SinglePaletteSortMode.HSV)
	sorted_indices_luminance['orig'], _ = color_sorter.sort(SinglePaletteSortMode.Luminance)

	sorted_indices_hsv['anchored'] = sorted_indices_hsv['orig']
	sorted_indices_dccw['anchored']	= sorted_indices_dccw['orig'][sorted_indices_dccw['orig'].index(sorted_indices_hsv['anchored'][0]):] \
									+ sorted_indices_dccw['orig'][:sorted_indices_dccw['orig'].index(sorted_indices_hsv['anchored'][0])]
	sorted_indices_luminance['anchored'] = sorted_indices_luminance['orig'][sorted_indices_luminance['orig'].index(sorted_indices_hsv['anchored'][0]):] \
									+ sorted_indices_luminance['orig'][:sorted_indices_luminance['orig'].index(sorted_indices_hsv['anchored'][0])]

	# =========================
	# For Figures=============
	print(' '.join(color_palette.to_hex_list()))

	# for key, lex_sorted_index in lex_sorted_indices.items():
		# sorted_print_hex = [color_palette.to_hex_list()[i] for i in lex_sorted_index['orig']]
		# print(key, ' '.join(sorted_print_hex))

	sorted_print_hex = [color_palette.to_hex_list()[i] for i in sorted_indices_dccw['orig']]
	print(' '.join(sorted_print_hex))

	sorted_print_hex = [color_palette.to_hex_list()[i] for i in sorted_indices_hsv['orig']]
	print(' '.join(sorted_print_hex))

	sorted_print_hex = [color_palette.to_hex_list()[i] for i in sorted_indices_luminance['orig']]
	print(' '.join(sorted_print_hex))
	# For Figures=============
	# =========================

	return render(request, 'paletteInterpolator.html', {
	'hex_data': color_palette.to_hex_list(),
	'sorted_indices_dccw': sorted_indices_dccw, 
	'sorted_indices_luminance' : sorted_indices_luminance, 
	'sorted_indices_hsv': sorted_indices_hsv, 
	})


def paletteNavigation(request):
	top_query_count = 3

	source_palette = _get_random_palette_for_palette_navigator()
	source_palette_hex_list = _make_a_list_from_serialized_hex_string(source_palette.hexes_list)

	dataset_names = ['palettes', 'Unsplash', 'mcs', 'wikiart-5', 'wikiart-10']
	query_results = {}

	for dataset_name in dataset_names:
		palette_distance_candidates = PaletteNavigatorDistanceModel.objects.filter((Q(palette1=source_palette) & Q(palette2__data_source__startswith=dataset_name)) | (Q(palette2=source_palette) & Q(palette1__data_source__startswith=dataset_name))).order_by('distance')

		query_result_array = []
		for result_palette_distance in palette_distance_candidates[:top_query_count]:
			target_palette = result_palette_distance.palette1
			if result_palette_distance.palette1 == source_palette:
				target_palette = result_palette_distance.palette2

			each_result_dic = {}
			each_result_dic['id'] = target_palette.palette_id
			each_result_dic['sorted_hex'] = _make_a_list_from_serialized_hex_string(target_palette.hexes_list)
			each_result_dic['distance'] = result_palette_distance.distance
			
			query_result_array.append(each_result_dic)
			
		query_results[dataset_name] = query_result_array

	return render(request, 'paletteNavigator.html', {
		'source_data': source_palette_hex_list,
		'query_results': query_results,
	})

# https://pynative.com/python-serialize-numpy-ndarray-into-json/
class NumpyArrayEncoder(JSONEncoder):
	def default(self, obj):
		if isinstance(obj, numpy.ndarray):
			return obj.tolist()
		return JSONEncoder.default(self, obj)

class RecoloringInterimResultMode(enum.Enum):
	source_palette = {'id': 1, 'is_matrix': False}
	image_data = {'id': 2, 'is_matrix': False}
	image_size = {'id': 3, 'is_matrix': False}
	RGBXY_mixing_weights = {'id': 4, 'is_matrix': True}

class RecoloringResult(enum.Enum):
	tan_et_al = {'id': 1, 'tag': 'tan-et-al-2018'}
	ours = {'id': 2, 'tag': 't2s-buffer'}

def _get_recoloring_file_names(mode):
	image_name = 'recoloring-%s.png' % mode.value['tag']
	palette_image_name = 'input_palette-%s.png' % mode.value['tag']
	return image_name, palette_image_name

def _get_interim_file_path(image_uid, mode):
	extension = 'npz' if mode.value['is_matrix'] else 'txt'
	return os.path.join(settings.MEDIA_ROOT, image_uid, '%s.%s' % (mode.name, extension))

def _serialize_and_save_numpy_array(image_uid, mode, np_array):
	output_path = _get_interim_file_path(image_uid, mode)
	print('serialize:', output_path)
	with open(output_path, 'w') as outfile:
		json.dump({"array": np_array}, outfile, cls=NumpyArrayEncoder)

def _serialize_and_save_sparse_matrix(image_uid, mode, matrix):
	output_path = _get_interim_file_path(image_uid, mode)
	print('serialize:', output_path)
	sparse.save_npz(output_path, matrix, compressed=True)

def _deserialize_and_load_numpy_array(image_uid, mode):
	# print('deserialize: ', output_path)
	output_path = _get_interim_file_path(image_uid, mode)
	result_np_array = None
	with open(output_path) as serialized_array:
		result_np_array = np.asarray(json.loads(serialized_array.read())['array'])

	return result_np_array

def _deserialize_and_load_sparse_matrix(image_uid, mode):
	# print('deserialize: ', output_path)
	output_path = _get_interim_file_path(image_uid, mode)
	return sparse.load_npz(output_path)

def _data_exist(image_uid):
	return os.path.isfile(_get_interim_file_path(image_uid, RecoloringInterimResultMode.source_palette)) and \
		os.path.isfile(_get_interim_file_path(image_uid, RecoloringInterimResultMode.image_data)) and \
		os.path.isfile(_get_interim_file_path(image_uid, RecoloringInterimResultMode.image_size)) and \
		os.path.isfile(_get_interim_file_path(image_uid, RecoloringInterimResultMode.RGBXY_mixing_weights))


def imageRecoloring(request):
	file_form = ImageRecoloringForm(request.POST, request.FILES)

	if request.method == 'POST':
		if request.FILES.get('source_image'):
			if file_form.is_valid():
				image_uid = get_random_string(length=32)
				output_dir_path = os.path.join(settings.MEDIA_ROOT, image_uid)
				os.makedirs(output_dir_path, exist_ok=True)

				input_image = Image.open(file_form.cleaned_data['source_image'])
				
				resize_length = 250
				if input_image.width > resize_length or input_image.height > resize_length:
					print("resize")
					if input_image.width > input_image.height:
						input_image = input_image.resize((resize_length, int(input_image.height * resize_length / input_image.width)))
					else:
						input_image = input_image.resize((int(input_image.width * resize_length / input_image.height), resize_length))
				print(input_image.width, input_image.height)

				input_image.save(os.path.join(output_dir_path, 'source_image.png'), optimize=False, quality=100)
				np_input_image = np.asfarray(input_image.convert('RGBA'))
				
				request.session['image_uid'] = image_uid
				
				if _data_exist(output_dir_path):
					source_palette = _deserialize_and_load_numpy_array(image_uid, RecoloringInterimResultMode.source_palette)
				else:
					data_hull, RGBXY_mixing_weights, source_palette = _calculate_convex_hull_data_of_image(np_input_image, output_dir_path)
					_serialize_and_save_numpy_array(image_uid, RecoloringInterimResultMode.source_palette, source_palette)
					_serialize_and_save_numpy_array(image_uid, RecoloringInterimResultMode.image_size, np_input_image.shape)
					_serialize_and_save_numpy_array(image_uid, RecoloringInterimResultMode.image_data, (np_input_image[:, :, :3].reshape((-1, 3))[data_hull.vertices]).reshape((-1, 1, 3)) / 255.0)
					_serialize_and_save_sparse_matrix(image_uid, RecoloringInterimResultMode.RGBXY_mixing_weights, RGBXY_mixing_weights)
				
				request.session['image_loaded'] = True
				request.session['image_file_name'] = file_form.files['source_image'].name
				return render(request, 'imageRecoloring.html', {
					'file_form': ImageRecoloringForm(),
					'current_file_name': request.session['image_file_name'],
					'source_image_path': os.path.join(settings.MEDIA_URL, image_uid, 'source_image.png'),
					'source_palette': _rgb_to_hex_list(source_palette),
				})

		elif request.POST.get("recolor") or request.POST.get("random"):
			if not request.session['image_loaded']:
				return render(request, 'imageRecoloring.html', {
					'file_form': ImageRecoloringForm()
				})

			image_uid = request.session['image_uid']
			source_palette = _deserialize_and_load_numpy_array(image_uid, RecoloringInterimResultMode.source_palette)

			if request.POST.get("recolor"):
				target_palette_hex_str = request.POST.get('hex_input')
				validation_result = view_helper.validate_input(target_palette_hex_str, 1, 4)
				
				if not validation_result['result']:
					print("[Error]", validation_result["message"])
					messages.info(request, validation_result["message"])
					return render(request, 'imageRecoloring.html', {
						'file_form': ImageRecoloringForm(),
						'current_file_name': request.session['image_file_name'],
						'source_image_path': os.path.join(settings.MEDIA_URL, image_uid, 'source_image.png'),
						'source_palette': _rgb_to_hex_list(source_palette),
					})
				
				target_palette_hex_list = _make_a_list_from_serialized_hex_string(target_palette_hex_str)
				target_color_palette = ColorPalette(auto_fetched=False, colors=target_palette_hex_list)
			else:
				target_color_palette = ColorPalette(auto_fetched=True, palette_length=random.randint(5, 10))

			
			output_dir_path = os.path.join(settings.MEDIA_ROOT, image_uid)			

			sorted_source_palette, sorted_target_palette, t2s_rgb_palette, tanstal_recolored_image_path, t2sbuffer_recolored_image_path = _recoloring(image_uid, source_palette, target_color_palette)

			with open(os.path.join(output_dir_path, 'hex_data.txt'), 'w') as f:
				f.write('extracted_palette\t' + ' '.join(_rgb_to_hex_list(sorted_source_palette)) + '\n')
				f.write('sorted_target_palette\t' + ' '.join(_rgb_to_hex_list(sorted_target_palette)) + '\n')
				f.write('t2s_palette\t' + ' '.join(_rgb_to_hex_list(t2s_rgb_palette)) + '\n')

			print('\textracted_palette', _rgb_to_hex_list(sorted_source_palette))
			print('\tsorted_target_palette', _rgb_to_hex_list(sorted_target_palette))
			print('\tt2s_palette', _rgb_to_hex_list(t2s_rgb_palette))

			return render(request, 'imageRecoloring.html', {
				'file_form': ImageRecoloringForm(),
				'current_file_name': request.session['image_file_name'],
				'source_image_path': os.path.join(settings.MEDIA_URL, image_uid, 'source_image.png'),
				'source_palette': _rgb_to_hex_list(source_palette),
				'target_palette': target_color_palette.to_hex_list(),
				't2s_palette':_rgb_to_hex_list(t2s_rgb_palette),
				'tanetal_recoloring_path': os.path.join(settings.MEDIA_URL, image_uid, _get_recoloring_file_names(RecoloringResult.tan_et_al)[0]),
				't2sbuffer_recoloring_path': os.path.join(settings.MEDIA_URL, image_uid, _get_recoloring_file_names(RecoloringResult.ours)[0])
			})
		else:
			print("[ERROR] No such post name in imageRecoloring")
	else:
		request.session['image_loaded'] = False
		return render(request, 'imageRecoloring.html', {'file_form': ImageRecoloringForm()})	
	

def _make_a_list_from_serialized_hex_string(hex_string):
	hex_list = hex_string.split('#')[1:]
	return ['#' + h for h in hex_list]


def _get_random_palette_for_palette_navigator():
	# reference: https://django-orm-cookbook-ko.readthedocs.io/en/latest/random.html
	return PaletteNavigatorPaletteModel.objects.filter(data_source='palettes_colourLoversTop100').order_by("?").first()


def _calculate_convex_hull_data_of_image(np_input_image, output_dir_path, rereconstructed=False):
	# Compute RGBXY_mixing_weights.
	print("\tComputing RGBXY mixing weights ...")
	X, Y = np.mgrid[0:np_input_image.shape[0], 0:np_input_image.shape[1]]
	XY = np.dstack((X * 1.0 / np_input_image.shape[0], Y * 1.0 / np_input_image.shape[1]))
	RGBXY_data = np.dstack((np_input_image[:, :, :3] / 255.0, XY))
	print("\t\tConvexHull 5D...")
	data_hull = ConvexHull(RGBXY_data.reshape((-1, 5)))
	
	print("\t\tComputing W_RGBXY...")
	RGBXY_mixing_weights = Additive_mixing_layers_extraction.recover_ASAP_weights_using_scipy_delaunay(data_hull.points[data_hull.vertices], data_hull.points, option=3)
	print("\t\tFinish!")
	rgb_palette = None

	if not rereconstructed:
		# Using determined palette number
		data = np_input_image[:, :, :3].reshape((-1, 3))/255.0
		rgb_palette = Additive_mixing_layers_extraction.Hull_Simplification_determined_version(data, output_dir_path)
		rgb_palette = (np.asarray(rgb_palette) * 255).tolist()

	return data_hull, RGBXY_mixing_weights, rgb_palette
	

def _recoloring(image_uid, source_palette, target_color_palette):
	# sort w/ color palette and input image's palette
	print('\t\t\tDCCW calculation')
	color_palettes_hex_list = [_rgb_to_hex_list(source_palette), target_color_palette.to_hex_list()]
	source_to_target_closest_points, target_to_source_closest_points, sorted_indices = _get_dccw_closest_points(color_palettes_hex_list)
	
	sorted_source_palette = [source_palette[i] for i in sorted_indices[0]]

	print('\t\t\tSave Tans')
	# Result of Tan et al.'s recoloring
	sorted_target_palette = target_color_palette.get_values_in_order('RGB', sorted_indices[1])
	tanstal_recolored_image_path = _save_result_of_recoloring(sorted_target_palette, image_uid, RecoloringResult.tan_et_al)
	
	print('\t\t\tSave t2sbuffer')
	# Result of Target-to-source recoloring
	t2s_rgb_palette = _lab_to_rgb_list(target_to_source_closest_points)
	t2sbuffer_recolored_image_path = _save_result_of_recoloring(sorted_target_palette+t2s_rgb_palette, image_uid, RecoloringResult.ours)

	# save original palettes
	print('\t\t\tSave original')
	_save_image_from_hex_list(_rgb_to_hex_list(source_palette), os.path.join(settings.MEDIA_ROOT, image_uid, 'source_palette.png'))

	return sorted_source_palette, sorted_target_palette, t2s_rgb_palette, tanstal_recolored_image_path, t2sbuffer_recolored_image_path


def _save_result_of_recoloring(target_palette, image_uid, mode):
	layers = _reflect_weights_to_image_pixels(target_palette, image_uid)
	recolored_image = _reconstruct_image_from_colors_and_weights(target_palette, layers)

	image_name, palette_image_name = _get_recoloring_file_names(mode)
	recolored_image_path = os.path.join(settings.MEDIA_ROOT, image_uid, image_name)
	recolored_image.save(recolored_image_path, optimize=False, quality=100)
	_save_image_from_hex_list(_rgb_to_hex_list(target_palette), os.path.join(settings.MEDIA_ROOT, image_uid, palette_image_name))

	return recolored_image_path


def _reflect_weights_to_image_pixels(target_palette, image_uid):
	image_size = _deserialize_and_load_numpy_array(image_uid, RecoloringInterimResultMode.image_size)
	image_data = _deserialize_and_load_numpy_array(image_uid, RecoloringInterimResultMode.image_data)
	RGBXY_mixing_weights = _deserialize_and_load_sparse_matrix(image_uid, RecoloringInterimResultMode.RGBXY_mixing_weights)
	output_dir_path = os.path.join(settings.MEDIA_ROOT, image_uid)

	target_palette = np.array(target_palette) / 255
	num_layers = target_palette.shape[0]
	w_rgb = Additive_mixing_layers_extraction.Get_ASAP_weights_using_Tan_2016_triangulation_and_then_barycentric_coordinates(image_data, target_palette, output_dir_path, order=0)
	w_rgb = w_rgb.reshape((-1, num_layers))

	if USE_OPENCL:
		w_rgbxy_values = RGBXY_mixing_weights.data
		w_rgbxy_values = w_rgbxy_values.reshape((-1,6))
		w_rgbxy_indices = RGBXY_mixing_weights.indices.reshape((-1,6))

		mult, _ = pyopencl_example.prepare_openCL_multiplication(w_rgb, w_rgbxy_values, w_rgbxy_indices)
		final_mixing_weights = mult(w_rgb)

	else:
		final_mixing_weights = RGBXY_mixing_weights.dot(w_rgb)

	layers = final_mixing_weights.reshape((image_size[0], image_size[1], num_layers))
	
	return layers


def _lab_to_rgb_list(lab_list):
	palette = []
	for lab in lab_list:
		L, a, b = lab.tolist()
		lab = LabColor(L, a, b)
		rgb = convert_color(lab, sRGBColor, target_illuminant='d65')
		rgb = [_rgb_clamp(x) for x in rgb.get_upscaled_value_tuple()]
		palette.append(rgb)

	return palette


def _rgb_to_hex_list(rgb_list):
	hex_list = []
	for r, g, b in rgb_list:
		hex_str = "#{0:02x}{1:02x}{2:02x}".format(_rgb_clamp(r), _rgb_clamp(g), _rgb_clamp(b))
		hex_list.append(hex_str)
	return hex_list


def _rgb_clamp(x): 
	return int(max(0, min(round(x), 255)))


def _reconstruct_image_from_colors_and_weights(colors, weights):
	colors = np.array(colors) / 255
	height, width, weight_len = weights.shape
	reconstructed_image = Image.new("RGB", (width, height), color=0)
	reconstructed_image_pixels = reconstructed_image.load()

	result = np.zeros((height, width, 3))

	for weight_index in range(weight_len):
		cur_layer = weights[:, :, weight_index]
		for i in range(3):
			result[:, :, i] += 255 * cur_layer * colors[weight_index][i]

	reconstructed_image = Image.fromarray(result.astype('uint8'), 'RGB')
	return reconstructed_image


def _save_image_from_hex_list(hex_list, file_name):
	color_palette = ColorPalette(auto_fetched=False, colors=hex_list)
	color_block_width = 100

	bulked_up = color_palette.bulk_up(color_block_width)
	Image.fromarray(bulked_up).save(file_name)


def _get_dccw_closest_points(color_palettes_hex_list):
	palettes = ColorPalettes(
	    auto_fetched=False, color_palettes_list=color_palettes_hex_list, is_hex_list=True)
	source_palette = palettes.get_single_palettes_list()[0]
	target_palette = palettes.get_single_palettes_list()[1]

	similarity_measurer = SimilarityMeasurer(source_palette, target_palette, LabDistanceMode.CIEDE2000)

	source_to_target_closest_points, target_to_source_closest_points = similarity_measurer.closest_points_by_dccw()
	return source_to_target_closest_points, target_to_source_closest_points, similarity_measurer.get_palette_sorted_indices()
	