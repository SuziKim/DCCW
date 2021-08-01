import random
import requests, json, math
import numpy as np 
import requests
import json
from colormath.color_diff import delta_e_cie2000

from dccw.color import *

class ColorPalette:
	def __init__(self, auto_fetched=True, palette_length=1, colors=None, original_palette_group=None):
		# colors: hex list
		# original_palette_group: This is for the multiple palettes sorter. It shows the each color's palette number before the palettes are merged together.

		self.colors = []

		if auto_fetched:
			palette = self._fetch_palette(palette_length)
			for c in palette:
				self.colors.append(Color(c))
		else:
			for c in colors:
				self.colors.append(Color(c, is_RGB=False))

			if original_palette_group:
				self.original_palette_group = original_palette_group


	#================
	# Helpers
	#================
	def to_serialized_hex_string(self):
		return ''.join([c.HEX_value() for c in self.colors])

	def to_hex_list(self, order=None):
		# return shape: ['#0e638d', '#7ba9a0', '#e6d6cf', '#e3a07f']
		if order:
			return [c for c in self.get_values_in_order('hex', order)]
		else:
			return [c for c in self.get_values('hex')]

	def length(self):
		return len(self.colors)

	def variance(self, color_space, is_geo=True):
		values = self.get_values(color_space, is_geo)
		values_np = np.array(values)
		return values_np.var()

	def get_original_palette_group(self):
		return self.original_palette_group

	def get_geo_coord(self):
		geo_coord = self.get_values('lab', is_geo=True)
		return [list(gc_tuple) for gc_tuple in geo_coord]

	def bulk_up(self, width=100, add_white_line=False):
		rgb_list = np.array(self.get_values('rgb'))
		rgb_list = rgb_list.reshape((rgb_list.shape[0], 1, rgb_list.shape[1])).astype('uint8')

		bulked_up = np.repeat(np.repeat(np.moveaxis(rgb_list, 0, 1), width, 0), width, 1)

		if add_white_line:
			white_width = 5
			for i in reversed(range(1,self.length())):
				bulked_up = np.insert(bulked_up, [i*width for k in range(white_width)], [255], axis=1)
		
		return bulked_up


	def change_partial_color_with_random(self, random_count):
		for color_index in range(self.length()-1, max(0, self.length() - random_count), -1):
			while True:
				# change color unless the color is not existing in current palette
				self.colors[color_index].change_color_to_random()
				changed_color = self.colors[color_index].RGB_value(is_geo=False)
				cur_colors = [color.RGB_value(is_geo=False) for index, color in enumerate(self.colors) if index != color_index]
				if changed_color not in cur_colors:
					break
			
	def jitter_colors(self, jitter_offset):
		for color_index, color in enumerate(self.colors):
			while True:
				color.jitter(jitter_offset)
				jittered_color = color.RGB_value(is_geo=False)
				cur_colors = [c.RGB_value(is_geo=False) for index, c in enumerate(self.colors) if index != color_index]
				if jittered_color not in cur_colors:
					break

	def get_graph_length_in_order(self, order):
		return sum(self.get_graph_length_list_in_order(order))
		
	def get_graph_length_list_in_order(self, order):
		colors = self.get_color_objects_in_order('lab', order)
		lengths = []
		for i in range(len(colors) - 1):
			cur_color = colors[i]
			next_color = colors[i+1]
			distance = self._get_CIEDE2000_distance_between_two_labs(
				cur_color, next_color)
			lengths.append(distance)

		return lengths

	def _get_CIEDE2000_distance_between_two_labs(self, lab_a, lab_b):
		return delta_e_cie2000(lab_a, lab_b)


	#================
	# Getters : Values
	#================
	def get_values_in_order(self, color_space, order, is_geo=False):
		return self._get_values(color_space, order, is_geo)

	def get_values(self, color_space, is_geo=False):
		return self._get_values(color_space, list(range(len(self.colors))), is_geo)

	def _get_values(self, color_space, order, is_geo):
		if color_space.lower() == 'rgb':
			return self._get_RGB_values(order, is_geo)
			
		elif color_space.lower() == 'hex':
			return self._get_HEX_values(order)
			
		elif color_space.lower() == 'lab':
			return self._get_LAB_values(order, is_geo)
			
		elif color_space.lower() == 'hsl':
			return self._get_HSL_values(order, is_geo)
			
		elif color_space.lower() == 'hsv':
			return self._get_HSV_values(order, is_geo)

		elif color_space.lower() == 'vhs':
			return self._get_VHS_values(order, is_geo)
			
		elif color_space.lower() == 'lch':
			return self._get_LCH_values(order, is_geo)

		else:
			print("[ColorPalette] Wrong Color Space..")
			return None


	def _get_RGB_values(self, order, is_geo=False):
		result = []
		for idx in order:
			result.append(self.colors[idx].RGB_value(is_geo))
		return result

	def _get_HEX_values(self, order):
		result = []
		for idx in order:
			result.append(self.colors[idx].HEX_value())
		return result

	def _get_LAB_values(self, order, is_geo=False):
		result = []
		for idx in order:
			result.append(self.colors[idx].LAB_value(is_geo=is_geo))
		return result

	def _get_LCH_values(self, order, is_geo=False):
		result = []
		for idx in order:
			result.append(self.colors[idx].LCH_value(is_geo=is_geo))
		return result

	def _get_HSL_values(self, order, is_geo=False):
		result = []
		for idx in order:
			result.append(self.colors[idx].HSL_value(is_geo=is_geo))
		return result

	def _get_HSV_values(self, order, is_geo=False):
		result = []
		for idx in order:
			result.append(self.colors[idx].HSV_value(is_geo=is_geo))
		return result

	def _get_VHS_values(self, order, is_geo=False):
		result = []
		for idx in order:
			result.append(self.colors[idx].VHS_value(is_geo=is_geo))
		return result


	#================
	# Getters : Objects
	#================
	def get_color_objects_in_order(self, color_space, order):
		return self._get_color_objects(color_space, order)

	def get_color_objects(self, color_space):
		return self._get_color_objects(color_space, list(range(len(self.colors))))

	def _get_color_objects(self, color_space, order):
		if color_space.lower() == 'rgb':
			return self._get_RGB_color_objects(order)
			
		elif color_space.lower() == 'lab':
			return self._get_LAB_color_objects(order)
			
		elif color_space.lower() == 'hsl':
			return self._get_HSL_color_objects(order)
			
		elif color_space.lower() == 'hsv':
			return self._get_HSV_color_objects(order)
			
		elif color_space.lower() == 'lch':
			return self._get_LCH_color_objects(order)

		else:
			print("[ColorPalette] Wrong Color Space..")
			return None

	def _get_RGB_color_objects(self, order):
		result = []
		for idx in order:
			result.append(self.colors[idx].RGB())
		return result

	def _get_LAB_color_objects(self, order):
		result = []
		for idx in order:
			result.append(self.colors[idx].LAB())
		return result

	def _get_LCH_color_objects(self, order):
		result = []
		for idx in order:
			result.append(self.colors[idx].LCH())
		return result

	def _get_HSL_color_objects(self, order):
		result = []
		for idx in order:
			result.append(self.colors[idx].HSL())
		return result

	def _get_HSV_color_objects(self, order):
		result = []
		for idx in order:
			result.append(self.colors[idx].HSV())
		return result


	#================
	# Fetch Palettes
	#================
	def _fetch_palette(self, count):
		# Return color array in RGB [0, 255]
		# source #1: http://colormind.io/api-access/

		print("[FetchColorScheme] Start to fetch ", count, "-colors...")

		colormind_url = "http://colormind.io/api/"		
		colormind_status_code = requests.get(colormind_url).status_code
		result = []

		
		while len(result) < count:
			if colormind_status_code != 200:
				print("[FetchColorScheme] Use Local")
				result = result + [[random.randint(0, 255) for i in range(3)]]				
			else:
				data_dict = {'model': 'default'} 
				res = requests.post(colormind_url, data=json.dumps(data_dict))
				sub_result = json.loads(res.text)['result']
				result = result + sub_result

		print('[FetchColorScheme] Results: %s' % result[:count])

		fetched_result = result[:count]
		random.shuffle(fetched_result)
		return fetched_result
