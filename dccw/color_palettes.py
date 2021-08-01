from dccw.color_palette import *

class ColorPalettes:
	def __init__(self, auto_fetched=True, palette_count=2, palette_lengths=[5,5], color_palettes_list=None, is_hex_list=False):
		# palette_length: length of the individual palette
		# color_palette_objects: a list of ColorPalette objects
		# colors_hex_list: a list of colors' hex

		# Mendatory combination:
		# 1. (auto_fetched=True, palette_count=k, palette_length=m)
		# 2. (auto_fetched=False, color_palettes_list=[], is_hex_list=True)
		#     ex) [['#232323', '#ababab'], ['#111111', '#ffffff']]
		# 3. (auto_fetched=False, color_palettes_list=[], is_hex_list=Fale)
		#     ex) [ColorPalette, ColorPalette]

		self.palettes = []

		if auto_fetched:
			for i in range(palette_count):
				palette = ColorPalette(auto_fetched=True, palette_length=palette_lengths[i])
				self.palettes.append(palette)

		else:			
			for palette_colors in color_palettes_list:
				if is_hex_list:
					palette = ColorPalette(auto_fetched=False, colors=palette_colors)
					self.palettes.append(palette)
				else:
					self.palettes.append(palette_colors)

	def get_single_palettes_list(self):
		return self.palettes

	def to_serialized_hex_string(self):
		# return shape: #0e638d#7ba9a0#e6d6cf,#e3a07f#d74649#000000
		return ','.join(self.to_hex_string_list())

	def to_hex_string_list(self):
		# return shape: ['#0e638d#7ba9a0', '#e6d6cf#e3a07f']
		return [p.to_serialized_hex_string() for p in self.palettes]

	def to_hex_list(self):
		# return shape: [['#0e638d', '#7ba9a0'], ['#e6d6cf', '#e3a07f']]
		return [p.to_hex_list() for p in self.palettes]

	def to_merged_hex_list(self):
		# return shape: ['#0e638d', '#7ba9a0', '#e6d6cf', '#e3a07f']
		multi_dimension_array = self.to_hex_list()
		flattened = []
		for arr in multi_dimension_array:
			flattened += arr
		return flattened

	def get_geo_coords(self):
		geo_coords = []
		for palette in self.palettes:
			geo_coord = palette.get_geo_coord()
			geo_coords.append([list(gc_tuple) for gc_tuple in geo_coord])

		return geo_coords

	def get_hue_values(self):
		hue_values = []

		for palette in self.palettes:
			hsl_list = palette.get_values('hsl', is_geo=False)
			hue_values.append([h for h, s, l in hsl_list])

		return hue_values

	def merge_to_single_palette(self):
		merged_hex_list = []
		original_palette_group = []

		for index, palette in enumerate(self.palettes):
			merged_hex_list += palette.get_values('hex')
			original_palette_group += [index for i in range(palette.length())]

		return ColorPalette(auto_fetched=False, colors=merged_hex_list, original_palette_group=original_palette_group)

	def maximum_variance_palette(self):
		# return palette with the maximum vairance 

		maximum_variance = - math.inf
		maximum_variance_palette = None
		maximum_index = -1
		for palette_index, palette in enumerate(self.palettes):
			variance = palette.variance('lab')
			if variance > maximum_variance:
				maximum_variance_palette = palette
				maximum_index = palette_index

		print('[maximum_variance_palette] basis palette: ', maximum_index)
		return maximum_variance_palette
