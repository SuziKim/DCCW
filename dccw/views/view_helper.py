from dccw.color_palettes import *

import string

def fetch_new_palettes(palette_count, color_count, is_color_count_a_list=False):
	color_counts = color_count
	if not is_color_count_a_list:
		color_counts = [color_count for i in range(palette_count)]

	return ColorPalettes(auto_fetched=True, 
						palette_count=palette_count, 
						palette_lengths=color_counts)

def validate_input(input_string, palette_count, minimum_color_in_palette_count=3):
	palettes = list(filter(None, input_string.split(',')))
	
	if palette_count == -1:
		if len(palettes) < 2:
			return {'result': False, 'message': "Enter more than two palettes separated with comma"}	
	elif len(palettes) != palette_count:
		return {'result': False, 'message': "Invalid palette counts"}
	
	for palette in palettes:
		hexes = list(filter(None, palette.split('#')))

		if len(hexes) < minimum_color_in_palette_count: 
			return {'result': False, 'message': "At least %d hex colors are required." % minimum_color_in_palette_count}
			
		if palette[0] != '#':
			return {'result': False, 'message': "Hex input is not starting with #."}

		if all(all(k in string.hexdigits for k in hex) for hex in hexes) is False:
			return {'result': False, 'message': "Invalid hex input"}

		if all(len(hex)==6 for hex in hexes) is False:
			return {'result': False, 'message': "Invalid hex input"}
	
	return {'result': True}
