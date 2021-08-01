import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from dccw.single_palette_sorter import * 

class ComprehensiveSinglePaletteSorter:
	def __init__(self, palette, target_spaces):
		# palette: ColorPalette Object
		# target_space: ['rgb', 'hsl', 'hsv', 'lab', 'lch']

		self.standard_sorter = SinglePaletteSorter(palette)

		self.palette_sorters = {}
		for target_space in target_spaces:
			self.palette_sorters[target_space] = SinglePaletteSorter(palette, target_space)
		
	#================
	# Sort Functions
	#================
	def lex_sort(self):
		sorted_result = {}

		for target_space, sorter in self.palette_sorters.items():
			sorted_result[target_space.lower()] = sorter.lex_sort()

		return sorted_result

	def standard_sort(self):
		return self.standard_sorter.standard_sort()
		