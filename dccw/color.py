import math
import random
from colormath.color_conversions import *
from colormath.color_objects import *


class Color:
	def __init__(self, source, is_RGB = True):
		"""
		Initialize the color object.

		@type	source:	list or string
		@param	source:	rgb vector w/ range [0,255] or hex string
		@type	is_RGB:	Boolean
		@param	is_RGB:	True for RGB and False for HEX

		@rtype	:	None
		@return :	None
		"""

		if is_RGB:
			self.rgb = sRGBColor(source[0], source[1], source[2], is_upscaled=True)
		else:
			self.rgb = sRGBColor.new_from_rgb_hex(source)

		self.illuminant = 'd65'
		self.observer = '2'

	def change_color_to_random(self):
		rd = lambda: random.randint(0,255)
		hex_string = '#%02X%02X%02X' % (rd(),rd(),rd())
		self.rgb = sRGBColor.new_from_rgb_hex(hex_string)

	def jitter(self, jitter_offset):
		rd = lambda: random.randint(0,jitter_offset) - jitter_offset*2 # range: [-jitter_offset, jitter_offset]
		cap = lambda k: min(max(0,k), 255)

		r, g, b = self.RGB_value(is_geo=False)
		hex_string = '#%02X%02X%02X' % (cap(rd()+r), cap(rd()+g), cap(rd()+b))
		self.rgb = sRGBColor.new_from_rgb_hex(hex_string)

	#=============
	# RGB
	#=============
	def RGB(self):
		return self.rgb

	def RGB_value(self, is_geo, scale_up=True):
		if is_geo:
			return self._RGB_geo()
		else:
			return self._RGB_lex(scale_up)

	def _RGB_lex(self, scale_up):
		if scale_up:
			return self.RGB().get_upscaled_value_tuple()
		else:
			return self.RGB().get_value_tuple()

	def _RGB_geo(self):
		return tuple([v - 0.5 for v in self._RGB_lex(False)])


	#=============
	# HEX
	#=============
	def HEX_value(self):
		return self.RGB().get_rgb_hex()


	#=============
	# LAB
	#=============
	def LAB(self):
		return convert_color(self.rgb, LabColor)

	def LAB_value(self, is_geo):
		if is_geo:
			return self._LAB_geo()
		else:
			return self._LAB_lex()

	def _LAB_lex(self):
		return self.LAB().get_value_tuple()

	def _LAB_geo(self):
		L, A, B = self._LAB_lex()

		x = A / 256
		y = B / 256
		z = (L / 100) - 0.5

		return (x, y, z)


	#=============
	# LCH
	#=============
	def LCH(self):
		return convert_color(self.rgb, LCHabColor, target_illuminant=self.illuminant)

	def LCH_value(self, is_geo):
		if is_geo:
			return self._LCH_geo()
		else:
			return self._LCH_lex()

	def _LCH_lex(self):
		return self.LCH().get_value_tuple()

	def _LCH_geo(self):
		L, C, H = self._LCH_lex()

		theta = H / 360 * 2 * math.pi
		r = C / 200 / 2
		x = r * math.cos(theta)
		y = r * math.sin(theta)
		z = L / 100 - 0.5

		return (x, y, z)
		

	#=============
	# HSL
	#=============
	def HSL(self):
		return convert_color(self.rgb, HSLColor)

	def HSL_value(self, is_geo):
		if is_geo:
			return self._HSL_geo()
		else:
			return self._HSL_lex()

	def _HSL_lex(self):
		return self.HSL().get_value_tuple()

	def _HSL_geo(self):
		H, S, L = self._HSL_lex()

		r = S / 2
		theta = H / 360 * 2 * math.pi
		x = r * math.cos(theta)
		y = r * math.sin(theta)
		z = L - 0.5

		return (x, y, z)


	#=============
	# HSV
	#=============
	def HSV(self):
		return convert_color(self.rgb, HSVColor)

	def HSV_value(self, is_geo):
		if is_geo:
			return self._HSV_geo()
		else:
			return self._HSV_lex()

	def _HSV_lex(self):
		return self.HSV().get_value_tuple()

	def _HSV_geo(self):
		H, S, V = self._HSV_lex()
		
		r = S / 2
		theta = H / 360 * 2 * math.pi
		x = r * math.cos(theta)
		y = r * math.sin(theta)
		z = V - 0.5

		return (x, y, z)
		
	#=============
	# VHS
	#=============
	def VHS(self):
		return convert_color(self.rgb, HSVColor)

	def VHS_value(self, is_geo):
		if is_geo:
			return self._VHS_geo()
		else:
			return self._VHS_lex()

	def _VHS_lex(self):
		H, S, V = self.VHS().get_value_tuple()
		return (V, H, S)

	def _VHS_geo(self):
		V, H, S = self._VHS_lex()
		
		r = S / 2
		theta = H / 360 * 2 * math.pi
		x = r * math.cos(theta)
		y = r * math.sin(theta)
		z = V - 0.5

		return (z, x, y)
		