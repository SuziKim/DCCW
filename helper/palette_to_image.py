import argparse
import os
import PIL.Image as Image  

import sys
sys.path.append(".")
sys.path.append("..")

from dccw.color_palette import *

def save(hex_list, file_name):
	image_dir = 'palette_images'
	os.makedirs(image_dir, exist_ok=True)

	color_palette = ColorPalette(auto_fetched=False, colors=hex_list)
	color_block_width = 100

	bulked_up = color_palette.bulk_up(color_block_width)
	image_name = '%s.png' % (file_name)
	path = os.path.join(image_dir, image_name) 
	Image.fromarray(bulked_up).save(path)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--hexes', nargs="+", required=True)
	parser.add_argument('--filename', required=False, default="palette")

	args = parser.parse_args()
	# hexes: "aa3820 e7a557 d9d6ac 0c8fa7 255a58 373a3c 5b6057 6e9d95 e5d7af ebbb6d"
	hex_list = ["#%s" % hex for hex in args.hexes]

	save(hex_list, args.filename)