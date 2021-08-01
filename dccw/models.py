from django.utils.timezone import now
from django.db import models
from django.utils.crypto import get_random_string

import os

class ColorPaletteModel(models.Model):
	colorCount = models.IntegerField()
	hexes = models.TextField(primary_key=True)
	created_at = models.DateTimeField(default=now)
	
	def __str__(self):
		return self.hexes


class ColorPalettesModel(models.Model):
	colorCount = models.IntegerField()
	paletteCount = models.IntegerField()
	hexes_list = models.TextField(primary_key=True)
	created_at = models.DateTimeField(default=now)

	def __str__(self):
		return self.hexes_list

class SimilarityMeasurementModel(models.Model):
	colorCount = models.IntegerField() # no of colors in a single palette
	targetPalettesCount = models.IntegerField() # no of target palettes
	hexes_list = models.TextField(primary_key=True)
	created_at = models.DateTimeField(default=now)

class PaletteNavigatorPaletteModel(models.Model):
	color_count = models.IntegerField(default=0)
	hexes_list = models.TextField(default='')
	palette_id = models.TextField(default='')
	data_source = models.TextField(default='')
	
	class Meta:
		unique_together = (("palette_id", "color_count"),)


class PaletteNavigatorDistanceModel(models.Model):
	palette1 = models.ForeignKey('PaletteNavigatorPaletteModel', on_delete=models.CASCADE, related_name='palette1')
	palette2 = models.ForeignKey('PaletteNavigatorPaletteModel', on_delete=models.CASCADE, related_name='palett2')
	distance = models.FloatField()

	class Meta:
		unique_together = (("palette1", "palette2"), )
	
	
class ImageRecoloringModel(models.Model):
	source_image = models.ImageField()