from django.contrib import admin
from .models import ColorPaletteModel, ColorPalettesModel, SimilarityMeasurementModel, PaletteNavigatorPaletteModel, PaletteNavigatorDistanceModel, ImageRecoloringModel

admin.site.register(ColorPaletteModel)
admin.site.register(ColorPalettesModel)
admin.site.register(SimilarityMeasurementModel)
admin.site.register(PaletteNavigatorPaletteModel)
admin.site.register(PaletteNavigatorDistanceModel)
admin.site.register(ImageRecoloringModel)