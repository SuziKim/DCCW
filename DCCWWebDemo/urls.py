"""DCCWWebDemo URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from dccw.views import *

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', menu, name='home'),
    path('sortSinglePalette', sortSinglePalette, name='sortSinglePalette'),
    path('sortMultiplePalettes', sortMultiplePalettes, name='sortMultiplePalettes'),
    path('measureSimilarity', measureSimilarity, name='measureSimilarity'),
	path('exp1/<int:palette_length>', experiment1, name='exp1'),
	path('exp2/<str:khtp_type>', experiment2, name='exp2'),
	path('exp3/<str:khtp_type>', experiment3, name='exp3'),
	path('exp4', experiment4, name='exp4'),
	path('exp5', experiment5, name='exp5'),
	path('exp6/<str:lhsp_type>', experiment6, name='exp6'),
	path('paletteInterpolation', paletteInterpolation),
	# path('paletteNavigation', paletteNavigation),
	path('imageRecoloring', imageRecoloring),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
