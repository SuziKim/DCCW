from django import forms 
from dccw.models import ImageRecoloringModel

class ImageRecoloringForm(forms.ModelForm):
    class Meta:
        model = ImageRecoloringModel
        fields = ["source_image",]
        labels = {
            "source_image": "",
        }
        widgets = {
            "source_image": forms.FileInput(attrs={'class': 'w-full flex-shrink-0 bg-indigo-500 hover:bg-indigo-700 border-indigo-500 hover:border-indigo-700 text-sm border-4 text-white py-1 px-2 rounded'}),
        }
        