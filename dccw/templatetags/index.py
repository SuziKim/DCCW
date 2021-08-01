from django.utils.safestring import mark_safe
from django import template
register = template.Library()

@register.simple_tag
def merge_to_array(first_array, second_array):
    return mark_safe([first_array, second_array])

@register.simple_tag
def value_of_double_indices(indexable, first_index, second_index):    
	return indexable[first_index][second_index]

@register.filter
def index(indexable, i):
	return indexable[i]

@register.filter
def get_cspace_item(dictionary, key):
	list_of_tuple = dictionary.get(key)
	return [list(ele) for ele in list_of_tuple] 
