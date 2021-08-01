import os
import PIL.Image as Image  

class SimilarityDataset:
	def __init__(self, name, query_palettes, retrieval_palettes, swatches=None, save_images=False):
		# param palettes: [['source', 'numcolors', palette], ...]
		# ex) [['xxx.jpg', 3, color_palette, ...]
		self.name = name
		self.dataset_type = name.split('_')[0]
		
		self.query_palettes = query_palettes
		self.retrieval_palettes = retrieval_palettes
		self.swatches = swatches

		if save_images:
			self._save_dataset_images()


	def get_name(self):
		return self.name
	
	def get_query_palettes(self):
		return self.query_palettes

	def get_retrieval_palettes(self):
		return self.retrieval_palettes

	def get_gt_count(self, source):
		return len([1 for _,v in enumerate(self.retrieval_palettes) if v[0]==source])

	def _save_dataset_images(self):
		color_block_width = 100

		# make dataset directory
		dataset_image_dir = os.path.join('experiments', 'dataset_images')
		os.makedirs(dataset_image_dir, exist_ok=True)
		
		cur_datset_image_dir = os.path.join(dataset_image_dir, self.dataset_type)
		os.makedirs(cur_datset_image_dir, exist_ok=True)

		cur_dataset_dir = os.path.join(cur_datset_image_dir, self.name)
		os.makedirs(cur_dataset_dir, exist_ok=True)

		# save query palettes
		self._save_dataset_image(cur_dataset_dir, 'querys', self.query_palettes)

		# save retrieval palettes
		self._save_dataset_image(cur_dataset_dir, 'retrievals', self.retrieval_palettes)

		# save swatches
		self._save_dataset_image(cur_dataset_dir, 'swatches', self.swatches)
		
		
	def _save_dataset_image(self, cur_dataset_dir, data_type, data_list):
		color_block_width = 100
		datset_dir = os.path.join(cur_dataset_dir, data_type)
		os.makedirs(datset_dir, exist_ok=True)

		for index, (palette_source, _, palette) in enumerate(data_list):
			bulked_up = palette.bulk_up(color_block_width)
			image_name = '%d-%s.png' % (index, palette_source)
			path = os.path.join(datset_dir, image_name) 
			Image.fromarray(bulked_up).save(path)