# -*- coding: utf-8 -*-
from __future__ import print_function, division

import time
import warnings
import json
import time
import PIL.Image as Image
import scipy.sparse
import scipy.optimize
import scipy
import os

from .Convexhull_simplification import *
from .trimesh import *

import pyximport
pyximport.install(reload_support=True)
from .GteDistPointTriangle import *


global DEMO
DEMO=False


### assume data is in range(0,1)
def Hull_Simplification_unspecified_M(data, output_dir_path, start_save=10):
#     hull=ConvexHull(data.reshape((-1,3)), qhull_options="Qs")
	hull=ConvexHull(data.reshape((-1,3)))
	origin_vertices=hull.points[ hull.vertices ]
	print ("original hull vertices number: ", len(hull.vertices))
	# with open( output_prefix+"-original_hull_vertices.js", 'w' ) as myfile:
	#     json.dump({'vs': (hull.points[ hull.vertices ].clip(0.0,1.0)*255).tolist(),'faces': (hull.points[ hull.simplices ].clip(0.0,1.0)*255).tolist()}, myfile, indent = 4 )
    
	output_rawhull_obj_file = os.path.join(output_dir_path, "mesh_obj_files.obj")
	write_convexhull_into_obj_file(hull, output_rawhull_obj_file)    
    
	max_loop=5000
	for i in range(max_loop):
		mesh=TriMesh.FromOBJ_FileName(output_rawhull_obj_file)
		old_num=len(mesh.vs)
		old_vertices=mesh.vs
		mesh=remove_one_edge_by_finding_smallest_adding_volume_with_test_conditions(mesh,option=2)
		#         newhull=ConvexHull(mesh.vs, qhull_options="Qs")
		hull=ConvexHull(mesh.vs)
		write_convexhull_into_obj_file(hull, output_rawhull_obj_file)

		if len(hull.vertices) <= start_save:
			name = os.path.join(output_dir_path, "%02d.js" % len(hull.vertices))
			with open( name, 'w' ) as myfile:
				json.dump({'vs': (hull.points[ hull.vertices ].clip(0.0,1.0)*255).tolist(),'faces': (hull.points[ hull.simplices ].clip(0.0,1.0)*255).tolist()}, myfile, indent = 4 )

		if len(hull.vertices)==old_num or len(hull.vertices)==4:
			return


def Hull_Simplification_old(arr, M, output_dir_path):
	hull = ConvexHull(arr.reshape((-1,3)))
	# print hull.points[hull.vertices].shape
	output_rawhull_obj_file = os.path.join(output_dir_path, "mesh_obj_files.obj")
	write_convexhull_into_obj_file(hull, output_rawhull_obj_file)
	mesh = TriMesh.FromOBJ_FileName(output_rawhull_obj_file)
		
	max_loop = 1000
	for i in range(max_loop):
		old_num = len(mesh.vs)
		mesh = TriMesh.FromOBJ_FileName(output_rawhull_obj_file)
		mesh = remove_one_edge_by_finding_smallest_adding_volume_with_test_conditions(mesh,option=2)
		newhull = ConvexHull(mesh.vs)
		write_convexhull_into_obj_file(newhull, output_rawhull_obj_file)
		
		if len(mesh.vs) == M or len(newhull.vertices) == old_num or len(newhull.vertices) == 4:
			Final_hull=newhull
			break

	Hull_vertices = Final_hull.points[Final_hull.vertices].clip(0,1)
	return Hull_vertices

def get_unique_colors_and_their_counts(arr):
	unique_colors, counts = np.unique(arr, axis=0, return_counts=True)
	return unique_colors, counts
		
def outsidehull_points_distance_unique_data_version(hull_vertices, points, counts):
	######### here, points are unique pixel colors, it will be faster than directly give all pixel colors.

	hull = ConvexHull(hull_vertices)
	de = Delaunay(hull_vertices)
	ind = de.find_simplex(points, tol=1e-8)
	total_distance = []
	for i in range(points.shape[0]):
		if ind[i] < 0:
			dist_list = []
			for j in range(hull.simplices.shape[0]):
				result = DCPPointTriangle( points[i], hull.points[hull.simplices[j]] )
				dist_list.append(result['distance'])
			total_distance.append(min(dist_list))
	total_distance = np.asarray(total_distance)
		
	return (((total_distance**2)*counts[ind<0]).sum()/counts.sum()) ** 0.5


def Get_ASAP_weights_using_Tan_2016_triangulation_and_then_barycentric_coordinates(img_label_origin, origin_order_tetra_prime, output_dir_path, order=0):
	img_label = img_label_origin.copy() ### do not modify img_label_origin
		
	if isinstance(order, (list, tuple, np.ndarray)):
		pass
		
	elif order == 0: ## use black as first pigment
		diff=abs(origin_order_tetra_prime-np.array([[0, 0, 0]])).sum(axis=-1)
		order=np.argsort(diff)
		
	elif order == 1: ## use white
		diff=abs(origin_order_tetra_prime-np.array([[1, 1, 1]])).sum(axis=-1)
		order=np.argsort(diff)

	tetra_prime = origin_order_tetra_prime[order]
	# print (tetra_prime[0])

	img_shape = img_label.shape
	img_label = img_label.reshape((-1, 3))
	img_label_backup = img_label.copy()

	hull = ConvexHull(tetra_prime)
	test_inside = Delaunay(tetra_prime)
	label = test_inside.find_simplex(img_label,tol=1e-8)
	# print len(label[label==-1])

	### modify img_label[] to make all points are inside the simplified convexhull
	for i in range(img_label.shape[0]):
	#	 print i
		if label[i] < 0:
			dist_list = []
			cloest_points = []
			for j in range(hull.simplices.shape[0]):
				result = DCPPointTriangle( img_label[i], hull.points[hull.simplices[j]] )
				dist_list.append(result['distance'])
				cloest_points.append(result['closest'])
			dist_list = np.asarray(dist_list)
			index = np.argmin(dist_list)
			img_label[i] = cloest_points[index]

	### assert
	test_inside = Delaunay(tetra_prime)
	label = test_inside.find_simplex(img_label, tol=1e-8)
	assert(len(label[label == -1]) == 0)

	### colors2xy dict
	colors2xy = {}
	unique_image_label = list(set(list(tuple(element) for element in img_label)))

	for element in unique_image_label:
		colors2xy.setdefault(tuple(element), [])
		
	for index in range(len(img_label)):
		element = img_label[index]
		colors2xy[tuple(element)].append(index)

	unique_colors = np.array(list(colors2xy.keys()))
	unique_image_label = unique_colors.copy()
	vertices_list = tetra_prime

	tetra_pixel_dict = {}
	for face_vertex_ind in hull.simplices:
		# print face_vertex_ind
		if (face_vertex_ind != 0).all():
			i, j, k = face_vertex_ind
			tetra_pixel_dict.setdefault(tuple((i,j,k)),[])

	index_list = np.array(list(np.arange(len(unique_image_label))))

	for face_vertex_ind in hull.simplices:
		if (face_vertex_ind != 0).all():
			# print face_vertex_ind
			i, j, k = face_vertex_ind
			tetra = np.array([vertices_list[0], vertices_list[i], vertices_list[j], vertices_list[k]])
			try:
				#### use try here, because sometimes the tetra is nearly flat, will cause qhull error to stop, we do not want to stop, we just skip.
	#			 print (tetra)
				test_Del = Delaunay(tetra)
				# print len(index_list)
				if len(index_list) != 0:
					label = test_Del.find_simplex(unique_image_label[index_list],tol=1e-8)
					chosen_index = list(index_list[label>=0])
					tetra_pixel_dict[tuple((i,j,k))] += chosen_index
					index_list = np.array(list(set(index_list) - set(chosen_index)))
			except Exception as e:
				pass
				# print (tetra)
				# print (e)

	# print index_list
	assert(len(index_list) == 0)

	pixel_num = 0
	for key in tetra_pixel_dict:
		pixel_num += len(tetra_pixel_dict[key])
	assert(pixel_num == unique_image_label.shape[0])

	### input is like (0,1,2,3,4) then shortest_path_order is (1,2,3,4), 0th is background color, usually is white
	shortest_path_order = tuple(np.arange(len(tetra_prime))[1:])

	unique_weights_list = np.zeros((unique_image_label.shape[0],len(tetra_prime)))

	for vertice_tuple in tetra_pixel_dict:
		vertice_index_inglobalorder = np.asarray(shortest_path_order)[np.asarray(sorted(list(shortest_path_order).index(s) for s in vertice_tuple))]
		vertice_index_inglobalorder_tuple = tuple(list(vertice_index_inglobalorder))
				
		colors = np.array([vertices_list[0],
						 vertices_list[vertice_index_inglobalorder_tuple[0]],
						 vertices_list[vertice_index_inglobalorder_tuple[1]],
						 vertices_list[vertice_index_inglobalorder_tuple[2]]
						])
						
		pixel_index = np.array(tetra_pixel_dict[vertice_tuple])
		if len(pixel_index) != 0:
			arr = unique_image_label[pixel_index]
			Y = recover_ASAP_weights_using_scipy_delaunay(colors, arr)
			unique_weights_list[pixel_index[:,None], np.array([0]+list(vertice_index_inglobalorder_tuple))] = Y.reshape((arr.shape[0],-1))

	#### from unique weights to original shape weights
	mixing_weights = np.zeros((len(img_label), len(tetra_prime)))
	for index in range(len(unique_image_label)):
		element = unique_image_label[index]
		index_list = colors2xy[tuple(element)]
		mixing_weights[index_list, :] = unique_weights_list[index, :]

	origin_order_mixing_weights = np.ones(mixing_weights.shape)
	#### to make the weights order is same as orignal input vertex order
	origin_order_mixing_weights[:, order] = mixing_weights

	origin_order_mixing_weights = origin_order_mixing_weights.reshape((img_shape[0], img_shape[1], -1))
	temp = (origin_order_mixing_weights.reshape((img_shape[0], img_shape[1], -1, 1))*origin_order_tetra_prime.reshape((1, 1, -1, 3))).sum(axis=2)
	img_diff = temp.reshape(img_label_origin.shape)*255 - img_label_origin*255
	diff = square(img_diff.reshape((-1, 3))).sum(axis=-1)

	# print ('max diff: ', sqrt(diff).max())
	# print ('median diff', median(sqrt(diff)))
	# print ('RMSE: ', sqrt(diff.sum()/diff.shape[0]))

	mixing_weights_filename = os.path.join(output_dir_path, str(len(origin_order_tetra_prime)) + "-RGB_ASAP-using_Tan2016_triangulation_and_then_barycentric_coordinates-linear_mixing-weights.js")
	with open(mixing_weights_filename, 'w') as myfile:
		json.dump({'weights': origin_order_mixing_weights.tolist()}, myfile)

	for i in range(origin_order_mixing_weights.shape[-1]):
		mixing_weights_map_filename = os.path.join(output_dir_path, str(len(origin_order_tetra_prime))+ "-RGB_ASAP-using_Tan2016_triangulation_and_then_barycentric_coordinates-linear_mixing-weights_map-%02d.png" % i)
		Image.fromarray((origin_order_mixing_weights[:,:,i]*255).round().clip(0,255).astype(uint8)).save(mixing_weights_map_filename)

	return origin_order_mixing_weights


def recover_ASAP_weights_using_scipy_delaunay(Hull_vertices, data, option=1):
	###modified from https://codereview.stackexchange.com/questions/41024/faster-computation-of-barycentric-coordinates-for-many-points (Gareth Rees)
	# Load points
	points = Hull_vertices
	# Load targets
	targets = data
	ntargets = len(targets)

	# Compute Delaunay triangulation of points.
	tri = Delaunay(points)

	# Find the tetrahedron containing each target (or -1 if not found)
	tetrahedra = tri.find_simplex(targets, tol=1e-6)

	# Affine transformation for tetrahedron containing each target
	X = tri.transform[tetrahedra, :data.shape[1]]

	# Offset of each target from the origin of its containing tetrahedron
	Y = targets - tri.transform[tetrahedra, data.shape[1]]

	# First three barycentric coordinates of each target in its tetrahedron.
	# The fourth coordinate would be 1 - b.sum(axis=1), but we don't need it.
	b = np.einsum('...jk,...k->...j', X, Y)
	barycoords = np.c_[b, 1-b.sum(axis=1)]
		
	############# this is slow for large size weights like N*1000
	if option == 1:
		weights_list = np.zeros((targets.shape[0], points.shape[0]))
		num_tetra = len(tri.simplices)
		all_index = np.arange(len(targets))
		for i in range(num_tetra):
			weights_list[all_index[tetrahedra == i][:,None], np.array(tri.simplices[i])]= barycoords[all_index[tetrahedra == i],:]

	elif option == 2:
		rows = np.repeat(np.arange(len(data)).reshape((-1,1)), len(tri.simplices[0]), 1).ravel().tolist()
		cols = []
		vals = []

		for i in range(len(data)):
			cols += tri.simplices[tetrahedra[i]].tolist()
			vals += barycoords[i].tolist()
		weights_list = scipy.sparse.coo_matrix( ( vals, ( rows, cols ) ), shape = ( len(data), len(Hull_vertices)) ).tocsr()
		
	elif option == 3:
		rows = np.repeat(np.arange(len(data)).reshape((-1,1)), len(tri.simplices[0]), 1).ravel()
		
		cols = tri.simplices[tetrahedra].ravel()
		vals = barycoords.ravel()
		weights_list = scipy.sparse.coo_matrix( ( vals, ( rows, cols ) ), shape = ( len(data), len(Hull_vertices)) ).tocsr()

	return weights_list


def Hull_Simplification_determined_version(data, output_dir_path, num_thres=0.1, error_thres=2.0/255.0, SAVE=True, option="use_quantitized_colors"):
	hull = ConvexHull(data.reshape((-1,3)))
	origin_vertices = hull.points[ hull.vertices ]
		
	output_rawhull_obj_file = os.path.join(output_dir_path, "mesh_obj_files.obj")
	write_convexhull_into_obj_file(hull, output_rawhull_obj_file)	
		
	if option == "unique_pixel_colors":
		unique_data, pixel_counts = get_unique_colors_and_their_counts(data.reshape((-1,3)))
		
	elif option == "use_quantitized_colors":
		new_data=(((data*255).round().astype(np.uint8)//8)*8+4)/255.0
		unique_data, pixel_counts=get_unique_colors_and_their_counts(new_data.reshape((-1,3)))
	   
		
	max_loop = 1000
	for i in range(max_loop):
		# if i % 10 == 0:
			# print ("loop: ", i)

		mesh = TriMesh.FromOBJ_FileName(output_rawhull_obj_file)
		old_num = len(mesh.vs)
		old_vertices = mesh.vs
		
		mesh = remove_one_edge_by_finding_smallest_adding_volume_with_test_conditions(mesh, option=2)
		hull = ConvexHull(mesh.vs)
		write_convexhull_into_obj_file(hull, output_rawhull_obj_file)
		
		if len(hull.vertices) <= 10:
			if option == "all_pixel_colors": ### basic one.
				reconstruction_errors = outsidehull_points_distance(hull.points[ hull.vertices ].clip(0.0, 1.0), data.reshape((-1, 3)))		

			elif option == "unique_pixel_colors": ### results should be same with above opiton, but faster
				reconstruction_errors = outsidehull_points_distance_unique_data_version(hull.points[ hull.vertices ].clip(0.0, 1.0), unique_data, pixel_counts)

			elif option == "origin_convexhull_vertices": 
				reconstruction_errors = outsidehull_points_distance_for_using_origin_hull_vertices(hull.points[ hull.vertices ].clip(0.0, 1.0), data.reshape((-1, 3)), origin_vertices.reshape((-1, 3))) ### may use 5/255.0 to be threshold.

			elif option == "use_quantitized_colors":
				reconstruction_errors = outsidehull_points_distance_unique_data_version(hull.points[ hull.vertices ].clip(0.0, 1.0), unique_data, pixel_counts)
				
			# print reconstruction_errors
			if reconstruction_errors > error_thres:
				oldhull = ConvexHull(old_vertices)
				if SAVE:
					name = os.path.join(output_dir_path, "%02d.js" % len(oldhull.vertices))
					with open( name, 'w' ) as myfile:
						json.dump({'vs': (oldhull.points[ oldhull.vertices ].clip(0.0, 1.0)*255).tolist(),'faces': (oldhull.points[ oldhull.simplices ].clip(0.0, 1.0)*255).tolist()}, myfile, indent = 4 )

				return oldhull.points[ oldhull.vertices ].clip(0.0, 1.0)
	   
		if len(hull.vertices) == old_num or len(hull.vertices) == 4:
			if SAVE:
				name = os.path.join(output_dir_path, "%02d.js" % len(hull.vertices))
				with open( name, 'w' ) as myfile:
					json.dump({'vs': (hull.points[ hull.vertices ].clip(0.0, 1.0)*255).tolist(),'faces': (hull.points[ hull.simplices ].clip(0.0, 1.0)*255).tolist()}, myfile, indent = 4 )

			return hull.points[ hull.vertices ].clip(0.0, 1.0)
		
		
