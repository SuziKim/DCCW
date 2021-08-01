import * as THREE from "/static/js/three.module.js";
import { OrbitControls } from '/static/js/threejs/OrbitControls.js';
import { Line2 } from '/static/js/threejs/Line2.js';
import { LineMaterial } from '/static/js/threejs/LineMaterial.js';
import { LineGeometry } from '/static/js/threejs/LineGeometry.js';

let allowed_color_space = ['rgb', 'hsl', 'hsv', 'lch', 'lab', 'vhs'];

export function load_graph(canvas_id, color_space, color_hexes, geo_coords) {
	// error check
	try{
		if (!allowed_color_space.includes(color_space.toLowerCase())) {
			throw new Error('There is no such a color space: ' + color_space);
		}

		init(canvas_id, color_space.toLowerCase(), color_hexes, geo_coords);
		animate();

	} catch (e) {
	  alert(e.name + ": " + e.message);
	}
}

export function load_multiple_palettes_graphs(canvas_id, color_hexes, geo_coords) {
	init_multiple_palettes(canvas_id, color_hexes, geo_coords);
	animate();
}

export function load_multiple_palettes_mask_graphs(canvas_id, geo_coords) {
    
    let color_count = geo_coords[0].length;
    let color_hexes = [Array(color_count).fill('#ff0000'), Array(color_count).fill('#0000ff')];
    
    init_multiple_palettes(canvas_id, color_hexes, geo_coords);
    animate();
}

export function add_lines(canvas_id, color_space, coord_indices, geo_coords, line_color) {
	let graph = find_graph(canvas_id);
	try {
		if (graph == null) {
			throw new Error('There is no graph named color space: ' + color_space);
		}
	} catch (e) {
		alert(e.name + ": " + e.message);
	}

	// Connect Spheres: orig indices
	let yz_converted_colors_coords = [];
	coord_indices.forEach(function(coord_index){
		let geo_coord = geo_coords[coord_index];
		yz_converted_colors_coords.push([geo_coord[0], geo_coord[2], geo_coord[1]])
	});

	graph.add_order_line_from_coords(yz_converted_colors_coords, line_color)
}


export function add_multiple_palettes_lines(canvas_id, palettes_coords_indices, geo_coords_palettes, line_colors) {
	// palettes_coords_indices: [[1,2,3,4,5],[1,2,3,4,5]]
	// geo_coords_palettes: palette_count * palette_length * 3
	// line_colors: [0x7e995377, 0xd9a73977, 0xbd080877]

	let graph = find_graph(canvas_id);
	try {
		if (graph == null) {
			throw new Error('There is no graph named color id: ' + canvas_id);
		}
	} catch (e) {
		alert(e.name + ": " + e.message);
	}

	// Connect Spheres: orig indices
	palettes_coords_indices.forEach(function(palette_coords_indices, palette_index){
		let yz_converted_colors_coords = [];

		palette_coords_indices.forEach(function(palette_coord_index) {
			let geo_coord = geo_coords_palettes[palette_index][palette_coord_index];
			yz_converted_colors_coords.push([geo_coord[0], geo_coord[2], geo_coord[1]])	
		});

		graph.add_order_line_from_coords(yz_converted_colors_coords, line_colors[palette_index])
	});
}


// ================================
// ================================

class Model {
	constructor(geometry, material) {
		this.geometry = geometry;
		this.material = material;
		this.mesh = new THREE.Mesh( geometry, material );
		this.visibility = true;
	}
}

class Grid {
	constructor() {
		this.yAxis = this.generate_y_axis();
	}

	generate_y_axis() {
		let material = new THREE.LineBasicMaterial( { color: 0x80AAD6 } );
		let points = [];
		points.push( new THREE.Vector3( 0, -0.8, 0));
		points.push( new THREE.Vector3( 0, 0.8, 0 ) );

		let geometry = new THREE.BufferGeometry().setFromPoints( points );
		return new THREE.Line( geometry, material );
	}
}

class Graph {
	constructor(camera, scene, renderer, color_space, canvas_id) {
		this.camera = camera;
		this.scene = scene;
		this.renderer = renderer;

		this.models = [];
		this.controls = new OrbitControls( camera, renderer.domElement );
		this.controls.update();

		this.color_space = color_space;
		this.canvas_id = canvas_id;

		this.update_color_polyhedron();

		this.grid = new Grid();
		this.scene.add(this.grid.yAxis);
	}

	// get color_space() {
	// 	return this.color_space;
	// }

	// get renderer() {
	// 	return this.renderer;
	// }

	// get camera() {
	// 	return this.camera;
	// }

	// get scene() {
	// 	return this.scene;
	// }

	is_canvas_id(canvas_id) {
		return this.canvas_id == canvas_id;
	}

	add_model(model) {
		this.models.push(model);

		if (model.visibility) {
			this.scene.add(model.mesh);
		}
	}

	add_order_line_from_coords(colors_coords, line_color) {
		let line_material = new THREE.LineBasicMaterial({ color: line_color });

		let points = [];
		colors_coords.forEach(function(color_coord) {
			points.push(new THREE.Vector3(color_coord[0], color_coord[1], color_coord[2]));
		});

		// let line_geometry = new THREE.BufferGeometry().setFromPoints( points );
		// let line = new THREE.Line( line_geometry, line_material );

		let line_geometry = new LineGeometry();
		line_geometry.setPositions( colors_coords.flat() );

		let colors = [];
		let color = new THREE.Color( line_color );

		for (let i=0; i<colors_coords.length; i++) {
			colors.push(color.r, color.g, color.b)
		}
		
		line_geometry.setColors(colors);
		line_material = new LineMaterial( {
			color: line_color,
			linewidth: 0.005, // in pixels
			vertexColors: THREE.VertexColors,
			dashed: false
		});

		let line = new Line2( line_geometry, line_material );
		line.computeLineDistances();
		// line.scale.set( 1, 1, 1 );
		this.scene.add(line);
	}

	update_color_polyhedron() {
		let geometry;
		let material = new THREE.MeshBasicMaterial({color: 0xd0d0d0, wireframe: true, wireframeLinewidth: 0.05});

		switch (this.color_space) {
			case 'rgb':
				geometry = new THREE.BoxGeometry( 1, 1, 1);
				break;
			case 'hsv':
				geometry = new THREE.CylinderGeometry(0.5, 0.5, 1, 8);				
				break;
			case 'vhs':
				geometry = new THREE.CylinderGeometry(0.5, 0.5, 1, 8);
				break;
			case 'hsl':
				geometry = new THREE.CylinderGeometry(0.5, 0.5, 1, 8);
				break;
			case 'lab':
				geometry = new THREE.SphereGeometry(0.5);
				break;
			case 'lch':
				let geometry_top = new THREE.ConeGeometry(0.5, 0.5);
				geometry_top.translate(0, 0.25, 0)
				let geometry_bottom = new THREE.ConeGeometry(0.5, 0.5);
				geometry_bottom.rotateX(Math.PI)
				geometry_bottom.translate(0, -0.25, 0)

				geometry = new THREE.Geometry();
				geometry.merge( geometry_top, geometry_top.matrix );
				geometry.merge( geometry_bottom, geometry_bottom.matrix);
				break;
			case 'cielch':
				geometry = new THREE.CylinderGeometry(0.5, 0.5, 1, 8);
				break;
		}

		let model = new Model(geometry, material);
		this.color_polyhedron = model;

		if (model.visibility) {
			this.scene.add(model.mesh)
		}
	}

	render() {
		this.controls.update();
		this.renderer.render(this.scene, this.camera)
	}
}

let graphs = [];

function init(canvas_id, color_space, colors_hexes, geo_coords) {
	let canvas = document.getElementById(canvas_id);
	
	// let camera = new THREE.PerspectiveCamera( 45, window.innerWidth / window.innerHeight, 0.01, 10 );
	let width = 1.5;
	let height = 1.5;
	let camera = new THREE.OrthographicCamera( width / - 2, width / 2, height / 2, height / - 2, 1, 1000 );
	camera.position.x = 2;
	camera.position.y = 2;
	camera.position.z = 2;

	let scene = new THREE.Scene();
	scene.background = new THREE.Color( 0xf0f0f0 );

	let renderer = new THREE.WebGLRenderer({ antialias: true, canvas: canvas, preserveDrawingBuffer: true} );
	renderer.setPixelRatio(window.devicePixelRatio);
	// renderer.setSize( window.innerWidth, window.innerHeight );

	const graph = new Graph(camera, scene, renderer, color_space, canvas_id);
	
	// Generate Spheres
	geo_coords.forEach(function(geo_coord, index){
		let geometry = new THREE.SphereGeometry(0.02);
		geometry.translate(geo_coord[0], geo_coord[2], geo_coord[1]);
		let material = new THREE.MeshBasicMaterial({ color: colors_hexes[index] });
		const model = new Model(geometry, material);
		graph.add_model(model);
	});	

	graphs.push(graph);
}

function init_multiple_palettes(canvas_id, colors_hexes, geo_coords_palettes) {
	// colors_hexes: [['#0e638d', '#7ba9a0'], ['#e6d6cf', '#e3a07f']]
	// geo_coords_palettes: palette_count * palette_length * 3

	let canvas = document.getElementById(canvas_id);
	
	// let camera = new THREE.PerspectiveCamera( 45, window.innerWidth / window.innerHeight, 0.01, 10 );
	let width = 1.2;
	let height = 1.2;
	let camera = new THREE.OrthographicCamera( width / - 2, width / 2, height / 2, height / - 2, 1, 1000 );
	camera.position.x = 1.5;
	camera.position.y = 0;
	camera.position.z = 0;
	camera.lookAt(new THREE.Vector3(0, 0, 0)); // Set look at coordinate like this
	camera.updateProjectionMatrix();

	let scene = new THREE.Scene();
	scene.background = new THREE.Color( 0xf0f0f0 );

	let renderer = new THREE.WebGLRenderer({ antialias: true, canvas: canvas, preserveDrawingBuffer: true });
	renderer.setPixelRatio(window.devicePixelRatio);
	// renderer.setPixelRatio(window.devicePixelRatio * 3);
	// renderer.setSize(900, 900); // Uncomment this line only if want to large up the canvas 

	const graph = new Graph(camera, scene, renderer, 'lab', canvas_id);
	
	// Generate Spheres
	geo_coords_palettes.forEach(function(geo_coord_palette, palette_index){
		geo_coord_palette.forEach(function(geo_coord, geo_coord_index) {
			let geometry = new THREE.SphereGeometry(0.02);
			geometry.translate(geo_coord[0], geo_coord[2], geo_coord[1]);
			let material = new THREE.MeshBasicMaterial({ color: colors_hexes[palette_index][geo_coord_index] });
			const model = new Model(geometry, material);
			graph.add_model(model);
		})
	});	

	graphs.push(graph);
}


function find_graph(canvas_id) {
	for (let graph_idx in graphs) {
		let graph = graphs[graph_idx];
		if (graph.is_canvas_id(canvas_id)) {
			return graph;
		}
	}

	return null;
}

export function screenshot(canvas_id) {
	let graph = find_graph(canvas_id);
	let dataURL = graph.renderer.domElement.toDataURL();
	
	let link = document.createElement("a");
	link.download = canvas_id;
	link.href = dataURL;
	link.target = "_blank";
	link.click();
}

function animate() {

	requestAnimationFrame( animate );

	// mesh.rotation.x += 0.01;
	// mesh.rotation.y += 0.02;
	graphs.forEach(function(graph) {
		graph.render();
		// graph.renderer.render(graph.scene, graph.camera);
	});

}