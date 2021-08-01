export function draw_polygon_interpolation(canvas_id, color_hexes, sorted_index) {
	let canvas = document.getElementById(canvas_id);
	if (canvas.getContext) {
		let ctx = canvas.getContext('2d');

		const cnum = color_hexes.length;
		const canvas_width = canvas.clientWidth;
		const canvas_height = canvas.clientHeight;
		const radius = canvas_width * 0.4;
		const step_count = 5;
		const padding_ratio = 0.15;

		// translate the coordinates
		ctx.translate(canvas_width * 0.5, canvas_height * 0.5); 

		let side_length = calculateSideLength(radius, cnum);
		let circle_radius = side_length / (step_count * 2);

		let angle = ((Math.PI * 2) / cnum);
		for (let cidx = 0; cidx < cnum; cidx++) {
			let x_cur = radius * Math.cos(Math.PI + angle * cidx);
			let y_cur = radius * Math.sin(Math.PI + angle * cidx);
			let c_cur = color_hexes[sorted_index[cidx]];

			let x_next = radius * Math.cos(Math.PI + angle * (cidx + 1));
			let y_next = radius * Math.sin(Math.PI + angle * (cidx + 1));
			let c_next = color_hexes[sorted_index[(cidx + 1) % cnum]];
			
			// interpolate till next vertex
			for (let step = 0; step < step_count; step++) {
				let x_interpolated = x_cur + step / step_count * (x_next - x_cur);
				let y_interpolated = y_cur + step / step_count * (y_next - y_cur);
				let c_interpolated = interpolateColor(c_cur, c_next, step/step_count);

				drawCircle(ctx, x_interpolated, y_interpolated, circle_radius, c_interpolated);
				
			}
		}
	}
}

export function draw_half_disk_interpolation(canvas_id, color_hexes, sorted_index) {
	let canvas = document.getElementById(canvas_id);
	if (canvas.getContext) {
		let ctx = canvas.getContext('2d');

		const cnum = color_hexes.length;
		const canvas_width = canvas.clientWidth;
		const canvas_height = canvas.clientHeight;
		const radius = canvas_width * 0.4;
		const step_count = Math.min(4, cnum);

		let x_center = canvas_width * 0.5;
		let y_center = (canvas_height + radius) * 0.5;
		
		let ordered_color = [];
		for (let i=0; i< cnum; i++) {
			ordered_color.push(color_hexes[sorted_index[i]]);
		}

		drawRecursiveDisk(ctx, x_center, y_center, radius, cnum, ordered_color, 1, Math.PI);

		// draw white disk
		let cur_radius = radius / 5;
		drawCircleSector(ctx, x_center, y_center, cur_radius, Math.PI, Math.PI * 2, '#ffffff');
	}
}

function drawRecursiveDisk(ctx, x, y, radius, cnum, color, depth, full_angle) {
	let depth_limit = 5;
	let full_disk = true;
	if (depth == depth_limit) return;

	let interpolated_colors = [];
	for (let i=0; i < color.length-1; i++) {
		interpolated_colors.push(interpolateColor(color[i], color[i + 1], 0.5));
	}

	let angle = full_angle / cnum;
	drawRecursiveDisk(ctx, x, y, radius, cnum - 1, interpolated_colors, depth + 1, full_angle - angle);

	for (let i = 1; i <= cnum; i++) {
		let initial_angle = Math.PI + 0.5 * (Math.PI - full_angle);

		let start_angle = initial_angle + angle * (i - 1);
		let end_angle = initial_angle + angle * i;

		if (full_disk) {
			if (i == 1) start_angle = Math.PI;
			if (i == cnum) end_angle = Math.PI * 2;
		}

		let cur_radius = radius * (depth + 1) / depth_limit;

		drawCircleSector(ctx, x, y, cur_radius, start_angle, end_angle, color[i - 1]);
	}
}

function calculateSideLength(radius, side_num) {
	// reference: https://www.mathsisfun.com/geometry/regular-polygons.html

	// let apothem = radius * Math.cos(Math.PI / side_num)
	// return 2 * apothem * Math.tan(Math.PI / side_num);
	return 2 * radius * Math.sin(Math.PI / side_num);
}

function interpolateColor(hex_color1, hex_color2, ratio) {
	let lab1 = chroma(hex_color1).lab();
	let lab2 = chroma(hex_color2).lab();

	let interpolated_lab = lab1.map((e, i) => (1 - ratio) * e + ratio * lab2[i]);
	return chroma.lab(interpolated_lab[0], interpolated_lab[1], interpolated_lab[2]);
}

function drawCircle(ctx, x, y, radius, color) {
	ctx.save();
	
	ctx.fillStyle = color;

	ctx.beginPath();
	ctx.arc(x, y, radius, 0, 2 * Math.PI);
	ctx.fill();
	// ctx.stroke();
	ctx.restore();
}

function drawCircleSector(ctx, x, y, radius, start_angle, end_angle, color) {
	ctx.save();

	ctx.fillStyle = color;

	ctx.beginPath();
	ctx.arc(x, y, radius, start_angle, end_angle);
	ctx.lineTo(x, y);
	ctx.fill();
	
	ctx.restore();
}

function regularpolygon(ctx, x, y, radius, sides, color) {
	// code reference: https://www.arungudelli.com/html5/html5-canvas-polygon/

	ctx.save();
	if (sides < 3) return;
	ctx.fillStyle = color;

	ctx.beginPath();
	let a = ((Math.PI * 2) / sides);
	ctx.translate(x, y);
	ctx.moveTo(radius, 0);
	for (let i = 1; i < sides; i++) {
		ctx.lineTo(radius * Math.cos(a * i), radius * Math.sin(a * i));
	}
	ctx.closePath();
	ctx.fill();
	ctx.stroke();
	ctx.restore();
}
