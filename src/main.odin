package main

import "base:intrinsics"
import "core:fmt"
import glm "core:math/linalg/glsl"
import "core:mem"
import vmem "core:mem/virtual"
import "core:os"
import gl "vendor:OpenGL"
import gltf "vendor:cgltf"
import glfw "vendor:glfw"
import stbi "vendor:stb/image"

window: glfw.WindowHandle
shader: u32

init_window :: proc() {
	glfw.Init()
	glfw.WindowHint(glfw.CONTEXT_VERSION_MAJOR, 4)
	glfw.WindowHint(glfw.CONTEXT_VERSION_MINOR, 6)
	glfw.WindowHint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

	window = glfw.CreateWindow(1024, 1024, "glTF Example in Odin", nil, nil)
	glfw.MakeContextCurrent(window)
	glfw.SwapInterval(1)
	gl.load_up_to(4, 6, glfw.gl_set_proc_address)

	screen_width, screen_height := glfw.GetWindowSize(window)
	gl.Viewport(0, 0, screen_width, screen_height)
	gl.Enable(gl.CULL_FACE)
	gl.Enable(gl.DEPTH_TEST)

	shader, _ = gl.load_shaders_source(#load("./base.vert"), #load("./base.frag"))
	gl.UseProgram(shader)
}

Vertex :: struct {
	position: glm.vec3,
	normal:   glm.vec3,
	joints:   [4]u16,
	weights:  glm.vec4,
}

Skin :: struct {
	ssbo:                  u32,
	inverse_bind_matrices: [dynamic]glm.mat4,
	joint_node_idxs:       [dynamic]i32,
}

Material :: struct {
	base_color_factor: glm.vec4,
}

Primitive :: struct {
	vao:          u32,
	vbo:          u32,
	ebo:          u32,
	vertices:     [dynamic]Vertex,
	indices:      [dynamic]u32,
	material_idx: i32,
}

Mesh :: struct {
	primitives: [dynamic]Primitive,
}

Node :: struct {
	has_matrix:   bool,
	matrix_:      glm.mat4,
	world_matrix: glm.mat4,
	translation:  glm.vec3,
	scale:        glm.vec3,
	rotation:     glm.quat,
	mesh_idx:     i32,
	skin_idx:     i32,
	child_idxs:   [dynamic]i32,
}

Animation :: struct {
	duration: f32,
	samplers: [dynamic]AnimationSampler,
	channels: [dynamic]AnimationChannel,
}

AnimationSampler :: struct {
	input:         [dynamic]f32,
	output:        [dynamic]f32,
	interpolation: gltf.interpolation_type,
}

AnimationChannel :: struct {
	sampler_idx:     i32,
	target_node_idx: i32,
	target_path:     gltf.animation_path_type,
}

Model :: struct {
	meshes:         [dynamic]Mesh,
	nodes:          [dynamic]Node,
	skins:          [dynamic]Skin,
	materials:      [dynamic]Material,
	animations:     [dynamic]Animation,
	root_node_idxs: [dynamic]i32,
	arena:          vmem.Arena,
}

load_model :: proc(file_path: cstring) -> Model {
	arena: vmem.Arena
	arena_err := vmem.arena_init_growing(&arena)
	ensure(arena_err == nil)
	allocator := vmem.arena_allocator(&arena)
	context.allocator = allocator

	model: Model = {}

	options := gltf.options{}
	data, status := gltf.parse_file(options, file_path)
	if status != .success {
		fmt.eprintf("Failed to parse gltf file: %v\n", file_path)
		os.exit(1)
	}
	defer gltf.free(data)

	status = gltf.load_buffers(options, data, file_path)
	if status != .success {
		fmt.eprintf("Failed to load gltf buffers: %v\n", file_path)
		os.exit(1)
	}

	// -------------------------------------------------------------------
	// Load meshes
	meshes := make([dynamic]Mesh)
	for mesh in data.meshes {
		primitives := make([dynamic]Primitive)

		for primitive in mesh.primitives {
			if primitive.type != .triangles {continue}

			// Get data accessors
			position_accessor: ^gltf.accessor
			normal_accessor: ^gltf.accessor
			joints_accessor: ^gltf.accessor
			weights_accessor: ^gltf.accessor
			indices_accessor: ^gltf.accessor = primitive.indices

			for attribute in primitive.attributes {
				#partial switch attribute.type {
				case .position:
					position_accessor = attribute.data
				case .normal:
					normal_accessor = attribute.data
				case .joints:
					if attribute.index == 0 {joints_accessor = attribute.data}
				case .weights:
					if attribute.index == 0 {weights_accessor = attribute.data}
				}
			}

			// Unpack vertices flat data
			if position_accessor == nil {continue}
			n_vertices := position_accessor.count

			positions_flat := make([dynamic]f32, n_vertices * 3)
			_ = gltf.accessor_unpack_floats(
				position_accessor,
				raw_data(positions_flat),
				len(positions_flat),
			)

			normals_flat: [dynamic]f32
			if normal_accessor != nil {
				normals_flat = make([dynamic]f32, n_vertices * 3)
				_ = gltf.accessor_unpack_floats(
					normal_accessor,
					raw_data(normals_flat),
					len(normals_flat),
				)
			}

			joints_flat: [dynamic]u32
			if joints_accessor != nil {
				component_size := gltf.component_size(joints_accessor.component_type)
				joints_flat = make([dynamic]u32, n_vertices * 4)

				for i in 0 ..< n_vertices {
					_ = gltf.accessor_read_uint(
						joints_accessor,
						i,
						&joints_flat[i * 4],
						component_size * 4,
					)
				}
			}

			weights_flat: [dynamic]f32
			if weights_accessor != nil {
				weights_flat = make([dynamic]f32, n_vertices * 4)
				_ = gltf.accessor_unpack_floats(
					weights_accessor,
					raw_data(weights_flat),
					len(weights_flat),
				)
			}

			indices: [dynamic]u32
			if indices_accessor != nil {
				indices = make([dynamic]u32, indices_accessor.count)
				_ = gltf.accessor_unpack_indices(
					indices_accessor,
					raw_data(indices),
					size_of(u32),
					indices_accessor.count,
				)
			} else {
				indices = make([dynamic]u32, n_vertices)
				for i in 0 ..< n_vertices {indices[i] = u32(i)}
			}

			// Collect vertices from flat data
			vertices := make([dynamic]Vertex)
			for i in 0 ..< n_vertices {
				position := glm.vec3 {
					positions_flat[i * 3 + 0],
					positions_flat[i * 3 + 1],
					positions_flat[i * 3 + 2],
				}

				normal := glm.vec3(0.0)
				if len(normals_flat) != 0 {
					normal = {
						normals_flat[i * 3 + 0],
						normals_flat[i * 3 + 1],
						normals_flat[i * 3 + 2],
					}
				}

				joints: [4]u16
				if len(joints_flat) != 0 {
					joints = {
						u16(joints_flat[i * 4 + 0]),
						u16(joints_flat[i * 4 + 1]),
						u16(joints_flat[i * 4 + 2]),
						u16(joints_flat[i * 4 + 3]),
					}
				}

				weights := glm.vec4(0.0)
				if len(weights_flat) != 0 {
					weights = {
						weights_flat[i * 4 + 0],
						weights_flat[i * 4 + 1],
						weights_flat[i * 4 + 2],
						weights_flat[i * 4 + 3],
					}
				}

				append(
					&vertices,
					Vertex {
						position = position,
						normal = normal,
						joints = joints,
						weights = weights,
					},
				)
			}

			material_idx: i32 = -1
			if primitive.material != nil {
				material_idx = i32(gltf.material_index(data, primitive.material))
			}

			append(
				&primitives,
				Primitive{vertices = vertices, indices = indices, material_idx = material_idx},
			)

			delete(positions_flat)
			delete(normals_flat)
			delete(joints_flat)
			delete(weights_flat)
		}

		append(&meshes, Mesh{primitives = primitives})
	}

	// -------------------------------------------------------------------
	// Load materials
	materials := make([dynamic]Material)
	for material in data.materials {
		pbr := material.pbr_metallic_roughness
		base_color_factor: glm.vec4 = pbr.base_color_factor
		append(&materials, Material{base_color_factor = base_color_factor})
	}

	// -------------------------------------------------------------------
	// Load skins
	skins := make([dynamic]Skin)
	for skin in data.skins {
		ibm_accessor := skin.inverse_bind_matrices
		inverse_bind_matrices: [dynamic]glm.mat4
		if ibm_accessor != nil {
			inverse_bind_matrices = make([dynamic]glm.mat4, ibm_accessor.count)
			_ = gltf.accessor_unpack_floats(
				ibm_accessor,
				&inverse_bind_matrices[0][0][0],
				ibm_accessor.count * 16,
			)
		}

		joint_node_idxs := make([dynamic]i32)
		for joint_node in skin.joints {
			append(&joint_node_idxs, i32(gltf.node_index(data, joint_node)))
		}

		append(
			&skins,
			Skin{inverse_bind_matrices = inverse_bind_matrices, joint_node_idxs = joint_node_idxs},
		)
	}

	// -------------------------------------------------------------------
	// Load animations
	animations := make([dynamic]Animation)
	for &animation in data.animations {

		duration: f32 = 0.0
		samplers := make([dynamic]AnimationSampler)
		for sampler in animation.samplers {
			input_accessor := sampler.input
			output_accessor := sampler.output

			input := make([dynamic]f32, input_accessor.count)
			_ = gltf.accessor_unpack_floats(input_accessor, raw_data(input), input_accessor.count)

			output := make(
				[dynamic]f32,
				output_accessor.count *
				gltf.calc_size(output_accessor.type, output_accessor.component_type) /
				size_of(f32),
			)
			_ = gltf.accessor_unpack_floats(output_accessor, raw_data(output), len(output))

			duration = max(duration, input[len(input) - 1])
			append(
				&samplers,
				AnimationSampler {
					input = input,
					output = output,
					interpolation = sampler.interpolation,
				},
			)
		}

		channels := make([dynamic]AnimationChannel)
		for channel, channel_idx in animation.channels {
			sampler_idx := i32(gltf.animation_sampler_index(&animation, channel.sampler))
			target_node_idx := i32(gltf.node_index(data, channel.target_node))

			append(
				&channels,
				AnimationChannel {
					sampler_idx = sampler_idx,
					target_node_idx = target_node_idx,
					target_path = channel.target_path,
				},
			)
		}

		append(
			&animations,
			Animation{duration = duration, samplers = samplers, channels = channels},
		)
	}

	// -------------------------------------------------------------------
	// Load nodes
	root_node_idxs := make([dynamic]i32)
	nodes := make([dynamic]Node)
	for node, node_idx in data.nodes {
		matrix_ := glm.mat4(1.0)
		if node.has_matrix {
			matrix_ = transmute(glm.mat4)node.matrix_
		}

		translation := glm.vec3{0.0, 0.0, 0.0}
		if node.has_translation {
			translation = node.translation
		}

		scale := glm.vec3{1.0, 1.0, 1.0}
		if node.has_scale {
			scale = node.scale
		}

		rotation := transmute(glm.quat)[4]f32{0, 0, 0, 1}
		if node.has_rotation {
			rotation = transmute(glm.quat)node.rotation
		}

		mesh_idx: i32 = -1
		if node.mesh != nil {
			mesh_idx = i32(gltf.mesh_index(data, node.mesh))
		}

		skin_idx: i32 = -1
		if node.skin != nil {
			skin_idx = i32(gltf.skin_index(data, node.skin))
		}

		child_idxs := make([dynamic]i32)
		for child in node.children {
			child_idx := i32(gltf.node_index(data, child))
			append(&child_idxs, child_idx)
		}

		if node.parent == nil {
			append(&root_node_idxs, i32(node_idx))
		}

		append(
			&nodes,
			Node {
				has_matrix = bool(node.has_matrix),
				matrix_ = matrix_,
				world_matrix = glm.mat4(1.0),
				translation = translation,
				scale = scale,
				rotation = rotation,
				mesh_idx = mesh_idx,
				skin_idx = skin_idx,
				child_idxs = child_idxs,
			},
		)
	}

	return Model {
		meshes = meshes,
		nodes = nodes,
		skins = skins,
		materials = materials,
		animations = animations,
		root_node_idxs = root_node_idxs,
		arena = arena,
	}
}

find_keyframe_interval :: proc(
	sampler: ^AnimationSampler,
	time: f32,
) -> (
	i: int,
	frac: f32,
	delta: f32,
) {
	n_keys := len(sampler.input)
	if n_keys < 2 {
		return 0, 0.0, 0.0
	}

	// Loop time
	t := time
	max_t := sampler.input[n_keys - 1]
	if t >= max_t {
		t = glm.mod(t, max_t)
	}

	// Find i where input[i] <= t < input[i+1]
	i = 0
	for j in 0 ..< n_keys - 1 {
		if t >= sampler.input[j] && t < sampler.input[j + 1] {
			i = j
			break
		}
	}
	if t >= sampler.input[n_keys - 1] {
		i = n_keys - 2
	}

	delta = sampler.input[i + 1] - sampler.input[i]
	frac = (t - sampler.input[i]) / delta if delta > 0 else 0.0
	return
}

sample_vec3 :: proc(sampler: ^AnimationSampler, time: f32) -> glm.vec3 {
	n_keys := len(sampler.input)
	if n_keys == 0 {
		return {0, 0, 0}
	}

	i, s, h := find_keyframe_interval(sampler, time)

	components := 3
	if sampler.interpolation == .cubic_spline {
		stride := components * 3
		value0_off := i * stride + 1 * components
		out_tan0_off := i * stride + 2 * components
		in_tan1_off := (i + 1) * stride + 0 * components
		value1_off := (i + 1) * stride + 1 * components

		p0 := glm.vec3 {
			sampler.output[value0_off + 0],
			sampler.output[value0_off + 1],
			sampler.output[value0_off + 2],
		}
		p1 := glm.vec3 {
			sampler.output[value1_off + 0],
			sampler.output[value1_off + 1],
			sampler.output[value1_off + 2],
		}
		m0 := glm.vec3 {
			sampler.output[out_tan0_off + 0],
			sampler.output[out_tan0_off + 1],
			sampler.output[out_tan0_off + 2],
		}
		m1 := glm.vec3 {
			sampler.output[in_tan1_off + 0],
			sampler.output[in_tan1_off + 1],
			sampler.output[in_tan1_off + 2],
		}

		s2 := s * s
		s3 := s2 * s
		return(
			(2 * s3 - 3 * s2 + 1) * p0 +
			(s3 - 2 * s2 + s) * h * m0 +
			(-2 * s3 + 3 * s2) * p1 +
			(s3 - s2) * h * m1 \
		)
	} else {
		v0_off := i * components
		v1_off := (i + 1) * components
		v0 := glm.vec3 {
			sampler.output[v0_off + 0],
			sampler.output[v0_off + 1],
			sampler.output[v0_off + 2],
		}
		v1 := glm.vec3 {
			sampler.output[v1_off + 0],
			sampler.output[v1_off + 1],
			sampler.output[v1_off + 2],
		}

		a := s if sampler.interpolation == .linear else 0.0
		return glm.mix(v0, v1, a)
	}
}

sample_quat :: proc(sampler: ^AnimationSampler, time: f32) -> glm.quat {
	n_keys := len(sampler.input)
	if n_keys == 0 {
		return quaternion(x = 0, y = 0, z = 0, w = 1)
	}

	i, s, h := find_keyframe_interval(sampler, time)

	components := 4
	if sampler.interpolation == .cubic_spline {
		stride := components * 3
		value0_off := i * stride + 1 * components
		out_tan0_off := i * stride + 2 * components
		in_tan1_off := (i + 1) * stride + 0 * components
		value1_off := (i + 1) * stride + 1 * components

		res: [4]f32
		for c in 0 ..< 4 {
			p0 := sampler.output[value0_off + c]
			p1 := sampler.output[value1_off + c]
			m0 := sampler.output[out_tan0_off + c]
			m1 := sampler.output[in_tan1_off + c]

			s2 := s * s
			s3 := s2 * s
			res[c] =
				(2 * s3 - 3 * s2 + 1) * p0 +
				(s3 - 2 * s2 + s) * h * m0 +
				(-2 * s3 + 3 * s2) * p1 +
				(s3 - s2) * h * m1
		}
		q := quaternion(x = res[0], y = res[1], z = res[2], w = res[3])
		return glm.normalize(q)
	} else {
		q0_off := i * components
		q1_off := (i + 1) * components
		q0_arr := [4]f32 {
			sampler.output[q0_off + 0],
			sampler.output[q0_off + 1],
			sampler.output[q0_off + 2],
			sampler.output[q0_off + 3],
		}
		q1_arr := [4]f32 {
			sampler.output[q1_off + 0],
			sampler.output[q1_off + 1],
			sampler.output[q1_off + 2],
			sampler.output[q1_off + 3],
		}
		q0 := transmute(glm.quat)q0_arr
		q1 := transmute(glm.quat)q1_arr

		if sampler.interpolation == .step {
			return q0
		}
		return glm.quatSlerp(q0, q1, s)
	}
}

apply_animation :: proc(model: ^Model, anim: ^Animation, time: f32) {
	for channel in anim.channels {
		sampler := &anim.samplers[channel.sampler_idx]
		node := &model.nodes[channel.target_node_idx]

		#partial switch channel.target_path {
		case .translation:
			node.translation = sample_vec3(sampler, time)
		case .rotation:
			node.rotation = sample_quat(sampler, time)
		case .scale:
			node.scale = sample_vec3(sampler, time)
		case .weights:
		// TODO: Implement morph targets
		}
	}
}

sync_skin :: proc(skin: ^Skin) {
	if skin.ssbo != 0 {
		gl.DeleteBuffers(1, &skin.ssbo)
		skin.ssbo = 0
	}

	buffer_size := len(skin.joint_node_idxs) * size_of(glm.mat4)

	gl.GenBuffers(1, &skin.ssbo)
	gl.BindBuffer(gl.SHADER_STORAGE_BUFFER, skin.ssbo)
	gl.BufferData(gl.SHADER_STORAGE_BUFFER, buffer_size, nil, gl.DYNAMIC_DRAW)
}

sync_primitive :: proc(primitive: ^Primitive) {
	if primitive.vao != 0 {
		gl.DeleteVertexArrays(1, &primitive.vao)
		primitive.vao = 0
	}

	if primitive.vbo != 0 {
		gl.DeleteBuffers(1, &primitive.vbo)
		primitive.vbo = 0
	}

	if primitive.ebo != 0 {
		gl.DeleteBuffers(1, &primitive.ebo)
		primitive.ebo = 0
	}

	gl.GenVertexArrays(1, &primitive.vao)
	gl.BindVertexArray(primitive.vao)

	gl.GenBuffers(1, &primitive.vbo)
	gl.BindBuffer(gl.ARRAY_BUFFER, primitive.vbo)
	gl.BufferData(
		gl.ARRAY_BUFFER,
		len(primitive.vertices) * size_of(Vertex),
		raw_data(primitive.vertices),
		gl.STATIC_DRAW,
	)

	gl.GenBuffers(1, &primitive.ebo)
	gl.BindBuffer(gl.ELEMENT_ARRAY_BUFFER, primitive.ebo)
	gl.BufferData(
		gl.ELEMENT_ARRAY_BUFFER,
		len(primitive.indices) * size_of(u32),
		raw_data(primitive.indices),
		gl.STATIC_DRAW,
	)

	// a_position
	gl.EnableVertexAttribArray(0)
	gl.VertexAttribPointer(0, 3, gl.FLOAT, gl.FALSE, size_of(Vertex), offset_of(Vertex, position))

	// a_normal
	gl.EnableVertexAttribArray(1)
	gl.VertexAttribPointer(1, 3, gl.FLOAT, gl.FALSE, size_of(Vertex), offset_of(Vertex, normal))

	// a_joints
	gl.EnableVertexAttribArray(2)
	gl.VertexAttribIPointer(2, 4, gl.UNSIGNED_SHORT, size_of(Vertex), offset_of(Vertex, joints))

	// a_weights
	gl.EnableVertexAttribArray(3)
	gl.VertexAttribPointer(3, 4, gl.FLOAT, gl.FALSE, size_of(Vertex), offset_of(Vertex, weights))

	gl.BindVertexArray(0)
	gl.BindBuffer(gl.ARRAY_BUFFER, 0)
	gl.BindBuffer(gl.ELEMENT_ARRAY_BUFFER, 0)
}

sync_model :: proc(model: ^Model) {
	for &mesh in model.meshes {
		for &primitive in mesh.primitives {
			sync_primitive(&primitive)
		}
	}

	for &skin in model.skins {
		sync_skin(&skin)
	}
}

unload_model :: proc(model: ^Model) {
	for &mesh in model.meshes {
		for &primitive in mesh.primitives {
			if primitive.vao != 0 {
				gl.DeleteVertexArrays(1, &primitive.vao)
				primitive.vao = 0
			}
			if primitive.vbo != 0 {
				gl.DeleteBuffers(1, &primitive.vbo)
				primitive.vbo = 0
			}
			if primitive.ebo != 0 {
				gl.DeleteBuffers(1, &primitive.ebo)
				primitive.ebo = 0
			}
		}
	}

	for &skin in model.skins {
		if skin.ssbo != 0 {
			gl.DeleteBuffers(1, &skin.ssbo)
			skin.ssbo = 0
		}
	}

	vmem.arena_destroy(&model.arena)
}

draw_model :: proc(model: ^Model, mat: glm.mat4 = glm.mat4(1.0)) {
	update_node_world_matrix :: proc(node_idx: i32, model: ^Model, mat: glm.mat4 = glm.mat4(1.0)) {
		node := &model.nodes[node_idx]
		node_matrix: glm.mat4
		if node.has_matrix {
			node_matrix = node.matrix_
		} else {
			t := glm.mat4Translate(node.translation)
			r := glm.mat4FromQuat(node.rotation)
			s := glm.mat4Scale(node.scale)
			node_matrix = t * r * s
		}

		node.world_matrix = mat * node_matrix
		for child_idx in node.child_idxs {
			update_node_world_matrix(child_idx, model, node.world_matrix)
		}
	}

	for root_node_idx in model.root_node_idxs {
		update_node_world_matrix(root_node_idx, model, mat)
	}


	for &node in model.nodes {
		draw_node(&node, model)
	}
}

draw_node :: proc(node: ^Node, model: ^Model) {
	if node.mesh_idx != -1 {
		set_mat4_uniform(&node.world_matrix, "u_model")

		if node.skin_idx != -1 {
			gl.Uniform1i(gl.GetUniformLocation(shader, "u_is_skinned"), 1)
			skin := &model.skins[node.skin_idx]
			joint_matrices := make([]glm.mat4, len(skin.joint_node_idxs))

			for i in 0 ..< len(skin.joint_node_idxs) {
				joint_node_idx := skin.joint_node_idxs[i]
				joint_matrices[i] =
					model.nodes[joint_node_idx].world_matrix * skin.inverse_bind_matrices[i]
			}

			gl.BindBuffer(gl.SHADER_STORAGE_BUFFER, skin.ssbo)
			gl.BufferSubData(
				gl.SHADER_STORAGE_BUFFER,
				0,
				len(joint_matrices) * size_of(glm.mat4),
				raw_data(joint_matrices),
			)
			gl.BindBufferBase(gl.SHADER_STORAGE_BUFFER, 0, skin.ssbo)

			delete(joint_matrices)
		} else {
			gl.Uniform1i(gl.GetUniformLocation(shader, "u_is_skinned"), 0)
		}

		mesh := model.meshes[node.mesh_idx]
		for &primitive in mesh.primitives {
			base_color_factor := glm.vec4{1.0, 1.0, 1.0, 1.0}

			if primitive.material_idx != -1 {
				material := &model.materials[primitive.material_idx]
				base_color_factor = material.base_color_factor
			}

			gl.Uniform4f(
				gl.GetUniformLocation(shader, "u_base_color_factor"),
				base_color_factor.r,
				base_color_factor.g,
				base_color_factor.b,
				base_color_factor.a,
			)

			gl.BindVertexArray(primitive.vao)
			gl.DrawElements(
				gl.TRIANGLES,
				i32(len(primitive.indices)),
				gl.UNSIGNED_INT,
				rawptr(uintptr(0)),
			)
		}
	}
}

set_mat4_uniform :: proc(mat: ^glm.mat4, name: cstring) {
	gl.UniformMatrix4fv(gl.GetUniformLocation(shader, name), 1, gl.FALSE, &mat[0][0])
}

main :: proc() {
	init_window()

	model := load_model("resources/Adventurer.glb")
	sync_model(&model)

	translation_mat := glm.mat4Translate({0.0, 0.0, 0.0})
	rotation_mat := glm.mat4FromQuat(quaternion(x = 0, y = 0, z = 0, w = 1))
	scale_mat := glm.mat4Scale({4.0, 4.0, 4.0})
	world_mat := translation_mat * rotation_mat * scale_mat

	view_mat := glm.mat4LookAt(
		eye = {5.0, 8.0, 5.0},
		centre = {0.0, 5.0, 0.0},
		up = {0.0, 1.0, 0.0},
	)

	screen_width, screen_height := glfw.GetWindowSize(window)
	proj_mat := glm.mat4Perspective(
		fovy = glm.radians_f32(70.0),
		aspect = f32(screen_width) / f32(screen_height),
		near = 0.1,
		far = 1000.0,
	)

	last_time := glfw.GetTime()
	current_time: f32 = 0.0
	active_anim: ^Animation = nil
	if len(model.animations) > 0 {
		active_anim = &model.animations[4]
	}

	for !glfw.WindowShouldClose(window) {
		glfw.PollEvents()

		now := glfw.GetTime()
		delta := now - last_time
		last_time = now
		current_time += f32(delta)

		gl.ClearColor(0.25, 0.0, 0.0, 1.0)
		gl.Clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

		set_mat4_uniform(&view_mat, "u_view")
		set_mat4_uniform(&proj_mat, "u_proj")

		if active_anim != nil {
			anim_time :=
				glm.mod(current_time, active_anim.duration) if active_anim.duration > 0 else 0.0
			apply_animation(&model, active_anim, anim_time)
		}

		draw_model(&model, world_mat)

		glfw.SwapBuffers(window)
	}

	unload_model(&model)
	gl.DeleteProgram(shader)
	glfw.DestroyWindow(window)
	glfw.Terminate()
}

