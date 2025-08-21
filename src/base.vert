#version 460 core

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in uvec4 a_joints;
layout(location = 3) in vec4 a_weights;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform bool u_is_skinned;

layout(std430, binding = 0) buffer SkinData { mat4 joint_matrices[]; }
skin;

out vec3 vs_normal;

void main() {
    vec4 position = vec4(a_position, 1.0);
    vec3 normal = a_normal;

    mat4 model = u_model;
    if (u_is_skinned) {
        model = a_weights.x * skin.joint_matrices[a_joints.x] +
                a_weights.y * skin.joint_matrices[a_joints.y] +
                a_weights.z * skin.joint_matrices[a_joints.z] +
                a_weights.w * skin.joint_matrices[a_joints.w];
    }

    gl_Position = u_proj * u_view * model * position;
    vs_normal = normalize(transpose(inverse(mat3(model))) * normal);
}
