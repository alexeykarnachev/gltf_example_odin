#version 460 core

in vec3 vs_normal;

out vec4 fs_color;

void main() {
    vec3 normal = normalize(vs_normal);
    vec3 color = normal;
    fs_color = vec4(color, 1.0);
}
