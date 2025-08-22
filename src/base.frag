#version 460 core

in vec3 vs_normal;

uniform vec4 u_base_color_factor;

out vec4 fs_color;

void main() {
    vec3 light_dir = normalize(vec3(-1.0, -1.0, -1.0));
    vec3 light_color = vec3(2.0);
    vec3 ambient_color = vec3(0.5);

    vec3 normal = normalize(vs_normal);
    vec3 color = u_base_color_factor.rgb;

    vec3 ambient = ambient_color * color;

    float diffuse_factor = max(dot(normal, -light_dir), 0.0);
    vec3 diffuse = diffuse_factor * light_color * color;

    fs_color = vec4(ambient + diffuse, u_base_color_factor.a);
}
