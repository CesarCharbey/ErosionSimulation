// evaporation.glsl
// Compute Shader for Evaporation
// Based on Mei et al. [2007] Equation 15
// 3.5 Evaporation

#version 430 core

layout(local_size_x = 16, local_size_y = 16) in;

layout(r32f, binding = 0) uniform readonly image2D waterHeight_in;
layout(r32f, binding = 1) uniform writeonly image2D waterHeight_out;

uniform float dt;
uniform float Ke; // Evaporation constant

void main() {
    ivec2 coords = ivec2(gl_GlobalInvocationID.xy);
    
    if (coords.x >= imageSize(waterHeight_in).x || coords.y >= imageSize(waterHeight_in).y) {
        return;
    }
    
    float d = imageLoad(waterHeight_in, coords).r;
    
    // Equation 15: d_new = d * (1 - Ke * dt)
    float d_new = d * (1.0 - Ke * dt);
    d_new = max(d_new, 0.0);
    
    imageStore(waterHeight_out, coords, vec4(d_new, 0.0, 0.0, 0.0));
}