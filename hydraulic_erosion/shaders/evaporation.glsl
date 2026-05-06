// evaporation.glsl
// Compute Shader for Evaporation
// Based on Mei et al. [2007] Equation 15
// 3.5 Evaporation
#version 430 core

layout(local_size_x = 16, local_size_y = 16) in;

layout(r32f, binding = 0) uniform readonly image2D waterHeight_in;
layout(r32f, binding = 1) uniform writeonly image2D waterHeight_out;
layout(rg32f, binding = 2) uniform readonly image2D velocityField;

uniform float dt;
uniform float Ke; // Evaporation constant
uniform float KeStagnant; // Evaporation constant for stagnant water (no velocity)

void main() {
    ivec2 coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 gridDim = imageSize(waterHeight_in);

    if (coords.x >= gridDim.x || coords.y >= gridDim.y) {
        return;
    }

    float d = imageLoad(waterHeight_in, coords).r;
    vec2 v = imageLoad(velocityField, coords).rg;

    float speed = length(v);
    float flowThreshold = 0.1;
    float stagnantFactor = 1.0 + KeStagnant * (1.0 - smoothstep(0.0, flowThreshold, speed));

    // Equation 15: d_new = d * (1 - Ke * dt)
    float d_new = d * (1.0 - Ke * stagnantFactor * dt);
    d_new = max(d_new, 0.0);

    // At the border of the map, we can set water to zero to prevent artifacts from sampling outside the grid
    if (coords.x <= 2 || coords.x >= gridDim.x - 3 || coords.y <= 2 || coords.y >= gridDim.y - 3) {
        d_new = 0.0;
    }

    imageStore(waterHeight_out, coords, vec4(d_new, 0.0, 0.0, 0.0));
}
