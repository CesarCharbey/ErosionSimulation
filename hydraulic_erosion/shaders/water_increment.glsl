// water_increment.glsl
// Compute Shader for Water Increment (Rain)
// Based on Mei et al. [2007] Equation 1
// 3.1 Water Increment
#version 430 core

layout(local_size_x = 16, local_size_y = 16) in;

layout(r32f, binding = 0) uniform readonly image2D waterHeight_in;
layout(r32f, binding = 1) uniform writeonly image2D waterHeight_out;

uniform float dt;
uniform float rainRate;
uniform float u_Time;

//River parameters
uniform int enableRiver;
uniform vec2 RiverSourcePos;
uniform float RiverRadius;
uniform float RiverRate;

// Pseudo-random number generator returning a value between 0.0 and 1.0
float hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

void main() {
    ivec2 coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 gridSize = imageSize(waterHeight_in);

    if (coords.x >= gridSize.x || coords.y >= gridSize.y) {
        return;
    }

    float d = imageLoad(waterHeight_in, coords).r;

    float randVal = hash(vec2(coords) + u_Time);

    float addedWater = 0.0;

    // 5% chance for a raindrop to hit this specific cell during this frame
    if (randVal < 0.05) {
        addedWater = rainRate * dt * 20.0;
    }

    // Equation 1: d1 = d + dt * r
    float d_new = d + addedWater;

    if (enableRiver == 1) {
        float dist = distance(vec2(coords), RiverSourcePos);

        if (dist < RiverRadius) {
            d_new += dt * RiverRate;
        }
    }

    imageStore(waterHeight_out, coords, vec4(d_new, 0.0, 0.0, 0.0));
}
