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

//River parameters
uniform int enableRiver; 
uniform vec2 RiverSourcePos;
uniform float RiverRadius;
uniform float RiverRate;

void main() {
    ivec2 coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 gridSize = imageSize(waterHeight_in);
    
    if (coords.x >= gridSize.x || coords.y >= gridSize.y) {
        return;
    }
    
    float d = imageLoad(waterHeight_in, coords).r;
    
    // Equation 1: d1 = d + dt * r
    float d_new = d + dt * rainRate;
    
    if (enableRiver == 1){
        float dist = distance(vec2(coords), RiverSourcePos);

        if (dist < RiverRadius){
            d_new += dt * RiverRate;
        }
    }
    
    imageStore(waterHeight_out, coords, vec4(d_new, 0.0, 0.0, 0.0));
}