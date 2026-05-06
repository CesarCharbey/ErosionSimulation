// sediment_transport.glsl
// Compute Shader for Sediment Transportation
// Based on Mei et al. [2007] Equation 14 (Semi-Lagrangian)
// 3.4 Sediment Transportation
#version 430 core

layout(local_size_x = 16, local_size_y = 16) in;

layout(r32f, binding = 0) uniform readonly image2D sediment_in;
layout(r32f, binding = 1) uniform writeonly image2D sediment_out;
layout(rg32f, binding = 2) uniform readonly image2D velocityField;

uniform float dt;
uniform int gridSize;
uniform float cellSize;

// Bilinear interpolation
float sampleBilinear(ivec2 gridDim, vec2 texCoord) {
    vec2 coord = texCoord * vec2(gridDim - 1);
    vec2 f = fract(coord);
    ivec2 i = ivec2(floor(coord));

    // Clamp to valid range
    i = clamp(i, ivec2(0), gridDim - 2);

    float s00 = imageLoad(sediment_in, i).r;
    float s10 = imageLoad(sediment_in, i + ivec2(1, 0)).r;
    float s01 = imageLoad(sediment_in, i + ivec2(0, 1)).r;
    float s11 = imageLoad(sediment_in, i + ivec2(1, 1)).r;

    return mix(mix(s00, s10, f.x), mix(s01, s11, f.x), f.y);
}

void main() {
    ivec2 coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 gridDim = imageSize(sediment_in);

    if (coords.x >= gridDim.x || coords.y >= gridDim.y) {
        return;
    }

    vec2 v = imageLoad(velocityField, coords).rg;

    // Equation 14: Semi-Lagrangian advection
    // s_new(x, y) = s_old(x - u*dt, y - v*dt)
    // divide by cellSize to convert velocity to grid units
    vec2 sourcePos = vec2(coords) - (v * dt / cellSize);

    // Convert to normalized texture coordinates
    vec2 texCoord = (sourcePos + 0.5) / vec2(gridDim);
    texCoord = clamp(texCoord, 0.0, 1.0);

    // Sample with bilinear interpolation
    float s_new = sampleBilinear(gridDim, texCoord);

    // To simulate sediment settling and prevent infinite suspension, we can apply a simple decay factor to the transported sediment.
    s_new *= 0.999;
    // If we are near the borders, we can set sediment to zero to prevent artifacts from sampling outside the grid
    if (coords.x <= 2 || coords.x >= gridDim.x - 3 || coords.y <= 2 || coords.y >= gridDim.y - 3) {
        s_new = 0.0;
    }

    imageStore(sediment_out, coords, vec4(s_new, 0.0, 0.0, 0.0));
}
