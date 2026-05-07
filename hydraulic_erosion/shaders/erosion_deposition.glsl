// erosion_deposition.glsl
// Compute Shader for Erosion and Deposition Process
// Based on Mei et al. [2007] Equations 10-12
// 3.3 Erosion and Deposition
#version 430 core

layout(local_size_x = 16, local_size_y = 16) in;

// Input textures (read-only)
layout(r32f, binding = 0) uniform readonly image2D terrainHeight_in;
layout(r32f, binding = 2) uniform readonly image2D sediment_in;
layout(rg32f, binding = 4) uniform readonly image2D velocityField;
layout(r32f, binding = 5) uniform image2D waterHeight;

// Output textures (write-only)
layout(r32f, binding = 1) uniform writeonly image2D terrainHeight_out;
layout(r32f, binding = 3) uniform writeonly image2D sediment_out;

// Uniforms
uniform float Kc; // Sediment capacity constant
uniform float Ks; // Dissolving constant
uniform float Kd; // Deposition constant
uniform float dt; // Time step
uniform float cellSize;

void main() {
    ivec2 coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 gridSize = imageSize(terrainHeight_in);

    // Avoid borders to prevent artifacts
    int border = 1;
    if (coords.x < border || coords.x >= gridSize.x - border ||
            coords.y < border || coords.y >= gridSize.y - border) {
        float b = imageLoad(terrainHeight_in, coords).r;
        float s = imageLoad(sediment_in, coords).r;
        imageStore(terrainHeight_out, coords, vec4(b, 0.0, 0.0, 0.0));
        imageStore(sediment_out, coords, vec4(s, 0.0, 0.0, 0.0));
        return;
    }

    // Read current values
    float b = imageLoad(terrainHeight_in, coords).r;
    float s = imageLoad(sediment_in, coords).r;
    vec2 v = imageLoad(velocityField, coords).rg;
    float d = imageLoad(waterHeight, coords).r;

    float speed = length(v);

    // Equation 10 updated : C = Kc * sin(alpha) * |v| * l_max  (depp water brakes erosion)
    const float Kdmax = 4.0;
    float l_max = clamp(1.0 - (d / Kdmax), 0.0, 1.0);

    // Equation 11 : 3D collision impact between water and terrain
    float hL = imageLoad(terrainHeight_in, coords + ivec2(-1, 0)).r;
    float hR = imageLoad(terrainHeight_in, coords + ivec2(1, 0)).r;
    float hT = imageLoad(terrainHeight_in, coords + ivec2(0, -1)).r;
    float hB = imageLoad(terrainHeight_in, coords + ivec2(0, 1)).r;

    // 3D Normal
    vec3 normal = normalize(vec3((hL - hR) / (2.0 * cellSize), 1.0, (hT - hB) / (2.0 * cellSize)));

    float slopeY = (hL - hR) * v.x + (hT - hB) * v.y;

    vec3 V3d = vec3(0.0, -1.0, 0.0);
    if (speed > 0.0001) {
        V3d = normalize(vec3(v.x, slopeY, v.y));
    }

    // Impact Force
    float collision = max(0.15, dot(-V3d, normal));

    float C = Kc * collision * speed * l_max;

    // Equation 10: Sediment transport capacity
    // C = Kc * sin(alpha) * |v|
    // float velocity_magnitude = length(v);
    // float C = Kc * sin_alpha * velocity_magnitude;

    // Erosion-Deposition process
    float b_new = b;
    float s_new = s;
    float d_new = d;

    if (C > s) {
        // Erosion: dissolve soil into water
        // Equation 11a: b_new = b - Ks * (C - s)
        // Equation 11b: s_new = s + Ks * (C - s)
        float erosion_amount = Ks * (C - s) * dt;
        // Maximum erosion per time step to prevent excessive terrain removal
        const float MAX_EROSION_RATE = 1.0;
        erosion_amount = min(erosion_amount, MAX_EROSION_RATE * dt);
        float virtualWaterDepth = max(d, 0.01);
        erosion_amount = min(erosion_amount, virtualWaterDepth);

        b_new = b - erosion_amount;
        s_new = s + erosion_amount;

        // Equation 12c  Water depth update due to erosion (water volume increases as it dissolves soil)
        d_new = d + erosion_amount;
    } else {
        // Deposition: sediment settles on terrain
        // Equation 12a: b_new = b + Kd * (s - C)
        // Equation 12b: s_new = s - Kd * (s - C)
        float deposition_amount = Kd * (s - C) * dt;

        b_new = b + deposition_amount;
        s_new = s - deposition_amount;
        d_new = d - deposition_amount; // Water volume decreases as sediment settles (Equation 13c)
    }

    // Ensure non-negative values
    b_new = max(b_new, 0.0);
    s_new = max(s_new, 0.0);
    d_new = max(d_new, 0.0);
    // Write results
    imageStore(terrainHeight_out, coords, vec4(b_new, 0.0, 0.0, 0.0));
    imageStore(sediment_out, coords, vec4(s_new, 0.0, 0.0, 0.0));
    imageStore(waterHeight, coords, vec4(d_new, 0.0, 0.0, 0.0));
}
