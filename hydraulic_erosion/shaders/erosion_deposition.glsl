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
layout(r32f, binding = 5) uniform readonly image2D waterHeight_in;

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
    float d = imageLoad(waterHeight_in, coords).r;

    // Calculate local tilt angle (alpha)
    // We estimate the gradient using central differences
    float dh_dx = 0.0;
    float dh_dy = 0.0;

    if (coords.x > 0 && coords.x < gridSize.x - 1) {
        float b_left = imageLoad(terrainHeight_in, ivec2(coords.x - 1, coords.y)).r;
        float b_right = imageLoad(terrainHeight_in, ivec2(coords.x + 1, coords.y)).r;
        dh_dx = (b_right - b_left) / (2.0 * cellSize);
    }

    if (coords.y > 0 && coords.y < gridSize.y - 1) {
        float b_top = imageLoad(terrainHeight_in, ivec2(coords.x, coords.y - 1)).r;
        float b_bottom = imageLoad(terrainHeight_in, ivec2(coords.x, coords.y + 1)).r;
        dh_dy = (b_bottom - b_top) / (2.0 * cellSize);
    }

    // Compute slope magnitude
    float slope = sqrt(dh_dx * dh_dx + dh_dy * dh_dy);

    // Calculate local tilt angle
    // sin(alpha) = slope for small angles
    // Minimum slope threshold to ensure non-zero sediment capacity
    const float MIN_SLOPE_THRESHOLD = 0.05;
    float sin_alpha = min(slope, 1.0);
    sin_alpha = max(sin_alpha, MIN_SLOPE_THRESHOLD);

    // Equation 10: Sediment transport capacity
    // C = Kc * sin(alpha) * |v|
    float velocity_magnitude = length(v);
    float C = Kc * sin_alpha * velocity_magnitude;

    // Erosion-Deposition process
    float b_new = b;
    float s_new = s;

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
    } else {
        // Deposition: sediment settles on terrain
        // Equation 12a: b_new = b + Kd * (s - C)
        // Equation 12b: s_new = s - Kd * (s - C)
        float deposition_amount = Kd * (s - C) * dt;

        b_new = b + deposition_amount;
        s_new = s - deposition_amount;
    }

    // Ensure non-negative values
    b_new = max(b_new, 0.0);
    s_new = max(s_new, 0.0);

    // Write results
    imageStore(terrainHeight_out, coords, vec4(b_new, 0.0, 0.0, 0.0));
    imageStore(sediment_out, coords, vec4(s_new, 0.0, 0.0, 0.0));
}
