// flux_compute.glsl
// Compute Shader for Outflow Flux Calculation (Pipe Model)
// Based on Mei et al. [2007] Equations 2-5
// 3.2.1 Outflow Flux Calculation (Pipe Model)
#version 430 core

layout(local_size_x = 16, local_size_y = 16) in;

// Input textures (read-only)
layout(r32f, binding = 0) uniform readonly image2D terrainHeight;
layout(r32f, binding = 1) uniform readonly image2D waterHeight;

// Output/Input texture (read-write)
layout(rgba32f, binding = 2) uniform image2D fluxField;

// Uniforms
uniform float dt;
uniform float g; // Gravity
uniform float pipeArea; // Cross-sectional area A
uniform float pipeLength; // Length l
uniform float cellSize;

void main() {
    ivec2 coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 gridSize = imageSize(terrainHeight);

    // Boundary check
    if (coords.x >= gridSize.x || coords.y >= gridSize.y) {
        return;
    }

    // Read current values
    float b = imageLoad(terrainHeight, coords).r;
    float d = imageLoad(waterHeight, coords).r;
    vec4 flux_old = imageLoad(fluxField, coords);

    // Total height at current cell
    float h = b + d;

    // Initialize new flux
    vec4 flux_new = vec4(0.0);

    // Compute flux for each direction
    // Left (flux_new.x)
    if (coords.x > 0) {
        ivec2 neighbor = ivec2(coords.x - 1, coords.y);
        float b_n = imageLoad(terrainHeight, neighbor).r;
        float d_n = imageLoad(waterHeight, neighbor).r;
        float h_n = b_n + d_n;
        float delta_h = h - h_n;

        // Equation 2: f_new = max(0, f_old + dt * A * g * delta_h / l)
        flux_new.x = max(0.0, flux_old.x + dt * pipeArea * g * delta_h / pipeLength);
    }

    // Right (flux_new.y)
    if (coords.x < gridSize.x - 1) {
        ivec2 neighbor = ivec2(coords.x + 1, coords.y);
        float b_n = imageLoad(terrainHeight, neighbor).r;
        float d_n = imageLoad(waterHeight, neighbor).r;
        float h_n = b_n + d_n;
        float delta_h = h - h_n;

        flux_new.y = max(0.0, flux_old.y + dt * pipeArea * g * delta_h / pipeLength);
    }

    // Top (flux_new.z)
    if (coords.y > 0) {
        ivec2 neighbor = ivec2(coords.x, coords.y - 1);
        float b_n = imageLoad(terrainHeight, neighbor).r;
        float d_n = imageLoad(waterHeight, neighbor).r;
        float h_n = b_n + d_n;
        float delta_h = h - h_n;

        flux_new.z = max(0.0, flux_old.z + dt * pipeArea * g * delta_h / pipeLength);
    }

    // Bottom (flux_new.w)
    if (coords.y < gridSize.y - 1) {
        ivec2 neighbor = ivec2(coords.x, coords.y + 1);
        float b_n = imageLoad(terrainHeight, neighbor).r;
        float d_n = imageLoad(waterHeight, neighbor).r;
        float h_n = b_n + d_n;
        float delta_h = h - h_n;

        flux_new.w = max(0.0, flux_old.w + dt * pipeArea * g * delta_h / pipeLength);
    }

    // Equation 4: Scaling factor K to prevent negative water height
    // K = min(1, d * lX * lY / (sum_flux * dt))
    float sum_flux = flux_new.x + flux_new.y + flux_new.z + flux_new.w;
    float K = min(1.0, d * cellSize * cellSize / (sum_flux * dt));

    if (sum_flux > 0.0001) {
        // Water volume in the cell
        float waterVol = d * cellSize * cellSize;

        // Volume that would be moved by the fluxes
        float fluxVol = sum_flux * dt;

        // If flux volume exceeds water volume, scale down
        if (fluxVol > waterVol) {
            K = waterVol / fluxVol;
            K = clamp(K, 0.0, 1.0);
        }
    }

    // Equation 5 : scale the flux
    flux_new *= K;

    // Write result
    imageStore(fluxField, coords, flux_new);
}
