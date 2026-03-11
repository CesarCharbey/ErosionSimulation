// water_update.glsl
// Compute Shader for Water Surface & Velocity Update
// Based on Mei et al. [2007] Equations 6-9
// 3.2.2 Water Surface and Velocity Field Update

#version 430 core

layout(local_size_x = 16, local_size_y = 16) in;

layout(r32f, binding = 0) uniform image2D waterHeight; // Read-write
layout(rgba32f, binding = 1) uniform readonly image2D fluxField;
layout(rg32f, binding = 2) uniform writeonly image2D velocityField;

uniform float dt;
uniform float cellSize;

void main() {
    ivec2 coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 gridSize = imageSize(waterHeight);
    
    if (coords.x >= gridSize.x || coords.y >= gridSize.y) {
        return;
    }
    
    float d1 = imageLoad(waterHeight, coords).r;
    vec4 flux_out = imageLoad(fluxField, coords);
    
    // Calculate inflow from neighbors
    // Flux notation: .x=Left, .y=Right, .w=Bottom, .z=Top
    float f_in_L = 0.0;
    float f_in_R = 0.0;
    float f_in_T = 0.0;
    float f_in_B = 0.0;
    
    // Left neighbor (x-1) contributes its right flux (.y)
    if (coords.x > 0) {
        f_in_L = imageLoad(fluxField, ivec2(coords.x - 1, coords.y)).y;
    }
    
    // Right neighbor (x+1) contributes its left flux (.x)
    if (coords.x < gridSize.x - 1) {
        f_in_R = imageLoad(fluxField, ivec2(coords.x + 1, coords.y)).x;
    }
    
    // Top neighbor (y-1) contributes its bottom flux (.w)
    if (coords.y > 0) {
        f_in_T = imageLoad(fluxField, ivec2(coords.x, coords.y - 1)).w;
    }
    
    // Bottom neighbor (y+1) contributes its top flux (.z)
    if (coords.y < gridSize.y - 1) {
        f_in_B = imageLoad(fluxField, ivec2(coords.x, coords.y + 1)).z;
    }
    
    // Equation 6: Calculate volume change
    // Delta_V = dt * (sum_inflow - sum_outflow)
    float sum_inflow = f_in_L + f_in_R + f_in_T + f_in_B;
    float sum_outflow = flux_out.x + flux_out.y + flux_out.z + flux_out.w;
    float delta_V = dt * (sum_inflow - sum_outflow);
    
    // Equation 7: Update water height
    // d2 = d1 + (Delta_V / (lX * lY))
    float d2 = d1 + (delta_V / (cellSize * cellSize));
    d2 = max(d2, 0.0); // Ensure non-negative water
    
    // Write updated water height
    imageStore(waterHeight, coords, vec4(d2, 0.0, 0.0, 0.0));
    
    // Equations 8-9: Calculate velocity field
    // Average water height during this step
    float d_avg = (d1 + d2) * 0.5;
    
    vec2 velocity = vec2(0.0);
    
    if (d_avg > 0.01) {
        // X direction: Delta_WX = (In_L - Out_L + Out_R - In_R) / 2
        float delta_WX = (f_in_L - flux_out.x + flux_out.y - f_in_R) * 0.5;     

        // Y direction: Delta_WY = (In_T - Out_T + Out_B - In_B) / 2
        float delta_WY = (f_in_T - flux_out.z + flux_out.w - f_in_B) * 0.5;
        
        // Velocity = Delta_W / (cellSize * d_avg)
        velocity.x = delta_WX / (cellSize * d_avg);
        velocity.y = delta_WY / (cellSize * d_avg);

        // Clamp velocity to avoid numerical explosions
        // Maximum velocity to prevent CFL violations and numerical instability
        const float MAX_VELOCITY = 100.0;
        float len = length(velocity);
        if(len > MAX_VELOCITY) {
            velocity = (velocity / len) * MAX_VELOCITY;
        }
    } else {
        velocity = vec2(0.0);
    }
    imageStore(velocityField, coords, vec4(velocity, 0.0, 0.0));
}