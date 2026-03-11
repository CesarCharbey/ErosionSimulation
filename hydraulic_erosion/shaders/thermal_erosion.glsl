// thermal_erosion.glsl
// Compute Shader for Thermal Erosion (Our Improvement)
// Smooths sharp slopes by material slippage
#version 430 core

layout(local_size_x = 16, local_size_y = 16) in;

layout(r32f, binding = 0) uniform readonly image2D terrainHeight_in;
layout(r32f, binding = 1) uniform writeonly image2D terrainHeight_out;

uniform float talusAngle;    // Maximum stable slope (radians)
uniform float erosionRate;   // Transfer rate
uniform float dt;            // Time step
uniform float cellSize;

void main() {
    ivec2 coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 gridSize = imageSize(terrainHeight_in);
    
    // Boundary check
    if (coords.x < 0 || coords.x >= gridSize.x || 
        coords.y < 0 || coords.y >= gridSize.y) {
        return;
    }
    
    float h = imageLoad(terrainHeight_in, coords).r;

    // We can't erode borders without sampling outside, 
    // so we just keep them static to act as a container.
    if (coords.x == 0 || coords.x == gridSize.x - 1 || 
        coords.y == 0 || coords.y == gridSize.y - 1) {
        imageStore(terrainHeight_out, coords, vec4(h, 0.0, 0.0, 0.0));
        return;
    }

    // Calculate the maximum height difference allowed before slippage occurs.
    // tan(angle) = dy / dx  ->  dy = dx * tan(angle)
    float tanTalus = tan(talusAngle);
    float maxDiff = cellSize * tanTalus;

    float heightChange = 0.0;
    
    // Von Neumann Neighborhood (4 neighbors)
    // We explicitly list offsets to avoid nested loops/branches for performance
    ivec2 offsets[4] = ivec2[](
        ivec2(-1, 0), ivec2(1, 0), 
        ivec2(0, -1), ivec2(0, 1)
    );

    for (int i = 0; i < 4; i++) {
        ivec2 neighborCoord = coords + offsets[i];
        
        // Sample neighbor height
        float h_n = imageLoad(terrainHeight_in, neighborCoord).r;
        
        // Calculate difference: (My Height) - (Neighbor Height)
        float d = h - h_n;
        
        // PHYSICS LOGIC:
        // If d > maxDiff: I am too high. Material slides FROM me TO neighbor. (I lose)
        // If d < -maxDiff: Neighbor is too high. Material slides FROM neighbor TO me. (I gain)
        
        if (d > maxDiff) {
            // Erosion: I lose material
            // The amount is proportional to the excess height difference
            float excess = d - maxDiff;
            // Cap transfer to d/2 to prevent oscillation
            float transfer = excess * erosionRate * dt;
            // Ensure we don't transfer more than half the difference to maintain stability
            transfer = min(transfer, excess * 0.5);
            
            heightChange -= transfer;
            
        } else if (d < -maxDiff) {
            // Deposition: I gain material
            // d is negative here, so we take abs(d) or just negate
            float excess = -d - maxDiff;
            float transfer = excess * erosionRate * dt;
            transfer = min(transfer, excess * 0.5);
            
            heightChange += transfer;
        }
    }
    
    // Apply changes
    float h_new = h + heightChange;
    
    // Write result
    imageStore(terrainHeight_out, coords, vec4(h_new, 0.0, 0.0, 0.0));
}