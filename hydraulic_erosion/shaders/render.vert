#version 430 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aTexCoord;

out vec2 TexCoord;
out vec3 WorldPos;
out float WaterDepth;
out float TerrainHeight;
out float SedimentAmount;

uniform mat4 viewProj;
uniform sampler2D terrainTex;
uniform sampler2D waterTex;
uniform sampler2D sedimentTex;

void main() {
    TexCoord = aTexCoord;
    
    // Sample all data needed for rendering
    WaterDepth = texture(waterTex, aTexCoord).r;
    TerrainHeight = texture(terrainTex, aTexCoord).r;
    SedimentAmount = texture(sedimentTex, aTexCoord).r;

    // If water is deep, we render the WATER surface. 
    // If dry, we render the TERRAIN surface.
    float surfaceHeight = TerrainHeight + max(WaterDepth, 0.0);    

    // Calculate world position
    WorldPos = aPos;
    WorldPos.y = surfaceHeight;
    
    gl_Position = viewProj * vec4(WorldPos, 1.0);
}