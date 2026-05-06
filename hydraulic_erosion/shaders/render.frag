#version 430 core

in vec2 TexCoord;
in vec3 WorldPos;
in float WaterDepth;
in float TerrainHeight;
in float SedimentAmount;

out vec4 FragColor;

uniform sampler2D terrainTex;
uniform int gridSize;
uniform float cellSize;
uniform vec3 u_ViewPos;
uniform float u_Time;
uniform int u_ShowSediment; // 0 = Beautiful mode, 1 = Scientific mode
uniform vec2 u_RiverPos;
uniform float u_RiverRadius;
uniform int u_ShowRiverPreview;

// Simple Noise for water
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}
float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x),
        mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
}

void main() {
    // Calculate normal using finite differences (dynamic!)
    vec2 texelSize = 1.0 / vec2(gridSize);

    // Sample neighboring heights
    float hL = texture(terrainTex, TexCoord + vec2(-texelSize.x, 0.0)).r;
    float hR = texture(terrainTex, TexCoord + vec2(texelSize.x, 0.0)).r;
    float hD = texture(terrainTex, TexCoord + vec2(0.0, -texelSize.y)).r;
    float hU = texture(terrainTex, TexCoord + vec2(0.0, texelSize.y)).r;

    // Calculate gradient
    vec3 terrainNormal;
    terrainNormal.x = (hL - hR) / (2.0 * cellSize);
    terrainNormal.z = (hD - hU) / (2.0 * cellSize);
    terrainNormal.y = 1.0;
    terrainNormal = normalize(terrainNormal);

    // Lighting direction (sun from above-right)
    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.3));

    // Diffuse lighting
    float diff = max(dot(terrainNormal, lightDir), 0.0);

    // Mode 1 : Scientific mode
    if (u_ShowSediment == 1) {
        // Base Terrain : Simple Clay
        vec3 color = vec3(0.7, 0.65, 0.5) * (0.3 + 0.7 * diff);

        float waterMix = smoothstep(0.0001, 0.02, WaterDepth);
        if (waterMix > 0.0) {
            vec3 waterCol = vec3(0.0, 0.0, 1.0);
            vec3 sedimentCol = vec3(0.0, 1.0, 0.0);

            //Mix based on sediment amount
            float sRatio = clamp(SedimentAmount * 10.0, 0.0, 1.0);
            vec3 finalFluid = mix(waterCol, sedimentCol, sRatio);
            color = mix(color, finalFluid, waterMix * 0.8);
        }
        FragColor = vec4(color, 1.0);
        return;
    }

    // Mode 2 : Beautiful mode
    //Slope calculation for terrain shading (1.0 = flat, 0.0 = vertical wall)
    float slope = dot(terrainNormal, vec3(0.0, 1.0, 0.0));

    vec3 grassColor = vec3(0.21, 0.34, 0.23);
    vec3 rockColor = vec3(0.42, 0.35, 0.28);
    vec3 sandColor = vec3(0.58, 0.61, 0.37);
    vec3 snowColor = vec3(0.90, 0.90, 0.95);
    vec3 groundColor;

    float rockFactor = smoothstep(0.75, 0.50, slope);

    if (TerrainHeight < 2.5) {
        float sandMix = smoothstep(0.5, 2.5, TerrainHeight);
        groundColor = mix(sandColor, grassColor, sandMix);
    } else if (TerrainHeight < 45.0) {
        groundColor = mix(grassColor, rockColor, rockFactor);
    } else {
        float snowLine = smoothstep(50.0, 65.0, TerrainHeight);
        float snowStickiness = smoothstep(0.5, 0.6, slope);
        float effectiveSnow = snowLine * snowStickiness;

        vec3 mountainBase = rockColor;
        groundColor = mix(mountainBase, snowColor, effectiveSnow);
    }

    // Apply lighting
    vec3 ambient = vec3(0.2);
    groundColor = groundColor * (ambient + vec3(diff));

    float waterAlpha = smoothstep(0.001, 0.02, WaterDepth);
    if (waterAlpha > 0.0) {
        // Water normal (animated)
        vec2 rippleCoords = WorldPos.xz * 0.5 + u_Time * 0.5;
        float ripple = noise(rippleCoords);
        vec3 waterNormal = normalize(vec3(0.0, 1.0, 0.0) + vec3(ripple * 0.05, 0.0, ripple * 0.05));

        // Fresnel schlick's approximation
        vec3 viewDir = normalize(u_ViewPos - WorldPos);
        float NdotV = max(dot(waterNormal, viewDir), 0.0);
        float fresnel = 0.02 + (1.0 - 0.02) * pow(1.0 - NdotV, 5.0);

        // Water color
        vec3 deepColor = vec3(0.0, 0.05, 0.2);
        vec3 shallowColor = vec3(0.0, 0.4, 0.5);
        float depthFactor = clamp(WaterDepth * 0.3, 0.0, 1.0);
        vec3 waterBase = mix(shallowColor, deepColor, depthFactor);

        //Specular highlight
        vec3 reflectDir = reflect(-lightDir, waterNormal);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 64.0);
        vec3 specular = vec3(1.0) * spec;

        // Sky reflection
        vec3 skyColor = vec3(0.6, 0.7, 0.8);
        vec3 reflection = mix(waterBase, skyColor, fresnel);

        // Combine
        float finalOpacity = clamp(waterAlpha * (0.3 + depthFactor * 0.7), 0.0, 0.95);
        vec3 fluidColor = reflection + specular;
        FragColor = vec4(mix(groundColor, fluidColor, finalOpacity), 1.0);
    } else {
        FragColor = vec4(groundColor, 1.0);
    }

    if (u_ShowRiverPreview == 1) {
        vec2 gridPos = WorldPos.xz / cellSize;
        float distToRiver = distance(gridPos, u_RiverPos);
        if (distToRiver < u_RiverRadius) {
            vec3 previewColor = vec3(0.1, 0.6, 1.0);
            float alpha = smoothstep(u_RiverRadius, u_RiverRadius - 1.0, distToRiver) * 0.5;
            FragColor = vec4(mix(FragColor.rgb, previewColor, alpha), 1.0);
        }
    }
}
