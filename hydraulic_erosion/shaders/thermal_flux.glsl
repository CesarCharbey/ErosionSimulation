#version 430 core
layout(local_size_x = 16, local_size_y = 16) in;

layout(r32f, binding = 0) uniform readonly image2D terrainHeight;
layout(rgba32f, binding = 1) uniform writeonly image2D fluxCard; // L, R, T, B
layout(rgba32f, binding = 2) uniform writeonly image2D fluxDiag; // TL, TR, BL, BR
// NOUVEAU : Lecture de l'eau
layout(r32f, binding = 3) uniform readonly image2D waterHeight;

uniform float talusAngle;
uniform float erosionRate;
uniform float dt;
uniform float cellSize;

// Fonction de bruit pseudo-aléatoire pour générer des veines de roches
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
    ivec2 coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 gridSize = imageSize(terrainHeight);

    vec4 fCard = vec4(0.0);
    vec4 fDiag = vec4(0.0);

    if (coords.x <= 0 || coords.x >= gridSize.x - 1 ||
            coords.y <= 0 || coords.y >= gridSize.y - 1) {
        imageStore(fluxCard, coords, fCard);
        imageStore(fluxDiag, coords, fDiag);
        return;
    }

    float h = imageLoad(terrainHeight, coords).r;
    float w = imageLoad(waterHeight, coords).r; // Quantité d'eau actuelle

    // ------------------------------------------------------------------
    // CALCUL DE LA DURETÉ LOCALE
    // ------------------------------------------------------------------
    float altitudeFactor = clamp(h / 60.0, 0.0, 1.0);
    float noiseFactor = noise(vec2(coords) * 0.1);

    // hardness : 0.0 = sable (s'effondre vite, pente douce) | 1.0 = roche (s'effondre lentement, falaises)
    float hardness = clamp(altitudeFactor * 0.7 + noiseFactor * 0.3, 0.0, 1.0);

    // La roche dure peut soutenir des angles beaucoup plus raides (+150%)
    float localTanTalus = tan(talusAngle) * (1.0 + hardness * 1.5);

    float d1 = cellSize;
    float d2 = cellSize * 1.4142135; // sqrt(2)

    float T1 = d1 * localTanTalus;
    float T2 = d2 * localTanTalus;

    // Hauteurs des 8 voisins
    float hL = imageLoad(terrainHeight, coords + ivec2(-1, 0)).r;
    float hR = imageLoad(terrainHeight, coords + ivec2(1, 0)).r;
    float hT = imageLoad(terrainHeight, coords + ivec2(0, -1)).r;
    float hB = imageLoad(terrainHeight, coords + ivec2(0, 1)).r;

    float hTL = imageLoad(terrainHeight, coords + ivec2(-1, -1)).r;
    float hTR = imageLoad(terrainHeight, coords + ivec2(1, -1)).r;
    float hBL = imageLoad(terrainHeight, coords + ivec2(-1, 1)).r;
    float hBR = imageLoad(terrainHeight, coords + ivec2(1, 1)).r;

    float eL = max(0.0, (h - hL) - T1);
    float eR = max(0.0, (h - hR) - T1);
    float eT = max(0.0, (h - hT) - T1);
    float eB = max(0.0, (h - hB) - T1);

    float eTL = max(0.0, (h - hTL) - T2);
    float eTR = max(0.0, (h - hTR) - T2);
    float eBL = max(0.0, (h - hBL) - T2);
    float eBR = max(0.0, (h - hBR) - T2);

    float totalExcess = eL + eR + eT + eB + eTL + eTR + eBL + eBR;
    float maxExcess = max(max(max(eL, eR), max(eT, eB)), max(max(eTL, eTR), max(eBL, eBR)));

    if (totalExcess > 0.0) {

        // ------------------------------------------------------------------
        // CONTRÔLE DE VITESSE ET PROTECTION DES RIVIÈRES
        // ------------------------------------------------------------------

        // 1. La roche dure s'effondre beaucoup plus lentement (jusqu'à 20x plus lent)
        float speedMultiplier = mix(1.0, 0.05, hardness);
        float localErosionRate = erosionRate * speedMultiplier;

        // 2. PROTECTION : Si de l'eau coule ici, on "gèle" l'érosion thermique !
        // Si w > 0.01 (rivière active), l'érosion thermique est divisée par 20.
        // Cela permet à l'eau de sculpter des ravins verticaux sans être rebouchée.
        float riverProtection = mix(0.05, 1.0, smoothstep(0.01, 0.001, w));
        localErosionRate *= riverProtection;

        // Volume final
        float transfer = localErosionRate * dt * maxExcess;

        // Sécurité anti-oscillation
        float minNeighborH = min(min(min(hL, hR), min(hT, hB)), min(min(hTL, hTR), min(hBL, hBR)));
        transfer = min(transfer, (h - minNeighborH) * 0.5);

        // Distribution proportionnelle
        fCard.x = transfer * (eL / totalExcess);
        fCard.y = transfer * (eR / totalExcess);
        fCard.z = transfer * (eT / totalExcess);
        fCard.w = transfer * (eB / totalExcess);

        fDiag.x = transfer * (eTL / totalExcess);
        fDiag.y = transfer * (eTR / totalExcess);
        fDiag.z = transfer * (eBL / totalExcess);
        fDiag.w = transfer * (eBR / totalExcess);
    }

    imageStore(fluxCard, coords, fCard);
    imageStore(fluxDiag, coords, fDiag);
}
