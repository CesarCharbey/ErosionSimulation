#version 430 core
layout(local_size_x = 16, local_size_y = 16) in;

layout(r32f, binding = 0) uniform readonly image2D terrainHeight_in;
layout(rgba32f, binding = 1) uniform readonly image2D fluxCard;
layout(rgba32f, binding = 2) uniform readonly image2D fluxDiag;
layout(r32f, binding = 3) uniform writeonly image2D terrainHeight_out;

void main() {
    ivec2 coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 gridSize = imageSize(terrainHeight_in);

    if (coords.x <= 0 || coords.x >= gridSize.x - 1 ||
            coords.y <= 0 || coords.y >= gridSize.y - 1) {
        float h = imageLoad(terrainHeight_in, coords).r;
        imageStore(terrainHeight_out, coords, vec4(h, 0.0, 0.0, 0.0));
        return;
    }

    float h = imageLoad(terrainHeight_in, coords).r;

    // 1. Ce qui sort de MA cellule
    vec4 myCard = imageLoad(fluxCard, coords);
    vec4 myDiag = imageLoad(fluxDiag, coords);
    float sumOut = myCard.x + myCard.y + myCard.z + myCard.w +
            myDiag.x + myDiag.y + myDiag.z + myDiag.w;

    // 2. Ce qui entre dans MA cellule depuis les voisins
    float inL = imageLoad(fluxCard, coords + ivec2(-1, 0)).y; // Le droit du voisin de gauche
    float inR = imageLoad(fluxCard, coords + ivec2(1, 0)).x; // Le gauche du voisin de droite
    float inT = imageLoad(fluxCard, coords + ivec2(0, -1)).w; // Le bas du voisin du haut
    float inB = imageLoad(fluxCard, coords + ivec2(0, 1)).z; // Le haut du voisin du bas

    float inTL = imageLoad(fluxDiag, coords + ivec2(-1, -1)).w; // BR du voisin TL
    float inTR = imageLoad(fluxDiag, coords + ivec2(1, -1)).z; // BL du voisin TR
    float inBL = imageLoad(fluxDiag, coords + ivec2(-1, 1)).y; // TR du voisin BL
    float inBR = imageLoad(fluxDiag, coords + ivec2(1, 1)).x; // TL du voisin BR

    float sumIn = inL + inR + inT + inB + inTL + inTR + inBL + inBR;

    // 3. Mise à jour finale
    float h_new = h + sumIn - sumOut;

    imageStore(terrainHeight_out, coords, vec4(h_new, 0.0, 0.0, 0.0));
}
