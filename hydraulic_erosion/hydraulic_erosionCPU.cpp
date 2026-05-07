#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <glm/glm.hpp>

// Helper mathématique (équivalent au smoothstep de GLSL)
inline float smoothstep(float edge0, float edge1, float x)
{
    float t = std::max(0.0f, std::min((x - edge0) / (edge1 - edge0), 1.0f));
    return t * t * (3.0f - 2.0f * t);
}

// Helper mathématique (équivalent au mix de GLSL)
inline float mix(float a, float b, float t)
{
    return a * (1.0f - t) + b * t;
}

// Fonction de bruit pseudo-aléatoire sur CPU (équivalente au shader)
inline float hash(glm::vec2 p)
{
    float dot = p.x * 12.9898f + p.y * 78.233f;
    float res = std::fmod(std::sin(dot) * 43758.5453f, 1.0f);
    return res < 0.0f ? res + 1.0f : res; // C++ fmod can return negative
}

inline float noise(glm::vec2 p)
{
    glm::vec2 i(std::floor(p.x), std::floor(p.y));
    glm::vec2 f(p.x - i.x, p.y - i.y);
    glm::vec2 u(f.x * f.x * (3.0f - 2.0f * f.x), f.y * f.y * (3.0f - 2.0f * f.y));

    float res = mix(
        mix(hash(i), hash(i + glm::vec2(1, 0)), u.x),
        mix(hash(i + glm::vec2(0, 1)), hash(i + glm::vec2(1, 1)), u.x),
        u.y);
    return res;
}

class HydraulicErosionCPU
{
public:
    int gridSize;
    float cellSize, dt;
    float gravity, pipeArea, pipeLength;
    float Kc, Ks, Kd, Ke;
    float KeStagnant = 3.0f;
    float rainRate, riverRate, riverRadius;
    glm::vec2 riverPos;
    bool enableRiver;
    float thermalErosionRate, talusAngle;

    std::vector<float> terrain[2];
    std::vector<float> water[2];
    std::vector<float> sediment[2];

    std::vector<glm::vec4> flux;
    std::vector<glm::vec2> velocity;

    float timePhysics = 0.0f;
    float timeThermal = 0.0f;

    int currentBuffer = 0;

    HydraulicErosionCPU(int size, float cell) : gridSize(size), cellSize(cell)
    {
        int totalCells = size * size;
        terrain[0].resize(totalCells, 0.0f);
        terrain[1].resize(totalCells, 0.0f);
        water[0].resize(totalCells, 0.0f);
        water[1].resize(totalCells, 0.0f);
        sediment[0].resize(totalCells, 0.0f);
        sediment[1].resize(totalCells, 0.0f);
        flux.resize(totalCells, glm::vec4(0.0f));
        velocity.resize(totalCells, glm::vec2(0.0f));
    }

    inline int IDX(int x, int y) const
    {
        x = std::max(0, std::min(x, gridSize - 1));
        y = std::max(0, std::min(y, gridSize - 1));
        return y * gridSize + x;
    }

    float sampleBilinear(const std::vector<float> &data, float x, float y)
    {
        int ix = std::floor(x);
        int iy = std::floor(y);
        float fx = x - ix;
        float fy = y - iy;

        float s00 = data[IDX(ix, iy)];
        float s10 = data[IDX(ix + 1, iy)];
        float s01 = data[IDX(ix, iy + 1)];
        float s11 = data[IDX(ix + 1, iy + 1)];

        float top = s00 * (1.0f - fx) + s10 * fx;
        float bottom = s01 * (1.0f - fx) + s11 * fx;
        return top * (1.0f - fy) + bottom * fy;
    }

    void simulationStep(bool isPlacingRiver)
    {
        int READ = currentBuffer;
        int WRITE = 1 - currentBuffer;

        auto startPhysics = std::chrono::high_resolution_clock::now();

        // PASSE 1 : WATER INCREMENT (Pluie + Rivière)
        for (int y = 0; y < gridSize; y++)
        {
            for (int x = 0; x < gridSize; x++)
            {
                int i = IDX(x, y);
                float d_new = water[READ][i] + (dt * rainRate);

                if (enableRiver && !isPlacingRiver)
                {
                    float dist = glm::distance(glm::vec2(x, y), riverPos);
                    if (dist < riverRadius)
                        d_new += dt * riverRate;
                }
                water[WRITE][i] = d_new;
            }
        }

        // PASSE 2 : FLUX COMPUTATION
        for (int y = 0; y < gridSize; y++)
        {
            for (int x = 0; x < gridSize; x++)
            {
                int i = IDX(x, y);
                float h = terrain[READ][i] + water[WRITE][i];
                glm::vec4 f_old = flux[i];
                glm::vec4 f_new(0.0f);

                auto calcFlux = [&](int nx, int ny, float old_f)
                {
                    int ni = IDX(nx, ny);
                    float h_n = terrain[READ][ni] + water[WRITE][ni];
                    return std::max(0.0f, old_f + dt * pipeArea * gravity * (h - h_n) / pipeLength);
                };

                if (x > 0)
                    f_new.x = calcFlux(x - 1, y, f_old.x);
                if (x < gridSize - 1)
                    f_new.y = calcFlux(x + 1, y, f_old.y);
                if (y > 0)
                    f_new.z = calcFlux(x, y - 1, f_old.z);
                if (y < gridSize - 1)
                    f_new.w = calcFlux(x, y + 1, f_old.w);

                float sum_flux = f_new.x + f_new.y + f_new.z + f_new.w;
                float K = 1.0f;
                if (sum_flux > 0.0001f)
                {
                    float waterVol = water[WRITE][i] * cellSize * cellSize;
                    float fluxVol = sum_flux * dt;
                    if (fluxVol > waterVol)
                        K = std::min(1.0f, waterVol / fluxVol);
                }
                flux[i] = f_new * K;
            }
        }

        // PASSE 3 : WATER SURFACE & VELOCITY UPDATE
        for (int y = 0; y < gridSize; y++)
        {
            for (int x = 0; x < gridSize; x++)
            {
                int i = IDX(x, y);
                float d1 = water[WRITE][i];
                glm::vec4 f_out = flux[i];

                float f_in_L = (x > 0) ? flux[IDX(x - 1, y)].y : 0.0f;
                float f_in_R = (x < gridSize - 1) ? flux[IDX(x + 1, y)].x : 0.0f;
                float f_in_T = (y > 0) ? flux[IDX(x, y - 1)].w : 0.0f;
                float f_in_B = (y < gridSize - 1) ? flux[IDX(x, y + 1)].z : 0.0f;

                float sum_inflow = f_in_L + f_in_R + f_in_T + f_in_B;
                float sum_outflow = f_out.x + f_out.y + f_out.z + f_out.w;

                float delta_V = dt * (sum_inflow - sum_outflow);
                float d2 = std::max(0.0f, d1 + (delta_V / (cellSize * cellSize)));
                water[WRITE][i] = d2; // L'eau est mise à jour sur place

                float d_avg = (d1 + d2) * 0.5f;
                glm::vec2 vel(0.0f);
                if (d_avg > 0.01f)
                {
                    float dWX = (f_in_L - f_out.x + f_out.y - f_in_R) * 0.5f;
                    float dWY = (f_in_T - f_out.z + f_out.w - f_in_B) * 0.5f;
                    vel.x = dWX / (cellSize * d_avg);
                    vel.y = dWY / (cellSize * d_avg);
                    float len = glm::length(vel);
                    if (len > 20.0f)
                        vel = (vel / len) * 20.0f;
                }
                velocity[i] = vel;
            }
        }

        // PASSE 4 : EROSION & DEPOSITION
        for (int i = 0; i < gridSize * gridSize; i++)
        {
            terrain[WRITE][i] = terrain[READ][i];
            sediment[WRITE][i] = sediment[READ][i];
        }

        for (int y = 1; y < gridSize - 1; y++)
        {
            for (int x = 1; x < gridSize - 1; x++)
            {
                int i = IDX(x, y);
                float b = terrain[READ][i];
                float s = sediment[READ][i];
                float d = water[WRITE][i]; // On utilise la nouvelle eau
                glm::vec2 v = velocity[i];
                float speed = glm::length(v);

                // Frein en eaux profondes
                float l_max = std::max(0.0f, std::min(1.0f - (d / 4.0f), 1.0f));

                // Lecture voisins
                float hL = terrain[READ][IDX(x - 1, y)];
                float hR = terrain[READ][IDX(x + 1, y)];
                float hT = terrain[READ][IDX(x, y - 1)];
                float hB = terrain[READ][IDX(x, y + 1)];

                // Normale & Vitesse 3D
                glm::vec3 normal = glm::normalize(glm::vec3((hL - hR) / (2.0f * cellSize), 1.0f, (hT - hB) / (2.0f * cellSize)));
                float penteY = (hL - hR) * v.x + (hT - hB) * v.y;

                glm::vec3 V3d(0.0f, -1.0f, 0.0f);
                if (speed > 0.0001f)
                    V3d = glm::normalize(glm::vec3(v.x, penteY, v.y));

                float collision = std::max(0.15f, glm::dot(-V3d, normal));
                float C = Kc * collision * speed * l_max;

                float b_new = b;
                float s_new = s;
                float d_new = d;

                if (C > s)
                {
                    float erosion = Ks * (C - s) * dt;
                    erosion = std::min(erosion, 1.0f * dt);
                    erosion = std::min(erosion, std::max(d, 0.01f));

                    b_new -= erosion;
                    s_new += erosion;
                    d_new += erosion; // Couplage de masse Eau-Sédiment
                }
                else
                {
                    float deposition = Kd * (s - C) * dt;
                    b_new += deposition;
                    s_new -= deposition;
                    d_new -= deposition; // Couplage de masse Eau-Sédiment
                }

                terrain[WRITE][i] = std::max(0.0f, b_new);
                sediment[WRITE][i] = std::max(0.0f, s_new);
                water[WRITE][i] = std::max(0.0f, d_new);
            }
        }

        // PASSE 5 : SEDIMENT TRANSPORT (Advection + Lessivage)
        for (int y = 0; y < gridSize; y++)
        {
            for (int x = 0; x < gridSize; x++)
            {
                int i = IDX(x, y);
                glm::vec2 v = velocity[i];
                float sourceX = (float)x - (v.x * dt / cellSize);
                float sourceY = (float)y - (v.y * dt / cellSize);

                sourceX = std::max(0.0f, std::min(sourceX, (float)gridSize - 1.001f));
                sourceY = std::max(0.0f, std::min(sourceY, (float)gridSize - 1.001f));

                float s_new = sampleBilinear(sediment[WRITE], sourceX, sourceY);
                s_new *= 0.999f; // Micro-lessivage

                // Drain aux frontières
                if (x <= 2 || x >= gridSize - 3 || y <= 2 || y >= gridSize - 3)
                    s_new = 0.0f;

                sediment[READ][i] = s_new;
            }
        }

        // PASSE 6 : EVAPORATION
        for (int y = 0; y < gridSize; y++)
        {
            for (int x = 0; x < gridSize; x++)
            {
                int i = IDX(x, y);
                float d = water[WRITE][i];
                float speed = glm::length(velocity[i]);

                float stagnantFactor = 1.0f + KeStagnant * (1.0f - smoothstep(0.0f, 0.1f, speed));
                float d_new = std::max(0.0f, d * (1.0f - Ke * stagnantFactor * dt));

                // Drain d'eau aux frontières
                if (x <= 2 || x >= gridSize - 3 || y <= 2 || y >= gridSize - 3)
                    d_new = 0.0f;

                water[READ][i] = d_new;
            }
        }

        auto endPhysics = std::chrono::high_resolution_clock::now();
        timePhysics = std::chrono::duration<float, std::milli>(endPhysics - startPhysics).count();

        // PASSE 7 : THERMAL EROSION (Advanced 2-Pass Moore)
        auto startThermal = std::chrono::high_resolution_clock::now();

        std::vector<glm::vec4> fluxCardT(gridSize * gridSize, glm::vec4(0.0f));
        std::vector<glm::vec4> fluxDiagT(gridSize * gridSize, glm::vec4(0.0f));

        float d1 = cellSize;
        float d2 = cellSize * 1.4142135f;

        // Passe 7.1 : Calcul des flux
        for (int y = 1; y < gridSize - 1; y++)
        {
            for (int x = 1; x < gridSize - 1; x++)
            {
                int i = IDX(x, y);
                float h = terrain[WRITE][i];
                float w = water[READ][i];

                float altitudeFactor = std::max(0.0f, std::min(h / 60.0f, 1.0f));
                float noiseFactor = noise(glm::vec2(x, y) * 0.1f);
                float hardness = std::max(0.0f, std::min(altitudeFactor * 0.7f + noiseFactor * 0.3f, 1.0f));

                float localTanTalus = std::tan(talusAngle) * (1.0f + hardness * 1.5f);
                float T1 = d1 * localTanTalus;
                float T2 = d2 * localTanTalus;

                float hL = terrain[WRITE][IDX(x - 1, y)];
                float hR = terrain[WRITE][IDX(x + 1, y)];
                float hT = terrain[WRITE][IDX(x, y - 1)];
                float hB = terrain[WRITE][IDX(x, y + 1)];

                float hTL = terrain[WRITE][IDX(x - 1, y - 1)];
                float hTR = terrain[WRITE][IDX(x + 1, y - 1)];
                float hBL = terrain[WRITE][IDX(x - 1, y + 1)];
                float hBR = terrain[WRITE][IDX(x + 1, y + 1)];

                float eL = std::max(0.0f, (h - hL) - T1);
                float eR = std::max(0.0f, (h - hR) - T1);
                float eT = std::max(0.0f, (h - hT) - T1);
                float eB = std::max(0.0f, (h - hB) - T1);

                float eTL = std::max(0.0f, (h - hTL) - T2);
                float eTR = std::max(0.0f, (h - hTR) - T2);
                float eBL = std::max(0.0f, (h - hBL) - T2);
                float eBR = std::max(0.0f, (h - hBR) - T2);

                float totalExcess = eL + eR + eT + eB + eTL + eTR + eBL + eBR;
                float maxExcess = std::max({eL, eR, eT, eB, eTL, eTR, eBL, eBR});

                if (totalExcess > 0.0f)
                {
                    float speedMult = mix(1.0f, 0.05f, hardness);
                    float localErosionRate = thermalErosionRate * speedMult;

                    float riverProtection = mix(0.05f, 1.0f, smoothstep(0.01f, 0.001f, w));
                    localErosionRate *= riverProtection;

                    float transfer = localErosionRate * dt * maxExcess;

                    float minH = std::min({hL, hR, hT, hB, hTL, hTR, hBL, hBR});
                    transfer = std::min(transfer, (h - minH) * 0.5f);

                    fluxCardT[i] = glm::vec4(transfer * (eL / totalExcess), transfer * (eR / totalExcess), transfer * (eT / totalExcess), transfer * (eB / totalExcess));
                    fluxDiagT[i] = glm::vec4(transfer * (eTL / totalExcess), transfer * (eTR / totalExcess), transfer * (eBL / totalExcess), transfer * (eBR / totalExcess));
                }
            }
        }

        // Sécurité bordures
        for (int i = 0; i < gridSize * gridSize; i++)
            terrain[READ][i] = terrain[WRITE][i];

        // Passe 7.2 : Application
        for (int y = 1; y < gridSize - 1; y++)
        {
            for (int x = 1; x < gridSize - 1; x++)
            {
                int i = IDX(x, y);
                float h = terrain[WRITE][i];

                glm::vec4 myCard = fluxCardT[i];
                glm::vec4 myDiag = fluxDiagT[i];
                float sumOut = myCard.x + myCard.y + myCard.z + myCard.w + myDiag.x + myDiag.y + myDiag.z + myDiag.w;

                float inL = fluxCardT[IDX(x - 1, y)].y;
                float inR = fluxCardT[IDX(x + 1, y)].x;
                float inT = fluxCardT[IDX(x, y - 1)].w;
                float inB = fluxCardT[IDX(x, y + 1)].z;

                float inTL = fluxDiagT[IDX(x - 1, y - 1)].w;
                float inTR = fluxDiagT[IDX(x + 1, y - 1)].z;
                float inBL = fluxDiagT[IDX(x - 1, y + 1)].y;
                float inBR = fluxDiagT[IDX(x + 1, y + 1)].x;

                float sumIn = inL + inR + inT + inB + inTL + inTR + inBL + inBR;

                terrain[READ][i] = h + sumIn - sumOut;
            }
        }

        auto endThermal = std::chrono::high_resolution_clock::now();
        timeThermal = std::chrono::duration<float, std::milli>(endThermal - startThermal).count();
    }
};