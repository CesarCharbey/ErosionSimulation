#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <random>
#include <glm/glm.hpp>

class HydraulicErosionCPU
{
public:
    int gridSize;
    float cellSize, dt;
    float gravity, pipeArea, pipeLength;
    float Kc, Ks, Kd, Ke;
    float rainRate, riverRate, riverRadius;
    glm::vec2 riverPos;
    bool enableRiver;
    float thermalErosionRate, talusAngle;

    // Vecteurs pour simuler les Textures (Double-Buffering)
    std::vector<float> terrain[2];
    std::vector<float> water[2];
    std::vector<float> sediment[2];

    // Vecteurs pour Flux (vec4: L, R, T, B) et Vitesse (vec2: u, v)
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

    // Helper pour accéder aux indices 1D à partir de (x, y)
    inline int IDX(int x, int y) const
    {
        // Clamp pour éviter les crashs aux bords
        x = std::max(0, std::min(x, gridSize - 1));
        y = std::max(0, std::min(y, gridSize - 1));
        return y * gridSize + x;
    }

    // Méthode d'interpolation bilinéaire (pour l'advection)
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

        // --- PASSE 1 : WATER INCREMENT (Pluie + Rivière) ---
        static std::mt19937 gen(std::random_device{}());
        static std::uniform_real_distribution<float> dis(0.0f, 1.0f);

        for (int y = 0; y < gridSize; y++)
        {
            for (int x = 0; x < gridSize; x++)
            {
                int i = IDX(x, y);
                float d = water[READ][i];
                float addedWater = 0.0f;

                if (dis(gen) < 0.05f)
                {
                    addedWater = rainRate * dt * 20.0f;
                }

                float d_new = d + addedWater;

                if (enableRiver && !isPlacingRiver)
                {
                    float dist = glm::distance(glm::vec2(x, y), riverPos);
                    if (dist < riverRadius)
                        d_new += dt * riverRate;
                }
                water[WRITE][i] = d_new;
            }
        }

        // --- PASSE 2 : FLUX COMPUTATION ---
        for (int y = 0; y < gridSize; y++)
        {
            for (int x = 0; x < gridSize; x++)
            {
                int i = IDX(x, y);
                float b = terrain[READ][i];
                float d = water[WRITE][i];
                float h = b + d;
                glm::vec4 f_old = flux[i];
                glm::vec4 f_new(0.0f);

                // Helper lambda pour calculer le flux vers un voisin
                auto calcFlux = [&](int nx, int ny, float old_f)
                {
                    int ni = IDX(nx, ny);
                    float h_n = terrain[READ][ni] + water[WRITE][ni];
                    return std::max(0.0f, old_f + dt * pipeArea * gravity * (h - h_n) / pipeLength);
                };

                if (x > 0)
                    f_new.x = calcFlux(x - 1, y, f_old.x); // Left
                if (x < gridSize - 1)
                    f_new.y = calcFlux(x + 1, y, f_old.y); // Right
                if (y > 0)
                    f_new.z = calcFlux(x, y - 1, f_old.z); // Top
                if (y < gridSize - 1)
                    f_new.w = calcFlux(x, y + 1, f_old.w); // Bottom

                // Scale factor K
                float sum_flux = f_new.x + f_new.y + f_new.z + f_new.w;
                float K = 1.0f;
                if (sum_flux > 0.0001f)
                {
                    float waterVol = d * cellSize * cellSize;
                    float fluxVol = sum_flux * dt;
                    if (fluxVol > waterVol)
                        K = std::min(1.0f, waterVol / fluxVol);
                }
                flux[i] = f_new * K;
            }
        }

        // --- PASSE 3 : WATER SURFACE & VELOCITY UPDATE ---
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
                water[WRITE][i] = d2;

                // Vitesse
                float d_avg = (d1 + d2) * 0.5f;
                glm::vec2 vel(0.0f);
                if (d_avg > 0.01f)
                {
                    float dWX = (f_in_L - f_out.x + f_out.y - f_in_R) * 0.5f;
                    float dWY = (f_in_T - f_out.z + f_out.w - f_in_B) * 0.5f;
                    vel.x = dWX / (cellSize * d_avg);
                    vel.y = dWY / (cellSize * d_avg);
                    float len = glm::length(vel);
                    if (len > 100.0f)
                        vel = (vel / len) * 100.0f; // MAX_VELOCITY
                }
                velocity[i] = vel;
            }
        }

        // --- PASSE 4 : EROSION & DEPOSITION ---
        // Sécurité : On copie d'abord tout pour préserver les bordures
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
                glm::vec2 v = velocity[i];
                float d = water[READ][i];

                float dh_dx = (terrain[READ][IDX(x + 1, y)] - terrain[READ][IDX(x - 1, y)]) / (2.0f * cellSize);
                float dh_dy = (terrain[READ][IDX(x, y + 1)] - terrain[READ][IDX(x, y - 1)]) / (2.0f * cellSize);
                float slope = std::sqrt(dh_dx * dh_dx + dh_dy * dh_dy);

                float sin_alpha = std::max(0.05f, std::min(slope, 1.0f));
                float C = Kc * sin_alpha * glm::length(v);

                float b_new = b;
                float s_new = s;

                if (C > s)
                {
                    float erosion = Ks * (C - s) * dt;
                    erosion = std::min(erosion, 1.0f * dt);
                    erosion = std::min(erosion, d * 0.8f);
                    b_new -= erosion;
                    s_new += erosion;
                }
                else
                {
                    float deposition = Kd * (s - C) * dt;
                    b_new += deposition;
                    s_new -= deposition;
                }

                terrain[WRITE][i] = std::max(0.0f, b_new);
                sediment[WRITE][i] = std::max(0.0f, s_new);
            }
        }

        // --- PASSE 5 : SEDIMENT TRANSPORT (Advection) ---
        for (int y = 0; y < gridSize; y++)
        {
            for (int x = 0; x < gridSize; x++)
            {
                int i = IDX(x, y);
                glm::vec2 v = velocity[i];
                float sourceX = (float)x - (v.x * dt / cellSize);
                float sourceY = (float)y - (v.y * dt / cellSize);

                // Clamp aux bords de la grille
                sourceX = std::max(0.0f, std::min(sourceX, (float)gridSize - 1.001f));
                sourceY = std::max(0.0f, std::min(sourceY, (float)gridSize - 1.001f));

                sediment[READ][i] = sampleBilinear(sediment[WRITE], sourceX, sourceY);
            }
        }

        // --- PASSE 6 : EVAPORATION ---
        for (int i = 0; i < gridSize * gridSize; i++)
        {
            water[READ][i] = std::max(0.0f, water[WRITE][i] * (1.0f - Ke * dt));
        }

        auto endPhysics = std::chrono::high_resolution_clock::now();
        timePhysics = std::chrono::duration<float, std::milli>(endPhysics - startPhysics).count();

        auto startThermal = std::chrono::high_resolution_clock::now();

        // --- PASSE 7 : THERMAL EROSION ---
        float tanTalus = std::tan(talusAngle);
        float maxDiff = cellSize * tanTalus;

        // Sécurité : Copier pour préserver les bordures
        for (int i = 0; i < gridSize * gridSize; i++)
        {
            terrain[READ][i] = terrain[WRITE][i];
        }

        for (int y = 1; y < gridSize - 1; y++)
        {
            for (int x = 1; x < gridSize - 1; x++)
            {
                int i = IDX(x, y);
                float h = terrain[WRITE][i];
                float heightChange = 0.0f;

                int neighbors[4] = {IDX(x - 1, y), IDX(x + 1, y), IDX(x, y - 1), IDX(x, y + 1)};

                for (int n = 0; n < 4; n++)
                {
                    float diff = h - terrain[WRITE][neighbors[n]];
                    if (diff > maxDiff)
                    {
                        float transfer = std::min((diff - maxDiff) * thermalErosionRate * dt, (diff - maxDiff) * 0.2f);
                        heightChange -= transfer;
                    }
                    else if (diff < -maxDiff)
                    {
                        float transfer = std::min((-diff - maxDiff) * thermalErosionRate * dt, (-diff - maxDiff) * 0.2f);
                        heightChange += transfer;
                    }
                }
                terrain[READ][i] = h + heightChange;
            }
        }
        auto endThermal = std::chrono::high_resolution_clock::now();
        timeThermal = std::chrono::duration<float, std::milli>(endThermal - startThermal).count();
    }
};