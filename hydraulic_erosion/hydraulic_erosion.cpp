// hydraulic_erosion.cpp
// Modern GPU-based Hydraulic Erosion System
// Based on Mei et al. [2007] with Compute Shader implementation
#include "imgui.h"
#include "hydraulic_erosionCPU.cpp"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include <sstream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int TERRAIN_MODE = 1;
int CURRENT_HEIGHTMAP = 0;
const char *HEIGHTMAP_FILES[] = {
    "heightmaps/heightmap1.png",
    "heightmaps/AlpeEcraseTaMere.png",
    "heightmaps/AlpeUnPeuMoinsEcrasseTaMere.png",

};

std::string loadShaderSourceFromFile(const char *filepath)
{
    std::ifstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "ERROR: Could not open shader file: " << filepath << std::endl;
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// Simulation parameters
const int GRID_SIZE = 1024;
const float CELL_SIZE = 1.0f;
float DT = 0.0015f;
float GRAVITY = 9.81f;
float PIPE_AREA = 1.5f;
float PIPE_LENGTH = CELL_SIZE;

// River parameters
bool ENABLE_RIVER = false;
float RIVER_RATE = 2.5f;
float RIVER_RADIUS = 3.0f;
glm::vec2 RIVER_SOURCE_POS(GRID_SIZE * 0.7f, GRID_SIZE * 0.7f); // Default position of the river source

// Erosion
float KC = 0.8f;  // Sediment capacity
float KS = 0.05f; // Dissolving constant
float KD = 0.1f;  // Deposition constant
float KE = 0.01f; // Evaporation constant (1%/s)
float RAIN_RATE = 0.008f;
float THERMAL_EROSION_RATE = 0.5f;
float TALUS_ANGLE = glm::radians(60.0f);

// Terrain Generation Parameters
float TERRAIN_SEED = 0.0f;
float MOUNTAIN_FREQ = 1.5f;    // Frequencies of mountain ranges
float MOUNTAIN_HEIGHT = 80.0f; // Max height of mountains
float BASE_FREQ = 4.0f;        // Frequencies of base terrain
float BASE_HEIGHT = 10.0f;     // Base height variation

// Simulation speed
int ITERATIONS_PER_FRAME = 1;

bool useGPU = true;
float lastPhysicsTime = 0.0f;
float cpuTransferTime = 0.0f;
HydraulicErosionCPU *cpuSim = nullptr;

class HydraulicErosion
{
private:
    // Texture IDs for simulation
    GLuint fluxTex;     // Outflow flux (vec4: L,R,T,B)
    GLuint velocityTex; // Velocity field (vec2: u,v)

    // Compute shader programs
    GLuint waterIncrementShader;
    GLuint fluxComputeShader;
    GLuint waterUpdateShader;
    GLuint erosionShader;
    GLuint sedimentTransportShader;
    GLuint evaporationShader;
    GLuint thermalErosionShader;

    // Rendering
    GLuint renderProgram;
    GLuint meshVAO, meshVBO, meshEBO;
    int meshIndexCount;

    // Evaluation
    GLuint timeQueries[3]; // [0] = Physics, [1] = Thermal, [2] = Render
    // Variables to store the results in milliseconds
    float gpuTimePhysics = 0.0f;
    float gpuTimeThermal = 0.0f;
    float gpuTimeRender = 0.0f;

public:
    GLuint terrainTex[2];  // Double-buffered terrain height
    GLuint waterTex[2];    // Double-buffered water height
    GLuint sedimentTex[2]; // Double-buffered sediment
    int currentBuffer;

    HydraulicErosion() : currentBuffer(0) {}

    // Getters for Evaluation
    float getPhysicsTime() const { return gpuTimePhysics; }
    float getThermalTime() const { return gpuTimeThermal; }
    float getRenderTime() const { return gpuTimeRender; }

    bool initialize()
    {
        // Initialize textures
        createTextures();

        // Load and compile shaders
        if (!loadShaders())
        {
            std::cerr << "Failed to load shaders" << std::endl;
            return false;
        }

        // Create mesh for rendering
        createMesh();

        // Initialize terrain with some height data
        initializeTerrain();

        // Initialize time queries for evaluation
        glGenQueries(3, timeQueries);
        for (int i = 0; i < 3; ++i)
        {
            glBeginQuery(GL_TIME_ELAPSED, timeQueries[i]);
            glEndQuery(GL_TIME_ELAPSED);
        }

        return true;
    }

    void createTextures()
    {
        // Helper lambda for creating 2D float textures
        auto createTexture = [](GLenum internalFormat)
        {
            GLuint tex;
            glGenTextures(1, &tex);
            glBindTexture(GL_TEXTURE_2D, tex);
            glTexStorage2D(GL_TEXTURE_2D, 1, internalFormat, GRID_SIZE, GRID_SIZE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            return tex;
        };

        // Create double-buffered textures
        terrainTex[0] = createTexture(GL_R32F);
        terrainTex[1] = createTexture(GL_R32F);
        waterTex[0] = createTexture(GL_R32F);
        waterTex[1] = createTexture(GL_R32F);
        sedimentTex[0] = createTexture(GL_R32F);
        sedimentTex[1] = createTexture(GL_R32F);

        // Single-buffered textures
        fluxTex = createTexture(GL_RGBA32F);
        velocityTex = createTexture(GL_RG32F);
    }

    void initializeTerrain()
    {
        std::vector<float> heightData(GRID_SIZE * GRID_SIZE, 0.0f);

        if (TERRAIN_MODE == 1) // HeightMap
        {
            int width, height, channels;
            // 1 channel reading (grayscale)
            unsigned char *img = stbi_load(HEIGHTMAP_FILES[CURRENT_HEIGHTMAP], &width, &height, &channels, 1);

            if (img)
            {
                std::cout << "Heightmap loaded : " << HEIGHTMAP_FILES[CURRENT_HEIGHTMAP] << " (" << width << "x" << height << ")" << std::endl;
                for (int y = 0; y < GRID_SIZE; y++)
                {
                    for (int x = 0; x < GRID_SIZE; x++)
                    {
                        int imgX = (x * width) / GRID_SIZE;
                        int imgY = (y * height) / GRID_SIZE;
                        float val = img[imgY * width + imgX] / 255.0f;
                        float h = val * MOUNTAIN_HEIGHT;
                        if (x <= 1 || x >= GRID_SIZE - 2 || y <= 1 || y >= GRID_SIZE - 2)
                            h = 0.0f;
                        if (h < 1.0f)
                            h = 1.0f;

                        heightData[y * GRID_SIZE + x] = h;
                    }
                }
                stbi_image_free(img);
            }
            else
            {
                std::cerr << "ERROR : Failed to load heightmap : " << HEIGHTMAP_FILES[CURRENT_HEIGHTMAP] << std::endl;
                std::cerr << "Reason : " << stbi_failure_reason() << std::endl;
                TERRAIN_MODE = 0;
            }
        }

        if (TERRAIN_MODE == 0) // Procedural
        {
            // Procedural terrain generation (FBM + Ridge Noise)
            auto hash = [](float n)
            { return std::fmod(std::sin(n) * 43758.5453f, 1.0f); };
            auto noise = [&](float x, float y)
            {
                float ix = std::floor(x), iy = std::floor(y);
                float fx = x - ix, fy = y - iy;
                float ux = fx * fx * (3.0f - 2.0f * fx);
                float uy = fy * fy * (3.0f - 2.0f * fy);
                float n = ix + iy * 57.0f;
                return (hash(n) * (1.0f - ux) + hash(n + 1.0f) * ux) * (1.0f - uy) +
                       (hash(n + 57.0f) * (1.0f - ux) + hash(n + 58.0f) * ux) * uy;
            };
            auto fbm = [&](float x, float y, int octaves)
            {
                float v = 0.0f, a = 0.5f, shift = 100.0f;
                for (int i = 0; i < octaves; ++i)
                {
                    v += a * noise(x, y);
                    x = x * 2.0f + shift;
                    y = y * 2.0f + shift;
                    a *= 0.5f;
                }
                return v;
            };
            auto ridge = [&](float x, float y, int octaves)
            {
                float v = 0.0f, a = 0.5f, prev = 1.0f, shift = 100.0f;
                for (int i = 0; i < octaves; ++i)
                {
                    float n = 1.0f - std::abs(noise(x, y) * 2.0f - 1.0f);
                    n = pow(n, 2.0f);
                    v += n * a * prev;
                    prev = n;
                    x = x * 2.0f + shift;
                    y = y * 2.0f + shift;
                    a *= 0.5f;
                }
                return v;
            };

            float seed = TERRAIN_SEED;
            for (int y = 0; y < GRID_SIZE; y++)
            {
                for (int x = 0; x < GRID_SIZE; x++)
                {
                    float nx = (float)x / GRID_SIZE * 2.0f - 1.0f;
                    float ny = (float)y / GRID_SIZE * 2.0f - 1.0f;
                    float mountains = ridge((nx + seed) * MOUNTAIN_FREQ, (ny + seed) * MOUNTAIN_FREQ, 8);
                    float base = fbm((nx + seed) * BASE_FREQ, (ny + seed) * BASE_FREQ, 6);
                    float h = mountains * MOUNTAIN_HEIGHT + base * BASE_HEIGHT;
                    float dist = sqrt(nx * nx + ny * ny);
                    float distort = fbm((nx + seed) * 5.0f, (ny + seed) * 5.0f, 3) * 0.25f;
                    float mask = 1.0f - glm::smoothstep(0.65f, 0.95f, dist + distort);
                    h *= mask;
                    h += noise(nx * 80.0f, ny * 80.0f) * 0.4f;
                    if (h < 1.0f)
                        h = 1.0f;
                    if (x <= 1 || x >= GRID_SIZE - 2 || y <= 1 || y >= GRID_SIZE - 2)
                        h = 0.0f;
                    heightData[y * GRID_SIZE + x] = h;
                }
            }
        }

        // Sending texture to GPU
        glBindTexture(GL_TEXTURE_2D, terrainTex[0]);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, GRID_SIZE, GRID_SIZE, GL_RED, GL_FLOAT, heightData.data());
        glBindTexture(GL_TEXTURE_2D, terrainTex[1]);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, GRID_SIZE, GRID_SIZE, GL_RED, GL_FLOAT, heightData.data());

        // Water and sediment start at 0
        std::vector<float> zeroData(GRID_SIZE * GRID_SIZE, 0.0f);
        glBindTexture(GL_TEXTURE_2D, waterTex[0]);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, GRID_SIZE, GRID_SIZE, GL_RED, GL_FLOAT, zeroData.data());
        glBindTexture(GL_TEXTURE_2D, waterTex[1]);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, GRID_SIZE, GRID_SIZE, GL_RED, GL_FLOAT, zeroData.data());
        glBindTexture(GL_TEXTURE_2D, sedimentTex[0]);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, GRID_SIZE, GRID_SIZE, GL_RED, GL_FLOAT, zeroData.data());
        glBindTexture(GL_TEXTURE_2D, sedimentTex[1]);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, GRID_SIZE, GRID_SIZE, GL_RED, GL_FLOAT, zeroData.data());
    }

    void simulationStep(bool isPlacingRiver)
    {
        const int workGroupsX = (GRID_SIZE + 15) / 16;
        const int workGroupsY = (GRID_SIZE + 15) / 16;

        // STRATEGY :
        // READ  : The buffer containing the validated state from the previous frame.
        // WRITE : The temporary buffer for intermediate calculations.
        // At the end, we bring everything back to READ.
        int READ = currentBuffer;
        int WRITE = 1 - currentBuffer;

        // PHASE 1: PHYSICS (OUTBOUND: READ -> WRITE)
        // We compute Rain, Flux, Water Update, and Erosion.
        // Results are stored in the WRITE buffer.
        // Pass 1: Water Increment (Rain)
        // Reads READ, Writes to WRITE
        glBeginQuery(GL_TIME_ELAPSED, timeQueries[0]); // Start physics query
        glUseProgram(waterIncrementShader);
        glBindImageTexture(0, waterTex[READ], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
        glBindImageTexture(1, waterTex[WRITE], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);

        int actuallyEmitWater = (ENABLE_RIVER && !isPlacingRiver) ? 1 : 0;
        glUniform1i(glGetUniformLocation(waterIncrementShader, "enableRiver"), actuallyEmitWater);

        glUniform1f(glGetUniformLocation(waterIncrementShader, "dt"), DT);
        glUniform1f(glGetUniformLocation(waterIncrementShader, "rainRate"), RAIN_RATE);
        glUniform1i(glGetUniformLocation(waterIncrementShader, "enableRiver"), ENABLE_RIVER ? 1 : 0);
        glUniform1f(glGetUniformLocation(waterIncrementShader, "RiverRate"), RIVER_RATE);
        glUniform1f(glGetUniformLocation(waterIncrementShader, "RiverRadius"), RIVER_RADIUS);
        glUniform2f(glGetUniformLocation(waterIncrementShader, "RiverSourcePos"), RIVER_SOURCE_POS.x, RIVER_SOURCE_POS.y);

        glDispatchCompute(workGroupsX, workGroupsY, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

        // Pass 2: Flux Computation
        // Reads Terrain[READ], Water[WRITE] (updated with rain), Writes to Flux
        glUseProgram(fluxComputeShader);
        glBindImageTexture(0, terrainTex[READ], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
        glBindImageTexture(1, waterTex[WRITE], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
        glBindImageTexture(2, fluxTex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
        glUniform1f(glGetUniformLocation(fluxComputeShader, "dt"), DT);
        glUniform1f(glGetUniformLocation(fluxComputeShader, "g"), GRAVITY);
        glUniform1f(glGetUniformLocation(fluxComputeShader, "pipeArea"), PIPE_AREA);
        glUniform1f(glGetUniformLocation(fluxComputeShader, "pipeLength"), PIPE_LENGTH);
        glUniform1f(glGetUniformLocation(fluxComputeShader, "cellSize"), CELL_SIZE);
        glDispatchCompute(workGroupsX, workGroupsY, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

        // Pass 3: Water Surface & Velocity Update
        // Reads/Writes Water[WRITE] (In-place update on temp buffer)
        glUseProgram(waterUpdateShader);
        glBindImageTexture(0, waterTex[WRITE], 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F);
        glBindImageTexture(1, fluxTex, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
        glBindImageTexture(2, velocityTex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG32F);
        glUniform1f(glGetUniformLocation(waterUpdateShader, "dt"), DT);
        glUniform1f(glGetUniformLocation(waterUpdateShader, "cellSize"), CELL_SIZE);
        glDispatchCompute(workGroupsX, workGroupsY, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

        // Pass 4: Erosion & Deposition
        // Reads READ, Writes to WRITE
        glUseProgram(erosionShader);
        glBindImageTexture(0, terrainTex[READ], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
        glBindImageTexture(1, terrainTex[WRITE], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
        glBindImageTexture(2, sedimentTex[READ], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
        glBindImageTexture(3, sedimentTex[WRITE], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
        glBindImageTexture(4, velocityTex, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F);
        glBindImageTexture(5, waterTex[READ], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
        glUniform1f(glGetUniformLocation(erosionShader, "Kc"), KC);
        glUniform1f(glGetUniformLocation(erosionShader, "Ks"), KS);
        glUniform1f(glGetUniformLocation(erosionShader, "Kd"), KD);
        glUniform1f(glGetUniformLocation(erosionShader, "dt"), DT);
        glUniform1f(glGetUniformLocation(erosionShader, "cellSize"), CELL_SIZE);
        glDispatchCompute(workGroupsX, workGroupsY, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

        // PHASE 2: POST-PROCESS (RETURN: WRITE -> READ)
        // We take the temporary results (WRITE) and finalize them back into (READ).
        // This ensures 'currentBuffer' (pointing to READ) always holds the final data.

        // Pass 5: Sediment Transport
        // Reads Sediment[WRITE] (from erosion pass), Writes to Sediment[READ] (final)
        glUseProgram(sedimentTransportShader);
        glBindImageTexture(0, sedimentTex[WRITE], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
        glBindImageTexture(1, sedimentTex[READ], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
        glBindImageTexture(2, velocityTex, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F);
        glUniform1f(glGetUniformLocation(sedimentTransportShader, "dt"), DT);
        glUniform1i(glGetUniformLocation(sedimentTransportShader, "gridSize"), GRID_SIZE);
        glDispatchCompute(workGroupsX, workGroupsY, 1);

        // Pass 6: Evaporation
        // Reads Water[WRITE] (from Pass 3), Writes to Water[READ] (final)
        glUseProgram(evaporationShader);
        glBindImageTexture(0, waterTex[WRITE], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
        glBindImageTexture(1, waterTex[READ], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
        glUniform1f(glGetUniformLocation(evaporationShader, "dt"), DT);
        glUniform1f(glGetUniformLocation(evaporationShader, "Ke"), KE);
        glDispatchCompute(workGroupsX, workGroupsY, 1);
        glEndQuery(GL_TIME_ELAPSED); // End physics query

        // Pass 7: Thermal Erosion
        // Reads Terrain[WRITE] (from erosion pass), Writes to Terrain[READ] (final)
        glBeginQuery(GL_TIME_ELAPSED, timeQueries[1]); // Start thermal erosion query
        glUseProgram(thermalErosionShader);
        glBindImageTexture(0, terrainTex[WRITE], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
        glBindImageTexture(1, terrainTex[READ], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
        glUniform1f(glGetUniformLocation(thermalErosionShader, "talusAngle"), TALUS_ANGLE);
        glUniform1f(glGetUniformLocation(thermalErosionShader, "erosionRate"), THERMAL_EROSION_RATE);
        glUniform1f(glGetUniformLocation(thermalErosionShader, "dt"), DT);
        glUniform1f(glGetUniformLocation(thermalErosionShader, "cellSize"), CELL_SIZE);
        glDispatchCompute(workGroupsX, workGroupsY, 1);
        glEndQuery(GL_TIME_ELAPSED); // End thermal erosion query
        // Final Synchronization
        glMemoryBarrier(GL_ALL_BARRIER_BITS);

        // We read from READ, processed in WRITE, and brought the final result back to READ.
        // currentBuffer remains valid for rendering.
    }

    void render(const glm::mat4 &viewProj, float time, const glm::vec3 &camPos, bool showSediment, bool isPlacingRiver, const glm::vec2 &hoverPos)
    {
        // Retrieve the results from previous evaluation
        GLuint64 elapsed;
        glGetQueryObjectui64v(timeQueries[0], GL_QUERY_RESULT, &elapsed);
        gpuTimePhysics = (float)elapsed / 1000000.0f;

        glGetQueryObjectui64v(timeQueries[1], GL_QUERY_RESULT, &elapsed);
        gpuTimeThermal = (float)elapsed / 1000000.0f;

        glGetQueryObjectui64v(timeQueries[2], GL_QUERY_RESULT, &elapsed);
        gpuTimeRender = (float)elapsed / 1000000.0f;

        // Start render query
        glBeginQuery(GL_TIME_ELAPSED, timeQueries[2]);

        glUseProgram(renderProgram);

        // Bind terrain and water textures
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, terrainTex[currentBuffer]);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, waterTex[currentBuffer]);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, sedimentTex[currentBuffer]);

        glUniformMatrix4fv(glGetUniformLocation(renderProgram, "viewProj"), 1, GL_FALSE, &viewProj[0][0]);
        glUniform1i(glGetUniformLocation(renderProgram, "terrainTex"), 0);
        glUniform1i(glGetUniformLocation(renderProgram, "waterTex"), 1);
        glUniform1i(glGetUniformLocation(renderProgram, "sedimentTex"), 2);
        glUniform1f(glGetUniformLocation(renderProgram, "cellSize"), CELL_SIZE);
        glUniform1i(glGetUniformLocation(renderProgram, "gridSize"), GRID_SIZE);
        glUniform1f(glGetUniformLocation(renderProgram, "u_Time"), time);
        glUniform3fv(glGetUniformLocation(renderProgram, "u_ViewPos"), 1, &camPos[0]);
        glUniform1i(glGetUniformLocation(renderProgram, "u_ShowSediment"), showSediment ? 1 : 0);

        glm::vec2 renderRiverPos = isPlacingRiver ? hoverPos : RIVER_SOURCE_POS;
        glUniform2fv(glGetUniformLocation(renderProgram, "u_RiverPos"), 1, &renderRiverPos[0]);
        glUniform1f(glGetUniformLocation(renderProgram, "u_RiverRadius"), RIVER_RADIUS);
        int showPreview = (isPlacingRiver) ? 1 : 0;
        glUniform1i(glGetUniformLocation(renderProgram, "u_ShowRiverPreview"), showPreview);

        glBindVertexArray(meshVAO);
        glDrawElements(GL_TRIANGLES, meshIndexCount, GL_UNSIGNED_INT, 0);

        glEndQuery(GL_TIME_ELAPSED); // End render query
    }

    bool loadShaders(); // Implementation below
    void createMesh();  // Implementation below

    ~HydraulicErosion()
    {
        // Cleanup
        glDeleteTextures(2, terrainTex);
        glDeleteTextures(2, waterTex);
        glDeleteTextures(2, sedimentTex);
        glDeleteTextures(1, &fluxTex);
        glDeleteTextures(1, &velocityTex);
        glDeleteProgram(waterIncrementShader);
        glDeleteProgram(fluxComputeShader);
        glDeleteProgram(waterUpdateShader);
        glDeleteProgram(erosionShader);
        glDeleteProgram(sedimentTransportShader);
        glDeleteProgram(evaporationShader);
        glDeleteProgram(thermalErosionShader);
        glDeleteProgram(renderProgram);
        glDeleteVertexArrays(1, &meshVAO);
        glDeleteBuffers(1, &meshVBO);
        glDeleteBuffers(1, &meshEBO);
    }
};

// Helper function to compile and link shaders
GLuint createShaderProgram(const char *vertexSource, const char *fragmentSource)
{
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexSource, NULL);
    glCompileShader(vertexShader);

    // Check vertex shader
    GLint success;
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        char infoLog[512];
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cerr << "Vertex Shader Error:\n"
                  << infoLog << std::endl;
        return 0;
    }

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
    glCompileShader(fragmentShader);

    // Check fragment shader
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        char infoLog[512];
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cerr << "Fragment Shader Error:\n"
                  << infoLog << std::endl;
        return 0;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    // Check linking
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success)
    {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        std::cerr << "Shader Linking Error:\n"
                  << infoLog << std::endl;
        return 0;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return program;
}

// Similar helper for compute shaders
GLuint createComputeShader(const char *computeSource)
{
    GLuint computeShader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(computeShader, 1, &computeSource, NULL);
    glCompileShader(computeShader);

    GLint success;
    glGetShaderiv(computeShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        char infoLog[512];
        glGetShaderInfoLog(computeShader, 512, NULL, infoLog);
        std::cerr << "Compute Shader Error:\n"
                  << infoLog << std::endl;
        return 0;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, computeShader);
    glLinkProgram(program);

    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success)
    {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        std::cerr << "Compute Program Linking Error:\n"
                  << infoLog << std::endl;
        return 0;
    }

    glDeleteShader(computeShader);
    return program;
}

void HydraulicErosion::createMesh()
{
    std::vector<float> vertices;
    std::vector<unsigned int> indices;

    // Generate grid vertices
    for (int y = 0; y < GRID_SIZE; y++)
    {
        for (int x = 0; x < GRID_SIZE; x++)
        {
            // Position (world space)
            float posX = (float)x * CELL_SIZE;
            float posY = 0.0f; // Will be displaced in vertex shader
            float posZ = (float)y * CELL_SIZE;

            // Texture coordinates (normalized 0-1)
            float u = (float)x / (float)(GRID_SIZE - 1);
            float v = (float)y / (float)(GRID_SIZE - 1);

            // Add vertex: position (3) + texcoords (2)
            vertices.push_back(posX);
            vertices.push_back(posY);
            vertices.push_back(posZ);
            vertices.push_back(u);
            vertices.push_back(v);
        }
    }

    // Generate indices for triangles
    for (int y = 0; y < GRID_SIZE - 1; y++)
    {
        for (int x = 0; x < GRID_SIZE - 1; x++)
        {
            int topLeft = y * GRID_SIZE + x;
            int topRight = topLeft + 1;
            int bottomLeft = (y + 1) * GRID_SIZE + x;
            int bottomRight = bottomLeft + 1;

            // First triangle
            indices.push_back(topLeft);
            indices.push_back(bottomLeft);
            indices.push_back(topRight);

            // Second triangle
            indices.push_back(topRight);
            indices.push_back(bottomLeft);
            indices.push_back(bottomRight);
        }
    }

    meshIndexCount = indices.size();

    // Create VAO
    glGenVertexArrays(1, &meshVAO);
    glBindVertexArray(meshVAO);

    // Create VBO
    glGenBuffers(1, &meshVBO);
    glBindBuffer(GL_ARRAY_BUFFER, meshVBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float),
                 vertices.data(), GL_STATIC_DRAW);

    // Create EBO
    glGenBuffers(1, &meshEBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, meshEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int),
                 indices.data(), GL_STATIC_DRAW);

    // Set vertex attributes
    // Position attribute (location 0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);

    // TexCoord attribute (location 1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                          (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

bool HydraulicErosion::loadShaders()
{
    std::string source;
    // Rendering Shaders
    // VERTEX SHADER for Rendering
    std::cout << "Loading Render Shaders..." << std::endl;
    source = loadShaderSourceFromFile("shaders/render.vert");
    if (source.empty())
        return false;

    // FRAGMENT SHADER for Rendering
    std::string fsSourceStr = loadShaderSourceFromFile("shaders/render.frag");
    if (fsSourceStr.empty())
        return false;

    renderProgram = createShaderProgram(source.c_str(), fsSourceStr.c_str());
    if (!renderProgram)
        return false;

    // Compute Shaders (Load all compute shader sources)
    // Water Increment
    std::cout << "Loading water_increment.glsl..." << std::endl;
    source = loadShaderSourceFromFile("shaders/water_increment.glsl");
    if (source.empty())
        return false;
    waterIncrementShader = createComputeShader(source.c_str());
    if (!waterIncrementShader)
        return false;

    // Flux Computation
    std::cout << "Loading flux_compute.glsl..." << std::endl;
    source = loadShaderSourceFromFile("shaders/flux_compute.glsl");
    if (source.empty())
        return false;
    fluxComputeShader = createComputeShader(source.c_str());
    if (!fluxComputeShader)
        return false;

    // Water Update
    std::cout << "Loading water_update.glsl..." << std::endl;
    source = loadShaderSourceFromFile("shaders/water_update.glsl");
    if (source.empty())
        return false;
    waterUpdateShader = createComputeShader(source.c_str());
    if (!waterUpdateShader)
        return false;

    // Erosion & Deposition
    std::cout << "Loading erosion_deposition.glsl..." << std::endl;
    source = loadShaderSourceFromFile("shaders/erosion_deposition.glsl");
    if (source.empty())
        return false;
    erosionShader = createComputeShader(source.c_str());
    if (!erosionShader)
        return false;

    // Sediment Transport
    std::cout << "Loading sediment_transport.glsl..." << std::endl;
    source = loadShaderSourceFromFile("shaders/sediment_transport.glsl");
    if (source.empty())
        return false;
    sedimentTransportShader = createComputeShader(source.c_str());
    if (!sedimentTransportShader)
        return false;

    // Evaporation
    std::cout << "Loading evaporation.glsl..." << std::endl;
    source = loadShaderSourceFromFile("shaders/evaporation.glsl");
    if (source.empty())
        return false;
    evaporationShader = createComputeShader(source.c_str());
    if (!evaporationShader)
        return false;

    // Thermal Erosion
    std::cout << "Loading thermal_erosion.glsl..." << std::endl;
    source = loadShaderSourceFromFile("shaders/thermal_erosion.glsl");
    if (source.empty())
        return false;
    thermalErosionShader = createComputeShader(source.c_str());
    if (!thermalErosionShader)
        return false;

    std::cout << "All external shaders loaded successfully!" << std::endl;
    return true;
}

// MAIN ENTRY POINT

// Camera variables
glm::vec3 cameraTarget(GRID_SIZE * CELL_SIZE * 0.5f, 0.0f, GRID_SIZE * CELL_SIZE * 0.5f);

float cameraDistance = 300.0f; // Zoom
float cameraAngleX = 45.0f;    // Horizontal Rotation
float cameraAngleY = 30.0f;    // Vertical Rotation

// Mouse state
bool isDraggingRotate = false; // Left Click
bool isDraggingPan = false;    // Right Click
bool firstMouse = true;
float lastX = 0.0f;
float lastY = 0.0f;

// Visualization parameters
bool showSediment = false; // Toggle sediment visualization

void mouse_callback(GLFWwindow *window, double xpos, double ypos)
{
    if (ImGui::GetIO().WantCaptureMouse)
    { // Ignore if ImGui is using the mouse
        firstMouse = true;
        return;
    }
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;

    // Left Click: ROTATION
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
    {
        cameraAngleX += xoffset * 0.5f;
        cameraAngleY += yoffset * 0.5f;

        if (cameraAngleY > 89.0f)
            cameraAngleY = 89.0f;
        if (cameraAngleY < 5.0f)
            cameraAngleY = 5.0f;
    }

    // Right Click: PAN
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
    {
        // Calculate forward and right vectors based on current angles
        float radX = glm::radians(cameraAngleX);
        // float radY = glm::radians(cameraAngleY);

        // Forward vector (projected on XZ plane)
        glm::vec3 forward(-sin(radX), 0.0f, -cos(radX));
        // Right vector (perpendicular to forward and up)
        glm::vec3 right = glm::cross(forward, glm::vec3(0, 1, 0));

        float speed = cameraDistance * 0.002f;

        cameraTarget -= right * xoffset * speed;
        cameraTarget -= forward * yoffset * speed;
    }
}

void scroll_callback(GLFWwindow * /*window*/, double /*xoffset*/, double yoffset)
{
    if (ImGui::GetIO().WantCaptureMouse)
        return;
    // Zoom
    cameraDistance -= yoffset * 20.0f;
    if (cameraDistance < 10.0f)
        cameraDistance = 10.0f;
    if (cameraDistance > 1000.0f)
        cameraDistance = 1000.0f;
}

// CFL Condition Checker
float calculateMaxDT(float cellSize, float maxVelocity = 100.0f)
{
    // CFL condition: dt <= cellSize / maxVelocity
    // safety factor of 0.5 for stability margin
    const float SAFETY_FACTOR = 0.5f;
    return (cellSize / maxVelocity) * SAFETY_FACTOR;
}

void checkCFLCondition(float dt, float cellSize, float maxVelocity = 100.0f)
{
    float maxDT = calculateMaxDT(cellSize, maxVelocity);
    if (dt > maxDT)
    {
        std::cout << "   WARNING: CFL Condition Violated!" << std::endl;
        std::cout << "   Current dt = " << dt << std::endl;
        std::cout << "   Recommended max dt = " << maxDT << std::endl;
        std::cout << "   Simulation may become unstable!" << std::endl;
        std::cout << "   Consider reducing dt or increasing cellSize." << std::endl;
    }
}

glm::vec3 getRayFromMouse(double mouseX, double mouseY, int screenWidth, int screenHeight, const glm::mat4 &viewMatrix, const glm::mat4 &projectionMatrix)
{
    float x = (2.0f * mouseX) / screenWidth - 1.0f;
    float y = 1.0f - (2.0f * mouseY) / screenHeight;
    glm::vec4 clipCoords(x, y, -1.0f, 1.0f);
    glm::vec4 eyeCoords = glm::inverse(projectionMatrix) * clipCoords;
    eyeCoords = glm::vec4(eyeCoords.x, eyeCoords.y, -1.0f, 0.0f);

    glm::vec3 rayWorld = glm::vec3(glm::inverse(viewMatrix) * eyeCoords);
    return glm::normalize(rayWorld);
}

int main()
{
    // Initialize GLFW
    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

    // Create window
    GLFWwindow *window = glfwCreateWindow(1280, 720, "GPU Hydraulic Erosion Simulation", NULL, NULL);
    if (!window)
    {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0); // Enable V-Sync (0 = off, 1 = on)

    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK)
    {
        std::cerr << "Failed to initialize GLEW: " << glewGetErrorString(err) << std::endl;
    }

    // ImGui Initialization
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Style Dark
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 430");

    // OpenGL settings
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.1f, 0.15f, 0.2f, 1.0f);

    // Initialize Erosion Simulation
    HydraulicErosion simulation;

    if (!simulation.initialize())
    {
        std::cerr << "Failed to initialize erosion simulation" << std::endl;
        return -1;
    }

    cpuSim = new HydraulicErosionCPU(GRID_SIZE, CELL_SIZE);

    std::cout << "=== Simulation Parameters ===" << std::endl;
    std::cout << "Grid Size: " << GRID_SIZE << "x" << GRID_SIZE << std::endl;
    std::cout << "Cell Size: " << CELL_SIZE << " meters" << std::endl;
    std::cout << "Time Step: " << DT << " seconds" << std::endl;
    checkCFLCondition(DT, CELL_SIZE);
    std::cout << "=============================" << std::endl
              << std::endl;

    bool isPlacingRiver = false;
    bool hasPlacedRiver = false;
    bool wasUsingGPU = true;
    glm::vec2 hoverGridPos = glm::vec2(0.0f);

    // Main Loop
    while (!glfwWindowShouldClose(window))
    {
        // Poll Events
        glfwPollEvents();

        // Input ESC
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        {
            ImGui::Begin("Hydraulic Erosion Simulation Controls");

            ImGui::Text("FPS: %.1f", io.Framerate);

            ImGui::Separator();

            ImGui::Text("Performance (1 iter)");
            if (useGPU)
            {
                ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.2f, 1.0f), "Mode : GPU (Compute Shaders)");
                ImGui::Text("Physics Simulation : %.3f ms", simulation.getPhysicsTime());
                ImGui::Text("Thermal Erosion    : %.3f ms", simulation.getThermalTime());
                ImGui::Text("Visual Rendering   : %.3f ms", simulation.getRenderTime());
                float totalCycle = simulation.getPhysicsTime() + simulation.getThermalTime() + simulation.getRenderTime();
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "CPS: %.3f ms", totalCycle);
            }
            else
            {
                ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Mode : CPU (C++)");
                ImGui::Text("Physics Simulation : %.3f ms", cpuSim->timePhysics);
                ImGui::Text("Thermal Erosion    : %.3f ms", cpuSim->timeThermal);
                float totalRender = cpuTransferTime + simulation.getRenderTime();
                ImGui::Text("Visual Rendering   : %.3f ms", totalRender);
                float totalCycle = cpuSim->timePhysics + cpuSim->timeThermal + totalRender;
                ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "CPS: %.3f ms", totalCycle);
            }
            ImGui::Separator();
            ImGui::Checkbox("Use GPU", &useGPU);
            ImGui::Separator();

            ImGui::Separator();
            if (ImGui::Checkbox("Show Sediment (Scientific Mode)", &showSediment))
            {
                // Toggle sediment visualization
            }
            ImGui::Spacing();

            if (ImGui::CollapsingHeader("Physics Simulation", ImGuiTreeNodeFlags_DefaultOpen))
            {
                ImGui::SliderInt("Iterations/Frame", &ITERATIONS_PER_FRAME, 0, 10);
                if (ImGui::SliderFloat("Time Step (DT)", &DT, 0.0001f, 0.01f, "%.4f"))
                {
                    checkCFLCondition(DT, CELL_SIZE);
                }
                float maxDT = calculateMaxDT(CELL_SIZE);
                bool cflSafe = (DT <= maxDT);
                ImGui::TextColored(
                    cflSafe ? ImVec4(0.0f, 1.0f, 0.0f, 1.0f) : ImVec4(1.0f, 0.0f, 0.0f, 1.0f),
                    "CFL Status: %s (max dt: %.4f)",
                    cflSafe ? "SAFE" : "VIOLATED",
                    maxDT);
                ImGui::SliderFloat("Rain", &RAIN_RATE, 0.0f, 0.1f);
                ImGui::SliderFloat("Evaporation (Ke)", &KE, 0.0f, 0.1f);
                ImGui::SliderFloat("Gravity", &GRAVITY, 1.0f, 20.0f);
                ImGui::SliderFloat("Pipe Area", &PIPE_AREA, 0.1f, 5.0f);
            }

            if (ImGui::CollapsingHeader("Erosion Parameters", ImGuiTreeNodeFlags_DefaultOpen))
            {
                ImGui::SliderFloat("Capacity (Kc)", &KC, 0.0f, 5.0f);
                ImGui::SliderFloat("Dissolution (Ks)", &KS, 0.0f, 5.0f);
                ImGui::SliderFloat("Deposition (Kd)", &KD, 0.0f, 1.0f);
                ImGui::SliderFloat("Thermal Erosion", &THERMAL_EROSION_RATE, 0.0f, 15.0f);
                ImGui::SliderFloat("Talus Angle", &TALUS_ANGLE, 0.0f, 1.5f);
            }

            if (ImGui::CollapsingHeader("Generation Terrain / Heightmap", ImGuiTreeNodeFlags_DefaultOpen))
            {
                bool changed = false;

                ImGui::Text("Terrain :");
                changed |= ImGui::RadioButton("Procedural (Noise)", &TERRAIN_MODE, 0);
                ImGui::SameLine();
                changed |= ImGui::RadioButton("Heightmap (Image)", &TERRAIN_MODE, 1);

                ImGui::Separator();

                if (TERRAIN_MODE == 1)
                {
                    changed |= ImGui::Combo("Heightmap List", &CURRENT_HEIGHTMAP, HEIGHTMAP_FILES, IM_ARRAYSIZE(HEIGHTMAP_FILES));
                    changed |= ImGui::SliderFloat("Max Height", &MOUNTAIN_HEIGHT, 10.0f, 300.0f);
                }
                else
                {
                    changed |= ImGui::SliderFloat("Seed", &TERRAIN_SEED, 0.0f, 100.0f);
                    changed |= ImGui::SliderFloat("Mountain Freq", &MOUNTAIN_FREQ, 0.1f, 5.0f);
                    changed |= ImGui::SliderFloat("Mountain Height", &MOUNTAIN_HEIGHT, 10.0f, 150.0f);
                    changed |= ImGui::SliderFloat("Base Frequency", &BASE_FREQ, 0.1f, 10.0f);
                    changed |= ImGui::SliderFloat("Base Height", &BASE_HEIGHT, 0.0f, 50.0f);
                }

                if (ImGui::Button("Generate / Reload Terrain") || changed)
                {
                    simulation.initializeTerrain();

                    if (!useGPU)
                    {
                        glBindTexture(GL_TEXTURE_2D, simulation.terrainTex[0]);
                        glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, cpuSim->terrain[0].data());
                        cpuSim->terrain[1] = cpuSim->terrain[0];

                        std::fill(cpuSim->water[0].begin(), cpuSim->water[0].end(), 0.0f);
                        std::fill(cpuSim->water[1].begin(), cpuSim->water[1].end(), 0.0f);
                        std::fill(cpuSim->sediment[0].begin(), cpuSim->sediment[0].end(), 0.0f);
                        std::fill(cpuSim->sediment[1].begin(), cpuSim->sediment[1].end(), 0.0f);
                    }
                }
            }

            if (ImGui::CollapsingHeader("River Parameters", ImGuiTreeNodeFlags_DefaultOpen))
            {

                if (ImGui::Checkbox("Enable River Source", &ENABLE_RIVER))
                {
                    if (ENABLE_RIVER && !hasPlacedRiver)
                    {
                        isPlacingRiver = true;
                    }
                }
                if (isPlacingRiver)
                {
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.4f, 0.0f, 1.0f));
                    if (ImGui::Button("Cancel Placement [ESC]"))
                    {
                        isPlacingRiver = false;
                        if (!hasPlacedRiver)
                            ENABLE_RIVER = false;
                    }
                    ImGui::PopStyleColor();
                    ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "Click on terrain to place source.");
                }
                else
                {
                    if (ImGui::Button("Place / Move River Source (Mouse)"))
                    {
                        isPlacingRiver = true;
                        ENABLE_RIVER = true;
                    }
                }
                if (ENABLE_RIVER)
                {
                    ImGui::SliderFloat2("Position (X,Y)", &RIVER_SOURCE_POS[0], 0.0f, (float)GRID_SIZE);
                    ImGui::SliderFloat("River Rate", &RIVER_RATE, 0.1f, 10.0f);
                    ImGui::SliderFloat("River Radius", &RIVER_RADIUS, 1.0f, 20.0f);

                    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Try to put the source on a slope to let gravity pull the water down");
                }
            }

            ImGui::End();
        }

        for (int i = 0; i < ITERATIONS_PER_FRAME; ++i)
        {
            if (useGPU)
            {
                simulation.simulationStep(isPlacingRiver);
            }
            else
            {
                cpuSim->dt = DT;
                cpuSim->gravity = GRAVITY;
                cpuSim->pipeArea = PIPE_AREA;
                cpuSim->pipeLength = PIPE_LENGTH;
                cpuSim->Kc = KC;
                cpuSim->Ks = KS;
                cpuSim->Kd = KD;
                cpuSim->Ke = KE;
                cpuSim->rainRate = RAIN_RATE;
                cpuSim->riverRate = RIVER_RATE;
                cpuSim->riverRadius = RIVER_RADIUS;
                cpuSim->riverPos = RIVER_SOURCE_POS;
                cpuSim->enableRiver = ENABLE_RIVER;
                cpuSim->thermalErosionRate = THERMAL_EROSION_RATE;
                cpuSim->talusAngle = TALUS_ANGLE;

                cpuSim->simulationStep(isPlacingRiver);
            }
        }

        if (useGPU != wasUsingGPU)
        {
            if (!useGPU)
            {
                int buf = simulation.currentBuffer;

                glBindTexture(GL_TEXTURE_2D, simulation.terrainTex[buf]);
                glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, cpuSim->terrain[0].data());
                cpuSim->terrain[1] = cpuSim->terrain[0];

                glBindTexture(GL_TEXTURE_2D, simulation.waterTex[buf]);
                glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, cpuSim->water[0].data());
                cpuSim->water[1] = cpuSim->water[0];

                glBindTexture(GL_TEXTURE_2D, simulation.sedimentTex[buf]);
                glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, cpuSim->sediment[0].data());
                cpuSim->sediment[1] = cpuSim->sediment[0];
            }

            wasUsingGPU = useGPU;
        }

        // Camera Setup
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);

        float camX = cameraDistance * cos(glm::radians(cameraAngleY)) * sin(glm::radians(cameraAngleX));
        float camY = cameraDistance * sin(glm::radians(cameraAngleY));
        float camZ = cameraDistance * cos(glm::radians(cameraAngleY)) * cos(glm::radians(cameraAngleX));

        glm::vec3 center(GRID_SIZE * CELL_SIZE * 0.5f, 10.0f, GRID_SIZE * CELL_SIZE * 0.5f);
        glm::vec3 cameraPos = cameraTarget + glm::vec3(camX, camY, camZ);

        glm::mat4 view = glm::lookAt(cameraPos, cameraTarget, glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)width / (float)height, 0.1f, 3000.0f);
        glm::mat4 viewProj = projection * view;

        if (isPlacingRiver)
        {
            if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            {
                isPlacingRiver = false;
            }

            if (!io.WantCaptureMouse)
            {
                double mouseX, mouseY;
                glfwGetCursorPos(window, &mouseX, &mouseY);

                glm::vec3 rayDir = getRayFromMouse(mouseX, mouseY, width, height, view, projection);

                if (rayDir.y < 0.0f)
                {
                    float t = (10.0f - cameraPos.y) / rayDir.y;
                    glm::vec3 hitPoint = cameraPos + rayDir * t;
                    hoverGridPos = glm::vec2(hitPoint.x / CELL_SIZE, hitPoint.z / CELL_SIZE);
                    hoverGridPos.x = glm::clamp(hoverGridPos.x, 0.0f, (float)GRID_SIZE);
                    hoverGridPos.y = glm::clamp(hoverGridPos.y, 0.0f, (float)GRID_SIZE);
                }

                if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
                {
                    RIVER_SOURCE_POS = hoverGridPos;
                    isPlacingRiver = false;
                    hasPlacedRiver = true;
                }
            }
        }
        // Render Scene
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // RAM transfer for CPU mode
        if (!useGPU)
        {
            auto tStart = std::chrono::high_resolution_clock::now();
            int buf = simulation.currentBuffer;

            glBindTexture(GL_TEXTURE_2D, simulation.terrainTex[buf]);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, GRID_SIZE, GRID_SIZE, GL_RED, GL_FLOAT, cpuSim->terrain[0].data());

            glBindTexture(GL_TEXTURE_2D, simulation.waterTex[buf]);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, GRID_SIZE, GRID_SIZE, GL_RED, GL_FLOAT, cpuSim->water[0].data());

            glBindTexture(GL_TEXTURE_2D, simulation.sedimentTex[buf]);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, GRID_SIZE, GRID_SIZE, GL_RED, GL_FLOAT, cpuSim->sediment[0].data());

            auto tEnd = std::chrono::high_resolution_clock::now();
            cpuTransferTime = std::chrono::duration<float, std::milli>(tEnd - tStart).count();
        }

        simulation.render(viewProj, (float)glfwGetTime(), cameraPos, showSediment, isPlacingRiver, hoverGridPos);

        // ImGui Render
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwTerminate();
    return 0;
}