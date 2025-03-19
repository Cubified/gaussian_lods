
#include <emscripten.h>
#include <emscripten/bind.h>
#include <vector>
#include <cmath>
#include <string>
#include <cstdint>
#include <cstring>
#include <stdexcept>

struct Vertex {
    float position[3];
    float scales[3];
    uint8_t rgba[4];
    uint8_t rot[4];
};

std::vector<Vertex> processPlyBuffer(const std::vector<uint8_t>& buffer) {
    const char* headerEnd = "end_header\n";
    const char* data = reinterpret_cast<const char*>(buffer.data());
    size_t headerEndPos = std::string(data, buffer.size()).find(headerEnd);
    if (headerEndPos == std::string::npos) {
        throw std::runtime_error("Unable to read .ply file header");
    }

    size_t vertexCount = 0;
    sscanf(data, "element vertex %zu", &vertexCount);

    size_t dataOffset = headerEndPos + strlen(headerEnd);

    std::vector<Vertex> vertices(vertexCount);

    for (size_t i = 0; i < vertexCount; ++i) {
        size_t offset = dataOffset + i * (3 * sizeof(float) + 3 * sizeof(float) + 4 + 4);

        float x = *reinterpret_cast<const float*>(&buffer[offset]);
        float y = *reinterpret_cast<const float*>(&buffer[offset + 4]);
        float z = *reinterpret_cast<const float*>(&buffer[offset + 8]);

        float scale_0 = *reinterpret_cast<const float*>(&buffer[offset + 12]);
        float scale_1 = *reinterpret_cast<const float*>(&buffer[offset + 16]);
        float scale_2 = *reinterpret_cast<const float*>(&buffer[offset + 20]);

        uint8_t r = buffer[offset + 24];
        uint8_t g = buffer[offset + 25];
        uint8_t b = buffer[offset + 26];
        uint8_t a = buffer[offset + 27];

        uint8_t rot_0 = buffer[offset + 28];
        uint8_t rot_1 = buffer[offset + 29];
        uint8_t rot_2 = buffer[offset + 30];
        uint8_t rot_3 = buffer[offset + 31];

        vertices[i] = {
            {x, y, z},
            {std::exp(scale_0), std::exp(scale_1), std::exp(scale_2)},
            {r, g, b, a},
            {rot_0, rot_1, rot_2, rot_3}
        };
    }

    // TODO: Build tree from the vertices here
    return vertices;
}

std::string test() {
    return "WASSUP MOTHAFUCKAS!\n";
}

EMSCRIPTEN_BINDINGS(wasm_gsplat) {
    // emscripten::register_vector<uint8_t>("VectorUint8");
    // emscripten::register_vector<Vertex>("VectorVertex");
    emscripten::function("processPlyBuffer", &processPlyBuffer);
    emscripten::function("test", &test);
}
