
#include <emscripten.h>
#include <emscripten/bind.h>
#include <vector>
#include <cmath>
#include <string>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <algorithm>

template<typename T>
void copyToVector(const emscripten::val &typedArray, std::vector<T> &vec) {
    unsigned int length = typedArray["length"].as<unsigned int>();
    emscripten::val heap = emscripten::val::module_property("HEAPU8");
    emscripten::val memory = heap["buffer"];
    vec.reserve(length);

    emscripten::val memoryView = typedArray["constructor"].new_(memory, reinterpret_cast<uintptr_t>(vec.data()), length);

    memoryView.call<void>("set", typedArray);
}

std::vector<uint32_t> runSort(emscripten::val vproj,
             emscripten::val fbuf,
             size_t lastVertexCount,
             size_t vertexCount) {
    std::vector<float> viewProj = emscripten::convertJSArrayToNumberVector<float>(vproj);
    std::vector<float> f_buffer;
    copyToVector(fbuf, f_buffer);
    int maxDepth = -INT32_MAX;
    int minDepth = INT32_MAX;
    std::vector<uint32_t> depthIndex(vertexCount, 0);
    std::vector<int32_t> sizeList(vertexCount);
    for (size_t i = 0; i < vertexCount; i++) {
        int depth =
            static_cast<int>((viewProj[2] * f_buffer[8 * i + 0] +
                viewProj[6] * f_buffer[8 * i + 1] +
                viewProj[10] * f_buffer[8 * i + 2]) *
                4096);
        sizeList[i] = depth;
        minDepth = std::min(minDepth, depth);
        maxDepth = std::max(maxDepth, depth);
    }
    float depthInv = (256 * 256 - 1) / static_cast<float>(maxDepth - minDepth);
    std::vector<size_t> counts0(256 * 256, 0);

    for (size_t i = 0; i < vertexCount; i++) {
        sizeList[i] = static_cast<int32_t>((sizeList[i] - minDepth) * depthInv);
        counts0[sizeList[i]]++;
    }
    std::vector<uint32_t> starts0(256 * 256, 0);
    for (size_t i = 1; i < 256 * 256; i++) {
        starts0[i] = starts0[i - 1] + counts0[i - 1];
    }

    for (size_t i = 0; i < vertexCount; i++) {
        depthIndex[starts0[sizeList[i]]++] = i;
    }
    return depthIndex;
}

std::vector<uint32_t> test2() {
    return {69420};
}

std::string test() {
    return "WASSUP MOTHAFUCKAS!\n";
}

EMSCRIPTEN_BINDINGS(wasm_gsplat) {
    emscripten::register_vector<uint8_t>("VectorUint8");
    // emscripten::register_vector<Vertex>("VectorVertex");
    emscripten::register_vector<uint32_t>("VectorUint32vector");
    emscripten::register_vector<float>("VectorFloat32");
    emscripten::function("runSort", &runSort);
}
