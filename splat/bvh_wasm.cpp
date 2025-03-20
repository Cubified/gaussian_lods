#include <array>
#include <memory>
#include <numeric>
#include <vector>
#include <iostream>

#include <Eigen/Eigen>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/bind.h>
#endif

constexpr size_t rowLength = 3 * 4 + 3 * 4 + 4 + 4;

template<typename T>
void copyToVector(const emscripten::val &typedArray, std::vector<T> &vec) {
    unsigned int length = typedArray["length"].as<unsigned int>();
    emscripten::val heap = emscripten::val::module_property("HEAPU8");
    emscripten::val memory = heap["buffer"];
    vec.reserve(length);

    emscripten::val memoryView = typedArray["constructor"].new_(memory, reinterpret_cast<uintptr_t>(vec.data()), length);

    memoryView.call<void>("set", typedArray);
}

class AABB {
public:
    AABB() { bounds = {{0, 0, 0, 0, 0, 0}}; }

    AABB(float minx, float maxx, float miny, float maxy, float minz, float maxz) {
        bounds = {{minx, maxx, miny, maxy, minz, maxz}};
    }

    inline float& minx() { return bounds[0]; }
    inline float& maxx() { return bounds[1]; }
    inline float& miny() { return bounds[2]; }
    inline float& maxy() { return bounds[3]; }
    inline float& minz() { return bounds[4]; }
    inline float& maxz() { return bounds[5]; }

    AABB merge(AABB &other) {
        return AABB(std::min(minx(), other.minx()),
                    std::max(maxx(), other.maxx()),
                    std::min(miny(), other.miny()),
                    std::max(maxy(), other.maxy()),
                    std::min(minz(), other.minz()),
                    std::max(maxz(), other.maxz()));
    }

    std::vector<Eigen::Vector3f> vertices3() {
        auto [minx, maxx, miny, maxy, minz, maxz] = bounds;
        return {
            {minx, miny, minz},
            {minx, miny, maxz},
            {minx, maxy, minz},
            {minx, maxy, maxz},
            {maxx, miny, minz},
            {maxx, miny, maxz},
            {maxx, maxy, minz},
            {maxx, maxy, maxz}
        };
    }

    std::vector<Eigen::Vector4f> vertices4() {
        auto [minx, maxx, miny, maxy, minz, maxz] = bounds;
        return {
            {minx, miny, minz, 1},
            {minx, miny, maxz, 1},
            {minx, maxy, minz, 1},
            {minx, maxy, maxz, 1},
            {maxx, miny, minz, 1},
            {maxx, miny, maxz, 1},
            {maxx, maxy, minz, 1},
            {maxx, maxy, maxz, 1}
        };
    }

    float surfaceArea() {
        auto [minx, maxx, miny, maxy, minz, maxz] = bounds;
        float x = maxx - minx;
        float y = maxy - miny;
        float z = maxz - minz;
        return 2 * (x * y + x * z + y * z);
    }
private:
    std::array<float, 6> bounds;
};

class GaussianCloud {
public:
    GaussianCloud() : vertexCount(0) {}
    GaussianCloud(void *buffer, size_t vertexCount) {
        positions.reserve(2 * vertexCount);
        scales.reserve(2 * vertexCount);
        rotations.reserve(2 * vertexCount);
        rgba.reserve(2 * vertexCount);

        for (size_t i = 0; i < vertexCount; i++) {
            parseElement(buffer, i);
        }
        this->vertexCount = vertexCount;
        positions.shrink_to_fit();
        scales.shrink_to_fit();
        rotations.shrink_to_fit();
        rgba.shrink_to_fit();
    }

    GaussianCloud(emscripten::val buftmp, size_t vertexCount) {
        std::vector<uint8_t> tmp;
        copyToVector(buftmp, tmp);
        void *buffer = tmp.data();
        positions.reserve(2 * vertexCount);
        scales.reserve(2 * vertexCount);
        rotations.reserve(2 * vertexCount);
        rgba.reserve(2 * vertexCount);

        for (size_t i = 0; i < vertexCount; i++) {
            parseElement(buffer, i);
        }
        this->vertexCount = vertexCount;
        positions.shrink_to_fit();
        scales.shrink_to_fit();
        rotations.shrink_to_fit();
        rgba.shrink_to_fit();
    }

    inline void parseElement(void *buffer, size_t i) {
        float *f_buffer = reinterpret_cast<float *>(buffer);
        uint8_t *u_buffer = reinterpret_cast<uint8_t *>(buffer);

        positions.push_back({
            f_buffer[8 * i + 0],
            f_buffer[8 * i + 1],
            f_buffer[8 * i + 2]
        });
        scales.push_back({
            f_buffer[8 * i + 3 + 0],
            f_buffer[8 * i + 3 + 1],
            f_buffer[8 * i + 3 + 2]
        });
        rgba.push_back({
            u_buffer[32 * i + 24 + 0],
            u_buffer[32 * i + 24 + 1],
            u_buffer[32 * i + 24 + 2],
            u_buffer[32 * i + 24 + 3]
        });
        // NOTE: Eigen::Quaternions have w term first
        rotations.push_back(Eigen::Quaternionf(
            (u_buffer[32 * i + 28 + 3] - 128) / static_cast<float>(128),
            (u_buffer[32 * i + 28 + 0] - 128) / static_cast<float>(128),
            (u_buffer[32 * i + 28 + 1] - 128) / static_cast<float>(128),
            (u_buffer[32 * i + 28 + 2] - 128) / static_cast<float>(128)
        ));
    }

    std::vector<uint8_t> toBuffer() {
        std::vector<uint8_t> buffervec(rowLength * vertexCount);
        uint8_t *buffer = buffervec.data();
        for (size_t i = 0; i < vertexCount; i++) {
            float *posP = reinterpret_cast<float *>(buffer + i * rowLength);
            float *scaP = reinterpret_cast<float *>(buffer + i * rowLength + 4 * 3);
            uint8_t *rgbaP = reinterpret_cast<uint8_t *>(buffer + i * rowLength + 4 * 3 + 4 * 3);
            uint8_t *rotP = reinterpret_cast<uint8_t *>(buffer + i * rowLength + 4 * 3 + 4 * 3 + 4);
            for (int j = 0; j < 3; j++) posP[j] = positions[i][j];
            for (int j = 0; j < 3; j++) scaP[j] = scales[i][j];
            for (int j = 0; j < 4; j++) rgbaP[j] = static_cast<uint8_t>(rgba[i][j]);
            // Store with w term last
            rotP[0] = static_cast<uint8_t>((128 * rotations[i].x()) + 128);
            rotP[1] = static_cast<uint8_t>((128 * rotations[i].y()) + 128);
            rotP[2] = static_cast<uint8_t>((128 * rotations[i].z()) + 128);
            rotP[3] = static_cast<uint8_t>((128 * rotations[i].w()) + 128);
        }
        // clear memory TODO: this is a hack, remove this if you ever re-use this code
        // Browsers limit each tab to ~2GB memory
        positions = std::vector<Eigen::Vector3f>();
        scales = std::vector<Eigen::Vector3f>();
        rotations = std::vector<Eigen::Quaternionf>();
        rgba = std::vector<Eigen::Vector4i>();
        return buffervec;
    }
    std::vector<Eigen::Vector3f> positions;
    std::vector<Eigen::Vector3f> scales;
    std::vector<Eigen::Quaternionf> rotations;
    std::vector<Eigen::Vector4i> rgba;
    size_t vertexCount;
};

AABB GaussianAABB(const GaussianCloud &cloud, size_t i) {
    const Eigen::Vector3f &pos = cloud.positions[i];
    const Eigen::Vector3f &scale = cloud.scales[i];
    const Eigen::Quaternionf &rotation = cloud.rotations[i];
    std::array<Eigen::Vector3f, 3> basis = {
        Eigen::Vector3f(3 * scale[0], 0, 0),
        Eigen::Vector3f(0, 3 * scale[1], 0),
        Eigen::Vector3f(0, 0, 3 * scale[2])
    };
    for (auto &v : basis) {
        v = rotation * v;
    }
    float minx = pos[0];
    float maxx = pos[0];
    float miny = pos[1];
    float maxy = pos[1];
    float minz = pos[2];
    float maxz = pos[2];
    for (auto &v : basis) {
        minx = std::min({minx, pos[0] + v[0], pos[0] - v[0]});
        maxx = std::max({maxx, pos[0] + v[0], pos[0] - v[0]});
        miny = std::min({miny, pos[1] + v[1], pos[1] - v[1]});
        maxy = std::max({maxy, pos[1] + v[1], pos[1] - v[1]});
        minz = std::min({minz, pos[2] + v[2], pos[2] - v[2]});
        maxz = std::max({maxz, pos[2] + v[2], pos[2] - v[2]});
    }
    return AABB(minx, maxx, miny, maxy, minz, maxz);
}

size_t MergeGaussian(GaussianCloud &cloud, size_t i, size_t j) {
    Eigen::Vector3f pos{0, 0, 0};
    Eigen::Vector3f scale{0, 0, 0};
    Eigen::Quaternionf rot{0, 0, 0, 0};
    Eigen::Vector4i rgba{0, 0, 0, 0};

    Eigen::Vector3f p1 = cloud.positions[i];
    Eigen::Vector3f s1 = cloud.scales[i];
    Eigen::Vector4i c1 = cloud.rgba[i];
    Eigen::Quaternionf r1 = cloud.rotations[i];
    Eigen::Matrix3f R1 = cloud.rotations[i].toRotationMatrix();
    Eigen::Matrix3f cov1;
    cov1 <<
        (s1.x() * s1.x()), 0, 0,
        0, (s1.y() * s1.y()), 0,
        0, 0, (s1.z() * s1.z());
    cov1 = R1.transpose() * cov1 * R1;
    float vol1 = cov1.determinant();

    Eigen::Vector3f p2 = cloud.positions[j];
    Eigen::Vector3f s2 = cloud.scales[j];
    Eigen::Vector4i c2 = cloud.rgba[j];
    Eigen::Quaternionf r2 = cloud.rotations[j];
    Eigen::Matrix3f R2 = cloud.rotations[j].toRotationMatrix();
    Eigen::Matrix3f cov2;
    cov2 <<
        (s2.x() * s2.x()), 0, 0,
        0, (s2.y() * s2.y()), 0,
        0, 0, (s2.z() * s2.z());
    cov2 = R2.transpose() * cov2 * R2;
    float vol2 = cov2.determinant();

    pos = vol1 * p1 + vol2 * p2;
    scale = {
        std::max(s1.x(), s2.x()),
        std::max(s1.y(), s2.y()),
        std::max(s1.z(), s2.z())
    };
    float total_volume = vol1 + vol2;

    rgba = {
        static_cast<int>((vol1 * c1.x() + vol2 * c2.x()) / total_volume),
        static_cast<int>((vol1 * c1.y() + vol2 * c2.y()) / total_volume),
        static_cast<int>((vol1 * c1.z() + vol2 * c2.z()) / total_volume),
        static_cast<int>((vol1 * c1.w() + vol2 * c2.w()) / total_volume)
    };
    rot = r1.slerp(vol1 / (total_volume), r2);
    pos /= total_volume;

    cloud.positions.push_back(pos);
    cloud.rgba.push_back(rgba);
    cloud.rotations.push_back(rot);
    cloud.scales.push_back(scale);

    return cloud.positions.size() - 1;
}

// size_t findOptimalSplit(
//     const GaussianCloud &cloud,
//     std::vector<size_t> &indices,
//     size_t l,
//     size_t r,
//     int axis
// ) {
//     size_t n = r - l;
//     std::vector<AABB> leftBounds(n);
//     std::vector<AABB> rightBounds(n);

//     leftBounds[0] = GaussianAABB(cloud, indices[l]);
//     for (size_t i = 1; i < n; ++i) {
//         AABB currentAABB = GaussianAABB(cloud, indices[l + i]);
//         leftBounds[i] = leftBounds[i - 1].merge(currentAABB);
//     }

//     rightBounds[n - 1] = GaussianAABB(cloud, indices[r - 1]);
//     for (int i = static_cast<int>(n) - 2; i >= 0; --i) {
//         AABB currentAABB = GaussianAABB(cloud, indices[l + i]);
//         rightBounds[i] = rightBounds[i + 1].merge(currentAABB);
//     }

//     float bestCost = std::numeric_limits<float>::max();
//     size_t bestSplit = l;
//     for (size_t i = 0; i < n - 1; ++i) {
//         float leftCost = leftBounds[i].surfaceArea() * (i + 1);
//         float rightCost = rightBounds[i + 1].surfaceArea() * (n - i - 1);
//         float cost = leftCost + rightCost;
//         if (cost < bestCost) {
//             bestCost = cost;
//             bestSplit = l + i + 1;
//         }
//     }

//     return bestSplit;
// }

// Gpt generated SAH cus im lazy as fuck
constexpr int NUM_BINS = 10;
size_t findOptimalSplit(
    const GaussianCloud &cloud,
    std::vector<size_t> &indices,
    size_t l,
    size_t r,
    int axis
) {
    size_t n = r - l;
    if (n <= 1) return l;

    // Step 1: Compute global bounds
    AABB globalBounds = GaussianAABB(cloud, indices[l]);
    for (size_t i = l + 1; i < r; ++i) {
        auto cur = GaussianAABB(cloud, indices[i]);
        globalBounds = globalBounds.merge(cur);
    }

    // Step 2: Sort based on axis
    std::nth_element(indices.begin() + l, indices.begin() + l + n / 2, indices.begin() + r,
        [&](size_t a, size_t b) {
            float va = (axis == 0) ? cloud.positions[a][0] : (axis == 1) ? cloud.positions[a][1] : cloud.positions[a][2];
            float vb = (axis == 0) ? cloud.positions[b][0] : (axis == 1) ? cloud.positions[b][1] : cloud.positions[b][2];
            return va < vb;
        });

    // Step 3: Bin-based SAH calculation
    int binSize = n / NUM_BINS;
    std::vector<AABB> leftBounds(NUM_BINS), rightBounds(NUM_BINS);

    // Left prefix bounds
    leftBounds[0] = GaussianAABB(cloud, indices[l]);
    for (int i = 1; i < NUM_BINS; ++i) {
        size_t idx = l + i * binSize;
        auto cur = GaussianAABB(cloud, indices[idx]);
        leftBounds[i] = leftBounds[i - 1].merge(cur);
    }

    // Right suffix bounds
    rightBounds[NUM_BINS - 1] = GaussianAABB(cloud, indices[r - 1]);
    for (int i = NUM_BINS - 2; i >= 0; --i) {
        size_t idx = l + (i + 1) * binSize;
        auto cur = GaussianAABB(cloud, indices[idx]);
        rightBounds[i] = rightBounds[i + 1].merge(cur);
    }

    // Step 4: Compute SAH for each bin
    float bestCost = std::numeric_limits<float>::infinity();
    size_t bestSplit = l;

    for (int i = 0; i < NUM_BINS - 1; ++i) {
        float leftSA = leftBounds[i].surfaceArea();
        float rightSA = rightBounds[i + 1].surfaceArea();

        int leftCount = (i + 1) * binSize;
        int rightCount = n - leftCount;

        float cost = leftCount * leftSA + rightCount * rightSA;
        if (cost < bestCost) {
            bestCost = cost;
            bestSplit = l + (i + 1) * binSize;
        }
    }

    // Step 5: Fallback to median split if SAH is unstable
    if (bestSplit == l) {
        bestSplit = l + n / 2;
    }

    return bestSplit;
}

class BVHNode {
public:
    BVHNode(
        GaussianCloud &cloud,
        std::vector<size_t> &indices,
        size_t l,
        size_t r,
        int axis
    ) {
        // No surface area heuristic because im a lazy fuck
        axis = axis % 3;
        if (r - l == 1) {
            index = indices[l];
            bbox = GaussianAABB(cloud, indices[l]);
            return;
        }

        sort(indices.begin() + l, indices.begin() + r, [&](const size_t &l, const size_t &r) {
            return cloud.positions[l][axis] < cloud.positions[r][axis];
        });
        size_t mid = (l + r) / 2;
        // surface area heuristic tree creationt is too fuckin slow for bigger scenes
        // size_t mid = findOptimalSplit(cloud, indices, l, r, axis);
        if (mid - l > 0) {
            left.reset(new BVHNode(cloud, indices, l, mid, axis + 1));
        }
        if (r - mid > 0) {
            right.reset(new BVHNode(cloud, indices, mid, r, axis + 1));
        }

        if (left && right) {
            bbox = left->bbox.merge(right->bbox);
            index = MergeGaussian(cloud, left->index, right->index);
        }
    }

    void getIndices(std::vector<size_t> &indices, const Eigen::Matrix4f viewProj) {
        auto vertices = bbox.vertices4();
        float minx, miny, minz;
        minx = miny = minz = INFINITY;
        float maxx, maxy, maxz;
        maxx = maxy = maxz = -INFINITY;

        for (int i = 0; i < vertices.size(); i++) {
            Eigen::Vector4f nc = viewProj * vertices[i];
            nc /= nc.w();
            minx = std::min(minx, nc.x());
            maxx = std::max(maxx, nc.x());
            miny = std::min(miny, nc.y());
            maxy = std::max(maxy, nc.y());
            minz = std::min(minz, nc.z());
            maxz = std::max(maxz, nc.z());
        }

        if (minz < 1 && maxz > -1 && minx < 1 && maxx > -1 && miny < 1 && maxy > -1) {
            if (!left || !right || maxx - minx < 0.01 || maxy - miny < 0.01) {
                indices.push_back(index);
            } else {
                left->getIndices(indices, viewProj);
                right->getIndices(indices, viewProj);
            }
        } else if (minz < 1.5 && maxz > -1.5 && minx < 1.5 && maxx > -1.5 && miny < 1.5 && maxy > -1.5) {
            if (!left || !right || maxx - minx < 0.25 || maxy - miny < 0.25) {
                indices.push_back(index);
            } else {
                left->getIndices(indices, viewProj);
                right->getIndices(indices, viewProj);
            }
        } else {
            if (!left || !right || maxx - minx < 0.7 || maxy - miny < 0.7) {
                indices.push_back(index);
            } else {
                left->getIndices(indices, viewProj);
                right->getIndices(indices, viewProj);
            }
        }
    }
    size_t index;
    AABB bbox;
    std::unique_ptr<BVHNode> left, right;
};

class BoundingVolumeHierarchy {
public:
    BoundingVolumeHierarchy(void *buffer, size_t vertexCount) {
        cloud = GaussianCloud(buffer, vertexCount);
        std::vector<size_t> indices(vertexCount);
        std::iota(indices.begin(), indices.end(), 0);
        root.reset(new BVHNode(cloud, indices, 0, vertexCount, 0));
        this->vertexCount = cloud.positions.size(); // TODO: cloud.vertexcount is not updated
    }
    BoundingVolumeHierarchy(emscripten::val buftmp, size_t vertexCount) {
        std::vector<uint8_t> tmp;
        copyToVector(buftmp, tmp);
        void *buffer = tmp.data();
        cloud = GaussianCloud(buffer, vertexCount);
        std::vector<size_t> indices(vertexCount);
        std::iota(indices.begin(), indices.end(), 0);
        root.reset(new BVHNode(cloud, indices, 0, vertexCount, 0));
        cloud.vertexCount = cloud.positions.size();
        this->vertexCount = cloud.positions.size(); // TODO: cloud.vertexcount is not updated
        cloud.positions.shrink_to_fit();
        cloud.scales.shrink_to_fit();
        cloud.rotations.shrink_to_fit();
        cloud.rgba.shrink_to_fit();
    }

    std::vector<size_t> getIndices(Eigen::Matrix4f viewProj) {
        if (!root) return {};
        std::vector<size_t> out;
        root->getIndices(out, viewProj);
        return out;
    }

    std::vector<size_t> getIndicesJS(emscripten::val viewProjJS) {
        std::vector<float> tmp = emscripten::convertJSArrayToNumberVector<float>(viewProjJS);
        Eigen::Matrix4f viewProj = Eigen::Map<Eigen::Matrix4f>(tmp.data());
        if (!root) return {};
        std::vector<size_t> out;
        root->getIndices(out, viewProj);
        return out;
    }

    std::vector<uint8_t> toBuffer() {
        return cloud.toBuffer();
    }

    size_t size() {
        return vertexCount;
    }
private:
    GaussianCloud cloud;
    std::unique_ptr<BVHNode> root;
    size_t vertexCount;
};

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_BINDINGS(wasm_gsplat) {
    emscripten::class_<GaussianCloud>("GaussianCloud")
        .constructor<emscripten::val, size_t>()
        .function("toBuffer", &GaussianCloud::toBuffer);
    emscripten::class_<BoundingVolumeHierarchy>("BoundingVolumeHierarchy")
        .constructor<emscripten::val, size_t>(emscripten::allow_raw_pointers())
        .function("getIndicesJS", &BoundingVolumeHierarchy::getIndicesJS)
        .function("size", &BoundingVolumeHierarchy::size)
        .function("toBuffer", &BoundingVolumeHierarchy::toBuffer);
    emscripten::register_vector<size_t>("VectorSizeT");
    emscripten::register_vector<uint8_t>("VectorUint8");
    // // emscripten::register_vector<Vertex>("VectorVertex");
    // emscripten::register_vector<uint32_t>("VectorUint32vector");
    // emscripten::register_vector<float>("VectorFloat32");
    // emscripten::function("runSort", &runSort);
}
#endif
