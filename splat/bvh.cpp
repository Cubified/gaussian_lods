#include <array>
#include <memory>
#include <numeric>
#include <vector>

#include <Eigen/Eigen>

// #include <emscripten.h>
// #include <emscripten/bind.h>

constexpr size_t rowLength = 3 * 4 + 3 * 4 + 4 + 4;

// template<typename T>
// void copyToVector(const emscripten::val &typedArray, std::vector<T> &vec) {
//     unsigned int length = typedArray["length"].as<unsigned int>();
//     emscripten::val heap = emscripten::val::module_property("HEAPU8");
//     emscripten::val memory = heap["buffer"];
//     vec.reserve(length);

//     emscripten::val memoryView = typedArray["constructor"].new_(memory, reinterpret_cast<uintptr_t>(vec.data()), length);

//     memoryView.call<void>("set", typedArray);
// }

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
private:
    std::array<float, 6> bounds;
};

class GaussianCloud {
public:
    GaussianCloud() : vertexCount(0) {}
    GaussianCloud(void *buffer, size_t vertexCount) {
        positions.reserve(vertexCount);
        scales.reserve(vertexCount);
        rotations.reserve(vertexCount);
        rgba.reserve(vertexCount);

        for (size_t i = 0; i < vertexCount; i++) {
            parseElement(buffer, i);
        }
        this->vertexCount = vertexCount;
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
        rotations.push_back({
            (u_buffer[32 * i + 28 + 3] - 128) / static_cast<float>(128),
            (u_buffer[32 * i + 28 + 0] - 128) / static_cast<float>(128),
            (u_buffer[32 * i + 28 + 1] - 128) / static_cast<float>(128),
            (u_buffer[32 * i + 28 + 2] - 128) / static_cast<float>(128)
        });
    }

    std::vector<uint8_t> toBuffer() {
        std::vector<uint8_t> buffervec(rowLength * vertexCount);
        void *buffer = buffervec.data();
        for (size_t i = 0; i < vertexCount; i++) {
            float *posP = reinterpret_cast<float *>(buffer + i * rowLength);
            float *scaP = reinterpret_cast<float *>(buffer + i * rowLength + 4 * 3);
            uint8_t *rgbaP = reinterpret_cast<uint8_t *>(buffer + i * rowLength + 4 * 3 + 4 * 3);
            uint8_t *rotP = reinterpret_cast<uint8_t *>(buffer + i * rowLength + 4 * 3 + 4 * 3 + 4);
            for (int j = 0; j < 3; j++) posP[j] = positions[i][j];
            for (int j = 0; j < 3; j++) scaP[j] = scales[i][j];
            for (int j = 0; j < 4; j++) rgbaP[j] = rgba[i][j];
            // Store with w term last
            rotP[0] = static_cast<uint8_t>((128 * rotations[i].x()) + 128);
            rotP[1] = static_cast<uint8_t>((128 * rotations[i].y()) + 128);
            rotP[2] = static_cast<uint8_t>((128 * rotations[i].z()) + 128);
            rotP[3] = static_cast<uint8_t>((128 * rotations[i].w()) + 128);
        }
        return buffervec;
    }
    std::vector<Eigen::Vector3f> positions;
    std::vector<Eigen::Vector3f> scales;
    std::vector<Eigen::Quaternionf> rotations;
    std::vector<Eigen::Vector4f> rgba;
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
    Eigen::Vector3f pos(0);
    Eigen::Vector3f scale(0);
    Eigen::Quaternionf rot(0);
    Eigen::Vector4f rgba(0);

    Eigen::Vector3f p1 = cloud.positions[i];
    Eigen::Vector3f s1 = cloud.scales[i];
    Eigen::Vector4f c1 = cloud.rgba[i];
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
    Eigen::Vector4f c2 = cloud.rgba[j];
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
    rgba = vol1 * c1 + vol2 * c2;
    rot = r1.slerp(vol1 / (vol1 + vol2), r2);

    float total_volume = vol1 + vol2;
    pos /= total_volume;
    rgba /= total_volume;

    cloud.positions.push_back(pos);
    cloud.rgba.push_back(rgba);
    cloud.rotations.push_back(rot);
    cloud.scales.push_back(scale);

    return cloud.positions.size() - 1;
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
        axis = axis % 3;
        if (indices.size() == 1) {
            index = indices[0];
            bbox = GaussianAABB(cloud, indices[0]);
            return;
        }

        sort(indices.begin() + l, indices.begin() + r, [&](const int &l, const int &r) {
            return cloud.positions[l][axis] < cloud.positions[r][axis];
        });

        size_t mid = (l + r) / 2;
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
            nc.x() /= nc.w();
            nc.y() /= nc.w();
            nc.z() /= nc.w();
            nc.w() = 1;
            minx = std::min(minx, nc.x());
            maxx = std::max(maxx, nc.x());
            miny = std::min(miny, nc.y());
            maxy = std::max(maxy, nc.y());
            minz = std::min(minz, nc.z());
            maxz = std::max(maxz, nc.z());
        }

        if (maxz < -1 || minz > 1 || maxx < -1 || minx > 1 || maxy < -1 || miny  > 1) {
            return;
        }

        if (!left || !right || maxx - minx < 0.1 || maxy - miny < 0.1) {
            indices.push_back(index);
        } else {
            left->getIndices(indices, viewProj);
            right->getIndices(indices, viewProj);
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
        this->vertexCount = vertexCount;
        std::vector<size_t> indices(vertexCount);
        std::iota(indices.begin(), indices.end(), 0);
        root.reset(new BVHNode(cloud, indices, 0, vertexCount, 0));
    }

    std::vector<size_t> getIndices(Eigen::Matrix4f viewProj) {
        if (!root) return {};
        std::vector<size_t> out;
        root->getIndices(out, viewProj);
        return out;
    }
private:
    GaussianCloud cloud;
    std::unique_ptr<BVHNode> root;
    size_t vertexCount;
};


