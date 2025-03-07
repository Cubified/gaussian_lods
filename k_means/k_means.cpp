#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <random>

#include "tqdm.hpp"

#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"

using namespace std;

struct float3 { float x, y, z; };
struct float4 { float x, y, z, w; };

struct Gaussian {
    float3 pos;
    float3 scale;
    float4 rot;
    float opacity;
    float3 sh;
    int32_t lod;
    bool invalid = false;

    Gaussian(const float3 &pos,
             const float3 &scale,
             const float4 &rot,
             const float opacity,
             const float3 &sh,
             const int lod)
        : pos(pos), scale(scale), rot(rot), opacity(opacity), sh(sh), lod(lod) {}
};

vector<Gaussian> load_ply(const string& path) {
    ifstream ss(path, ios::binary);
    tinyply::PlyFile file;
    file.parse_header(ss);

    shared_ptr<tinyply::PlyData> ply_verts = file.request_properties_from_element("vertex", {"x", "y", "z"});
    shared_ptr<tinyply::PlyData> ply_opacities = file.request_properties_from_element("vertex", {"opacity"});
    shared_ptr<tinyply::PlyData> ply_scales = file.request_properties_from_element("vertex", {"scale_0", "scale_1", "scale_2"});
    shared_ptr<tinyply::PlyData> ply_shs = file.request_properties_from_element("vertex", {"f_dc_0", "f_dc_1", "f_dc_2"});
    shared_ptr<tinyply::PlyData> ply_rots = file.request_properties_from_element("vertex", {"rot_0", "rot_1", "rot_2", "rot_3"});

    file.read(ss);

#define GET_AS_VEC(in, out, type) do { \
    const size_t n_bytes = in->buffer.size_bytes(); \
    out.reserve(in->count); \
    memcpy(out.data(), in->buffer.get(), n_bytes); \
} while (0)

    vector<float> opacities;
    vector<float3> verts, scales, shs;
    vector<float4> rots;

    GET_AS_VEC(ply_verts, verts, float3);
    GET_AS_VEC(ply_scales, scales, float3);
    GET_AS_VEC(ply_shs, shs, float3);
    GET_AS_VEC(ply_opacities, opacities, float);
    GET_AS_VEC(ply_rots, rots, float4);

    vector<Gaussian> gaussians;
    for (size_t i = 0; i < ply_verts->count; ++i) {
        gaussians.emplace_back(verts[i], scales[i], rots[i], opacities[i], shs[i], 0);
    }

    return gaussians;
}

inline float squared_euclidean_distance(const vector<float> &a, const vector<float> &b) {
    float dist = 0.0f;
#pragma omp unroll
    for (size_t i = 0; i < 6; ++i) {
    // for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

vector<int> cluster_gaussians(const vector<vector<float>> &data, int k, int max_iters = 100) {
    size_t n = data.size();
    size_t dim = data[0].size();
    vector<vector<float>> centroids(k, vector<float>(dim));
    vector<int> assignments(n, -1);

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<size_t> dist(0, n - 1);
    for (int i = 0; i < k; ++i) {
        centroids[i] = data[dist(gen)];
    }

    for (int iter = 0; iter < max_iters; ++iter) {
        // cout << "Iteration " << iter << ":" << endl;
        bool changed = false;

        for (size_t i : tq::trange(n)) {
        // for (size_t i = 0; i < n; ++i) {
            int best_cluster = 0;
            float best_distance = numeric_limits<float>::max();
#pragma omp parallel for
            for (int j = 0; j < k; ++j) {
                float dist = squared_euclidean_distance(data[i], centroids[j]);
                if (dist < best_distance) {
                    best_distance = dist;
                    best_cluster = j;
                }
            }
            if (assignments[i] != best_cluster) {
                assignments[i] = best_cluster;
                changed = true;
            }
        }

        cout << endl;

        if (!changed) break;

        vector<vector<float>> new_centroids(k, vector<float>(dim, 0.0f));
        vector<int> counts(k, 0);
#pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            int cluster = assignments[i];
            for (size_t d = 0; d < dim; ++d) {
                new_centroids[cluster][d] += data[i][d];
            }
            counts[cluster]++;
        }
#pragma omp parallel for
        for (int j = 0; j < k; ++j) {
            if (counts[j] > 0) {
                for (size_t d = 0; d < dim; ++d) {
                    centroids[j][d] = new_centroids[j][d] / counts[j];
                }
            }
        }
    }
    
    return assignments;
}

Gaussian merge_gaussians(const vector<Gaussian> &gaussians, int32_t lod = 1) {
    float total_opacity = 0;
    float3 pos = {0, 0, 0};
    float3 scale = {0, 0, 0};
    float4 rot = {0, 0, 0, 0};
    float3 sh = {0, 0, 0};

    for (const auto &g : gaussians) {
        total_opacity += g.opacity;

        pos.x += g.opacity * g.pos.x;
        pos.y += g.opacity * g.pos.y;
        pos.z += g.opacity * g.pos.z;

        scale.x += g.opacity * g.scale.x;
        scale.y += g.opacity * g.scale.y;
        scale.z += g.opacity * g.scale.z;

        rot.x += g.opacity * g.rot.x;
        rot.y += g.opacity * g.rot.y;
        rot.z += g.opacity * g.rot.z;
        rot.w += g.opacity * g.rot.z;

        sh.x += g.opacity * g.sh.x;
        sh.y += g.opacity * g.sh.y;
        sh.z += g.opacity * g.sh.z;
    }
    if (total_opacity == 0) {
        Gaussian g = Gaussian(pos, scale, rot, total_opacity, sh, lod);
        g.invalid = true;
        return g;
    }

    pos.x /= total_opacity;
    pos.y /= total_opacity;
    pos.z /= total_opacity;

    scale.x /= total_opacity;
    scale.y /= total_opacity;
    scale.z /= total_opacity;

    rot.x /= total_opacity;
    rot.y /= total_opacity;
    rot.z /= total_opacity;
    rot.w /= total_opacity;

    float rot_len = sqrt((rot.x * rot.x) + (rot.y * rot.y) + (rot.z * rot.z) + (rot.w * rot.w));
    rot.x /= rot_len;
    rot.y /= rot_len;
    rot.z /= rot_len;
    rot.w /= rot_len;

    sh.x /= total_opacity;
    sh.y /= total_opacity;
    sh.z /= total_opacity;

    return Gaussian(pos, scale, rot, total_opacity, sh, lod);
}

void save_ply(const string &path, const vector<Gaussian> &gaussians) {
    filebuf fb;
    fb.open(path, ios::out | ios::binary);
    ostream outstream(&fb);

    vector<float> data_f;
    vector<int32_t> data_i;

    for (auto g : gaussians) {
        data_f.push_back(g.pos.x);
        data_f.push_back(g.pos.y);
        data_f.push_back(g.pos.z);
        data_f.push_back(g.opacity);
        data_f.push_back(g.scale.x);
        data_f.push_back(g.scale.y);
        data_f.push_back(g.scale.z);
        data_f.push_back(g.sh.x);
        data_f.push_back(g.sh.y);
        data_f.push_back(g.sh.z);
        data_f.push_back(g.rot.x);
        data_f.push_back(g.rot.y);
        data_f.push_back(g.rot.z);
        data_f.push_back(g.rot.w);

        data_i.push_back(g.lod);
    }

    tinyply::PlyFile file;
    file.add_properties_to_element("vertex", {"x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "f_dc_0", "f_dc_1", "f_dc_2", "rot_0", "rot_1", "rot_2", "rot_3"},
        tinyply::Type::FLOAT32, data_f.size() / 14, reinterpret_cast<uint8_t*>(data_f.data()), tinyply::Type::INVALID, 0);
    file.add_properties_to_element("vertex", {"lod"},
        tinyply::Type::INT32, data_i.size(), reinterpret_cast<uint8_t*>(data_i.data()), tinyply::Type::INVALID, 0);

    file.write(outstream, true);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        cout << "Usage: k_means [in.ply] [out.ply]" << endl;
        return 0;
    }

    string input_path = string(argv[1]);
    string output_path = string(argv[2]);

    cout << "Loading... ";
    auto gaussians = load_ply(input_path);

    cout << gaussians.size() << " Gaussians" << endl;
    
    vector<Gaussian> output_gaussians;
    for (auto g : gaussians) {
        output_gaussians.push_back(g);
    }

    int32_t current_lod = 1;
    int n_labels = gaussians.size();
    for (;;) {
        n_labels /= 2;
        if (n_labels <= 1000) break;

        cout << "Level " << current_lod << ": " << n_labels << " Gaussians" << endl;

        vector<vector<float>> features(gaussians.size());
#pragma omp parallel for
        for (int i = 0; i < gaussians.size(); i++) {
            Gaussian &g = gaussians[i];
            vector<float> tmp;

            tmp.push_back(g.pos.x);
            tmp.push_back(g.pos.y);
            tmp.push_back(g.pos.z);

            tmp.push_back(g.sh.x);
            tmp.push_back(g.sh.y);
            tmp.push_back(g.sh.z);

            /*
            tmp.push_back(g.scale.x);
            tmp.push_back(g.scale.y);
            tmp.push_back(g.scale.z);

            tmp.push_back(g.opacity);
            */

            features[i] = tmp;
        }

        int n_iters = 2;
        cout << "  Clustering (" << n_iters << " iterations)..." << endl;
        vector<int> labels = cluster_gaussians(features, n_labels, n_iters);

        vector<vector<Gaussian>> clustered_gaussians(n_labels);
        for (int i = 0; i < n_labels; i++) {
            clustered_gaussians[i] = vector<Gaussian>();
        }
        for (int i = 0; i < gaussians.size(); i++) {
            clustered_gaussians[labels[i]].push_back(gaussians[i]);
        }

        cout << "  Merging..." << endl;
        for (int i = 0; i < n_labels; i++) {
            Gaussian g = merge_gaussians(clustered_gaussians[i], current_lod);
            if (!g.invalid) output_gaussians.push_back(g);
        }

        current_lod++;
    }

    cout << "Saving (" << output_gaussians.size() << " Gaussians)..." << endl;
    save_ply(output_path, output_gaussians);

    return 0;
}
