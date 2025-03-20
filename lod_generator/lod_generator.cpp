#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <random>
#include <set>

#include <Eigen/Eigen>

#include "tqdm.hpp"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define N_GAUSSIAN_FEATURES 6

#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"

using namespace std;

#define EPS (1.0e-8)

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

    Gaussian() { invalid = true; }
    Gaussian(const float3 &pos,
             const float3 &scale,
             const float4 &rot,
             const float opacity,
             const float3 &sh,
             const int lod)
        : pos(pos), scale(scale), rot(rot), opacity(opacity), sh(sh), lod(lod) {}
};

vector<Gaussian> load_ply(const string& path, int lod = 0) {
    ifstream ss(path, ios::binary);
    tinyply::PlyFile file;
    file.parse_header(ss);

    shared_ptr<tinyply::PlyData> ply_verts = file.request_properties_from_element("vertex", {"x", "y", "z"});
    shared_ptr<tinyply::PlyData> ply_opacities = file.request_properties_from_element("vertex", {"opacity"});
    shared_ptr<tinyply::PlyData> ply_scales = file.request_properties_from_element("vertex", {"scale_0", "scale_1", "scale_2"});
    shared_ptr<tinyply::PlyData> ply_shs = file.request_properties_from_element("vertex", {"f_dc_0", "f_dc_1", "f_dc_2"});
    shared_ptr<tinyply::PlyData> ply_rots = file.request_properties_from_element("vertex", {"rot_0", "rot_1", "rot_2", "rot_3"});

    file.read(ss);

#define GET_AS_VEC(in, out) do { \
    const size_t n_bytes = in->buffer.size_bytes(); \
    out.reserve(in->count); \
    memcpy(out.data(), in->buffer.get(), n_bytes); \
} while (0)

    vector<float> opacities;
    vector<float3> verts, scales, shs;
    vector<float4> rots;

    GET_AS_VEC(ply_verts,     verts);
    GET_AS_VEC(ply_scales,    scales);
    GET_AS_VEC(ply_shs,       shs);
    GET_AS_VEC(ply_opacities, opacities);
    GET_AS_VEC(ply_rots,      rots);

    vector<Gaussian> gaussians;
    for (int i = 0; i < ply_verts->count; ++i) {
        opacities[i] = 1.0 / (1.0 + exp(-opacities[i]));

        scales[i].x = exp(scales[i].x);
        scales[i].y = exp(scales[i].y);
        scales[i].z = exp(scales[i].z);

        gaussians.emplace_back(verts[i], scales[i], rots[i], opacities[i], shs[i], lod);
    }

    return gaussians;
}

inline float squared_euclidean_distance(const vector<float> &a, const vector<float> &b) {
    float dist = 0.0f;
#pragma omp unroll
    for (size_t i = 0; i < N_GAUSSIAN_FEATURES; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

vector<int> k_means(const vector<vector<float>> &data, int k, int max_iters = 100) {
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
        bool changed = false;

        for (size_t i : tq::trange(n)) {
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

vector<int> k_medoids(const vector<vector<float>>& data, int k, int max_iters = 100) {
    size_t n = data.size();
    vector<int> medoids(k);
    vector<int> assignments(n, -1);
    
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<size_t> dist(0, n - 1);
    for (int i = 0; i < k; ++i) {
        medoids[i] = dist(gen);
    }
    
    for (int iter = 0; iter < max_iters; ++iter) {
        bool changed = false;
        
        for (size_t i : tq::trange(n)) {
            int best_cluster = 0;
            float best_distance = numeric_limits<float>::max();
#pragma omp parallel for
            for (int j = 0; j < k; ++j) {
                float dist = squared_euclidean_distance(data[i], data[medoids[j]]);
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
        
#pragma omp parallel for
        for (int j = 0; j < k; ++j) {
            int best_medoid = medoids[j];
            float best_cost = 0.0f;
            for (size_t i = 0; i < n; ++i) {
                if (assignments[i] == j) {
                    best_cost += squared_euclidean_distance(data[i], data[best_medoid]);
                }
            }
            for (size_t i = 0; i < n; ++i) {
                if (assignments[i] == j) {
                    float swap_cost = 0.0f;
                    for (size_t m = 0; m < n; ++m) {
                        if (assignments[m] == j) {
                            swap_cost += squared_euclidean_distance(data[m], data[i]);
                        }
                    }
                    if (swap_cost < best_cost) {
                        best_medoid = i;
                        best_cost = swap_cost;
                    }
                }
            }
            medoids[j] = best_medoid;
        }
    }
    
    return assignments;
}

vector<int> cluster_gaussians(string method, const vector<vector<float>> &data, int k, int max_iters = 100) {
    if (method == "kmeans") {
        return k_means(data, k, max_iters);
    } else if (method == "kmedoids") {
        return k_medoids(data, k, max_iters);
    }

    cerr << "Error: Unrecognized clustering algorithm " << method << endl;
    return vector<int>();
}

Gaussian mean_merge(const vector<Gaussian> &gaussians, int32_t lod = 1) {
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

        sh.x += g.opacity * g.sh.x;
        sh.y += g.opacity * g.sh.y;
        sh.z += g.opacity * g.sh.z;

        scale.x = MAX(scale.x, g.scale.x);
        scale.y = MAX(scale.y, g.scale.y);
        scale.z = MAX(scale.z, g.scale.z);

        rot.x += g.opacity * g.rot.x;
        rot.y += g.opacity * g.rot.y;
        rot.z += g.opacity * g.rot.z;
        rot.w += g.opacity * g.rot.w;
    }
    if (fabs(total_opacity) < EPS) {
        return Gaussian();
    }

    pos.x /= total_opacity;
    pos.y /= total_opacity;
    pos.z /= total_opacity;

    sh.x /= total_opacity;
    sh.y /= total_opacity;
    sh.z /= total_opacity;

    float rot_len = sqrt((rot.x * rot.x) + (rot.y * rot.y) + (rot.z * rot.z) + (rot.w * rot.w));
    rot.x /= rot_len;
    rot.y /= rot_len;
    rot.z /= rot_len;
    rot.w /= rot_len;

    return Gaussian(pos, scale, rot, total_opacity, sh, lod);
}

Gaussian mean_volume_merge(const vector<Gaussian> &gaussians, int32_t lod = 1) {
    float total_opacity = 0;
    float total_volume = 0;
    float3 pos = {0, 0, 0};
    float3 scale = {0, 0, 0};
    float4 rot = {0, 0, 0, 0};
    float3 sh = {0, 0, 0};

    for (const auto &g : gaussians) {
        Eigen::Quaterniond q(g.rot.x, g.rot.y, g.rot.z, g.rot.w);
        Eigen::Matrix3d R = q.toRotationMatrix();

        Eigen::Matrix3d cov3D;
        cov3D <<
            (g.scale.x * g.scale.x), 0, 0,
            0, (g.scale.y * g.scale.y), 0,
            0, 0, (g.scale.z * g.scale.z);
        cov3D = R.transpose() * cov3D * R;

        float volume = cov3D.determinant();
        total_volume += volume;
        total_opacity += g.opacity;

        pos.x += volume * g.pos.x;
        pos.y += volume * g.pos.y;
        pos.z += volume * g.pos.z;

        sh.x += volume * g.sh.x;
        sh.y += volume * g.sh.y;
        sh.z += volume * g.sh.z;

        scale.x = MAX(scale.x, g.scale.x);
        scale.y = MAX(scale.y, g.scale.y);
        scale.z = MAX(scale.z, g.scale.z);

        rot.x += volume * g.rot.x;
        rot.y += volume * g.rot.y;
        rot.z += volume * g.rot.z;
        rot.w += volume * g.rot.w;
    }
    if (fabs(total_opacity) < EPS) {
        return Gaussian();
    }

    pos.x /= total_volume;
    pos.y /= total_volume;
    pos.z /= total_volume;

    sh.x /= total_volume;
    sh.y /= total_volume;
    sh.z /= total_volume;

    float rot_len = sqrt((rot.x * rot.x) + (rot.y * rot.y) + (rot.z * rot.z) + (rot.w * rot.w));
    rot.x /= rot_len;
    rot.y /= rot_len;
    rot.z /= rot_len;
    rot.w /= rot_len;

    return Gaussian(pos, scale, rot, total_opacity, sh, lod);
}

Gaussian max_merge(const vector<Gaussian> &gaussians, int32_t lod = 1) {
    float largest_volume = 0;
    Gaussian largest_gaussian;
    for (const auto &g : gaussians) {
        if (fabs(g.opacity) < EPS) continue;

        Eigen::Quaterniond q(g.rot.x, g.rot.y, g.rot.z, g.rot.w);
        Eigen::Matrix3d R = q.toRotationMatrix();

        Eigen::Matrix3d cov3D;
        cov3D <<
            (g.scale.x * g.scale.x), 0, 0,
            0, (g.scale.y * g.scale.y), 0,
            0, 0, (g.scale.z * g.scale.z);
        cov3D = R.transpose() * cov3D * R;

        float volume = cov3D.determinant();
        if (volume > largest_volume) {
            largest_volume = volume;
            largest_gaussian = g;
        }
    }

    return Gaussian(
        largest_gaussian.pos,
        largest_gaussian.scale,
        largest_gaussian.rot,
        largest_gaussian.opacity,
        largest_gaussian.sh,
        lod
    );
}

Gaussian merge_gaussians(string method, const vector<Gaussian> &gaussians, int32_t lod = 1) {
    if (method == "mean") {
        return mean_merge(gaussians, lod);
    } else if (method == "mean_volume") {
        return mean_volume_merge(gaussians, lod);
    } else if (method == "max") {
        return max_merge(gaussians, lod);
    }

    cerr << "Error: Unrecognized merging algorithm " << method << endl;
    return Gaussian();
}

void save_ply(const string &path, const vector<Gaussian> &gaussians) {
    filebuf fb;
    fb.open(path, ios::out | ios::binary);
    ostream outstream(&fb);

    vector<float> data_f;
    vector<int32_t> data_i;

    for (const auto &g : gaussians) {
        float rot_norm = sqrt((g.rot.x * g.rot.x) + (g.rot.y * g.rot.y) + (g.rot.z * g.rot.z) + (g.rot.w * g.rot.w));

        float opacity = 0;
        if ((1.0 / (g.opacity + EPS)) > 1.0) {
            opacity = -log((1.0 / (g.opacity + EPS)) - 1.0);
        }

        data_f.push_back(g.pos.x);
        data_f.push_back(g.pos.y);
        data_f.push_back(g.pos.z);
        data_f.push_back(opacity);
        data_f.push_back(log(MAX(EPS, g.scale.x)));
        data_f.push_back(log(MAX(EPS, g.scale.y)));
        data_f.push_back(log(MAX(EPS, g.scale.z)));
        data_f.push_back(g.sh.x);
        data_f.push_back(g.sh.y);
        data_f.push_back(g.sh.z);
        data_f.push_back(g.rot.x / rot_norm);
        data_f.push_back(g.rot.y / rot_norm);
        data_f.push_back(g.rot.z / rot_norm);
        data_f.push_back(g.rot.w / rot_norm);

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
        cout << "Usage: lod_generator <in.ply|merge> <out.ply> [kmeans|kmedoids] [mean|mean_volume|max]" << endl;
        return 0;
    }

    string input_path = string(argv[1]);
    string output_path = string(argv[2]);
    string cluster_method = string(argc == 3 ? "kmeans" : argv[3]);
    string merge_method = string(argc == 4 ? "mean" : argv[4]);

    if (input_path == "merge") {
        vector<Gaussian> gaussians;
        for (int i = 2; i < argc - 1; i++) {
            cout << "Loading \"" << argv[i] << "\"... ";
            auto tmp = load_ply(argv[i], i - 2);
            cout << tmp.size() << " Gaussians" << endl;
            gaussians.insert(gaussians.end(), tmp.begin(), tmp.end());
        }

        string output_path = string(argv[argc - 1]);
        cout << "Saving \"" << output_path << "\"... ";
        save_ply(output_path, gaussians);
        cout << gaussians.size() << " Gaussians" << endl;
        return 0;
    }

    cout << "<== Generating LODs ==>" << endl;
    cout << "Clustering method: " << cluster_method << endl;
    cout << "Merging method: " << merge_method << endl;
    cout << endl;

    cout << "Loading \"" << input_path << "\"... ";
    auto gaussians = load_ply(input_path);
    cout << gaussians.size() << " Gaussians" << endl;

    /*
    vector<Gaussian> tmp;
    for (int i = 0; i < gaussians.size(); i++) {
        if (gaussians[i].pos.x > 0.0 && gaussians[i].pos.z > 0.0) {
            tmp.push_back(gaussians[i]);
        }
    }
    gaussians = tmp;

    save_ply("tmp.ply", gaussians);
    return 0;
    */
    
    vector<Gaussian> output_gaussians;
    for (auto g : gaussians) {
        output_gaussians.push_back(g);
    }

    int32_t current_lod = 1;
    int n_labels = gaussians.size();
    for (;;) {
        n_labels /= 2;
        if (n_labels <= 10000) break;

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

            features[i] = tmp;
        }

        int n_iters = 1;
        cout << "  Clustering (" << n_iters << " iterations)..." << endl;
        vector<int> labels = cluster_gaussians(cluster_method, features, n_labels, n_iters);

        vector<vector<Gaussian>> clustered_gaussians(n_labels);
        for (int i = 0; i < n_labels; i++) {
            clustered_gaussians[i] = vector<Gaussian>();
        }
        for (int i = 0; i < gaussians.size(); i++) {
            clustered_gaussians[labels[i]].push_back(gaussians[i]);
        }

        cout << "  Merging..." << endl;
        for (int i = 0; i < n_labels; i++) {
            Gaussian g = merge_gaussians(merge_method, clustered_gaussians[i], current_lod);
            if (!g.invalid) output_gaussians.push_back(g);
        }

        current_lod++;
    }

    cout << "Saving \"" << output_path << "\"... " << output_gaussians.size() << " Gaussians" << endl;
    save_ply(output_path, output_gaussians);

    return 0;
}
