/**
 * Axis-Aligned Bounding Box
 */
class AABB {
    /**
     *
     * @param {number} minx
     * @param {number} maxx
     * @param {number} miny
     * @param {number} maxy
     * @param {number} minz
     * @param {number} maxz
     */
    constructor(minx, maxx, miny, maxy, minz, maxz) {
        this.bounds = [minx, maxx, miny, maxy, minz, maxz];
    }

    /**
     *
     * @param {Array<number>} bounds
     */
    constructor(bounds) {
        this.bounds = bounds;
    }

    minx() { return this.bounds[0]; }
    maxx() { return this.bounds[1]; }
    miny() { return this.bounds[2]; }
    maxy() { return this.bounds[3]; }
    minz() { return this.bounds[4]; }
    maxz() { return this.bounds[5]; }

    /**
     * Returns AABB vertices as an array in no particular order
     * @returns {Float32Array}
     */
    vertices() {
        const [minx, maxx, miny, maxy, minz, maxz] = this.bounds;
        return new Float32Array([
            minx, miny, minz,
            minx, miny, maxz,
            minx, maxy, minz,
            minx, maxy, maxz,
            maxx, miny, minz,
            maxx, miny, maxz,
            maxx, maxy, minz,
            maxx, maxy, maxz
        ]);
    }
}

/**
 * Merges two bounding boxes
 * @param {AABB} a
 * @param {AABB} b
 * @returns {AABB} Merged bounding box
 */
function MergeAABB(a, b) {
    return AABB(Math.min(a.minx, b.minx), Math.max(a.maxx, b.maxx),
                Math.min(a.miny, b.miny), Math.max(a.miny, a.maxy),
                Math.min(a.minz, b.minz), Math.max(a.maxz, b.maxz));
}

class GaussianCloud {
    /**
     *
     * @param {ArrayBuffer} buffer
     * @param {number} vertexCount Number of gaussians to process
     */
    constructor(buffer, vertexCount) {
        this.positions = [];
        this.rotations = [];
        this.scales = [];
        this.rgba = [];
        // No spherical harmonics :(
        const f_buffer = new Float32Array(buffer);
        const u_buffer = new Uint8Array(buffer);
        for (let i = 0; i < vertexCount; i++) {
            this.parseElement(f_buffer, u_buffer, i);
        }
    }

    /**
     * Parse splat i in buffer, add to gaussian cloud data structure
     * @param {Float32Array} f_buffer
     * @param {Uint8Array} u_buffer
     * @param {number} i
     */
    parseElement(f_buffer, u_buffer, i) {
        this.positions.push([
            f_buffer[8 * i + 0],
            f_buffer[8 * i + 1],
            f_buffer[8 * i + 2]]
        );
        this.rgba.push([
            u_buffer[32 * i + 24 + 0],
            u_buffer[32 * i + 24 + 1],
            u_buffer[32 * i + 24 + 2],
            u_buffer[32 * i + 24 + 3]]
        );
        this.scales.push([
            f_buffer[8 * i + 3 + 0],
            f_buffer[8 * i + 3 + 1],
            f_buffer[8 * i + 3 + 2],
        ]);
        this.rotations.push([
            (u_buffer[32 * i + 28 + 0] - 128) / 128,
            (u_buffer[32 * i + 28 + 1] - 128) / 128,
            (u_buffer[32 * i + 28 + 2] - 128) / 128,
            (u_buffer[32 * i + 28 + 3] - 128) / 128,
        ]);
    }
}

/**
 * spherical linear interpolation (slerp) between two quaternions
 * @param {Array<number>} q1 sirst quaternion [x, y, z, w]
 * @param {Array<number>} q2 second quaternion [x, y, z, w]
 * @param {number} t interpolation factor in [0, 1] inclusive
 * @returns {Array<number>} interpolated quaternion
 */
function slerp(q1, q2, t) {
    let dot = q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3];

    // If the dot product is negative, invert one quaternion to take the shorter path
    if (dot < 0.0) {
        q2 = q2.map(v => -v);
        dot = -dot;
    }

    // If the quaternions are very close, use linear interpolation
    if (dot > 0.9995) {
        const result = q1.map((v, i) => v + t * (q2[i] - v));
        const norm = Math.sqrt(result.reduce((sum, v) => sum + v * v, 0));
        return result.map(v => v / norm);
    }

    // Compute the angle between the quaternions
    const theta_0 = Math.acos(dot);
    const theta = theta_0 * t;

    const sin_theta = Math.sin(theta);
    const sin_theta_0 = Math.sin(theta_0);

    const s1 = Math.cos(theta) - dot * sin_theta / sin_theta_0;
    const s2 = sin_theta / sin_theta_0;

    return [
        s1 * q1[0] + s2 * q2[0],
        s1 * q1[1] + s2 * q2[1],
        s1 * q1[2] + s2 * q2[2],
        s1 * q1[3] + s2 * q2[3]
    ];
}


/**
 * Rotates a vector by a quaternion
 * @param {Array<number>} vec 3D vector
 * @param {Array<number>} q Quaternion [qx, qy, qz, qw]
 * @returns {Array<number>} Rotated vector
 */
function rotateVector(vec, q) {
    const [x, y, z] = vec;
    const [qx, qy, qz, qw] = q;

    // quaternion-vector multiplication (q * v * q^-1)
    const uvx = qw * x + qy * z - qz * y;
    const uvy = qw * y + qz * x - qx * z;
    const uvz = qw * z + qx * y - qy * x;
    const uuw = -qx * x - qy * y - qz * z;

    const rx = uvx * qw + uuw * -qx + uvy * -qz - uvz * -qy;
    const ry = uvy * qw + uuw * -qy + uvz * -qx - uvx * -qz;
    const rz = uvz * qw + uuw * -qz + uvx * -qy - uvy * -qx;

    return [rx, ry, rz];
}

/**
 * Gets AABB for a given gaussian up to 3-sigma away
 * @param {GaussianCloud} cloud gaussian cloud
 * @param {number} i index in gaussian cloud
 * @returns {AABB} Axis-Aligned Bounding Box
 */
function GaussianAABB(cloud, i) {
    const pos = cloud.positions[i];
    const scale = cloud.scales[i];
    const rotation = cloud.rotations[i];

    // 3-sigma range (covariance scaling)
    const sx = 3 * scale[0];
    const sy = 3 * scale[1];
    const sz = 3 * scale[2];

    // Basis vectors of the covariance ellipsoid
    const basis = [
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, sz]
    ];

    // Rotate the basis vectors using the quaternion
    const rotatedBasis = basis.map(v => rotateVector(v, rotation));

    // Compute bounds by projecting the rotated vectors onto the coordinate axes
    let minx = pos[0];
    let maxx = pos[0];
    let miny = pos[1];
    let maxy = pos[1];
    let minz = pos[2];
    let maxz = pos[2];

    for (const vec of rotatedBasis) {
        minx = Math.min(minx, pos[0] + vec[0], pos[0] - vec[0]);
        maxx = Math.max(maxx, pos[0] + vec[0], pos[0] - vec[0]);
        miny = Math.min(miny, pos[1] + vec[1], pos[1] - vec[1]);
        maxy = Math.max(maxy, pos[1] + vec[1], pos[1] - vec[1]);
        minz = Math.min(minz, pos[2] + vec[2], pos[2] - vec[2]);
        maxz = Math.max(maxz, pos[2] + vec[2], pos[2] - vec[2]);
    }

    return new AABB(minx, maxx, miny, maxy, minz, maxz);
}

/**
 * Merge two gaussians together, appends to gaussiancloud, and
 * returns index in gaussiancloud
 * @param {GaussianCloud} cloud
 * @param {number} i
 * @param {number} j
 * @returns {number} Index of new gaussian in gaussian cloud
 */
function MergeGaussian(cloud, i, j) {
    // Weight of i
    const i_w = cloud.rgba[i][3] * Math.sqrt(cloud.scales[i][0] * cloud.scales[i][0] +
                   cloud.scales[i][1] * cloud.scales[i][1] +
                   cloud.scales[i][2] * cloud.scales[i][2]);
    // Weight of j
    const j_w = cloud.rgba[j][3] * Math.sqrt(cloud.scales[j][0] * cloud.scales[j][0] +
                   cloud.scales[j][1] * cloud.scales[j][1] +
                   cloud.scales[j][2] * cloud.scales[j][2]);

    let new_pos = [i_w * cloud.positions[i][0] + j_w * cloud.positions[j][0],
                   i_w * cloud.positions[i][1] + j_w * cloud.positions[j][1],
                   i_w * cloud.positions[i][2] + j_w * cloud.positions[j][2]];
    new_pos.map(x => x / (i_w + j_w));

    let new_rgba = [i_w * cloud.rgba[i][0] + j_w * cloud.rgba[j][0],
                   i_w * cloud.rgba[i][1] + j_w * cloud.rgba[j][1],
                   i_w * cloud.rgba[i][2] + j_w * cloud.rgba[j][2],
                   i_w * cloud.rgba[i][3] + j_w * cloud.rgba[j][3]];
    new_rgba.map(x => x / (i_w + j_w));

    let new_rot = slerp(cloud[i].rotations, cloud[j].rotations, i_w / (i_w + j_w));

    // TODO: increase scale instead of interpolating?
    let new_scales = [i_w * cloud.scales[i][0] + j_w * cloud.scales[j][0],
                      i_w * cloud.scales[i][1] + j_w * cloud.scales[j][1],
                      i_w * cloud.scales[i][2] + j_w * cloud.scales[j][2]];
    new_scales.map(x => x / (i_w + j_w));

    cloud.positions.push(new_pos);
    cloud.rotations.push(new_rot);
    cloud.rgba.push(new_rgba);
    cloud.scales.push(new_scales);
    return cloud.length - 1;

}

/**
 *
 * @param {Array<number>} viewProj
 * @param {Array<number>} vec
 * @returns
 */
function multiplyVec(viewProj, vec) {
    let tmp = [viewProj[0] * vec[0] + viewProj[4] * vec[1] + viewProj[8] * vec[2] + viewProj[12] * vec[3],
               viewProj[1] * vec[0] + viewProj[5] * vec[1] + viewProj[9] * vec[2] + viewProj[13] * vec[3],
               viewProj[2] * vec[0] + viewProj[6] * vec[1] + viewProj[10] * vec[2] + viewProj[14] * vec[3],
               viewProj[3] * vec[0] + viewProj[7] * vec[1] + viewProj[11] * vec[2] + viewProj[15] * vec[3]]
    return tmp;
}

/**
 *
 * @param {Array<number>} vec
 * @returns
 */
function dehomogenize(vec) {
    if (vec.length !== 4) {
        return vec;
    }
    if (vec[3] !== 0) {
        tmp[0] /= tmp[3];
        tmp[1] /= tmp[3];
        tmp[2] /= tmp[3];
        tmp[3] = 1;
        return tmp;
    }
    return tmp;
}

class BVHNode {
    /**
     *
     * @param {GaussianCloud} cloud
     * @param {Array<number>} indices
     * @param {number} axis to sort against (between 0 and 2 inclusive)
     */
    constructor(cloud, indices, axis) {
        axis = axis % 3;
        this.indices = indices;
        if (indices.length === 1) {
            this.index = indices[0];
            this.bbox = GaussianAABB(cloud, indices[0]);
            this.left = this.right = undefined;
            return;
        }
        // TODO: surface-area heuristic, but im lazy :)
        indices.sort((a, b) => {
            cloud.positions[a][axis] - cloud.positions[b][axis];
        });
        let leftHalf = indices.slice(0, indices.length / 2);
        let rightHalf = indices.slice(indices.length / 2, -1);
        this.left = new BVHNode(cloud, leftHalf, axis + 1);
        this.right = new BVHNode(cloud, rightHalf, axis + 1);

        this.bbox = MergeAABB(this.left.bbox, this.right.bbox);
        this.index = MergeGaussian(cloud, this.left.index, this.right.index);
    }

    /**
     *
     * @param {Array<number>} indices
     * @param {Array<number>} viewProj
     */
    getIndices(indices, viewProj) {
        let vertices = this.bbox.vertices();
        // Screen space bounding box
        // minx maxx miny maxy
        let minx = Infinity;
        let maxx = -Infinity;
        let miny = Infinity;
        let maxy = -Infinity;
        let minz = Infinity;
        let maxz = -Infinity;
        for (let i = 0; i < 8; i++) {
            let vec = [vertices[i * 3 + 0], vertices[i * 3 + 1], vertices[i * 3 + 2]];
            let pvec = dehomogenize(multiplyVec(viewProj, vec));
            minx = Math.min(minx, pvec[0]);
            maxx = Math.max(maxx, pvec[0]);
            miny = Math.min(miny, pvec[1]);
            maxy = Math.max(maxy, pvec[1]);
            minz = Math.min(minz, pvec[2]);
            maxz = Math.min(maxz, pvec[2]);
        }
        if (maxz < -1 || minz > 1) {
            return;
        }
        if (!this.left || !this.right || maxx - minx < 0.005 || maxy - miny < 0.005) {
            indices.push(this.index);
        } else {
            this.left.intersect(indices, viewProj);
            this.right.intersect(indices, viewProj);
        }
    }
}

class BoundingVolumeHierarchy {
    /**
     * Constructs a BVH given an array buffer of splat info
     * @param {ArrayBuffer} buffer
     */
    constructor(buffer, vertexCount) {
        this.cloud = new GaussianCloud(buffer, vertexCount);
        let indices = Array.from(0, vertexCount);
        this.root = new BVHNode(this.cloud, indices, 0);
    }

    /**
     * @param {Array<number>} viewProj
     */
    getIndices(viewProj) {
        let indices = []
        return this.root.getIndices(indices, viewProj);
    }
}

export * from 'bvh';
