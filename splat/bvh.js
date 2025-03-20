let module;
importScripts('bvh_wasm.js');

bvh_WASM_Module().then(response => {
    module = response;
});

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
    return new AABB(Math.min(a.minx(), b.minx()), Math.max(a.maxx(), b.maxx()),
                Math.min(a.miny(), b.miny()), Math.max(a.miny(), a.maxy()),
                Math.min(a.minz(), b.minz()), Math.max(a.maxz(), b.maxz()));
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
        // this.rotations.push([
        //     (u_buffer[32 * i + 28 + 0]),
        //     (u_buffer[32 * i + 28 + 1]),
        //     (u_buffer[32 * i + 28 + 2]),
        //     (u_buffer[32 * i + 28 + 3]),
        // ]);
        this.rotations.push([
            (u_buffer[32 * i + 28 + 0] - 128) / 128,
            (u_buffer[32 * i + 28 + 1] - 128) / 128,
            (u_buffer[32 * i + 28 + 2] - 128) / 128,
            (u_buffer[32 * i + 28 + 3] - 128) / 128,
        ]);
    }

    /**
     * Converts the gaussian cloud to a buffer in the format worker.js likes
     * @returns {ArrayBuffer}
     */
    toBuffer() {
        // Copied from worker.js
        // 6*4 + 4 + 4 = 8*4
        // XYZ - Position (Float32)
        // XYZ - Scale (Float32)
        // RGBA - colors (uint8)
        // IJKL - quaternion/rot (uint8)
        const vertexCount = this.positions.length;
        const buffer = new ArrayBuffer(rowLength * vertexCount);

        for (let i = 0; i < vertexCount; i++) {
            const position = new Float32Array(buffer, i * rowLength, 3);
            const scales = new Float32Array(buffer, i * rowLength + 4 * 3, 3);
            const rgba = new Uint8ClampedArray(
                buffer,
                i * rowLength + 4 * 3 + 4 * 3,
                4,
            );
            const rot = new Uint8ClampedArray(
                buffer,
                i * rowLength + 4 * 3 + 4 * 3 + 4,
                4,
            );

            for (let j = 0; j < 3; j++) position[j] = this.positions[i][j];
            for (let j = 0; j < 4; j++) rgba[j] = this.rgba[i][j];
            for (let j = 0; j < 3; j++) scales[j] = this.scales[i][j];
            for (let j = 0; j < 4; j++) rot[j] = (this.rotations[i][j] * 128) + 128 | 0;
        }
        return buffer;
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
    const t = i_w / (i_w + j_w);

    let new_pos = [t * cloud.positions[i][0] + (1 - t) * cloud.positions[j][0],
                   t * cloud.positions[i][1] + (1 - t) * cloud.positions[j][1],
                   t * cloud.positions[i][2] + (1 - t) * cloud.positions[j][2]];

    let new_rgba = [t * cloud.rgba[i][0] + (1 - t) * cloud.rgba[j][0],
                   t * cloud.rgba[i][1] + (1 - t) * cloud.rgba[j][1],
                   t * cloud.rgba[i][2] + (1 - t) * cloud.rgba[j][2],
                   t * cloud.rgba[i][3] + (1 - t) * cloud.rgba[j][3]];

    let new_rot = slerp(cloud.rotations[i], cloud.rotations[j], t);

    // TODO: increase scale instead of interpolating?
    let new_scales = [t * cloud.scales[i][0] + (1 - t) * cloud.scales[j][0],
                      t * cloud.scales[i][1] + (1 - t) * cloud.scales[j][1],
                      t * cloud.scales[i][2] + (1 - t) * cloud.scales[j][2]];

    cloud.positions.push(new_pos);
    cloud.rotations.push(new_rot);
    cloud.rgba.push(new_rgba);
    cloud.scales.push(new_scales);
    return cloud.positions.length - 1;
}

/**37
 *
 * @param {Array<number>} viewProj
 * @param {Array<number>} vec
 * @returns
 */
function multiplyVec43(viewProj, vec) {
    let tmp = [viewProj[0] * vec[0] + viewProj[4] * vec[1] + viewProj[8] * vec[2] + viewProj[12],
               viewProj[1] * vec[0] + viewProj[5] * vec[1] + viewProj[9] * vec[2] + viewProj[13],
               viewProj[2] * vec[0] + viewProj[6] * vec[1] + viewProj[10] * vec[2] + viewProj[14],
               viewProj[3] * vec[0] + viewProj[7] * vec[1] + viewProj[11] * vec[2] + viewProj[15]];
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
        vec[0] /= vec[3];
        vec[1] /= vec[3];
        vec[2] /= vec[3];
        vec[3] = 1;
        return vec;
    }
    return vec;
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
        let rightHalf = indices.slice(indices.length / 2);
        if (leftHalf.length > 0) {
            this.left = new BVHNode(cloud, leftHalf, axis + 1);
        }
        if (rightHalf.length > 0) {
            this.right = new BVHNode(cloud, rightHalf, axis + 1);
        }


        if (this.left && this.right) {
            this.bbox = MergeAABB(this.left.bbox, this.right.bbox);
            this.index = MergeGaussian(cloud, this.left.index, this.right.index);
        }

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
            let pvec = dehomogenize(multiplyVec43(viewProj, vec));
            minx = Math.min(minx, pvec[0]);
            maxx = Math.max(maxx, pvec[0]);
            miny = Math.min(miny, pvec[1]);
            maxy = Math.max(maxy, pvec[1]);
            minz = Math.min(minz, pvec[2]);
            maxz = Math.max(maxz, pvec[2]);
        }

        // Thingy not in view, don't bother.
        // TODO: this WILL cause artifacts on edges when the camera rotates, relax these constraints later
        if (maxz < -1 || minz > 1 || maxx < -1 || minx > 1 || maxy < -1 || miny  > 1) {
            return [];
        }
        if (!this.left || !this.right || maxx - minx < 0.1 || maxy - miny < 0.1) {
            return [this.index];
        } else {
            return this.left.getIndices(indices, viewProj).concat(this.right.getIndices(indices, viewProj));
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
        let indices = [...Array(vertexCount).keys()];
        console.log("start building");
        this.root = new BVHNode(this.cloud, indices, 0);
        console.log("done building");
        this.vertexCount = this.cloud.positions.length;
    }

    /**
     * @param {Array<number>} viewProj
     */
    getIndices(viewProj) {
        let indices = []
        return this.root.getIndices(indices, viewProj);
    }
}

let bvh;
let port;
let vertexCount = 0;
const rowLength = 3 * 4 + 3 * 4 + 4 + 4;

function processPlyBuffer(inputBuffer) {
    const ubuf = new Uint8Array(inputBuffer);
    // 10KB ought to be enough for a header...
    const header = new TextDecoder().decode(ubuf.slice(0, 1024 * 10));
    const header_end = "end_header\n";
    const header_end_index = header.indexOf(header_end);
    if (header_end_index < 0)
        throw new Error("Unable to read .ply file header");
    const vertexCount = parseInt(/element vertex (\d+)\n/.exec(header)[1]);
    console.log("Vertex Count", vertexCount);
    let row_offset = 0,
        offsets = {},
        types = {};
    const TYPE_MAP = {
        double: "getFloat64",
        int: "getInt32",
        uint: "getUint32",
        float: "getFloat32",
        short: "getInt16",
        ushort: "getUint16",
        uchar: "getUint8",
    };
    for (let prop of header
        .slice(0, header_end_index)
        .split("\n")
        .filter((k) => k.startsWith("property "))) {
        const [p, type, name] = prop.split(" ");
        const arrayType = TYPE_MAP[type] || "getInt8";
        types[name] = arrayType;
        offsets[name] = row_offset;
        row_offset += parseInt(arrayType.replace(/[^\d]/g, "")) / 8;
    }
    console.log("Bytes per row", row_offset, types, offsets);

    let dataView = new DataView(
        inputBuffer,
        header_end_index + header_end.length,
    );
    let row = 0;
    const attrs = new Proxy(
        {},
        {
            get(target, prop) {
                if (!types[prop]) throw new Error(prop + " not found");
                return dataView[types[prop]](
                    row * row_offset + offsets[prop],
                    true,
                );
            },
        },
    );

    console.time("calculate importance");
    let sizeList = new Float32Array(vertexCount);
    let sizeIndex = new Uint32Array(vertexCount);
    let lodList = [];
    let prevLod = -1;
    for (row = 0; row < vertexCount; row++) {
        sizeIndex[row] = row;
        if (!types["scale_0"]) continue;
        const size =
            Math.exp(attrs.scale_0) *
            Math.exp(attrs.scale_1) *
            Math.exp(attrs.scale_2);
        const opacity = 1 / (1 + Math.exp(-attrs.opacity));
        sizeList[row] = size * opacity;

        if (types["lod"] && attrs.lod > prevLod) {
            lodList.push(row);
            prevLod = attrs.lod;
        }
    }
    while (lodList.length < 10) {
        lodList.push(lodList[lodList.length - 1]);
    }
    console.timeEnd("calculate importance");

    console.time("sort");
    sizeIndex.sort((b, a) => sizeList[a] - sizeList[b]);
    console.timeEnd("sort");

    // 6*4 + 4 + 4 = 8*4
    // XYZ - Position (Float32)
    // XYZ - Scale (Float32)
    // RGBA - colors (uint8)
    // IJKL - quaternion/rot (uint8)
    const rowLength = 3 * 4 + 3 * 4 + 4 + 4;
    const buffer = new ArrayBuffer(rowLength * vertexCount);

    console.time("build buffer");
    for (let j = 0; j < vertexCount; j++) {
        row = sizeIndex[j];

        const position = new Float32Array(buffer, j * rowLength, 3);
        const scales = new Float32Array(buffer, j * rowLength + 4 * 3, 3);
        const rgba = new Uint8ClampedArray(
            buffer,
            j * rowLength + 4 * 3 + 4 * 3,
            4,
        );
        const rot = new Uint8ClampedArray(
            buffer,
            j * rowLength + 4 * 3 + 4 * 3 + 4,
            4,
        );

        if (types["scale_0"]) {
            const qlen = Math.sqrt(
                attrs.rot_0 ** 2 +
                    attrs.rot_1 ** 2 +
                    attrs.rot_2 ** 2 +
                    attrs.rot_3 ** 2,
            );

            rot[0] = (attrs.rot_0 / qlen) * 128 + 128;
            rot[1] = (attrs.rot_1 / qlen) * 128 + 128;
            rot[2] = (attrs.rot_2 / qlen) * 128 + 128;
            rot[3] = (attrs.rot_3 / qlen) * 128 + 128;

            scales[0] = Math.exp(attrs.scale_0);
            scales[1] = Math.exp(attrs.scale_1);
            scales[2] = Math.exp(attrs.scale_2);
        } else {
            scales[0] = 0.01;
            scales[1] = 0.01;
            scales[2] = 0.01;

            rot[0] = 255;
            rot[1] = 0;
            rot[2] = 0;
            rot[3] = 0;
        }

        position[0] = attrs.x;
        position[1] = attrs.y;
        position[2] = attrs.z;

        if (types["f_dc_0"]) {
            const SH_C0 = 0.28209479177387814;
            rgba[0] = (0.5 + SH_C0 * attrs.f_dc_0) * 255;
            rgba[1] = (0.5 + SH_C0 * attrs.f_dc_1) * 255;
            rgba[2] = (0.5 + SH_C0 * attrs.f_dc_2) * 255;
        } else {
            rgba[0] = attrs.red;
            rgba[1] = attrs.green;
            rgba[2] = attrs.blue;
        }
        if (types["opacity"]) {
            rgba[3] = (1 / (1 + Math.exp(-attrs.opacity))) * 255;
        } else {
            rgba[3] = 255;
        }
    }
    // TODO: build tree?
    console.timeEnd("build buffer");
    return [buffer, lodList];
}

let indicesRunning = false;

const throttledIndices = (view) => {
    if (!indicesRunning && bvh && port) {
        indicesRunning = true;
        let lastView = view;
        console.log("starting indices");
        let indices = bvh.getIndicesJS(view);
        indices = new Uint32Array(Array.from({ length: indices.size() }, (_, i) => indices.get(i)));
        port.postMessage({ indices: indices });
        console.log("finished indices");

        setTimeout(() => {
            indicesRunning = false;
            if (lastView !== view) {
                throttledIndices();
            }
        }, 0)
    }
}

onmessage = (e) => {
    if (e.data.ply) {
        // This is unused mostly
        vertexCount = 0;
        let buffer;
        let lodList;
        [buffer, lodList] = processPlyBuffer(e.data.ply);
        vertexCount = Math.floor(buffer.byteLength / rowLength);
        // TODO: process buffer
        // bvh = new BoundingVolumeHierarchy(buffer, vertexCount);
        // buffer = bvh.cloud.toBuffer();
        if (port) {
            port.postMessage({
                buffer: buffer,
                vertexCount: vertexCount
            }, [buffer]);
        }
        postMessage({ buffer, lodList, save: false }, [buffer, lodList]);
    } else if (e.data.buffer) {
        while (!module); // gross but it works
        let arr = new Uint8Array(e.data.buffer);
        console.log('buildin\' bvh');
        bvh = new module.BoundingVolumeHierarchy(arr, e.data.vertexCount);
        e.data.buffer = undefined; // force garbage collection
        console.log('bvh complete');
        let buftmp = bvh.toBuffer();
        console.log('buffered');
        vertexCount = bvh.size();
        let buffer = new Uint8Array(buftmp.size());
        for (let i = 0; i < buftmp.size(); i++) {
            buffer[i] = buftmp.get(i);
        }
        console.log('actually buffered');
        buftmp = undefined; // free up some memory
        // console.log(`size: ${vertexCount}`);
        // let arr = new Uint8Array(e.data.buffer);
        // let cloud = new module.GaussianCloud(arr, e.data.vertexCount);
        // let buftmp = cloud.toBuffer();
        // let buffer = new Uint8Array(Array.from({ length: arr.length }, (_, i) => buftmp.get(i)));
        if (port) {
            port.postMessage({
                buffer: buffer.buffer,
                vertexCount: vertexCount
            }, [buffer.buffer]);
        }
    } else if (e.data.vertexCount) {
        vertexCount = e.data.vertexCount;
    } else if (e.data.view) {
        throttledIndices(e.data.view);
    } else if (e.data.port) {
        port = e.data.port;
    }
}
