#ifndef __BRESENHAM_H__
#define __BRESENHAM_H__

// Code adjusted from Will Navidson's (yamamushi) gist at:
// https://gist.github.com/yamamushi/5823518#file-bresenham3d

// This function does NOT draw a line but instead it uses Bresenham algorithm
// to check if all voxels along the line are non-zero.
// pos1 and pos2 are voxel positions
__device__
bool Bresenham3D(const uint3 pos1,              // input: voxel position of first particle
                 const uint3 pos2,              // input: voxel position of second particle
                 uint32_t *contour,             // input: contour array
                 const uint3 contourSize) {     // input: contour size

    int32_t i, dx, dy, dz, l, m, n, x_inc, y_inc, z_inc, err_1, err_2, dx2, dy2, dz2;

    uint3 point(pos1);

    dx = pos2.x - pos1.x;
    dy = pos2.y - pos1.y;
    dz = pos2.z - pos1.z;
    x_inc = (dx < 0) ? -1 : 1;
    l = abs(dx);
    y_inc = (dy < 0) ? -1 : 1;
    m = abs(dy);
    z_inc = (dz < 0) ? -1 : 1;
    n = abs(dz);
    dx2 = l << 1;
    dy2 = m << 1;
    dz2 = n << 1;

    if ((l >= m) && (l >= n)) {
        err_1 = dy2 - l;
        err_2 = dz2 - l;
        for (i = 0; i < l; i++) {
            if (contour[point.x + point.y * contourSize.x +
                        point.z * contourSize.x * contourSize.y] == 0) {
                return false;
            }
            if (err_1 > 0) {
                point.y += y_inc;
                err_1 -= dx2;
            }
            if (err_2 > 0) {
                point.z += z_inc;
                err_2 -= dx2;
            }
            err_1 += dy2;
            err_2 += dz2;
            point.x += x_inc;
        }
    }
    else if ((m >= l) && (m >= n)) {
        err_1 = dx2 - m;
        err_2 = dz2 - m;
        for (i = 0; i < m; i++) {
            if (contour[point.x + point.y * contourSize.x +
                        point.z * contourSize.x * contourSize.y] == 0) {
                return false;
            }
            if (err_1 > 0) {
                point.x += x_inc;
                err_1 -= dy2;
            }
            if (err_2 > 0) {
                point.z += z_inc;
                err_2 -= dy2;
            }
            err_1 += dx2;
            err_2 += dz2;
            point.y += y_inc;
        }
    }
    else {
        err_1 = dy2 - n;
        err_2 = dx2 - n;
        for (i = 0; i < n; i++) {
            if (contour[point.x + point.y * contourSize.x +
                        point.z * contourSize.x * contourSize.y] == 0) {
                return false;
            }
            if (err_1 > 0) {
                point.y += y_inc;
                err_1 -= dz2;
            }
            if (err_2 > 0) {
                point.x += x_inc;
                err_2 -= dz2;
            }
            err_1 += dy2;
            err_2 += dx2;
            point.z += z_inc;
        }
    }
    if (contour[point.x + point.y * contourSize.x +
                point.z * contourSize.x * contourSize.y] == 0) {
        return false;
    }
    return true;
}

// Checks if straight line joining p1, p2 is contained in the contour given.
// (Passes through non-zero contour fields.)
__device__
bool checkPathInContour(float3 p1,                      // input: first particle position
                        float3 p2,                      // input: second particle position
                        uint32_t *contour,              // input: contour array
                        uint3 contourSize,              // input: contour size
                        float3 voxelSize) {             // input: voxel size
    uint3 pv1 = make_uint3((uint32_t)floorf(p1.x / voxelSize.x),
                           (uint32_t)floorf(p1.y / voxelSize.y), (uint32_t)floorf(p1.z / voxelSize.z));
    uint3 pv2 = make_uint3((uint32_t)floorf(p2.x / voxelSize.x),
                           (uint32_t)floorf(p2.y / voxelSize.y), (uint32_t)floorf(p2.z / voxelSize.z));

#if defined(DEBUG) && (DEBUG >= 3)
    if(contour[pv1.x + pv1.y * contourSize.x + pv1.z * contourSize.x * contourSize.y] == 0) {
        printf("Particle 1 at (%f, %f, %f), voxel: (%u, %u, %u) (%f, %f, %f) outside contour!\n",
                p1.x, p1.y, p1.z, pv1.x, pv1.y, pv1.z,
                floorf(p1.x / voxelSize.x), floorf(p1.y / voxelSize.y), floorf(p1.z / voxelSize.z));
    }
    if(contour[pv2.x + pv2.y * contourSize.x + pv2.z * contourSize.x * contourSize.y] == 0) {
        printf("Particle 2 at (%f, %f, %f), voxel: (%u, %u, %u) (%f, %f, %f) outside contour!\n",
                p2.x, p2.y, p2.z, pv2.x, pv2.y, pv2.z,
                floorf(p2.x / voxelSize.x), floorf(p2.y / voxelSize.y), floorf(p2.z / voxelSize.z));
    }
#endif

    return Bresenham3D(pv1, pv2, contour, contourSize);
}

#endif
