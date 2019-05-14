#ifndef __BRESENHAM_H__
#define __BRESENHAM_H__

// Code adjusted from Will Navidson's (yamamushi) gist at:
// https://gist.github.com/yamamushi/5823518#file-bresenham3d
//
// Bresenham3D
//
// A slightly modified version of the source found at
// http://www.ict.griffith.edu.au/anthony/info/graphics/bresenham.procs
// Provided by Anthony Thyssen, though he does not take credit for the original implementation
//
// It is highly likely that the original Author was Bob Pendelton, as referenced here
//
// ftp://ftp.isc.org/pub/usenet/comp.sources.unix/volume26/line3d
//
// line3d was dervied from DigitalLine.c published as "Digital Line Drawing"
// by Paul Heckbert from "Graphics Gems", Academic Press, 1990
//
// 3D modifications by Bob Pendleton. The original source code was in the public
// domain, the author of the 3D version places his modifications in the
// public domain as well.
//
// line3d uses Bresenham's algorithm to generate the 3 dimensional points on a
// line from (x1, y1, z1) to (x2, y2, z2)
//

#include <helper_functions.h>

// This function does NOT draw a line but instead it uses Bresenham algorithm
// to check if all voxels along the line are non-zero.
// pos1 and pos2 are already voxel positions
__device__
bool Bresenham3D(uint3 pos1, const uint3 pos2, uint32_t *contour, uint3 contourSize) {

    int i, dx, dy, dz, l, m, n, x_inc, y_inc, z_inc, err_1, err_2, dx2, dy2, dz2;

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
                point.z * contourSize.x * contourSize.y] == 0) return false;
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
    } else if ((m >= l) && (m >= n)) {
        err_1 = dx2 - m;
        err_2 = dz2 - m;
        for (i = 0; i < m; i++) {
            if (contour[point.x + point.y * contourSize.x +
                point.z * contourSize.x * contourSize.y] == 0) return false;
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
    } else {
        err_1 = dy2 - n;
        err_2 = dx2 - n;
        for (i = 0; i < n; i++) {
            if (contour[point.x + point.y * contourSize.x +
                point.z * contourSize.x * contourSize.y] == 0) return false;
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
                   point.z * contourSize.x * contourSize.y] == 0) return false;
    return true;
}

// Checks if straight line joining p1, p2 is contained in the contour given.
// (Passes through non-zero contour fields.)
__device__
bool checkPathInContour(float3 p1, float3 p2, uint32_t *contour, uint3 contourSize, float3 voxelSize) {
    uint3 pv1 = make_uint3((uint)floorf(p1.x / voxelSize.x),
            (uint)floorf(p1.y / voxelSize.y), (uint)floorf(p1.z / voxelSize.z));
    uint3 pv2 = make_uint3((uint)floorf(p2.x / voxelSize.x),
                           (uint)floorf(p2.y / voxelSize.y), (uint)floorf(p2.z / voxelSize.z));

    if(contour[pv1.x + pv1.y * contourSize.x +
               pv1.z * contourSize.x * contourSize.y] == 0) {
        printf("Particle 1 at (%f, %f, %f), voxel: (%u, %u, %u) (%f, %f, %f) outside contour!\n", p1.x, p1.y, p1.z, pv1.x, pv1.y, pv1.z, floorf(p1.x / voxelSize.x), floorf(p1.y / voxelSize.y), floorf(p1.z / voxelSize.z));
    }
    if(contour[pv2.x + pv2.y * contourSize.x +
               pv2.z * contourSize.x * contourSize.y] == 0) {
        printf("Particle 2 at (%f, %f, %f), voxel: (%u, %u, %u) (%f, %f, %f) outside contour!\n", p2.x, p2.y, p2.z, pv2.x, pv2.y, pv2.z, floorf(p2.x / voxelSize.x), floorf(p2.y / voxelSize.y), floorf(p2.z / voxelSize.z));
    }

    bool res = Bresenham3D(pv1, pv2, contour, contourSize);
    //printf("Checking contour, p1: (%f, %f, %f), pv1: (%u, %u, %u), p2: (%f, %f, %f), pv2: (%u, %u, %u), result: %d\n",
    //       p1.x, p1.y, p1.z, pv1.x, pv1.y, pv1.z, p2.x, p2.y, p2.z, pv2.x, pv2.y, pv2.z, res);
    return res;
}

#endif
