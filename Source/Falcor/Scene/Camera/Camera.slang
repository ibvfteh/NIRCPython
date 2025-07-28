/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"

import Scene.Camera.CameraData;
__exported import Utils.Math.Ray;
import Utils.Math.MathHelpers;

struct Camera
{
    CameraData data;

    float3 getPosition() { return data.posW; }
    float4x4 getViewProj() { return data.viewProjMat; }

    /** Computes a camera ray for a given pixel assuming a pinhole camera model.
        The camera jitter is taken into account to compute the sample position on the image plane.
        \param[in] pixel Pixel coordinates with origin in top-left.
        \param[in] frameDim Image plane dimensions in pixels.
        \param[in] applyJitter true if jitter should be applied else false.
        \return Returns the camera ray.
    */
    Ray computeRayPinhole(uint2 pixel, uint2 frameDim, bool applyJitter = true)
    {
        if (data.useHemisphericalCamera != 0) {
            return computeRayHemispherical(pixel, frameDim, applyJitter);
        }
        Ray ray;

        // Compute the normalized ray direction assuming a pinhole camera.
        ray.origin = data.posW;
        ray.dir = normalize(computeNonNormalizedRayDirPinhole(pixel, frameDim, applyJitter));

        float invCos = 1.f / dot(normalize(data.cameraW), ray.dir);
        ray.tMin = data.nearZ * invCos;
        ray.tMax = data.farZ * invCos;

        return ray;
    }

    Ray computeRayHemispherical(uint2 pixel, uint2 frameDim, bool applyJitter = true)
    {
        Ray ray;

        // Compute the normalized ray direction assuming a hemispherical camera.
        ray.origin = data.posW;
        ray.dir = normalize(computeNonNormalizedRayDirHemispherical(pixel, frameDim, applyJitter));
        ray.tMin = 0.0;
        ray.tMax = 100000000;

        return ray;
    }

    float3 computeNonNormalizedRayDirHemispherical(uint2 pixel, uint2 frameDim, bool applyJitter = true)
    {
        // Normalize pixel coordinates to [0, 1] with origin at the top-left corner
        float2 p = (pixel + float2(0.5f, 0.5f)) / frameDim;
        if (applyJitter)
        {
            p += float2(-data.jitterX, data.jitterY); // Apply jitter
        }

        // Convert normalized device coordinates to spherical coordinates
        // Latitude (theta) ranges from 0 at the pole (top of hemisphere) to PI/2 at the equator (edge of the frame).
        // Longitude (phi) ranges from -PI to PI.
        float theta = p.y * M_PI;  // Only top hemisphere, maps y to [0, PI/2]
        //float phi = p.x * 2.0f * M_PI - M_PI; // Maps x to [-PI, PI]
        float phi = (p.x * M_PI - M_PI) + M_PI; // Maps x to [-PI, PI] and shifts by PI

        // Normalize phi to wrap around correctly by ensuring it remains within [-PI, PI]
        if (phi > M_PI)
            phi -= 2.0f * M_PI;

        // Convert spherical coordinates to Cartesian coordinates on the unit sphere
        float3 rayDir;
        rayDir.x = sin(theta) * cos(phi); // Rightward direction
        rayDir.y = sin(theta) * sin(phi); // Upward direction
        rayDir.z = cos(theta);            // Forward direction, aligning with the top pole

        // Transform the ray direction into the camera's coordinate space
        return rayDir.x * normalize(data.cameraU) + rayDir.z * normalize(data.cameraV) + rayDir.y * normalize(data.cameraW);
    }


    // Assume normalized input on +Z hemisphere.
    // Output is on [-1, 1].
    float2 float32x3_to_hemioct(in float3 v) {
        // Project the hemisphere onto the hemi-octahedron,
        // and then into the xy plane
        float2 p = v.xy * (1.0 / (abs(v.x) + abs(v.y) + v.z));
        // Rotate and scale the center diamond to the unit square
        return float2(p.x + p.y, p.x - p.y);
    }
    float3 hemioct_to_float32x3(float2 e) {
        // Rotate and scale the unit square back to the center diamond
        float2 temp = float2(e.x + e.y, e.x - e.y) * 0.5;
        float3 v = float3(temp, 1.0 - abs(temp.x) - abs(temp.y));
        return normalize(v);
    }

    /** Computes the primary ray's direction, non-normalized assuming pinhole camera model.
        The camera jitter is taken into account to compute the sample position on the image plane.
        \param[in] pixel Pixel coordinates with origin in top-left.
        \param[in] frameDim Image plane dimensions in pixels.
        \param[in] applyJitter True if jitter should be applied else false.
        \return Returns the non-normalized ray direction
    */
    float3 computeNonNormalizedRayDirPinhole(uint2 pixel, uint2 frameDim, bool applyJitter = true)
    {
        // Compute sample position in screen space in [0,1] with origin at the top-left corner.
        // The camera jitter offsets the sample by +-0.5 pixels from the pixel center.
        float2 p = (pixel + float2(0.5f, 0.5f)) / frameDim;
        if (applyJitter) p += float2(-data.jitterX, data.jitterY);
        float2 ndc = float2(2, -2) * p + float2(-1, 1);

        // Compute the non-normalized ray direction assuming a pinhole camera.
        return ndc.x * data.cameraU + ndc.y * data.cameraV + data.cameraW;
    }


    /** Computes a camera ray for a given pixel assuming a thin-lens camera model.
        The camera jitter is taken into account to compute the sample position on the image plane.
        \param[in] pixel Pixel coordinates with origin in top-left.
        \param[in] frameDim Image plane dimensions in pixels.
        \param[in] u Uniform 2D sample.
        \return Returns the camera ray.
    */
    Ray computeRayThinlens(uint2 pixel, uint2 frameDim, float2 u)
    {
        Ray ray;

        // Sample position in screen space in [0,1] with origin at the top-left corner.
        // The camera jitter offsets the sample by +-0.5 pixels from the pixel center.
        float2 p = (pixel + float2(0.5f, 0.5f)) / frameDim + float2(-data.jitterX, data.jitterY);
        float2 ndc = float2(2, -2) * p + float2(-1, 1);

        // Compute the normalized ray direction assuming a thin-lens camera.
        ray.origin = data.posW;
        ray.dir = ndc.x * data.cameraU + ndc.y * data.cameraV + data.cameraW;
        float2 apertureSample = sample_disk(u); // Sample lies in the unit disk [-1,1]^2
        float3 rayTarget = ray.origin + ray.dir;
        ray.origin += data.apertureRadius * (apertureSample.x * normalize(data.cameraU) + apertureSample.y * normalize(data.cameraV));
        ray.dir = normalize(rayTarget - ray.origin);

        float invCos = 1.f / dot(normalize(data.cameraW), ray.dir);
        ray.tMin = data.nearZ * invCos;
        ray.tMax = data.farZ * invCos;

        return ray;
    }
};
