/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
import Utils.Math.PackedFormats;
import Utils.Math.FormatConversion;
import Utils.Math.MathHelpers;


/** Struct holding guide data for denoising.
*/
struct MLDataOutput
{
    float3 radiance;
    float3 dir;
    float3 thp;
    float3 worldpos;
    float3 normal;
    float3 color;
    float3 directradiance;
    float3 view;
    float roughness;
    float pdf;
};


struct MLData
{
    float3 primaryWorldPos;
    float3 primaryColor;
    float3 primaryThp;
    float3 primaryRadiance;
    float2 primaryNormalPacked;
    float2 primaryScatterRayPacked;
    float2 primaryViewPacked;
    float rough;
    float pdf; // We store delta flag in pdf because of some anomalous compilation problems

    [mutating] void setPrimaryWorldPos(float3 pos) { primaryWorldPos = pos; }
    [mutating] void setPrimaryThp(float3 thp) { primaryThp = thp; }
    [mutating] void setPrimaryNormal(float3 n) { primaryNormalPacked = ndir_to_oct_snorm(n); }
    [mutating] void setPrimaryView(float3 v) { primaryViewPacked = ndir_to_oct_snorm(v); }
    [mutating] void setPrimaryColor(float3 c) { primaryColor = c; }
    [mutating] void setIsDelta(bool isDelta) { if (isDelta) pdf *= -1.0f; }
    [mutating] void setPDF(float _pdf)
    {
        pdf = (pdf < 0.0) ? -_pdf : _pdf;
    }
    [mutating] void setPrimaryRadiance(float3 rad) { primaryRadiance = rad; }
    [mutating] void setPrimaryRoughness(float r) { rough = r; }
    [mutating] void setPrimaryScatterDir(float3 dir) { primaryScatterRayPacked = ndir_to_oct_snorm(dir); }

    [mutating] void init()
    {
        pdf = -1.0;
        primaryScatterRayPacked = ndir_to_oct_snorm(float3(0.0f, 0.0f, 0.0f));
        primaryThp = float3(1.0f, 1.0f, 1.0f);
        primaryRadiance = float3(-10.0f, -10.0f, -100.0f);
        primaryNormalPacked = ndir_to_oct_snorm(float3(0.0f, 0.0f, 0.0f));
        primaryColor = float3(0.0f, 0.0f, 0.0f);
        primaryViewPacked = ndir_to_oct_snorm(float3(0.0f, 0.0f, 0.0f));
        rough = 0.0f;
    }

    MLDataOutput generateOutput(float3 radiance)
    {
        MLDataOutput out;

        float3 divider = (pdf < 0.0) ? float3(1.0f, 1.0f, 1.0f) : 1.0 / primaryThp;


        if(any(radiance < 0.0)){
            out.radiance  = radiance;
        }
        else
            out.radiance = clamp(radiance - primaryRadiance, 0.0, 100000.0);




        // Unpack the directions when writing to the global buffer
        out.dir = oct_to_ndir_snorm(primaryScatterRayPacked);
        out.thp = primaryThp;
        out.worldpos = primaryWorldPos;
        out.normal = oct_to_ndir_snorm(primaryNormalPacked);
        out.color = primaryColor;
        out.roughness = rough;
        out.view = oct_to_ndir_snorm(primaryViewPacked);
        out.pdf = abs(pdf);
        out.directradiance = primaryRadiance;

        return out;
    }
};



