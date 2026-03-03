#ifndef BSDF_H
#define BSDF_H

__device__ __forceinline__ float dot3(const float* a, const float* b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

__device__ __forceinline__ void normalize3(float* v) {
    float len = sqrtf(dot3(v, v));
    if (len > 0.0f) {
        v[0] /= len;
        v[1] /= len;
        v[2] /= len;
    }
}

__device__ __forceinline__ void evalGGX(float result[3],
                               const float wo[3],
                               const float wi[3],
                               const float specular_refl[3],
                               const float eta[3],
                               const float k[3],
                               const float alpha) {
    // Half vector (explicit scalars)
    float hx = wi[0] + wo[0];
    float hy = wi[1] + wo[1];
    float hz = wi[2] + wo[2];

    float hlen2 = hx*hx + hy*hy + hz*hz;
    if (hlen2 > 0.0f) {
        float inv = rsqrtf(hlen2);
        hx *= inv;
        hy *= inv;
        hz *= inv;
    }

    float NoV = fmaxf(wo[2], 0.0f);
    float NoL = fmaxf(wi[2], 0.0f);
    float NoH = fmaxf(hz,     0.0f);
    float VoH = fmaxf(wo[0]*hx + wo[1]*hy + wo[2]*hz, 0.0f);

    if (NoV <= 0.0f || NoL <= 0.0f) {
        result[0] = result[1] = result[2] = 0.0f;
        return;
    }

    float alpha2 = alpha * alpha;

    // GGX NDF
    float denom = NoH * NoH * (alpha2 - 1.0f) + 1.0f;
    float D = alpha2 / (M_PI * denom * denom);

    // Smith GGX (Schlick)
    float schlick = (alpha + 1.0f);
    schlick = (schlick * schlick) * 0.125f;

    float Gv = NoV / (NoV * (1.0f - schlick) + schlick);
    float Gl = NoL / (NoL * (1.0f - schlick) + schlick);
    float G  = Gv * Gl;

    float oneMinusVoH = 1.0f - VoH;
    float oneMinusVoH5 = oneMinusVoH * oneMinusVoH;
    oneMinusVoH5 *= oneMinusVoH5 * oneMinusVoH;

    const float F0r = ((eta[0] - 1) * (eta[0] - 1) + k[0] * k[0]) / ((eta[0] + 1) * (eta[0] + 1) + k[0] * k[0]);
    const float F0g = ((eta[1] - 1) * (eta[1] - 1) + k[1] * k[1]) / ((eta[1] + 1) * (eta[1] + 1) + k[1] * k[1]);
    const float F0b = ((eta[2] - 1) * (eta[2] - 1) + k[2] * k[2]) / ((eta[2] + 1) * (eta[2] + 1) + k[2] * k[2]);

    float Fr = F0r + (1.0f - F0r) * oneMinusVoH5;
    float Fg = F0g + (1.0f - F0g) * oneMinusVoH5;
    float Fb = F0b + (1.0f - F0b) * oneMinusVoH5;

    float scale = (D * G) / (4.0f * NoV * NoL);

    result[0] = scale * Fr * specular_refl[0];
    result[1] = scale * Fg * specular_refl[1];
    result[2] = scale * Fb * specular_refl[2];
}

__device__ __forceinline__ void evalDiffuse(float* result, float* albedo) {
    result[0] = albedo[0];
    result[1] = albedo[1];
    result[2] = albedo[2];
}

#endif