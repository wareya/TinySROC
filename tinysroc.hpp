// Tiny Software-rasterized Occlusion Culling

#include <vector>
#include <array>
#include <unordered_map>
#include <ranges>
#include <cstdint>
#include <cstring>
#include <algorithm> // for std::fill
#include <cstdlib>
#include <bit>

#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>
#include <glm/ext.hpp>

using namespace glm;

#ifdef TINYSROC_NAMESPACED
namespace tinysroc {
#endif

// clean up some common macro pollution if needed
#ifdef near
#undef near
#endif
#ifdef far
#undef far
#endif

typedef vec2 Vec2;
typedef vec3 Vec3;
typedef vec4 Vec4;
typedef mat4 Mat4;
typedef mat3 Mat3;

#if __cplusplus >= 202002L
// c++20
#define TINYSROC_RANGE std::views::iota
#else
// fallback
namespace _tsroc_private {
    class _RangeIterS {
        int64_t lo, hi;
    public:
        class _RangeIter {
            int64_t n;
        public:
            _RangeIter(int64_t argn) { n = argn; }
            int64_t operator*() { return n; }
            _RangeIter& operator++() { n++; return *this; }
            bool operator!=(_RangeIter other) { return n != other.n; }
        };
        _RangeIterS(int64_t _lo, int64_t _hi) { lo = _lo; hi = _hi; }
        auto begin() { return _RangeIter(lo); }
        auto end() { return _RangeIter(hi); }
    };
}
#define TINYSROC_RANGE _tsroc_private::_RangeIterS
#endif

#ifndef TINYSROC_MALLOC
#define TINYSROC_MALLOC malloc
#endif
#ifndef TINYSROC_FREE
#define TINYSROC_FREE free
#endif

struct Trimesh {
    Vec3 * verts; // layout: xyz XYZ ... (small: position. caps: normals.)
    Vec4 * verts_mm; // xyzw XYZW ... ; post-matmul.
    uint32_t * indexes; // indexes to actually draw
    size_t triangle_count;
    size_t vertex_count;
    bool owned;
};
struct Rect {
    Vec3 verts[4]; // Layout: xyz xyz xyz xyz. Clockwise winding order. MUST be coplanar.
};
struct Trilist {
    Vec3 * verts; // always owned. positions only.
    Vec4 * verts_mm; // post-matmul
    size_t triangle_count;
};

struct Occluder {
    int kind; // 0: trimesh. 1: rect. 2: trilist.
    bool conservative;
    bool backface_cull;
    Mat4 xform; // modelspace to worldspace transform
    Vec4 aabb_mid;
    Vec3 aabb_ext;
    
    union {
        Trimesh mesh;
        Rect rect;
        Trilist list;
    };
};

void occ_free(Occluder & occ)
{
    if (occ.kind == 0) // trimesh
    {
        if (occ.mesh.owned && occ.mesh.verts != 0)
        {
            TINYSROC_FREE(occ.mesh.verts);
            TINYSROC_FREE(occ.mesh.verts_mm);
            TINYSROC_FREE(occ.mesh.indexes);
            occ.mesh.verts = 0;
            occ.mesh.verts_mm = 0;
            occ.mesh.indexes = 0;
        }
    }
    else if (occ.kind == 2 && occ.list.verts != 0) // triangle list
    {
        TINYSROC_FREE(occ.list.verts);
        TINYSROC_FREE(occ.list.verts_mm);
        occ.list.verts = 0;
        occ.list.verts_mm = 0;
    }
}

struct World;

namespace _tsroc_private {
    static void _write_depth(size_t i, float z, uint32_t * output);
    static void _rasterize_scanline(int pitch, int w, int h, size_t y, Vec3 x1, Vec3 x2, uint32_t * output,
        int32_t skip_xstart = 0, int32_t skip_xend = 0);
    
    static void _rasterize_poly_core(
        float start, float end,
        Vec3 center, Vec3 normal_1z,
        std::array<std::array<Vec2, 2>, 4> edges, int edge_count,
        uint32_t * output, size_t w, size_t h, size_t pitch
        );

    static void _rasterize_poly(
        Vec3 center, Vec3 normal_1z,
        std::array<std::array<Vec2, 2>, 4> edges, int edge_count,
        uint32_t * output, size_t w, size_t h, size_t pitch
        );
}

struct World {
    uint32_t lowres_w;
    uint32_t lowres_h;
    std::vector<uint32_t> hires;
    std::vector<uint32_t> lowres;
    
    Mat4 xform_proj = Mat4(1.0f);
    Mat4 xform_view = Mat4(1.0f);
    
private:
    uint32_t w;
    uint32_t h;
    
    std::unordered_map<uint64_t, Occluder> occluders;
    std::unordered_map<uint64_t, Occluder> occluders_disabled;
    
    std::vector<uint64_t> deleted_ids;
    uint64_t next_id = 0;
    uint64_t new_id()
    {
        if (deleted_ids.size() > 0)
        {
            uint64_t r = deleted_ids.back();
            deleted_ids.pop_back();
            return r;
        }
        return next_id++;
    }
    
    int lr_ratio = 8;
public:
    void set_lowres_ratio(int r)
    {
        if (r < 1) r = 1;
        if (r > 32) r = 32;
    }
    
    /// You should use this *only* for occluders that are extremely unlikely to ever let you see their "naked edges".
    /// For example, terrain chunks from an open world.
    /// The conservative nudge on these is normals-based, not edge-based.
    /// In particular the conservative nudge on this is designed for terrain-like heightmaps.
    uint64_t add_occluder_trimesh_owned(float * a_verts, float * a_normals, size_t vertex_count, uint32_t * a_indexes, size_t triangle_count)
    {
        Vec3 * verts = (Vec3 *)TINYSROC_MALLOC(vertex_count * sizeof(Vec3) * 2);
        Vec4 * verts_mm = (Vec4 *)TINYSROC_MALLOC(vertex_count * sizeof(Vec4));
        uint32_t * indexes = (uint32_t *)TINYSROC_MALLOC(triangle_count * 3 * sizeof(uint32_t));
        float inf = 1.0f/0.0f;
        Vec3 aabb_lo = Vec3(inf, inf, inf);
        Vec3 aabb_hi = Vec3(-inf, -inf, -inf);
        for (size_t i = 0; i < vertex_count; i += 1)
        {
            verts[i*2].x = a_verts[i*3];
            verts[i*2].y = a_verts[i*3 + 1];
            verts[i*2].z = a_verts[i*3 + 2];
            verts[i*2+1].x = a_normals[i*3];
            verts[i*2+1].y = a_normals[i*3 + 1];
            verts[i*2+1].z = a_normals[i*3 + 2];
            aabb_hi = max(aabb_hi, verts[i*2]);
            aabb_lo = min(aabb_lo, verts[i*2]);
        }
        for (size_t i = 0; i < triangle_count*3; i += 1)
            indexes[i] = a_indexes[i];
        
        Vec4 aabb_mid = Vec4((aabb_lo + aabb_hi)*0.5f, 1.0f);
        Vec3 aabb_ext = (aabb_hi - aabb_lo)*0.5f;
        auto id = new_id();
        Occluder occ;
        occ.kind = 0;
        occ.conservative = true;
        occ.backface_cull = true;
        occ.xform = Mat4(1.0f);
        occ.aabb_mid = aabb_mid;
        occ.aabb_ext = aabb_ext;
        occ.mesh.verts = verts;
        occ.mesh.verts_mm = verts_mm;
        occ.mesh.indexes = indexes;
        occ.mesh.triangle_count = triangle_count;
        occ.mesh.vertex_count = vertex_count;
        occ.mesh.owned = true;
        occluders.insert({id, occ});
        //std::cout << triangle_count << "\n";
        return id;
    }
    /// a_verts must point to four vertexes (12 floats, XYZ order) and they must be as close to rectangular and coplanar as possible.
    ///
    /// Verts must be in clockwise/fan order, not Z/strip order.
    uint64_t add_occluder_rectangle(float * verts)
    {
        auto id = new_id();
        Occluder occ;
        
        float inf = 1.0f/0.0f;
        Vec3 aabb_lo = Vec3(inf, inf, inf);
        Vec3 aabb_hi = Vec3(-inf, -inf, -inf);
        for (size_t i = 0; i < 4; i += 1)
        {
            occ.rect.verts[i].x = verts[i*3];
            occ.rect.verts[i].y = verts[i*3 + 1];
            occ.rect.verts[i].z = verts[i*3 + 2];
            aabb_hi = max(aabb_hi, occ.rect.verts[i].x);
            aabb_lo = min(aabb_lo, occ.rect.verts[i].x);
        }
        
        Vec4 aabb_mid = Vec4((aabb_lo + aabb_hi)*0.5f, 1.0f);
        Vec3 aabb_ext = (aabb_hi - aabb_lo)*0.5f;
        occ.kind = 1;
        occ.conservative = true;
        occ.backface_cull = false;
        occ.xform = Mat4(1.0f);
        occ.aabb_mid = aabb_mid;
        occ.aabb_ext = aabb_ext;
        occluders.insert({id, occ});
        
        return id;
    }
    /// List of triangles, e.g. 6 verts for 2 triangles. Not indexed.
    ///
    /// Meant for small numbers of triangles pulled from large geometry, NOT for entire 100+ poly meshes or small triangles.
    ///
    /// Each triangle is rasterized conservatively and touching triangles will have a gap.
    uint64_t add_occluder_triangles(float * a_verts, size_t triangle_count)
    {
        Vec3 * verts = (Vec3 *)TINYSROC_MALLOC(triangle_count * 3 * sizeof(Vec3));
        Vec4 * verts_mm = (Vec4 *)TINYSROC_MALLOC(triangle_count * 3 * sizeof(Vec4));
        float inf = 1.0f/0.0f;
        Vec3 aabb_lo = Vec3(inf, inf, inf);
        Vec3 aabb_hi = Vec3(-inf, -inf, -inf);
        for (size_t i = 0; i < triangle_count * 3; i += 1)
        {
            verts[i].x = a_verts[i*3];
            verts[i].y = a_verts[i*3 + 1];
            verts[i].z = a_verts[i*3 + 2];
            aabb_hi = max(aabb_hi, verts[i]);
            aabb_lo = min(aabb_lo, verts[i]);
        }
        
        Vec4 aabb_mid = Vec4((aabb_lo + aabb_hi)*0.5f, 1.0f);
        Vec3 aabb_ext = (aabb_hi - aabb_lo)*0.5f;
        auto id = new_id();
        
        Occluder occ;
        occ.kind = 2;
        occ.conservative = true;
        occ.backface_cull = true;
        occ.xform = Mat4(1.0f);
        occ.aabb_mid = aabb_mid;
        occ.aabb_ext = aabb_ext;
        occ.list.verts = verts;
        occ.list.verts_mm = verts_mm;
        occ.list.triangle_count = triangle_count;
        occluders.insert({id, occ});
        
        return id;
    }
    void occluder_enable(uint64_t id)
    {
        if (occluders_disabled.count(id))
        {
            auto occ = occluders_disabled[id];
            occluders_disabled.erase(id);
            occluders.insert({id, occ});
        }
    }
    void occluder_disable(uint64_t id)
    {
        if (occluders.count(id))
        {
            auto occ = occluders[id];
            occluders.erase(id);
            occluders_disabled.insert({id, occ});
        }
    }
    void occluder_remove(uint64_t id)
    {
        if (occluders.count(id))
        {
            auto & occ = occluders.at(id);
            occ_free(occ);
            occluders.erase(id);
            deleted_ids.push_back(id);
        }
        else if (occluders_disabled.count(id))
        {
            auto & occ = occluders_disabled.at(id);
            occ_free(occ);
            occluders_disabled.erase(id);
            deleted_ids.push_back(id);
        }
    }
    void occluder_set_xform(uint64_t id, Mat4 xform)
    {
        if (occluders.count(id))
        {
            auto & occ = occluders.at(id);
            occ.xform = xform;
        }
    }
    size_t occluder_tricount(uint64_t id)
    {
        if (occluders.count(id))
        {
            auto & occ = occluders.at(id);
            return occ.mesh.triangle_count;
        }
        if (occluders_disabled.count(id))
        {
            auto & occ = occluders_disabled.at(id);
            return occ.mesh.triangle_count;
        }
        return 0;
    }
    
    Vec3 occluder_get_center(uint64_t id)
    {
        if (occluders.count(id))
        {
            auto & occ = occluders.at(id);
            return occ.xform[3].xyz() + occ.aabb_mid.xyz();
        }
        if (occluders_disabled.count(id))
        {
            auto & occ = occluders_disabled.at(id);
            return occ.xform[3].xyz() + occ.aabb_mid.xyz();
        }
        return Vec3(0.0f);
    }
    
private:
    float _hr_float_at(int x, int y)
    {
        auto & p = hires[y * w + x];
        float f;
        memcpy(&f, &p, 4);
        return f;
    }
    bool occluder_rect_query_hires(float _x0, float _y0, float _x1, float _y1, float near_distance)
    {
        int x0 = _x0 * w;
        int x1 = ceil(_x1 * w);
        int y0 = _y0 * h;
        int y1 = ceil(_y1 * h);
        x0 = x0 < 0 ? 0 : x0 >= (int32_t)w ? w-1 : x0;
        x1 = x1 < 0 ? 0 : x1 >= (int32_t)w ? w-1 : x1;
        y0 = y0 < 0 ? 0 : y0 >= (int32_t)h ? h-1 : y0;
        y1 = y1 < 0 ? 0 : y1 >= (int32_t)h ? h-1 : y1;
        if (x0 > x1) x0 = x1;
        if (y0 > y1) y0 = y1;
        
        float inv_near_distance = 1.0 / near_distance;
        
        for (size_t y = y0; y <= (size_t)y1; y++)
        {
            for (size_t x = x0; x <= (size_t)x1; x++)
            {
                if (inv_near_distance >= _hr_float_at(x, y))
                    return false;
            }
        }
        return true;
    }
    
    float _lr_float_at(int x, int y)
    {
        auto & p = lowres[y * lowres_w + x];
        float f;
        memcpy(&f, &p, 4);
        return f;
    }
    
public:
    bool _occluder_rect_query_allow_hr_fallback = true;
    /// Returns true if 2d AABB is fully occluded. AABB values are normalized from 0.0 to 1.0 across screen bounds.
    ///
    /// May return false even if the AABB is technically occluded.
    ///
    /// near_distance must be in forwards (not radial) world units from the origin of the camera (not the near plane).
    bool occluder_rect_query(float _x0, float _y0, float _x1, float _y1, float near_distance)
    {
        _x0 = clamp(_x0, 0.0f, 1.0f);
        _x1 = clamp(_x1, 0.0f, 1.0f);
        _y0 = clamp(_y0, 0.0f, 1.0f);
        _y1 = clamp(_y1, 0.0f, 1.0f);
        if (_occluder_rect_query_allow_hr_fallback && hires_owned)
        {
            int hrw = int(_x1 * w + 1.0f) - int(_x0 * w);
            int hrh = int(_y1 * h + 1.0f) - int(_y0 * h);
            if (hrw * hrh <= 16*16)
                return occluder_rect_query_hires(_x0, _y0, _x1, _y1, near_distance);
        }
        
        int x0 = _x0 * lowres_w;
        int x1 = ceil(_x1 * lowres_w);
        int y0 = _y0 * lowres_h;
        int y1 = ceil(_y1 * lowres_h);
        x0 = x0 < 0 ? 0 : x0 >= (int32_t)lowres_w ? lowres_w-1 : x0;
        x1 = x1 < 0 ? 0 : x1 >= (int32_t)lowres_w ? lowres_w-1 : x1;
        y0 = y0 < 0 ? 0 : y0 >= (int32_t)lowres_h ? lowres_h-1 : y0;
        y1 = y1 < 0 ? 0 : y1 >= (int32_t)lowres_h ? lowres_h-1 : y1;
        if (x0 > x1) x0 = x1;
        if (y0 > y1) y0 = y1;
        
        float inv_near_distance = 1.0 / near_distance;
        
        for (size_t y = y0; y <= (size_t)y1; y++)
        {
            for (size_t x = x0; x <= (size_t)x1; x++)
            {
                if (inv_near_distance >= _lr_float_at(x, y))
                    return false;
            }
        }
        return true;
    }
    
    bool occluder_aabb_query(Vec3 lo, Vec3 hi, Mat4 xform)
    {
        bool any_in_front = false;
        float inf = 1.0f/0.0f;
        Vec2 lo2 = Vec2(inf, inf);
        Vec2 hi2 = Vec2(-inf, -inf);
        float lo_depth = inf;
        auto mvp = camera * xform;
        for (int i = 0; i < 8; i++)
        {
            Vec4 vert = Vec4(((i&1) ? lo : hi).x, ((i&2) ? lo : hi).y, ((i&4) ? lo : hi).z, 1.0f);
            vert = mvp * vert;
            if (vert.w > 1e-37f)
                any_in_front = true;
            else
                vert.w = 1e-37f;
            
            float loz = vert.w;
            lo_depth = min(lo_depth, loz);
            
            vert *= 1.0f / vert.w;
            
            vert.x = vert.x * 0.5f + 0.5f;
            vert.y = vert.y * 0.5f + 0.5f;
            vert.y = 1.0f - vert.y;
            
            lo2 = min(lo2, vert.xy());
            hi2 = max(hi2, vert.xy());
        }
        
        if (!any_in_front) return false;
        
        return occluder_rect_query(lo2.x, lo2.y, hi2.x, hi2.y, lo_depth);
    }
    
    ~World()
    {
        for (auto & _occ : occluders)
        {
            auto & occ = _occ.second;
            occ_free(occ);
        }
    }
    
    void update_lowres(uint32_t * hr_output, int w, int h, int hr_pitch)
    {
        (void)w;
        std::fill(lowres.begin(), lowres.end(), ~0);
        for (int y = 0; y < h; y += 1)
        {
            auto yi2 = hr_pitch * y;
            auto yi = lowres_w * (y / lr_ratio);
            for (int x : TINYSROC_RANGE(0, (ptrdiff_t)lowres_w))
            {
                auto i = yi + x;
                auto i2 = yi2 + x * lr_ratio;
                
                auto n = min(lowres[i], hr_output[i2]);
                for (int xd : TINYSROC_RANGE(0, lr_ratio))
                    n = min(n, hr_output[i2 + xd]);
                lowres[i] = n;
            }
        }
    };
    
private:
    Mat4 proj = Mat4(1.0f);
    Mat4 view = Mat4(1.0f);
    Mat4 camera = Mat4(1.0f);
    float near = 0.01f;
    float far = 1000.0f;
    
public:
    /// Sets up camera matrices.
    ///
    /// Projection matrix must follow the coordinate system used by glm::perspectiveLH_ZO.
    void set_perspective_camera(Mat4 projection, Mat4 view_pos_and_orientation)
    {
        proj = projection;
        view = view_pos_and_orientation;
        camera = proj * view;
        
        float m22 = proj[2][2];
        float m32 = proj[3][2];

        near = -m32 / m22;
        far  = m32 / (1.0f - m22);
    }
    
private:
    std::vector<Occluder *> occ_sorted;
    bool hires_owned = false;
    
    
public:
    
    /// Rasterize the world to the given output buffer instead of to the `hires` field.
    ///
    /// The output is NOT cleared before rasterizing.
    ///
    /// Rounds dimensions down to a multiple of 16. If too small, renders nothing.
    ///
    /// Respects output pitch despite rounding dimensions.
    ///
    /// Also updates the projection matrix.
    ///
    /// Output contains reciprocal depth f32s, type-punned as u32.
    ///
    /// After running, this->lowres contains an 8x8 min-pooled downres'd version of output. Because of
    ///  the reciprocal format, this means the furthest pixel of each 8x8 block is chosen to represent
    ///  the entire 8x8 block, not the nearest, despite being a min-pool.
    void rasterize_to(int & w, int & h, int pitch, uint32_t * output, uint32_t framenum)
    {
        hires_owned = false;
        (void)framenum;
        
        w = w & ~15;
        h = h & ~15;
        if (w == 0 || h == 0) return;
        
        auto farnearval = (far - near) / far;
        
        auto xnudge = (3.5f / (float)w);
        auto ynudge = (3.5f / (float)h);
        
        auto cm = camera;
        
        // frustum planes
        auto fm_l = Vec4(cm[0][3]+cm[0][0], cm[1][3]+cm[1][0], cm[2][3]+cm[2][0], cm[3][3]+cm[3][0]);
        auto fm_r = Vec4(cm[0][3]-cm[0][0], cm[1][3]-cm[1][0], cm[2][3]-cm[2][0], cm[3][3]-cm[3][0]);
        auto fm_b = Vec4(cm[0][3]+cm[0][1], cm[1][3]+cm[1][1], cm[2][3]+cm[2][1], cm[3][3]+cm[3][1]);
        auto fm_u = Vec4(cm[0][3]-cm[0][1], cm[1][3]-cm[1][1], cm[2][3]-cm[2][1], cm[3][3]-cm[3][1]);
        auto fm_n = Vec4(cm[0][2], cm[1][2], cm[2][2], cm[3][2]);
        auto fm_f = Vec4(cm[0][3]-cm[0][2], cm[1][3]-cm[1][2], cm[2][3]-cm[2][2], cm[3][3]-cm[3][2]);
        // we need normalized plane normals for the radius hack to work properly
        fm_l /= max(1e-30f, length(fm_l.xyz()));
        fm_r /= max(1e-30f, length(fm_r.xyz()));
        fm_b /= max(1e-30f, length(fm_b.xyz()));
        fm_u /= max(1e-30f, length(fm_u.xyz()));
        fm_n /= max(1e-30f, length(fm_n.xyz()));
        fm_f /= max(1e-30f, length(fm_f.xyz()));
        
        Vec4 planes[6] = { fm_l, fm_r, fm_b, fm_u, fm_n, fm_f };
        
        occ_sorted.clear();
        for (auto & _occ : occluders)
        {
            auto & occ = _occ.second;
            
            auto is_outside = [&]() {
                for (int i = 0; i < 6; i++)
                {
                    Vec4 plane = planes[i];
                    plane = transpose(occ.xform) * plane;
                    auto r = dot(abs(plane.xyz()), occ.aabb_ext);
                    auto m = dot(plane, occ.aabb_mid);
                    if (m + r < 0.0f)
                        return true;
                }
                return false;
            };
            if (is_outside()) continue;
            
            occ_sorted.push_back(&occ);
        }
        
        std::sort(occ_sorted.begin(), occ_sorted.end(), [&](Occluder * a, Occluder * b) {
            Vec4 q1 = view * (a->aabb_mid + Vec4(a->xform[3].xyz(), 0.0f));
            Vec4 q2 = view * (b->aabb_mid + Vec4(b->xform[3].xyz(), 0.0f));
            return q1.z > q2.z;
        });
        
        auto & lowres = this->lowres;
        
        auto lr_w = max(1, w/lr_ratio);
        auto lr_h = max(1, (h + (lr_ratio-1))/lr_ratio);
        lowres.resize(lr_w * lr_h);
        std::fill(lowres.begin(), lowres.end(), ~0);
        lowres_w = lr_w;
        lowres_h = lr_h;
        
        bool lowres_updated = false;
        
        size_t _tri_n = 0;
        
        // near-clip distance
        //float clipdist = 0.0f;
        float clipdist = 1e-20f;
        //float clipdist = near;
        
        for (auto & _occ : occ_sorted)
        {
            auto & occ = *_occ;
            
            if (lowres_updated && occluder_aabb_query(occ.aabb_mid.xyz() - occ.aabb_ext, occ.aabb_mid.xyz() + occ.aabb_ext, occ.xform))
                continue;
            
            auto process_poly = [&](
                Vec4 * verts, int verts_n,
                bool backface_cull,
                bool * fudges)
            {
                for (int i = 0; i < verts_n; i++)
                    { verts[i].x *= 1.0f / verts[i].w; verts[i].y *= 1.0f / verts[i].w; }
                
                if (backface_cull)
                {
                    auto d_b = verts[1].xy() - verts[0].xy();
                    auto d_c = verts[2].xy() - verts[0].xy();
                    auto det = d_b.x * d_c.y - d_b.y * d_c.x;
                    //if (det <= 0.0f)
                    if (det <= 1e-30f) // numeric stability hack
                        return;
                }
                
                // convert clipspace z to worldspace 1/z for linear interpolation to be worldspace linear
                for (int i = 0; i < verts_n; i++)
                    verts[i].z = 1.0f / (verts[i].z * farnearval + near);
                
                float miny = 1.0f;
                float maxy = -1.0f;
                float minx = 1.0f;
                float maxx = -1.0f;
                for (int i = 0; i < verts_n; i++)
                {
                    miny = min(miny, verts[i].y);
                    maxy = max(maxy, verts[i].y);
                    minx = min(minx, verts[i].x);
                    maxx = max(maxx, verts[i].x);
                }
                
                // screen edge culling
                if (miny >= 1.0f || maxy <= -1.0f) { return; }
                if (minx >= 1.0f || maxx <= -1.0f) { return; }
                
                // convert clip/ndc space to screen space
                for (int i = 0; i < verts_n; i++)
                {
                    verts[i].x = (verts[i].x * 0.5f + 0.5f) * (float)w;
                    verts[i].y = (1.0 - (verts[i].y * 0.5f + 0.5f)) * float(h);
                }
                
                size_t tri_n_threshold = 512;
                _tri_n += 1;
                if (_tri_n == tri_n_threshold || _tri_n == tri_n_threshold * 4)
                {
                    update_lowres(output, w, h, pitch);
                    lowres_updated = true;
                }
                
                Vec3 normal = cross((verts[0]-verts[1]).xyz(), (verts[0]-verts[2]).xyz());
                if (normal.z < 0.0f)
                {
                    std::reverse(verts+1, verts+verts_n);
                    std::reverse(fudges, fudges+verts_n);
                }
                
                int edges_n = verts_n > 4 ? 4 : verts_n;
                
                std::array<std::array<Vec2, 2>, 4> edges;
                for (int i = 0; i < edges_n; i++)
                    edges[i] = {verts[i], verts[(i+1)%verts_n]};
                
                for (int i = 0; i < edges_n; i++)
                {
                    if (!fudges[i]) continue;
                    Vec2 x2 = edges[i][0]-edges[i][1];
                    x2 = Vec2(-x2.y, x2.x);
                    x2 /= max(abs(x2.x), abs(x2.y));
                    edges[i][0] -= x2*0.72f;
                    edges[i][1] -= x2*0.72f;
                }
                
                _tsroc_private::_rasterize_poly(verts[0].xyz(), normal, edges, edges_n, output, w, h, pitch);
            };
            
            auto mvp = camera * occ.xform;
            auto xlate = translate(Mat4(1.0f), vec3(1.0f/w, 1.0f/h, 0.0f));
            mvp = xlate * mvp;
            
            auto handle_triangle = [&](Vec4 a, Vec4 b, Vec4 c, bool conservative)
            {
                // We need to be able to cut up triangles that intersect the view frustum.
                auto apply_2cut = [&](auto ref, auto s, auto t)
                {
                    auto s_v = s - ref;
                    auto t_v = t - ref;
                    auto s_t = (ref.z - clipdist) / -s_v.z;
                    auto t_t = (ref.z - clipdist) / -t_v.z;
                    s = ref + s_v * s_t;
                    t = ref + t_v * t_t;
                    
                    std::array<Vec4, 3> v = {ref, s, t};
                    std::array<bool, 3> b = {conservative, false, conservative};
                    
                    process_poly(&v[0], 3, occ.backface_cull, &b[0]);
                };
                auto apply_1cut = [&](auto ref1, auto ref2, auto t)
                {
                    auto tv1 = t - ref1;
                    auto tv2 = t - ref2;
                    auto t_t1 = (ref1.z - clipdist) / -tv1.z;
                    auto t_t2 = (ref2.z - clipdist) / -tv2.z;
                    auto t1a = ref1 + tv1 * t_t1;
                    auto t2a = ref2 + tv2 * t_t2;
                    
                    std::array<Vec4, 4> v = {ref1, ref2, t2a, t1a};
                    std::array<bool, 4> b = {conservative, conservative, false, conservative};
                    
                    process_poly(&v[0], 4, occ.backface_cull, &b[0]);
                };
                
                // triangle is fully behind camera
                if (a.z <= clipdist && b.z <= clipdist && c.z <= clipdist) return;
                
                // triangle is entirely in front of camera
                if (a.z > clipdist && b.z > clipdist && c.z > clipdist)
                {
                    std::array<Vec4, 3> v = {a, b, c};
                    std::array<bool, 3> b = {conservative, conservative, conservative};
                    
                    process_poly(&v[0], 3, occ.backface_cull, &b[0]);
                }
                // various amounts of partial cut
                else if (a.z <= clipdist && b.z <= clipdist)
                    apply_2cut(c, a, b);
                else if (a.z <= clipdist && c.z <= clipdist)
                    apply_2cut(b, c, a);
                else if (b.z <= clipdist && c.z <= clipdist)
                    apply_2cut(a, b, c);
                else if (a.z <= clipdist)
                    apply_1cut(b, c, a);
                else if (b.z <= clipdist)
                    apply_1cut(c, a, b);
                else if (c.z <= clipdist)
                    apply_1cut(a, b, c);
            };
            
            auto handle_quad = [&](Vec4 * v, bool conservative)
            {
                // triangle is fully behind camera
                if (v[0].z <= clipdist && v[1].z <= clipdist && v[2].z <= clipdist && v[3].z <= clipdist) return;
                
                // triangle is entirely in front of camera
                if (v[0].z > clipdist && v[1].z > clipdist && v[2].z > clipdist && v[3].z > clipdist)
                {
                    std::array<bool, 4> b = {conservative, conservative, conservative, conservative};
                    process_poly(v, 4, occ.backface_cull, &b[0]);
                }
                
                int n_notvis = 0;
                int notvis_to_vis_idx = -1;
                
                for (int i = 0; i < 4; i++)
                {
                    auto a = v[i];
                    auto next = v[(i+1)%4];
                    if (a.z <= clipdist)
                    {
                        n_notvis += 1;
                        if (notvis_to_vis_idx == -1 && next.z > clipdist)
                            notvis_to_vis_idx = i;
                    }
                }
                if (notvis_to_vis_idx != 3)
                    std::rotate(v, v+(notvis_to_vis_idx+1)%4, v+4);
                
                if (n_notvis == 1)
                {
                    std::array<Vec4, 5> v2;
                    v2[1] = v[0];
                    v2[2] = v[1];
                    v2[3] = v[2];
                    
                    auto d0 = v[3] - v[0];
                    auto d2 = v[3] - v[2];
                    
                    auto t0 = (v[0].z - clipdist) / -d0.z;
                    auto t2 = (v[2].z - clipdist) / -d2.z;
                    
                    auto p0 = v[0] + d0*t0;
                    auto p2 = v[2] + d2*t2;
                    
                    v2[0] = p0;
                    v2[4] = p2;
                    
                    std::array<bool, 5> f2 = {true, true, true, true, false};
                    
                    process_poly(&v2[0], 5, occ.backface_cull, &f2[0]);
                }
                else if (n_notvis == 2)
                {
                    std::array<Vec4, 4> v2;
                    v2[1] = v[0];
                    v2[2] = v[1];
                    
                    auto d0 = v[3] - v[0];
                    auto d2 = v[2] - v[1];
                    
                    auto t0 = (v[0].z - clipdist) / -d0.z;
                    auto t2 = (v[1].z - clipdist) / -d2.z;
                    
                    auto p0 = v[0] + d0*t0;
                    auto p2 = v[1] + d2*t2;
                    
                    v2[0] = p0;
                    v2[3] = p2;
                    
                    std::array<bool, 4> f2 = {true, true, true, false};
                    
                    process_poly(&v2[0], 4, occ.backface_cull, &f2[0]);
                }
                else if (n_notvis == 3)
                {
                    std::array<Vec4, 3> v2;
                    v2[1] = v[0];
                    
                    auto d0 = v[3] - v[0];
                    auto d2 = v[1] - v[0];
                    
                    auto t0 = (v[0].z - clipdist) / -d0.z;
                    auto t2 = (v[0].z - clipdist) / -d2.z;
                    
                    auto p0 = v[0] + d0*t0;
                    auto p2 = v[0] + d2*t2;
                    
                    v2[0] = p0;
                    v2[2] = p2;
                    
                    std::array<bool, 3> f2 = {true, true, false};
                    
                    process_poly(&v2[0], 3, occ.backface_cull, &f2[0]);
                }
            };
            
            if (occ.kind == 0) // trimeshes
            {
                Vec3 * verts = occ.mesh.verts;
                Vec4 * verts_mm = occ.mesh.verts_mm;
                for (size_t j = 0; j < occ.mesh.vertex_count; j++)
                {
                    Vec3 pos = verts[j*2];
                    Vec3 norm = verts[j*2 + 1];
                    
                    auto a = mvp * Vec4(pos, 1.0f);
                    auto n_a = Mat3(mvp) * norm;
                    
                    // conservative nudge to shrink trimesh verts inwards by a couple pixels
                    a.x -= n_a.x * xnudge * abs(a.z);
                    a.y -= n_a.y * ynudge * abs(a.z);
                    
                    verts_mm[j] = a;
                }
                for (size_t j = 0; j < occ.mesh.triangle_count; j++)
                {
                    uint32_t * indexes = occ.mesh.indexes + j*3;
                    
                    Vec4 tri0 = verts_mm[indexes[0]];
                    Vec4 tri1 = verts_mm[indexes[1]];
                    Vec4 tri2 = verts_mm[indexes[2]];
                    
                    handle_triangle(tri0, tri1, tri2, false);
                }
            }
            else if (occ.kind == 1) // rects
            {
                Vec4 verts_mm[4];
                Vec3 * verts = occ.rect.verts;
                for (size_t j = 0; j < 4; j++)
                    verts_mm[j] = mvp * Vec4(verts[j], 1.0f);
                handle_quad(verts_mm, true);
            }
            else if (occ.kind == 2) // rects
            {
                Vec3 * verts = occ.list.verts;
                Vec4 * verts_mm = occ.list.verts_mm;
                for (size_t j = 0; j < occ.list.triangle_count * 3; j++)
                    verts_mm[j] = mvp * Vec4(verts[j], 1.0f);
                for (size_t j = 0; j < occ.list.triangle_count; j++)
                {
                    Vec4 tri0 = verts_mm[j*3 + 0];
                    Vec4 tri1 = verts_mm[j*3 + 1];
                    Vec4 tri2 = verts_mm[j*3 + 2];
                    
                    handle_triangle(tri0, tri1, tri2, true);
                }
            }
        }
        update_lowres(output, w, h, pitch);
    }
    /// Rasterize the world to the `hires` field. See rasterize_to for behavioral details.
    ///
    /// The output IS CLEARED before rasterizing.
    void rasterize(int & w, int & h, int pitch, uint32_t framenum)
    {
        w = w & ~15;
        h = h & ~15;
        if (w == 0 || h == 0) return;
        
        this->w = w;
        this->h = h;
        hires.resize(w * h);
        hires.assign(w * h, 0);
        
        rasterize_to(w, h, pitch, hires.data(), framenum);
        hires_owned = true;
        
        //auto & lowres = this->lowres;
        //auto lr_w = max(1, w/lr_ratio);
        //auto lr_h = max(1, (h + (lr_ratio-1))/lr_ratio);
        //lowres.resize(lr_w * lr_h, ~0);
        //lowres_w = lr_w;
        //lowres_h = lr_h;
        //update_lowres(hires.data(), w, h, pitch);
    }
};

namespace _tsroc_private {
    static void _write_depth(size_t i, float z, uint32_t * output) {
        int32_t n;
        memcpy(&n, &z, 4);
        int32_t ref;
        memcpy(&ref, &output[i], 4);
        ref = max(ref, n);
        memcpy(&output[i], &ref, 4);
    };

    static void _rasterize_scanline(int pitch, int w, int h, size_t y, Vec3 x1, Vec3 x2, uint32_t * output,
        int32_t skip_xstart, int32_t skip_xend
    ) {
        if (y >= (size_t)h) return;
        auto xd = 1.0f / (x2.x - x1.x);
        
        uint32_t start = clamp(((int32_t)x1.x) + skip_xstart, (int32_t)0, (int32_t)w);
        uint32_t end   = clamp(((int32_t)x2.x) - skip_xend, (int32_t)0, (int32_t)w);
        uint32_t count = end - start;
        if ((int32_t)count <= 0) return;
        
        uint32_t _x = 0;
        
        // TODO: figure out which one of these is FOR SURE correct for our pixel centering.
        // The 0.5f version caused phantom negative-depth pixels in rare situations.
        auto xtb = (float)start + 1.0f - x1.x;
        //auto xtb = (float)start + 0.5f - x1.x;
        
        auto xtbf = (x2 - x1) * xd;
        if (xtbf.z != xtbf.z) return; // locally force non-NaN to encourage better optimizations
        auto xtbfz = xtbf.z;
        auto xbase = x1.z + xtbfz * xtb;
        
        size_t xofft = _x + start + y*pitch;
        size_t xoffend = xofft + count;
        
        while (xofft < xoffend)
        {
            _write_depth(xofft, xbase, output);
            xofft += 1;
            xbase += xtbfz;
        }
    };

    static void _rasterize_poly_core(
        float start, float end,
        Vec3 center, Vec3 normal_1z,
        std::array<std::array<Vec2, 2>, 4> edges, int edge_count,
        uint32_t * output, size_t w, size_t h, size_t pitch
        )
    {
        auto get_z = [&] (float x, float y) {
            float xd = x - center.x;
            float yd = y - center.y;
            
            float z = normal_1z.x*xd + normal_1z.y*yd;
            
            return center.z - z;
        };
        
        auto y_to_x = [&](Vec2 lo, Vec2 hi, float y)
        {
            Vec2 vd = hi - lo;
            float yd = y - lo.y;
            float t = yd / vd.y;
            return lo.x + vd.x * t;
        };
        auto get_l = [&](float y)
        {
            float x = 0.0f;
            for (int i = 0; i < edge_count; i++)
            {
                float dy = edges[i][1].y - edges[i][0].y;
                if (dy < 0.0f)
                    x = max(x, y_to_x(edges[i][0], edges[i][1], y));
            }
            return x;
        };
        auto get_r = [&](float y)
        {
            float x = w;
            for (int i = 0; i < edge_count; i++)
            {
                float dy = edges[i][1].y - edges[i][0].y;
                if (dy > 0.0f)
                    x = min(x, y_to_x(edges[i][0], edges[i][1], y));
            }
            return x;
        };
        
        for (float y = start; y <= end; y++)
        {
            auto l = Vec3(get_l(y), y, 0.0f);
            auto r = Vec3(get_r(y), y, 0.0f);
            
            l.z = get_z(l.x, l.y);
            r.z = get_z(r.x, r.y);
            
            _rasterize_scanline(pitch, w, h, y, l, r, output);
        }
    }
    static void _rasterize_poly(
        Vec3 center, Vec3 normal_1z,
        std::array<std::array<Vec2, 2>, 4> edges, int edge_count,
        uint32_t * output, size_t w, size_t h, size_t pitch)
    {
        // degenerate
        if (normal_1z.z == 0.0f) return;
        normal_1z /= normal_1z.z;
        
        auto ystart = (float)h;
        auto yend = 0.0f;
        
        for (int i = 0; i < edge_count; i++)
        {
            ystart = min(ystart, edges[i][0].y);
            ystart = min(ystart, edges[i][1].y);
            yend = max(yend, edges[i][0].y);
            yend = max(yend, edges[i][1].y);
        }
        for (int i = 0; i < edge_count; i++)
        {
            if (edges[i][0].y == edges[i][1].y &&
                edges[i][0].x < edges[i][1].x)
                ystart = max(ystart, edges[i][0].y);
            if (edges[i][0].y == edges[i][1].y &&
                edges[i][0].x > edges[i][1].x)
                yend = min(yend, edges[i][0].y);
        }
        
        ystart = (uint32_t)clamp(ceil(ystart), 0.0f, (float)h);
        yend = (uint32_t)clamp(floor(yend), 0.0f, (float)h);
        
        _rasterize_poly_core(
            ystart, yend,
            center, normal_1z, edges, edge_count,
            output, w, h, pitch
        );
    };
}

#ifdef TINYSROC_NAMESPACED
}
#endif
