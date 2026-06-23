# TinySROC

TinySROC is a tiny software-rasterized occlusion culling library, header-only, for C++17 and later.

## Compile the demo

Go to `demo/` in a terminal and run:

C++17 `clang++ -std=c++17 -O3 -g -ggdb demo2.cpp -static $(pkg-config sdl3 sdl3-image --static -libs) -Wall -Wextra -Wfatal-errors -lfmt`

C++20 `clang++ -std=c++20 -O3 -g -ggdb demo2.cpp -static $(pkg-config sdl3 sdl3-image --static -libs) -Wall -Wextra -Wfatal-errors`

## API

Include `<tinysroc.hpp>` in a SINGLE source file in your project, make a World object, and use its API:

```c++
//////////////
// World setup
//////////////

    /// You should use this *only* for occluders that are extremely unlikely to ever let you see their "naked edges".
    /// For example, terrain chunks from an open world.
    /// The conservative nudge on these is normals-based, not edge-based.
    /// In particular the conservative nudge on this is designed for terrain-like heightmaps.
    uint64_t add_occluder_trimesh_owned(float * a_verts, float * a_normals, size_t vertex_count, uint32_t * a_indexes, size_t triangle_count)
    
    /// a_verts must point to four vertexes (12 floats, XYZ order) and they must be as close to rectangular and coplanar as possible.
    ///
    /// Verts must be in clockwise/fan order, not Z/strip order.
    uint64_t add_occluder_rectangle(float * verts)
    
    /// List of triangles, e.g. 6 verts for 2 triangles. Not indexed.
    ///
    /// Meant for small numbers of triangles pulled from large geometry, NOT for entire 100+ poly meshes or small triangles.
    ///
    /// Each triangle is rasterized conservatively and touching triangles will have a gap.
    uint64_t add_occluder_triangles(float * a_verts, size_t triangle_count)
    
//////////////
// World management
//////////////

    void occluder_enable(uint64_t id)
    void occluder_disable(uint64_t id)
    void occluder_remove(uint64_t id)
    void occluder_set_xform(uint64_t id, Mat4 xform)

//////////////
// Main API
//////////////

    /// Sets up camera matrices.
    ///
    /// Projection matrix must follow the coordinate system used by glm::perspectiveLH_ZO.
    void set_perspective_camera(Mat4 projection, Mat4 view_pos_and_orientation)
    
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
    
    /// Rasterize the world to the `hires` field. See rasterize_to for behavioral details.
    ///
    /// The output IS CLEARED before rasterizing.
    void rasterize(int & w, int & h, int pitch, uint32_t framenum)
    
//////////////
// Query 
//////////////

    /// Returns true if 2d AABB is fully occluded. AABB values are normalized from 0.0 to 1.0 across screen bounds.
    ///
    /// May return false even if the AABB is technically occluded.
    ///
    /// near_distance must be in forwards (not radial) world units from the origin of the camera (not the near plane).
    
    bool query_rect(float _x0, float _y0, float _x1, float _y1, float near_distance)
    bool query_aabb(Vec3 lo, Vec3 hi, Mat4 xform)
    
//////////////
// Feedback/debug
//////////////

    size_t occluder_tricount(uint64_t id)
    Vec3 occluder_get_center(uint64_t id)
```

## Design

TinySROC works by performing "conservative rasterization", and then letting you make rect or AABB queries against the rasterized depth buffer. Sphere queries will probably be supported in the future.

It also makes a downscaled copy of the depth buffer to use when querying very large rects or AABBs.

It supports trimesh (terrain), rectangle, and loose triangle occluders. It has a retained worldstate API, where you create occluders with given mesh data, and then later destroy them manually when you want to clean them up. TinySROC handles culling, transforming, and conservative-izing them internally on its own.

Enabling and disabling occluders without destroying them is low-cost, but you should try to limit the number of occluders that exist in TinySROC's world at once to several thousand, even if most of them are disabled. (Yes, several thousand is fine.) Very high numbers of occluders (e.g. millions) are supported, but will probably see degraded performance.

## Performance & Scaling to worse hardware

TinySROC is decently fast.

TinySROC only over rasterizes at the resolution you tell it to, and it handles the changes to conservativeness at various resolutions correctly. Asking for a 256x128 depth buffer works well for most hardware, but 64x32 might be more appropriate for very weak hardware (e.g. old phones). Reducing the resolutino reduces the rasterization cost, but not the setup and triangle transformation cost. Reducing the number of active and live occluders, and putting less data into them, should also be investigated for low-spec hardware.

Even on strong hardware, trimesh occluders (meant for terrain) shouldn't be fed to TinySROC at the original mesh detail level. You should subsample it to 1/4 or 1/16 the triangle count (1/2 or 1/4 the dimensions) with a min-pool or some other conservative terrain subsampling method, and make sure each terrain occluder has in the range of 500~2000 triangles (smaller is usually better), not 4000+.

## Copyright

Licensed under the MIT license.

Not AI slop.
