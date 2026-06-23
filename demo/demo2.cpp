#define SDL_MAIN_HANDLED
#include <SDL3/SDL.h>
#include <SDL3_image/SDL_image.h>
#include <vector>
#include <iostream>
#include <format>
#include <ranges>
#include <thread>
#include <memory>
#include <cstdio>
#include <functional>
#include <string>
#include <sstream>
#include <charconv>
#include <algorithm>

#include "glad.h"
#include "glad.c"

#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>
#include <glm/ext.hpp>

using namespace glm;

#define TINYSROC_NAMESPACED
#include "../tinysroc.hpp"

#if __cplusplus < 202002L
#include <fmt/core.h>
#endif

const char * shader_vert = R"xxx(
    #version 330 core
    in vec3 vertPos;
    in vec3 vertNorm;
    
    uniform mat4 matVP;
    uniform mat4 matM;
    
    out vec3 pos;
    out vec3 normal;
    
    void main()
    {
        gl_Position = (matVP * matM) * vec4(vertPos, 1.0);
        pos = vertPos + matM[3].xyz;
        normal = vertNorm;
    }
)xxx";

const char * shader_frag = R"xxx(
    #version 330 core
    out vec4 fragColor;
    
    uniform int normIsUV;
    
    in vec3 pos;
    in vec3 normal;
    
    uniform sampler2D tx;
    uniform float opacity;
    void main()
    {
        if (normIsUV != 0)
        {
            fragColor = texture(tx, vec2(normal.xy)) * opacity;
            return;
        }
        vec3 s = normalize(vec3(0.5, 1.0, 0.5));
        float l = dot(s, normalize(normal));
        vec2 uv = pos.xz;
        if (abs(normal.y) < 0.5)
        {
            if (abs(normal.x) > 0.7) uv = pos.zy;
            else uv = pos.xy;
        }
        vec4 tx1 = texture(tx, vec2(uv));
        vec4 tx2 = texture(tx, vec2(uv/8.0f));
        vec4 tx3 = texture(tx, vec2(uv/64.0f));
        vec4 m = textureLod(tx, vec2(0.0), 1000.0);
        vec4 outc = tx1+tx2*0.5+tx3*0.5-m;
        outc = 1.0/(1.0+exp((0.5-outc)*4.0));
        fragColor = outc * (clamp(l, 0.0, 1.0) + 0.2f);
        fragColor.a = 1.0 * opacity;
        //fragColor.rgb = normal.xyz * 0.5 + 0.5;
    }
)xxx";

struct RenderableMesh {
    std::vector<vec3> verts;
    std::vector<vec3> normals;
    std::vector<uint32_t> indexes;
    unsigned int VBO = 0;
    unsigned int VBO2 = 0;
    unsigned int EBO = 0;
    unsigned int VAO = 0;
    
    uint32_t texture = 0;
    
    vec4 aabb_mid = vec4(0.0f);
    vec3 aabb_ext = vec3(0.0f);
    
    RenderableMesh(
        unsigned int loc,
        unsigned int loc2,
        std::vector<vec3> _verts,
        std::vector<vec3> _normals,
        std::vector<uint32_t> _indexes,
        uint32_t _texture)
    {
        verts = _verts;
        normals = _normals;
        indexes = _indexes;
        texture = _texture;
        
        vec3 aabb_lo = vec3(1.0f/0.0f, 1.0f/0.0f, 1.0f/0.0f);
        vec3 aabb_hi = -vec3(1.0f/0.0f, 1.0f/0.0f, 1.0f/0.0f);
        
        for (auto & v : verts)
        {
            aabb_lo = min(aabb_lo, v);
            aabb_hi = max(aabb_hi, v);
        }
        
        aabb_mid = vec4((aabb_lo + aabb_hi) * 0.5f, 1.0f);
        aabb_ext = (aabb_hi - aabb_lo) * 0.5f;
        
        glGenVertexArrays(1, &VAO);
        glBindVertexArray(VAO);
        
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &VBO2);
        glGenBuffers(1, &EBO);
        
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(verts[0]), verts.data(), GL_STATIC_DRAW);
        
        glVertexAttribPointer(loc, 3, GL_FLOAT, false, 0, 0);
        glEnableVertexAttribArray(loc);
        
        glBindBuffer(GL_ARRAY_BUFFER, VBO2);
        glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(normals[0]), normals.data(), GL_STATIC_DRAW);
        
        glVertexAttribPointer(loc2, 3, GL_FLOAT, false, 0, 0);
        glEnableVertexAttribArray(loc2);
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexes.size() * sizeof(indexes[0]), indexes.data(), GL_STATIC_DRAW);
        
        glBindVertexArray(0);
    }
    void render()
    {
        glBindVertexArray(VAO);
        glBindTexture(GL_TEXTURE_2D, texture);
        glDrawElements(GL_TRIANGLES, indexes.size(), GL_UNSIGNED_INT, 0);
    }
    ~RenderableMesh()
    {
        glDeleteBuffers(1, &VBO);
        glDeleteBuffers(1, &VBO2);
        glDeleteBuffers(1, &EBO);
        glDeleteVertexArrays(1, &VAO);
    }
};

struct Renderable {
    std::shared_ptr<RenderableMesh> mesh;
    mat4 xform = mat4(1.0f);
    Renderable(std::shared_ptr<RenderableMesh> _mesh)
    {
        mesh = _mesh;
    }
    void render(uint32_t u)
    {
        glUniformMatrix4fv(u, 1, 0, (float *)&xform);
        mesh->render();
    }
};

SDL_Surface * IMG_LoadSafe(const char * path)
{
    SDL_Surface * surface = IMG_Load(path);
    if (!surface) return 0;
    SDL_Surface * surface_2 = SDL_ConvertSurface(surface, SDL_PIXELFORMAT_RGBA32);
    SDL_DestroySurface(surface);
    return surface_2;
}

uint32_t loadTexture(const char * path, bool linear)
{
    SDL_Surface * surface = IMG_LoadSafe(path);
    if (!surface) exit(0);
    
    uint32_t tx_id;
    glGenTextures(1, &tx_id);
    glBindTexture(GL_TEXTURE_2D, tx_id);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, surface->w, surface->h, 0, GL_RGBA, GL_UNSIGNED_BYTE, surface->pixels);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    if (linear)
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameterf(GL_TEXTURE_2D, 0x84FE, 16.0f); // GL_TEXTURE_MAX_ANISOTROPY_EXT
    }
    else
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
    glGenerateMipmap(GL_TEXTURE_2D);
    SDL_DestroySurface(surface);
    return tx_id;
}

void checkGlError()
{
    auto err = glGetError();
    bool errored = false;
    while (err != GL_NO_ERROR)
    {
        errored = true;
        printf("GL error: %d\n", err);
        fflush(stdout);
        err = glGetError();
    }
    if (errored) exit(0);
}
void checkGlShaderError(uint32_t s)
{
    int ok = 1;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char log[1024];
        memset(log, 0, 1024);
        glGetShaderInfoLog(s, 1024, 0, log);
        printf("Shader compilation error: %s\n", log);
        fflush(stdout);
        exit(0);
    }
}

int main()
{
    SDL_Init(SDL_INIT_VIDEO);
    
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_DEBUG_FLAG);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    
    int w = 1280;
    int h = 720;
    
    SDL_Window * window = SDL_CreateWindow("Occlusion Demo", w, h, SDL_WINDOW_OPENGL);
    SDL_SetWindowResizable(window, true);
    SDL_GLContext context = SDL_GL_CreateContext(window);
    SDL_GL_MakeCurrent(window, context);
    bool quit = false;
    
    gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress);
    checkGlError();
    
    glDebugMessageCallback([]
        (   GLenum source,
            GLenum type,
            GLuint id,
            GLenum severity,
            GLsizei length,
            const GLchar * message,
            const void * userParam
        ) APIENTRY -> void
        {
            (void)source;
            (void)type;
            (void)id;
            (void)severity;
            (void)userParam;
            for (GLsizei i = 0; i < length; i++)
                putc(message[i], stdout);
            putc('\n', stdout);
            fflush(stdout);
        }, 0
    );
    
    auto shadobj_vert = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(shadobj_vert, 1, &shader_vert, 0);
    checkGlError();
    glCompileShader(shadobj_vert);
    checkGlShaderError(shadobj_vert);
    checkGlError();
    
    auto shadobj_frag = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(shadobj_frag, 1, &shader_frag, 0);
    checkGlError();
    glCompileShader(shadobj_frag);
    checkGlShaderError(shadobj_frag);
    checkGlError();
    
    unsigned int program = glCreateProgram();
    glAttachShader(program, shadobj_vert);
    checkGlError();
    glAttachShader(program, shadobj_frag);
    checkGlError();
    glLinkProgram(program);
    checkGlError();
    glUseProgram(program);
    checkGlError();
    
    auto vloc = glGetAttribLocation(program, "vertPos");
    auto nloc = glGetAttribLocation(program, "vertNorm");
    
    uint32_t tx_brick = loadTexture("brick.tiff", true);
    uint32_t tx_grass = loadTexture("grass.tiff", true);
    uint32_t tx_font = loadTexture("font.png", false);
    
    SDL_Surface * hm = IMG_LoadSafe("__heightmap.jpg");
    int32_t hm_w = hm->w;
    int32_t hm_h = hm->h;
    int32_t hm_pitch = hm->pitch;
    uint8_t * hm_px = (uint8_t *)hm->pixels;
    auto get_px = [&](int x, int z) -> auto
    {
        x = clamp(x, 0, hm_w-1);
        z = clamp(z, 0, hm_h-1);
        z = hm_h - z - 1;
        return hm_px[z*hm_pitch + x*4] * 0.35f;
    };
    
    auto occ_world = tinysroc::OccWorld::build();
    tinysroc::OccWorldView occ_world_view(occ_world);
    
    std::vector<Renderable *> objects;
    
    const char * overlay_text = "Hello???\nasdf";
    
    int cw = 32;
    
    std::vector<uint64_t> occluders;
    
    for (int gz = 0; gz < hm_h; gz += cw)
    {
        for (int gx = 0; gx < hm_w; gx += cw)
        {
            std::vector<vec3> verts;
            std::vector<vec3> normals;
            std::vector<uint32_t> indices;
            
            for (int _z = 0; _z <= cw; _z++)
            {
                int z = gz + _z;
                for (int _x = 0; _x <= cw; _x++)
                {
                    int x = gx + _x;
                    auto get_vert = [&](int xd, int zd)
                    {
                        float y = get_px(x+xd, z+zd);
                        return vec3(_x+xd, y, _z+zd);
                    };
                    verts.push_back(get_vert(0, 0));
                    
                    vec3 u = get_vert(0, 0-1);
                    vec3 d = get_vert(0, 0+1);
                    vec3 l = get_vert(0-1, 0);
                    vec3 r = get_vert(0+1, 0);
                    
                    vec3 a = u-d;
                    vec3 b = l-r;
                    vec3 n = normalize(cross(a, b));
                    normals.push_back(n);
                }
            }
            for (int z = 0; z < cw; z++)
            {
                for (int x = 0; x < cw; x++)
                {
                    indices.push_back(z*(cw+1) + x);
                    indices.push_back(z*(cw+1) + x + 1);
                    indices.push_back(z*(cw+1) + x + 2 + cw);
                    indices.push_back(z*(cw+1) + x);
                    indices.push_back(z*(cw+1) + x + 2 + cw);
                    indices.push_back(z*(cw+1) + x + 1 + cw);
                }
            }
            
            // There are three alternatives:
            // 1) Build a low-res version, but with a source-wise min-pool to push everything down to the local minimum.
            //     - This sucks because it pushes peaks down, when peaks don't have to be pushed down, only divots.
            // 2) Build a low-res version, pushing things down as you go to prevent stick-out, based on whatever was
            //       previously built. This sucks because it's not starting-position-agnostic, so it's bad at chunking.
            // 3) A variant of 2 that works properly with chunking. That's what I implemented here.
            //       This can be extended to 1/4x and 1/8x etc. by applying it over itself recursively.
            //       The way normals are calculated can be improved, but I kept it source-based here for simplicity.
            auto build_conservative_terrain_occluder = [&]() -> uint64_t {
                std::vector<vec3> verts;
                int ez = cw;
                int ex = cw;
                for (int z = gz-2; z <= gz + ez+2; z += 2) {
                    for (int x = gx-2; x <= gx + ex+2; x += 2) {
                        auto get_vert = [&](int xd, int zd)
                        {
                            float y = get_px(x+xd, z+zd);
                            return vec3(x-gx+xd, y, z-gz+zd);
                        };
                        verts.push_back(get_vert(0, 0));
                    }
                }
                std::vector<vec3> normals;
                int _i = 0;
                for (int lz = gz-2; lz <= gz + ez+2; lz += 2) {
                    for (int lx = gx-2; lx <= gx + ex+2; lx += 2) {
                        if (lz != gz-2 && lz != gz + ez+2 &&
                            lx != gx-2 && lx != gx + ex+2)
                        {
                            vec3 l = verts[_i - 1];
                            vec3 r = verts[_i + 1];
                            vec3 u = verts[_i - (ex/2+3)];
                            vec3 d = verts[_i + (ex/2+3)];
                            vec3 a = u-d;
                            vec3 b = l-r;
                            vec3 n = cross(a, b);
                            n = normalize(n);
                            normals.push_back(n);
                        }
                        _i += 1;
                    }
                }
                std::vector<vec3> verts_interp;
                int vw = sqrt(verts.size());
                for (int lz = 0; lz < vw-1; lz++) {
                    for (int lx = 0; lx < vw-1; lx++) {
                        vec3 v1 = verts[lz * vw + lx];
                        vec3 v2 = verts[lz * vw + lx + 1];
                        verts_interp.push_back(v1);
                        verts_interp.push_back((v1+v2)*0.5f);
                    }
                    for (int lx = 0; lx < vw-1; lx++) {
                        vec3 v1 = verts[(lz+0) * vw + lx];
                        vec3 v2 = verts[(lz+0) * vw + lx + 1];
                        vec3 v3 = verts[(lz+1) * vw + lx];
                        vec3 v4 = verts[(lz+1) * vw + lx + 1];
                        
                        vec3 a = (v1+v4)*0.5f;
                        vec3 b = (v2+v3)*0.5f;
                        a.y = max(a.y, b.y);
                        verts_interp.push_back(a);
                        verts_interp.push_back((v1+v3)*0.5f);
                    }
                }
                int vwi = sqrt(verts_interp.size());
                int nw = sqrt(normals.size());
                std::vector<vec3> verts2;
                for (int lz = 1; lz < vw-1; lz++) {
                    for (int lx = 1; lx < vw-1; lx++) {
                        vec3 v = verts[lz * vw + lx];
                        
                        float max_stickout = 0.0f;
                        for (int _z = -1; _z <= 1; _z++) {
                            for (int _x = -1; _x <= 1; _x++) {
                                if (_x == 0 && _z == 0) continue;
                                float ref = get_px(
                                    gx-2 + lx*2 + _x,
                                    gz-2 + lz*2 + _z
                                );
                                vec3 interp_v = verts_interp[
                                    lz*2*vwi + _z*vwi +
                                    lx*2 + _x
                                ];
                                float interp = interp_v.y;
                                float stickout = interp - ref;
                                max_stickout = max(max_stickout, stickout);
                            }
                        }
                        
                        v.y -= max_stickout;
                        verts2.push_back(v);
                    }
                }
                
                std::vector<uint32_t> occ_indexes;
                for (int lz = 0; lz < nw-1; lz++) {
                    for (int lx = 0; lx < nw-1; lx++) {
                        uint32_t a1 = lz*nw + lx + 0;
                        uint32_t a2 = lz*nw + lx + 1;
                        uint32_t a3 = lz*nw + lx + 0 + nw;
                        uint32_t a4 = lz*nw + lx + 1 + nw;
                        
                        occ_indexes.push_back(a1);
                        occ_indexes.push_back(a2);
                        occ_indexes.push_back(a4);
                        occ_indexes.push_back(a1);
                        occ_indexes.push_back(a4);
                        occ_indexes.push_back(a3);
                    }
                }
                
                auto occ = occ_world->add_occluder_trimesh_owned(
                    (float *)verts2.data(), (float *)normals.data(), normals.size(),
                    (uint32_t *)occ_indexes.data(), occ_indexes.size() / 3
                );
                return occ;
            };
            auto rm = std::make_shared<RenderableMesh>(vloc, nloc, verts, normals, indices, tx_grass);
            auto renderable = new Renderable(rm);
            renderable->xform = translate(renderable->xform, vec3(gx - hm_w/2, 0.0f, gz - hm_h/2));
            
            if (true)
            {
                auto occ = build_conservative_terrain_occluder();
                occ_world->occluder_set_xform(occ, renderable->xform);
                occluders.push_back(occ);
            }
            
            objects.push_back(renderable);
        }
    }
    
    // rects
    {
        auto lcg = []() -> uint32_t {
            static uint64_t seed = 41;
            seed = ((seed * 901234591) + 1) ^ 5125266;
            return (seed >> 32) ^ seed;
        };
        auto get_px_worldspace = [&](float x, float y) {
            x += hm_w/2;
            y += hm_h/2;
            return get_px(x, y);
        };
        
        auto size = 4.0f;
        
        std::vector<vec3> verts = {
            vec3(-size, -size, 0.0f),
            vec3( size, -size, 0.0f),
            vec3( size,  size, 0.0f),
            vec3(-size,  size, 0.0f),
            
            vec3(-size, -size, 0.0f),
            vec3(-size,  size, 0.0f),
            vec3( size,  size, 0.0f),
            vec3( size, -size, 0.0f),
        };
        std::vector<vec3> normals = {
            vec3(0.0f, 0.0f, -1.0f),
            vec3(0.0f, 0.0f, -1.0f),
            vec3(0.0f, 0.0f, -1.0f),
            vec3(0.0f, 0.0f, -1.0f),
            
            vec3(0.0f, 0.0f, 1.0f),
            vec3(0.0f, 0.0f, 1.0f),
            vec3(0.0f, 0.0f, 1.0f),
            vec3(0.0f, 0.0f, 1.0f),
        };
        std::vector<uint32_t> indexes = {
            0, 1, 2, 0, 2, 3,
            4, 5, 6, 4, 6, 7,
        };
        
        auto rm = std::make_shared<RenderableMesh>(vloc, nloc, verts, normals, indexes, tx_brick);
        
        for (int _i = 0; _i < 3000; _i++)
        {
            float x = double(lcg()) / 4294967295.0;
            float y = double(lcg()) / 4294967295.0;
            
            vec3 coords[4] = {
                vec3(-size, -size, 0.0f),
                vec3( size, -size, 0.0f),
                vec3( size,  size, 0.0f),
                vec3(-size,  size, 0.0f),
            };
            
            auto occ = occ_world->add_occluder_rectangle(&coords[0].x);
    
            x -= 0.5f;
            y -= 0.5f;
            x *= hm_w;
            y *= hm_h;
            
            auto renderable = new Renderable(rm);
            renderable->xform = translate(renderable->xform, vec3(x, get_px_worldspace(x, y) + size, y));
            renderable->xform = rotate(renderable->xform, lcg() * 100.0f, vec3(0.0f, 1.0f, 0.0f));
            
            occ_world->occluder_set_xform(occ, renderable->xform);
            occluders.push_back(occ);
            
            objects.push_back(renderable);
        }
    }
    
    // loose triangles
    {
        auto lcg = []() -> uint32_t {
            static uint64_t seed = 4511;
            seed = ((seed * 901234591) + 1) ^ 5125266;
            return (seed >> 32) ^ seed;
        };
        auto get_px_worldspace = [&](float x, float y) {
            x += hm_w/2;
            y += hm_h/2;
            return get_px(x, y);
        };
        
        auto size = 4.0f;
        
        std::vector<vec3> verts = {
            vec3(-size, -size, 0.0f),
            vec3( size, -size, 0.0f),
            vec3( size,  size, 0.0f),
            
            vec3(-size+size, -size+size, 0.0f),
            vec3( size+size, -size+size, 0.0f),
            vec3( size+size,  size+size, 0.0f),
        };
        std::vector<vec3> normals = {
            vec3(0.0f, 0.0f, -1.0f),
            vec3(0.0f, 0.0f, -1.0f),
            vec3(0.0f, 0.0f, -1.0f),
            
            vec3(0.0f, 0.0f, -1.0f),
            vec3(0.0f, 0.0f, -1.0f),
            vec3(0.0f, 0.0f, -1.0f),
        };
        std::vector<uint32_t> indexes = {
            0, 1, 2,
            3, 4, 5,
        };
        
        auto rm = std::make_shared<RenderableMesh>(vloc, nloc, verts, normals, indexes, tx_brick);
        
        for (int _i = 0; _i < 3000; _i++)
        {
            float x = double(lcg()) / 4294967295.0;
            float y = double(lcg()) / 4294967295.0;
            
            vec3 coords[6] = {
                vec3(-size, -size, 0.0f),
                vec3( size, -size, 0.0f),
                vec3( size,  size, 0.0f),
                
                vec3(-size+size, -size+size, 0.0f),
                vec3( size+size, -size+size, 0.0f),
                vec3( size+size,  size+size, 0.0f),
            };
            
            auto occ = occ_world->add_occluder_triangles(&coords[0].x, 2);
            
            x -= 0.5f;
            y -= 0.5f;
            x *= hm_w;
            y *= hm_h;
            
            auto renderable = new Renderable(rm);
            renderable->xform = translate(renderable->xform, vec3(x, get_px_worldspace(x, y) + size, y));
            renderable->xform = rotate(renderable->xform, lcg() * 100.0f, vec3(0.0f, 1.0f, 0.0f));
            
            occ_world->occluder_set_xform(occ, renderable->xform);
            occluders.push_back(occ);
            
            objects.push_back(renderable);
        }
    }
    
    // boxes
    {
        auto boxfile = fopen("box1.obj", "rb");
        if (!boxfile)
        {
            puts("if (!boxfile)"), exit(1);
        }
        fseek(boxfile, 0, SEEK_END);
        auto size = ftell(boxfile);
        fseek(boxfile, 0, SEEK_SET);
        
        char * buffer = (char *)calloc(size + 1, 1);
        if (!buffer)
        {
            puts("if (!buffer)"), exit(1);
        }
        if (fread(buffer, size, 1, boxfile) != 1)
        {
            puts("if (fread(buffer, size, 1, boxfile) != 1)"), exit(1);
        }
        
        auto text = std::string(buffer);
        
        free(buffer);
        fclose(boxfile);
        
        std::vector<vec3> o_verts;
        std::vector<vec3> o_normals;
        
        std::vector<vec3> verts;
        std::vector<vec3> normals;
        std::vector<uint32_t> indexes;
        
        std::unordered_map<uint64_t, size_t> o_to_us;
        
        std::vector<std::string> lines;
        size_t start = 0;
        size_t end = 0;
        while ((end = text.find('\n', start)) != (size_t)-1)
        {
            lines.emplace_back(text.data() + start, end - start);
            start = end + 1;
        }
        lines.emplace_back(text.data() + start, text.size() - start);
        
        for (auto subrange : lines)
        {
            auto line = std::string(subrange.begin(), subrange.end());
            
            std::vector<std::string> tokens;
            size_t start = 0;
            size_t end = 0;
            while ((end = line.find(' ', start)) != (size_t)-1)
            {
                tokens.emplace_back(line.data() + start, end - start);
                start = end + 1;
            }
            tokens.emplace_back(line.data() + start, line.size() - start);
            
            if (tokens.size() <= 1) continue;
            auto tok = tokens[0];
            if (tok == "v" || tok == "vn")
            {
                float x = 0.0f;
                float y = 0.0f;
                float z = 0.0f;
                std::from_chars(tokens[1].data(), tokens[1].data() + tokens[1].size(), x);
                std::from_chars(tokens[2].data(), tokens[2].data() + tokens[2].size(), y);
                std::from_chars(tokens[3].data(), tokens[3].data() + tokens[3].size(), z);
                if (tok == "v")
                    o_verts.push_back({x, y, z});
                else
                    o_normals.push_back({x, y, z});
            }
            if (tok == "f")
            {
                auto to_indexes = [&](size_t i, uint64_t & a, uint64_t & b)
                {
                    std::vector<std::string> subs;
                    size_t start = 0;
                    size_t end = 0;
                    while ((end = tokens[i].find('/', start)) != (size_t)-1)
                    {
                        subs.emplace_back(tokens[i].data() + start, end - start);
                        start = end + 1;
                    }
                    subs.emplace_back(tokens[i].data() + start, tokens[i].size() - start);
                    
                    std::from_chars(subs[0].data(), subs[0].data() + subs[0].size(), a);
                    std::from_chars(subs[2].data(), subs[2].data() + subs[2].size(), b);
                    if (a > 0) a -= 1;
                    if (b > 0) b -= 1;
                };
                
                for (size_t _i = 1; _i < 4; _i++)
                {
                    size_t i = _i == 2 ? 3 : _i == 3 ? 2 : 1;
                    uint64_t a = 0;
                    uint64_t b = 0;
                    to_indexes(i, a, b);
                    uint64_t key = (a<<32) | b;
                    if (o_to_us.count(key))
                    {
                        indexes.push_back(o_to_us[key]);
                    }
                    else
                    {
                        auto idx = verts.size();
                        verts.push_back(o_verts[a]);
                        normals.push_back(o_normals[b]);
                        indexes.push_back(idx);
                        o_to_us.insert({key, idx});
                    }
                }
            }
        }
        
        auto rm = std::make_shared<RenderableMesh>(vloc, nloc, verts, normals, indexes, tx_brick);
        
        auto lcg = []() -> uint32_t {
            static uint64_t seed = 42;
            seed = ((seed * 901234591) + 1) ^ 5125266;
            return (seed >> 32) ^ seed;
        };
        auto get_px_worldspace = [&](float x, float y) {
            x += hm_w/2;
            y += hm_h/2;
            return get_px(x, y);
        };
        for (int _i = 0; _i < 30000; _i++)
        {
            auto renderable = new Renderable(rm);
            float x = double(lcg()) / 4294967295.0;
            float y = double(lcg()) / 4294967295.0;
            x -= 0.5f;
            y -= 0.5f;
            x *= hm_w;
            y *= hm_h;
            renderable->xform = translate(renderable->xform, vec3(x, get_px_worldspace(x, y) + 1.0f, y));
            objects.push_back(renderable);
        }
    }
    
    #define deg2rad(X) ((float)((X)*(3.141592653598783/180.0)))
    
    float cam_yaw = 0.0f;
    float cam_pitch = 15.0f;
    
    vec3 pos = vec3(-2.4f, -25.6f, 1.5f);
    pos.y = -get_px(hm_w/2 - 2.4f, hm_h/2 + 1.5f) - 20.0f;
    
    //cam_yaw = -29.875f;
    //cam_pitch = -9.5f;
    //pos = vec3(-2.632f, -45.627f, -9.026f);
    
    //cam_yaw = 34.25f;
    //cam_yaw = 0.0f;
    //cam_pitch = -8.375f;
    //pos = vec3(1.115f, -44.427f, -6.335f);
    
    //cam_yaw = 32.5f;
    //cam_pitch = 4.25f;
    //pos = vec3(1.214f, -44.444f, -6.182f);
    
    //cam_yaw = -30.625f;
    //cam_pitch = 4.75f;
    //pos = vec3(-4.988f, -44.033f, -2.577f);
    
    //cam_yaw = 97.75f;
    //cam_pitch = -4.5f;
    //pos = vec3(-28.574f, -49.431f, -12.59f);
    
    //cam_yaw = 180.0f;
    //cam_pitch = 7.0f;
    //pos = vec3(-36.184f, -50.42f, -24.4f);
    
    //cam_yaw = 104.125f;
    //cam_pitch = 10.875f;
    //pos = vec3(-26.526f, -50.833f, -12.727f);
    
    //cam_yaw = 110.5f;
    //cam_pitch = 3.625f;
    //pos = vec3(957.996f, -2.654f, -1026.676f);
    
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    
    std::mutex renderlock;
    bool grabbed = false;
    
    auto nisuv = glGetUniformLocation(program, "normIsUV");
    
    uint32_t r_VAO, r_VBO, r_VBO2;
    {
        glGenVertexArrays(1, &r_VAO);
        glBindVertexArray(r_VAO);
        
        glGenBuffers(1, &r_VBO);
        glGenBuffers(1, &r_VBO2);
        
        vec3 v[6] = {
            vec3( 0.0f,  0.0f, 0.1f),
            vec3( 1.0f, -1.0f, 0.1f),
            vec3( 1.0f,  0.0f, 0.1f),
            vec3( 0.0f,  0.0f, 0.1f),
            vec3( 0.0f, -1.0f, 0.1f),
            vec3( 1.0f, -1.0f, 0.1f),
        };
        glBindBuffer(GL_ARRAY_BUFFER, r_VBO);
        glBufferData(GL_ARRAY_BUFFER, 6 * sizeof(v[0]), v, GL_STATIC_DRAW);
        glVertexAttribPointer(vloc, 3, GL_FLOAT, false, 0, 0);
        glEnableVertexAttribArray(vloc);
        
        glBindBuffer(GL_ARRAY_BUFFER, r_VBO2);
        glBufferData(GL_ARRAY_BUFFER, 6 * sizeof(v[0]), v, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(nloc, 3, GL_FLOAT, false, 0, 0);
        glEnableVertexAttribArray(nloc);
        
        glBindVertexArray(0);
    }
    
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    auto time0 = SDL_GetTicksNS();
    
    uint32_t tx_occ_debug;
    glGenTextures(1, &tx_occ_debug);
    glBindTexture(GL_TEXTURE_2D, tx_occ_debug);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    
    // options
    bool occ_enabled = true;
    bool occ_debug = true;
    bool occ_smaller = false;
    bool wireframe_enabled = false;
    bool accurate_timing = true;
    
    const int n_gl_queries = 5;
    uint32_t gl_queries[n_gl_queries];
    glGenQueries(n_gl_queries, &gl_queries[0]);
    
    size_t framenum = 0;
    
    mat4 view = mat4(1.0f);
    auto _rerender = [&]() {
        renderlock.lock();
        framenum += 1;
        SDL_GetWindowSize(window, &w, &h);
        glViewport(0, 0, w, h);
        mat4 proj = perspectiveLH_ZO(deg2rad(75.0f), (float)w / (float)h, 0.01f, 10000.0f);
        
        auto time1 = SDL_GetTicksNS();
        auto dt = (time1-time0) / 1000000000.0f;
        time0 = time1;
        
        auto keys = SDL_GetKeyboardState(0);
        float delta = dt;
        float speed = 5.0f;
        if (keys[SDL_SCANCODE_LSHIFT])
            speed *= 16.0f;
        if (keys[SDL_SCANCODE_W])
            pos -= (vec4(0.0f, 0.0f, 1.0f, 0.0f) * view).xyz() * delta * speed;
        if (keys[SDL_SCANCODE_S])
            pos += (vec4(0.0f, 0.0f, 1.0f, 0.0f) * view).xyz() * delta * speed;
        if (keys[SDL_SCANCODE_D])
            pos -= (vec4(1.0f, 0.0f, 0.0f, 0.0f) * view).xyz() * delta * speed;
        if (keys[SDL_SCANCODE_A])
            pos += (vec4(1.0f, 0.0f, 0.0f, 0.0f) * view).xyz() * delta * speed;
        
        pos.x = round(pos.x * 1000.0f) / 1000.0f;
        pos.y = round(pos.y * 1000.0f) / 1000.0f;
        pos.z = round(pos.z * 1000.0f) / 1000.0f;
        
        view = mat4(1.0f);
        view = rotate(view, deg2rad(cam_pitch), vec3(-1.0f, 0.0f, 0.0f));
        view = rotate(view, deg2rad(cam_yaw)  , vec3(0.0f, -1.0f, 0.0f));
        view = translate(view, pos);
        
        glClearColor(0.5f, 0.75f, 1.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
        
        auto t_debug_0 = SDL_GetTicksNS();
        auto t_debug_01 = SDL_GetTicksNS();
        auto t_debug_02 = SDL_GetTicksNS();
        
        mat4 vp = proj * view;
        if (occ_enabled)
            occ_world_view.set_perspective_camera(proj, view);
        
        if (occ_enabled)
        {
            int occ_w = 256;
            int occ_h = 128;
            if (occ_smaller)
            {
                occ_w = 64;
                occ_h = 32;
            }
            for (auto occ : occluders)
            {
                vec3 a = occ_world->occluder_get_center(occ);
                vec3 b = -pos;
                float dist = length(a-b);
                if (dist < cw * 2.5f)
                    occ_world->occluder_enable(occ);
                else
                    occ_world->occluder_disable(occ);
            }
            t_debug_01 = SDL_GetTicksNS();
            occ_world_view.rasterize(occ_w, occ_h, occ_w, 0);
            t_debug_02 = SDL_GetTicksNS();
            if (occ_debug)
            {
                glBindTexture(GL_TEXTURE_2D, tx_occ_debug);
                uint8_t * opixbuf = (uint8_t *)malloc(occ_w*occ_h*4);
                for (int y = 0; y < occ_h; y++)
                {
                    for (int x = 0; x < occ_w; x++)
                    {
                        float d = 0.0f;
                        memcpy(&d, &occ_world_view.hires[y*occ_w + x], 4);
                        if (d != 0.0f) d = d*0.5f + 0.5f;
                        uint8_t c = clamp(round(d*255.0f), 0.0f, 255.0f);
                        opixbuf[(y*occ_w + x)*4 + 0] = c;
                        opixbuf[(y*occ_w + x)*4 + 1] = c;
                        opixbuf[(y*occ_w + x)*4 + 2] = c;
                        opixbuf[(y*occ_w + x)*4 + 3] = 255;
                    }
                }
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, occ_w, occ_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, opixbuf);
                free(opixbuf);
            }
        }
        else if (occ_debug)
        {
            char * opixbuf = (char *)malloc(256*128*4);
            volatile size_t asdf = (size_t)opixbuf;
            asdf = asdf;
            asdf = (size_t)opixbuf;
            free(opixbuf);
        }
        
        auto t_debug_1 = SDL_GetTicksNS();
        
        auto cm = vp;
        
        // frustum planes
        auto fm_l = vec4(cm[0][3]+cm[0][0], cm[1][3]+cm[1][0], cm[2][3]+cm[2][0], cm[3][3]+cm[3][0]);
        auto fm_r = vec4(cm[0][3]-cm[0][0], cm[1][3]-cm[1][0], cm[2][3]-cm[2][0], cm[3][3]-cm[3][0]);
        auto fm_b = vec4(cm[0][3]+cm[0][1], cm[1][3]+cm[1][1], cm[2][3]+cm[2][1], cm[3][3]+cm[3][1]);
        auto fm_u = vec4(cm[0][3]-cm[0][1], cm[1][3]-cm[1][1], cm[2][3]-cm[2][1], cm[3][3]-cm[3][1]);
        auto fm_n = vec4(cm[0][2], cm[1][2], cm[2][2], cm[3][2]);
        auto fm_f = vec4(cm[0][3]-cm[0][2], cm[1][3]-cm[1][2], cm[2][3]-cm[2][2], cm[3][3]-cm[3][2]);
        // we need normalized plane normals for the radius hack to work properly
        fm_l /= max(1e-30f, length(fm_l.xyz()));
        fm_r /= max(1e-30f, length(fm_r.xyz()));
        fm_b /= max(1e-30f, length(fm_b.xyz()));
        fm_u /= max(1e-30f, length(fm_u.xyz()));
        fm_n /= max(1e-30f, length(fm_n.xyz()));
        fm_f /= max(1e-30f, length(fm_f.xyz()));
        
        vec4 planes[6] = { fm_l, fm_r, fm_b, fm_u, fm_n, fm_f };
        
        static std::vector<Renderable *> sorted_objs;
        sorted_objs.clear();
        for (auto & obj : objects)
        {
            auto is_outside = [&]() {
                for (int i = 0; i < 6; i++)
                {
                    vec4 plane = planes[i];
                    plane = transpose(obj->xform) * plane;
                    auto r = dot(abs(plane.xyz()), obj->mesh->aabb_ext);
                    auto m = dot(plane, obj->mesh->aabb_mid);
                    if (m + r < 0.0f)
                        return true;
                }
                return false;
            };
            if (is_outside())
                continue;
            auto lo = obj->mesh->aabb_mid.xyz() - obj->mesh->aabb_ext;
            auto hi = lo + obj->mesh->aabb_ext*2.0f;
            if (occ_enabled && occ_world_view.query_aabb(lo, hi, obj->xform))
                continue;
            
            sorted_objs.push_back(obj);
        }
        
        auto t_debug_2 = SDL_GetTicksNS();
        
        std::sort(sorted_objs.begin(), sorted_objs.end(), [&](Renderable * a, Renderable * b) {
            vec4 q1 = view * a->mesh->aabb_mid;
            vec4 q2 = view * b->mesh->aabb_mid;
            return q1.z > q2.z;
        });
        
        auto t_debug_3 = SDL_GetTicksNS();
        uint64_t rast_time_taken_really = 0;
        if (framenum > n_gl_queries)
            glGetQueryObjectui64v(gl_queries[framenum % n_gl_queries], GL_QUERY_RESULT, &rast_time_taken_really);
        glBeginQuery(GL_TIME_ELAPSED, gl_queries[framenum % n_gl_queries]);

        glUniform1i(nisuv, 0);
        
        int calls = 0;
        auto uopa = glGetUniformLocation(program, "opacity");
        glUniform1f(uopa, 1.0f);
        
        if (wireframe_enabled)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        
        auto uf1 = glGetUniformLocation(program, "matM");
        auto uf2 = glGetUniformLocation(program, "matVP");
        glUniformMatrix4fv(uf2, 1, 0, (float *)&vp);
        for (auto renderable : sorted_objs)
        {
            renderable->render(uf1);
            calls += 1;
        }
        if (wireframe_enabled)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        
        glEndQuery(GL_TIME_ELAPSED);
        
        glDisable(GL_DEPTH_TEST);

        auto t_debug_4 = SDL_GetTicksNS();
        
        // draw occluder overlay
        if (occ_enabled && occ_debug)
        {
            glUniform1i(nisuv, 1);
            mat4 m_ident = mat4(1.0f);
            
            mat4 mat = translate(m_ident, vec3(-1.0f, 1.0f, 0.0f));
            mat = scale(mat, vec3(2.0f, 2.0f, 1.0f));
            vec3 normals[6] = {
                vec3( 0.0f, 0.0f, 0.0f),
                vec3( 1.0f, 1.0f, 0.0f),
                vec3( 1.0f, 0.0f, 0.0f),
                vec3( 0.0f, 0.0f, 0.0f),
                vec3( 0.0f, 1.0f, 0.0f),
                vec3( 1.0f, 1.0f, 0.0f),
            };
            
            glBindVertexArray(r_VAO);
            
            glUniform1f(uopa, 0.5f);
            
            glBindBuffer(GL_ARRAY_BUFFER, r_VBO2);
            glBufferData(GL_ARRAY_BUFFER, 6 * sizeof(normals[0]), normals, GL_DYNAMIC_DRAW);
            
            glBindTexture(GL_TEXTURE_2D, tx_occ_debug);
            glUniformMatrix4fv(uf1, 1, 0, (float *)&mat);
            glUniformMatrix4fv(uf2, 1, 0, (float *)&m_ident);
            glDrawArrays(GL_TRIANGLES, 0, 6);
            glBindVertexArray(0);
            
            glUniform1f(uopa, 1.0f);
            
            checkGlError();
        }
        
        auto t_debug_5 = SDL_GetTicksNS();
        
        float t = 1.0f - 0.05f;
        
        static float t01a = 0.0f;
        static float t01b = 0.0f;
        static float t01c = 0.0f;
        static float t12 = 0.0f;
        static float t23 = 0.0f;
        static float t34 = 0.0f;
        static float t45 = 0.0f;
        float ins_t01a = (t_debug_01 - t_debug_0)/1000000.0;
        float ins_t01b = (t_debug_02 - t_debug_01)/1000000.0;
        float ins_t01c = (t_debug_1 - t_debug_02)/1000000.0;
        float ins_t12 = (t_debug_2 - t_debug_1)/1000000.0;
        float ins_t23 = (t_debug_3 - t_debug_2)/1000000.0;
        float ins_t34;
        if (accurate_timing)
            ins_t34 = (rast_time_taken_really)/1000000.0;
        else
            ins_t34 = (t_debug_4 - t_debug_3)/1000000.0;
        float ins_t45 = (t_debug_5 - t_debug_4)/1000000.0;
        
        t01a = t01a*t + ins_t01a*(1.0f - t);
        t01b = t01b*t + ins_t01b*(1.0f - t);
        t01c = t01c*t + ins_t01c*(1.0f - t);
        t12 = t12*t + ins_t12*(1.0f - t);
        t23 = t23*t + ins_t23*(1.0f - t);
        t34 = t34*t + ins_t34*(1.0f - t);
        t45 = t45*t + ins_t45*(1.0f - t);
        
        static auto start = SDL_GetTicksNS();
        auto end = SDL_GetTicksNS();
        static float ft = 0.05f;
        float instantaneous_ft = (end-start+1)/1000000000.0;
        instantaneous_ft = max(0.0f, instantaneous_ft);
        ft = ft*t + instantaneous_ft*(1.0f - t);
        #if __cplusplus < 202002L
        auto ol = fmt::format(
        #else
        auto ol = std::format(
        #endif
            "FPS: {}\n"
            "Calls: {}\n"
            "SoftrastA: {}\n"
            "SoftrastB: {}\n"
            "SoftrastC: {}\n"
            "Cull: {}\n"
            "Sort: {}\n"
            "Render: {}\n"
            "Debug: {}\n"
            "cam y p: {} {}\n"
            "pos: {} {} {}\n"
            "---\n"
            "C: toggle occlusion\n"
            "V: toggle debug overlay\n"
            "R: toggle occlusion resolution\n"
            "T: toggle wireframe\n"
            "U: toggle measure Render latency (higher)\n               vs Render dispatch time (lower)\n"
            "ESC: grab/ungrab mouse\n"
            ,
            1.0f/ft,
            calls,
            round(t01a*1000.0f)/1000.0f,
            round(t01b*1000.0f)/1000.0f,
            round(t01c*1000.0f)/1000.0f,
            round(t12*1000.0f)/1000.0f,
            round(t23*1000.0f)/1000.0f,
            round(t34*1000.0f)/1000.0f,
            round(t45*1000.0f)/1000.0f,
            cam_yaw, cam_pitch,
            pos.x, pos.y, pos.z
        );
        start = end;
        overlay_text = ol.data();
        
        // draw overlay text
        {
            glUniform1i(nisuv, 1);
            mat4 m_ident = mat4(1.0f);
            const char * c = overlay_text;
            float x = 0.0f;
            float y = 0.0f;
            while (*c != 0)
            {
                if (*c == '\r') { c++; continue; }
                if (*c == '\n')
                {
                    c++;
                    y += 1.0f;
                    x = 0.0f;
                    continue;
                }
                mat4 mat = translate(m_ident, vec3((x+0.25f) * 20.0f/w - 1.0f, -(y+0.25f) * 36.0f/h + 1.0f, 0.0f));
                mat = scale(mat, vec3(16.0f/w, 32.0f/h, 1.0f));
                float u = (*c)%32;
                float v = (*c)/32;
                u = u / 32.0f;
                v = v / 8.0f;
                u += (0.01f/128.0f);
                v += (0.01f/64.0f);
                float nu = 1.0f / 32.0f;
                float nv = 1.0f / 8.0f;
                
                vec3 normals[6] = {
                    vec3(    u,    v, 0.0f),
                    vec3( u+nu, v+nv, 0.0f),
                    vec3( u+nu,    v, 0.0f),
                    vec3(    u,    v, 0.0f),
                    vec3(    u, v+nv, 0.0f),
                    vec3( u+nu, v+nv, 0.0f),
                };
                
                glBindVertexArray(r_VAO);
                
                glBindBuffer(GL_ARRAY_BUFFER, r_VBO2);
                glBufferData(GL_ARRAY_BUFFER, 6 * sizeof(normals[0]), normals, GL_DYNAMIC_DRAW);
                
                glBindTexture(GL_TEXTURE_2D, tx_font);
                glUniformMatrix4fv(uf1, 1, 0, (float *)&mat);
                glUniformMatrix4fv(uf2, 1, 0, (float *)&m_ident);
                glDrawArrays(GL_TRIANGLES, 0, 6);
                
                c += 1;
                x += 1.0f;
            }
            checkGlError();
        }
        
        glEnable(GL_DEPTH_TEST);
        glUniform1i(nisuv, 0);
        
        SDL_GL_SwapWindow(window);
        //SDL_Delay(1);
        renderlock.unlock();
    };
    
    std::function<void()> rerender = _rerender;
    
    auto event_pumper = [](void * userdata, SDL_Event * event) -> bool
    {
        auto rerender = (std::function<void()> *)userdata;
        if (event->type == SDL_EVENT_WINDOW_EXPOSED)
            (*rerender)();
        return true;
    };
    SDL_AddEventWatch((SDL_EventFilter)event_pumper, &rerender);
    
    int mx = 0;
    int my = 0;
    while (!quit)
    {
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            if (event.type == SDL_EVENT_QUIT)
                quit = true;
            if (event.type == SDL_EVENT_KEY_DOWN)
            {
                switch(event.key.key) {
                case SDLK_C: {
                    occ_enabled = !occ_enabled;
                    break;
                }
                case SDLK_V: {
                    occ_debug = !occ_debug;
                    break;
                }
                case SDLK_R: {
                    occ_smaller = !occ_smaller;
                    break;
                }
                case SDLK_T: {
                    wireframe_enabled = !wireframe_enabled;
                    break;
                }
                case SDLK_U: {
                    accurate_timing = !accurate_timing;
                    break;
                }
                case SDLK_ESCAPE: {
                    grabbed = !grabbed;
                    SDL_SetWindowRelativeMouseMode(window, grabbed);
                    if (grabbed)
                    {
                        SDL_Rect r;
                        r.x = mx;
                        r.y = my;
                        r.w = 1;
                        r.h = 1;
                        SDL_SetWindowMouseRect(window, &r);
                    }
                    else
                        SDL_SetWindowMouseRect(window, 0);
                    break;
                }
                default: {}
                }
            }
            if (event.type == SDL_EVENT_MOUSE_MOTION)
            {
                mx = event.motion.x;
                my = event.motion.y;
            }
            if (grabbed && event.type == SDL_EVENT_MOUSE_MOTION)
            {
                cam_yaw   += event.motion.xrel * (1.0f/8.0f);
                cam_pitch += event.motion.yrel * (1.0f/8.0f);
                cam_pitch = clamp(cam_pitch, -89.99f, 89.99f);
            }
        }
        
        rerender();
    }
    
    SDL_GL_DestroyContext(context);
    SDL_DestroyWindow(window);
    SDL_Quit();
}
