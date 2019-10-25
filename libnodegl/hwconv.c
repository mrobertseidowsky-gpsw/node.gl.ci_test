/*
 * Copyright 2018 GoPro Inc.
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <string.h>

#include "bstr.h"
#include "buffer.h"
#include "colorconv.h"
#include "hwconv.h"
#include "glincludes.h"
#include "glcontext.h"
#include "gctx.h"
#include "image.h"
#include "log.h"
#include "math_utils.h"
#include "memory.h"
#include "nodegl.h"
#include "nodes.h"
#include "pipeline.h"
#include "program.h"
#include "texture.h"
#include "topology.h"
#include "type.h"
#include "utils.h"

#define GLSL_ADD(bstr, ...) do {                  \
    int ret = ngli_bstr_print(bstr, __VA_ARGS__); \
    if (ret < 0)                                  \
        return ret;                               \
} while (0)

#define GLSL_XDD(bstr, str) do {                  \
    int ret = ngli_bstr_append(bstr, str);        \
    if (ret < 0)                                  \
        return ret;                               \
} while (0)

static int build_vertex_shader(struct bstr *bstr, struct glcontext *gl, int layout)
{
    const int backend = gl->backend;
    const int glsl_version = gl->glsl_version;

    GLSL_ADD(bstr, "#version %d%s\n", glsl_version, backend == NGL_BACKEND_OPENGLES ? " es" : "");
    if (backend == NGL_BACKEND_OPENGLES)
        GLSL_XDD(bstr, "precision highp float;\n");
    GLSL_ADD(bstr, "%s vec4 position;\n", glsl_version >= 130 ? "in"  : "attribute");
    GLSL_XDD(bstr, "uniform mat4 tex_coord_matrix;\n");
    GLSL_ADD(bstr, "%s vec2 tex_coord;\n", glsl_version >= 130 ? "out" : "varying");
    GLSL_XDD(bstr, "void main()\n{\n");
    GLSL_XDD(bstr, "    gl_Position = vec4(position.xy, 0.0, 1.0);\n");
    GLSL_XDD(bstr, "    tex_coord = (tex_coord_matrix * vec4(position.zw, 0.0, 1.0)).xy;\n");
    GLSL_XDD(bstr, "}\n");

    return 0;
}

static int build_fragment_shader(struct bstr *bstr, struct glcontext *gl, int layout)
{
    const int backend = gl->backend;
    const int gl_version = gl->version;
    const int glsl_version = gl->glsl_version;

    GLSL_ADD(bstr, "#version %d%s\n", glsl_version, backend == NGL_BACKEND_OPENGLES ? " es" : "");
    if (layout == NGLI_IMAGE_LAYOUT_MEDIACODEC) {
        const char *ext = glsl_version >= 310 ? "GL_OES_EGL_image_external_essl3" : "GL_OES_EGL_image_external";
        GLSL_ADD(bstr, "#extension %s: require\n", ext);
    }
    if (backend == NGL_BACKEND_OPENGLES)
        GLSL_XDD(bstr, "precision mediump float;\n");
    if (layout == NGLI_IMAGE_LAYOUT_DEFAULT)
        GLSL_XDD(bstr, "uniform sampler2D tex0;\n");
    else if (layout == NGLI_IMAGE_LAYOUT_MEDIACODEC) {
        GLSL_XDD(bstr, "uniform samplerExternalOES tex0;\n");
    } else if (layout == NGLI_IMAGE_LAYOUT_NV12) {
        GLSL_XDD(bstr, "uniform sampler2D tex0;\n");
        GLSL_XDD(bstr, "uniform sampler2D tex1;\n");
    } else if (layout == NGLI_IMAGE_LAYOUT_NV12_RECTANGLE) {
        GLSL_XDD(bstr, "uniform mediump sampler2DRect tex0;\n");
        GLSL_XDD(bstr, "uniform mediump sampler2DRect tex1;\n");
    }
    GLSL_ADD(bstr, "uniform mat4 color_matrix;\n");
    GLSL_ADD(bstr, "%s vec2 tex_coord;\n", glsl_version >= 130 ? "in" : "attribute");
    if (glsl_version >= 130)
        GLSL_ADD(bstr, "out vec4 frag_color;\n");
    GLSL_ADD(bstr, "void main(void)\n");
    GLSL_ADD(bstr, "{\n");

    const char *texture_func = glsl_version >= 130 ? "texture" : "texture2D";
    const char *rg_components = gl_version >= 300 ? "rg" : "ra";
    if (layout == NGLI_IMAGE_LAYOUT_DEFAULT ||
        layout == NGLI_IMAGE_LAYOUT_MEDIACODEC)  {
        GLSL_ADD(bstr, "    vec3 color = %s(tex0, tex_coord).rgb;\n", texture_func);
    } else if (layout == NGLI_IMAGE_LAYOUT_NV12) {
        GLSL_XDD(bstr, "    vec3 color;\n");
        GLSL_ADD(bstr, "    color.x = %s(tex0, tex_coord).r;\n", texture_func);
        GLSL_ADD(bstr, "    color.yz = %s(tex1, tex_coord).%s;\n", texture_func, rg_components);
    } else if (layout == NGLI_IMAGE_LAYOUT_NV12_RECTANGLE) {
        GLSL_XDD(bstr, "    vec3 color;\n");
        GLSL_ADD(bstr, "    color.x = %s(tex0, tex_coord).r;\n", texture_func);
        GLSL_ADD(bstr, "    color.yz = %s(tex1, tex_coord / vec2(2.0)).%s;\n", texture_func, rg_components);
    }
    GLSL_ADD(bstr, "    %s = color_matrix * vec4(color.rgb, 1.0);\n", glsl_version >= 130 ? "frag_color" : "gl_FragColor");
    GLSL_XDD(bstr, "}\n");
    return 0;
}

static const struct hwconv_desc {
    int nb_planes;
    const char *vertex_data;
    const char *fragment_data;
} hwconv_descs[] = {
    [NGLI_IMAGE_LAYOUT_MEDIACODEC] = {
        .nb_planes = 1,
    },
    [NGLI_IMAGE_LAYOUT_NV12] = {
        .nb_planes = 2,
    },
    [NGLI_IMAGE_LAYOUT_NV12_RECTANGLE] = {
        .nb_planes = 2,
    },
};

#include "utils.h"

int ngli_hwconv_init(struct hwconv *hwconv, struct ngl_ctx *ctx,
                     const struct image *dst_image,
                     const struct image_params *src_params)
{
    hwconv->ctx = ctx;
    hwconv->src_params = *src_params;

    if (dst_image->params.layout != NGLI_IMAGE_LAYOUT_DEFAULT) {
        LOG(ERROR, "unsupported output image layout: 0x%x", dst_image->params.layout);
        return NGL_ERROR_UNSUPPORTED;
    }

    struct glcontext *gl = ctx->glcontext;
    const struct texture *texture = dst_image->planes[0];

    struct rendertarget_params rt_params = {
        .width = dst_image->params.width,
        .height = dst_image->params.height,
        .nb_attachments = 1,
        .attachments = &texture,
    };
    int ret = ngli_rendertarget_init(&hwconv->rt, ctx, &rt_params);
    if (ret < 0)
        return ret;

    enum image_layout src_layout = src_params->layout;
    if (src_layout != NGLI_IMAGE_LAYOUT_NV12 &&
        src_layout != NGLI_IMAGE_LAYOUT_NV12_RECTANGLE &&
        src_layout != NGLI_IMAGE_LAYOUT_MEDIACODEC) {
        LOG(ERROR, "unsupported texture layout: 0x%x", src_layout);
        return NGL_ERROR_UNSUPPORTED;
    }
    const struct hwconv_desc *desc = &hwconv_descs[src_layout];

    int64_t t = ngli_gettime();
    for (int i = 0; i < 1000; i++) {
        struct bstr *vertex_bstr = ngli_bstr_create();
        ret = build_vertex_shader(vertex_bstr, gl, src_layout);
        if (ret < 0)
            return ret;

        ngli_bstr_freep(&vertex_bstr);
    }
    int64_t t2 = ngli_gettime();
    LOG(ERROR, "V took %fus", (t2 - t) / (double)1000);


    /* */
    t = ngli_gettime();
    for (int i = 0; i < 1000; i++) {
        struct bstr *vertex_bstr = ngli_bstr_create();
        ret = build_vertex_shader(vertex_bstr, gl, src_layout);
        if (ret < 0)
            return ret;

        struct bstr *fragment_bstr = ngli_bstr_create();
        ret = build_fragment_shader(fragment_bstr, gl, src_layout);
        if (ret < 0)
            return ret;

        ngli_bstr_freep(&vertex_bstr);
        ngli_bstr_freep(&fragment_bstr);
    }
    t2 = ngli_gettime();
    LOG(ERROR, "took %fus", (t2 - t) / (double)1000);
    /* */

    struct bstr *vertex_bstr = ngli_bstr_create();
    ret = build_vertex_shader(vertex_bstr, gl, src_layout);
    if (ret < 0)
        return ret;

    struct bstr *fragment_bstr = ngli_bstr_create();
    ret = build_fragment_shader(fragment_bstr, gl, src_layout);
    if (ret < 0)
        return ret;

    LOG(ERROR, "%s", ngli_bstr_strptr(vertex_bstr));
    LOG(ERROR, "%s", ngli_bstr_strptr(fragment_bstr));


    ret = ngli_program_init(&hwconv->program, ctx, ngli_bstr_strptr(vertex_bstr), ngli_bstr_strptr(fragment_bstr), NULL);
    if (ret < 0)
        return ret;

    static const float vertices[] = {
        -1.0f, -1.0f, 0.0f, 0.0f,
         1.0f, -1.0f, 1.0f, 0.0f,
         1.0f,  1.0f, 1.0f, 1.0f,
        -1.0f,  1.0f, 0.0f, 1.0f,
    };
    ret = ngli_buffer_init(&hwconv->vertices, ctx, sizeof(vertices), NGLI_BUFFER_USAGE_STATIC);
    if (ret < 0)
        return ret;

    ret = ngli_buffer_upload(&hwconv->vertices, vertices, sizeof(vertices));
    if (ret < 0)
        return ret;

    ngli_mat4_identity(hwconv->src_color_matrix);
    if (src_layout == NGLI_IMAGE_LAYOUT_NV12 ||
        src_layout == NGLI_IMAGE_LAYOUT_NV12_RECTANGLE) {
        ngli_colorconv_get_ycbcr_to_rgb_color_matrix(hwconv->src_color_matrix, &src_params->color_info);
    }

    const struct pipeline_uniform uniforms[] = {
        {.name = "tex_coord_matrix", .type = NGLI_TYPE_MAT4, .count = 1,               .data  = NULL},
        {.name = "tex_dimensions",   .type = NGLI_TYPE_VEC2, .count = desc->nb_planes, .data  = NULL},
        {.name = "color_matrix",     .type = NGLI_TYPE_MAT4, .count = 1,               .data  = hwconv->src_color_matrix},
    };

    const struct pipeline_texture textures[] = {
        {.name = "tex0", .texture = NULL},
        {.name = "tex1", .texture = NULL},
    };

    const struct pipeline_attribute attributes[] = {
        {.name = "position", .format = NGLI_FORMAT_R32G32B32A32_SFLOAT, .stride = 4 * 4, .buffer = &hwconv->vertices},
    };

    struct pipeline_params pipeline_params = {
        .type          = NGLI_PIPELINE_TYPE_GRAPHICS,
        .program       = &hwconv->program,
        .textures      = textures,
        .nb_textures   = desc->nb_planes,
        .uniforms      = uniforms,
        .nb_uniforms   = NGLI_ARRAY_NB(uniforms),
        .attributes    = attributes,
        .nb_attributes = NGLI_ARRAY_NB(attributes),
        .graphics      = {
            .topology    = NGLI_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN,
            .nb_vertices = 4,
        },
    };

    ret = ngli_pipeline_init(&hwconv->pipeline, ctx, &pipeline_params);
    if (ret < 0)
        return ret;

    hwconv->tex_indices[0] = ngli_pipeline_get_texture_index(&hwconv->pipeline, "tex0");
    hwconv->tex_indices[1] = ngli_pipeline_get_texture_index(&hwconv->pipeline, "tex1");
    hwconv->tex_coord_matrix_index = ngli_pipeline_get_uniform_index(&hwconv->pipeline, "tex_coord_matrix");
    hwconv->tex_dimensions_index = ngli_pipeline_get_uniform_index(&hwconv->pipeline, "tex_dimensions");

    return 0;
}

int ngli_hwconv_convert_image(struct hwconv *hwconv, const struct image *image)
{
    struct ngl_ctx *ctx = hwconv->ctx;
    ngli_assert(hwconv->src_params.layout == image->params.layout);

    struct rendertarget *rt = &hwconv->rt;
    struct rendertarget *prev_rt = ngli_gctx_get_rendertarget(ctx);
    ngli_gctx_set_rendertarget(ctx, rt);

    int prev_vp[4] = {0};
    ngli_gctx_get_viewport(ctx, prev_vp);

    const int vp[4] = {0, 0, rt->width, rt->height};
    ngli_gctx_set_viewport(ctx, vp);

    ngli_gctx_clear_color(ctx);

    const struct hwconv_desc *desc = &hwconv_descs[hwconv->src_params.layout];
    float dimensions[4] = {0};
    for (int i = 0; i < desc->nb_planes; i++) {
        struct texture *plane = image->planes[i];
        ngli_pipeline_update_texture(&hwconv->pipeline, hwconv->tex_indices[i], plane);

        const struct texture_params *params = &plane->params;
        dimensions[i*2 + 0] = params->width;
        dimensions[i*2 + 1] = params->height;
    }
    ngli_pipeline_update_uniform(&hwconv->pipeline, hwconv->tex_coord_matrix_index, image->coordinates_matrix);
    ngli_pipeline_update_uniform(&hwconv->pipeline, hwconv->tex_dimensions_index, dimensions);

    ngli_pipeline_exec(&hwconv->pipeline);

    ngli_gctx_set_rendertarget(ctx, prev_rt);
    ngli_gctx_set_viewport(ctx, prev_vp);

    return 0;
}

void ngli_hwconv_reset(struct hwconv *hwconv)
{
    struct ngl_ctx *ctx = hwconv->ctx;
    if (!ctx)
        return;

    ngli_pipeline_reset(&hwconv->pipeline);
    ngli_buffer_reset(&hwconv->vertices);
    ngli_program_reset(&hwconv->program);
    ngli_rendertarget_reset(&hwconv->rt);

    memset(hwconv, 0, sizeof(*hwconv));
}
