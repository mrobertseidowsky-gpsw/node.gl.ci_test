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

#ifndef BUFFER_H
#define BUFFER_H

#ifdef VULKAN_BACKEND
#include <vulkan/vulkan.h>
#else
#include "glincludes.h"
#endif

struct buffer {
    struct ngl_ctx *ctx;
    int size;
    int usage;
#ifdef VULKAN_BACKEND
    VkBuffer vkbuf;
    VkDeviceMemory vkmem;
#else
    GLuint id;
#endif
};

int ngli_buffer_init(struct buffer *s, struct ngl_ctx *ctx, int size, int usage);
int ngli_buffer_upload(struct buffer *s, void *data, int size);
void ngli_buffer_reset(struct buffer *s);
void *ngli_buffer_map(struct buffer *s);
void ngli_buffer_unmap(struct buffer *s);

#endif
