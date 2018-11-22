/*
 * Copyright 2017 GoPro Inc.
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

#ifndef HWUPLOAD_H
#define HWUPLOAD_H

#include <sxplayer.h>

#include "nodegl.h"

struct hwupload_config {
    int format;
    int width;
    int height;
    int linesize;
    int data_format;
};

int ngli_hwupload_upload_frame(struct ngl_node *node, struct sxplayer_frame *frame);
void ngli_hwupload_uninit(struct ngl_node *node);

#endif /* HWUPLOAD_H */
