#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2019 GoPro Inc.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#


import os
import os.path as op
import pynodegl as ngl
from pynodegl_utils.com import query_subproc, query_inplace


_TEST_MODULE = 'pynodegl_utils.examples'
_HSIZE = 8
_NB_KEYFRAMES = 5
_REF_DIR = op.join(op.dirname(__file__), "refs")


def _get_plane_hashes(buf):
    hashes = []
    linesize = _HSIZE + 1
    comp_bufs = (buf[x::4] for x in range(4))  # R, G, B, A
    for comp_buf in comp_bufs:
        comp_hash = 0
        for y in range(_HSIZE):
            for x in range(_HSIZE):
                pos = y * linesize + x
                px_ref = comp_buf[pos]
                px_xp1 = comp_buf[pos + 1]
                px_yp1 = comp_buf[pos + linesize]
                h_bit = px_ref < px_xp1
                v_bit = px_ref < px_yp1
                comp_hash = comp_hash << 2 | h_bit << 1 | v_bit
        hashes.append(comp_hash)
    return hashes


def _test_scene(scene_data, nb_times=_NB_KEYFRAMES):
    width, height = _HSIZE + 1, _HSIZE + 1
    capture_buffer = bytearray(width * height * 4)
    viewer = ngl.Viewer()
    assert viewer.configure(offscreen=1, width=width, height=height, capture_buffer=capture_buffer) == 0
    timescale = 1. / scene_data["duration"]
    viewer.set_scene_from_string(scene_data["scene"])
    hashes = []
    for t_id in range(nb_times):
        viewer.draw(t_id * timescale)
        h = _get_plane_hashes(capture_buffer)
        hashes.append(h)
    return hashes


def _get_output(hashes):
    return '\n'.join(' '.join(format(x, '032X') for x in comp_hashes) for comp_hashes in hashes) + '\n'


def _parse_output_line(line):
    return [int(x, 16) for x in line.split()]


def _diff_hash(hash1, hash2):
    linesize = _HSIZE + 1
    diff = hash1 ^ hash2
    diff_chars = '.v>+'  # identical, vertical diff, horizontal diff, vertical+horizontal diff
    ret = ''
    for y in range(_HSIZE):
        line = ''
        for x in range(_HSIZE):
            pos = y * linesize + x
            bits = diff >> (pos * 2) & 0b11
            line += ' {}'.format(diff_chars[bits])
        ret += line + '\n'
    return ret


def _diff_hashes(hashes1, hashes2):
    comp_names = "RGBA"
    for comp_name, hash1, hash2 in zip(comp_names, hashes1, hashes2):
        print("Component {}:".format(comp_name))
        print(_diff_hash(hash1, hash2))


def _print_diff(output1, output2):
    for i, (line1, line2) in enumerate(zip(output1.splitlines(), output2.splitlines())):
        hashes1 = _parse_output_line(line1)
        hashes2 = _parse_output_line(line2)
        ok = hashes1 == hashes2
        print('Frame #{}: {}'.format(i, "OK" if ok else "DIFF"))
        if ok:
            continue
        _diff_hashes(hashes1, hashes2)


def _run_tests():
    gen_ref = os.environ.get("GEN") == "1"

    ret = query_inplace(query='list', pkg=_TEST_MODULE)
    assert 'error' not in ret
    scenes = ret['scenes']

    results = []
    for module_name, sub_scenes in scenes:
        for scene_name, scene_doc, widgets_specs in sub_scenes:
            cfg = {
                'pkg': _TEST_MODULE,
                'scene': (module_name, scene_name),
            }
            ret = query_inplace(query='scene', **cfg)
            assert 'error' not in ret

            hashes = _test_scene(ret)
            new_output = _get_output(hashes)

            test_name = '{}_{}'.format(module_name, scene_name)
            ref_filepath = op.join(_REF_DIR, '{}.ref'.format(test_name))

            if gen_ref:
                with open(ref_filepath, "w") as ref_file:
                    ref_file.write(new_output)
                ref_output = new_output
            else:
                with open(ref_filepath, "r") as ref_file:
                    ref_output = ref_file.read()

            results.append((test_name, ref_output, new_output))

    nb_ok = 0
    for test_name, ref_output, new_output in results:
        ok = ref_output == new_output
        nb_ok += ok is True
        print("{}: {}".format(test_name, "OK" if ok else "FAIL"))
        if ref_output != new_output:
            _print_diff(ref_output, new_output)

    print("{}/{} tests passing".format(nb_ok, len(results)))
    assert nb_ok == len(results)

if __name__ == '__main__':
    _run_tests()
