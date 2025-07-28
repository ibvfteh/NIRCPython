import os
import sys
import argparse
import signal
import sys
from glob import glob
from pathlib import Path
import torch
import falcor
import numpy as np
import pyexr as exr
import json
from PIL import Image
import io


def load_scene(testbed: falcor.Testbed, scene_path: Path, aspect_ratio=1.0):
    flags = (
        falcor.SceneBuilderFlags.DontMergeMaterials
        | falcor.SceneBuilderFlags.RTDontMergeDynamic
        | falcor.SceneBuilderFlags.DontOptimizeMaterials
    )
    testbed.load_scene(scene_path, flags)
    testbed.scene.camera.aspectRatio = aspect_ratio
    testbed.scene.renderSettings.useAnalyticLights = False
    testbed.scene.renderSettings.useEnvLight = False
    return testbed.scene


def create_testbed(reso: (int, int)):
    device_id = 0
    testbed = falcor.Testbed(
        width=reso[0], height=reso[1], create_window=True, gpu=device_id
    )
    testbed.show_ui = False
    testbed.clock.time = 0
    testbed.clock.pause()
    return testbed


def create_passes(testbed: falcor.Testbed):
    render_graph = testbed.create_render_graph("StandardPathTracer")

    # Create the PathTracer pass.
    path_tracer_pass = render_graph.create_pass(
        "PathTracer",
        {
            "samplesPerPixel": 1
        }
    )

    # Create the VBufferRT pass.
    vbuffer_rt_pass = render_graph.create_pass(
        "VBufferRT",
        {
            "samplePattern": "Stratified",
            "sampleCount": 16,
            "useAlphaTest": True
        }
    )

    # Add edges to connect the passes.
    render_graph.add_edge("VBufferRT.vbuffer", "PathTracer.vbuffer")
    render_graph.add_edge("VBufferRT.viewW", "PathTracer.viewW")
    render_graph.add_edge("VBufferRT.mvec", "PathTracer.mvec")

    # Mark the output of the PathTracer pass.
    render_graph.mark_output("PathTracer.color")

    # Assign the configured render graph to the testbed.
    testbed.render_graph = render_graph

    # Return a dictionary of the created passes.
    return {
        "path_tracer": path_tracer_pass,
        "vbuffer_rt": vbuffer_rt_pass,
    }


def render_primal(spp: int, testbed: falcor.Testbed, passes):
    passes["war_diff_pt"].run_backward = 0
    passes["primal_accumulate"].reset()
    for i in range(spp):
        testbed.frame()

    img = testbed.render_graph.get_output("PrimalAccumulatePass.output").to_numpy()
    img = torch.from_numpy(img[:, :, :3]).cuda()
    return img


def render_grad(spp: int, testbed: falcor.Testbed, passes, dL_dI_buffer, grad_type):
    passes["war_diff_pt"].run_backward = 1
    passes["war_diff_pt"].dL_dI = dL_dI_buffer

    scene_gradients = passes["war_diff_pt"].scene_gradients
    scene_gradients.clear(testbed.device.render_context, grad_type)

    for _ in range(spp):
        testbed.frame()

    scene_gradients.aggregate(testbed.device.render_context, grad_type)

    grad_buffer = scene_gradients.get_grads_buffer(grad_type)
    grad = torch.tensor([0] * (grad_buffer.size // 4), dtype=torch.float32)
    grad_buffer.copy_to_torch(grad)
    testbed.device.render_context.wait_for_cuda()
    return grad / float(spp)
