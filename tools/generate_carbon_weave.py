#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the original, tileable 2x2 twill appearance maps (NumPy + Pillow).

One tile is four yarns wide, nominally 8 mm at the default 2 mm yarn pitch.
These are illustrative surface maps, not measured carbon reflectance/geometry.
Runtime uses the checked-in PNGs and does not need Pillow or this generator.
"""
from pathlib import Path

import numpy as np
from PIL import Image


def generate(output, resolution=512):
    output.mkdir(parents=True, exist_ok=True)
    # Pixel-centred samples of a strictly periodic four-yarn weave repeat.
    v, u = np.meshgrid((np.arange(resolution)+.5)*4/resolution,
                       (np.arange(resolution)+.5)*4/resolution, indexing='ij')
    ix, iy = np.floor(u), np.floor(v)
    x, y = u-ix, v-iy
    # Each yarn passes over two and under two; successive rows shift by one.
    warp = np.mod(ix+iy, 4) < 2
    across, along = np.where(warp, x, y), np.where(warp, v, u)
    crown = np.sin(np.pi*across)**.65
    # Subtle longitudinal filament relief, without baked directional lighting.
    fibers = .5+.5*np.cos(2*np.pi*(across*23+.14*np.sin(2*np.pi*along/4)))
    ends = .90+.10*np.cos(np.pi*(along+np.where(warp, ix, iy))/2)**2
    height = (.72*crown+.025*fibers)*ends
    gap = np.clip(crown/.24, 0, 1)
    value = (.105+.022*crown+.012*fibers)*(.4+.6*gap)
    color = np.stack((value*.96, value*.98, value), axis=-1)
    # Tangent-space normal, OpenGL +Y. Image rows increase in material V.
    # A modest 0.035 mm relief over an 8 mm repeat; no mesh displacement.
    du = (np.roll(height, -1, axis=1)-np.roll(height, 1, axis=1))/(2/resolution)
    dv = (np.roll(height, -1, axis=0)-np.roll(height, 1, axis=0))/(2/resolution)
    normal = np.stack((-du*.000035/.008, -dv*.000035/.008, np.ones_like(u)), axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    # File row zero is image top; flip together so mask, color and normals agree.
    maps = {'basecolor': color, 'normal': normal*.5+.5,
            'warp_mask': warp.astype(float), 'roughness': .54+.08*(1-crown)}
    for name, data in maps.items():
        pixels = np.rint(np.clip(np.flipud(data), 0, 1)*255).astype(np.uint8)
        Image.fromarray(pixels).save(output/f'carbon_twill_{name}.png', optimize=True)


if __name__ == '__main__':
    generate(Path(__file__).resolve().parents[1]/'assets'/'materials'/'dry_carbon_twill')
