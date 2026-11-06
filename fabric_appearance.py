# SPDX-License-Identifier: Apache-2.0
"""USD appearance for the cloth; deliberately independent of Newton mechanics."""
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parent/'assets'/'materials'/'dry_carbon_twill'
MATERIALS = ('dry_carbon_twill', 'legacy_red')


def cloth_vertex_normals(positions, triangles):
    """Area-weighted shading normals; never used by the physical model."""
    q = np.asarray(positions)
    faces = np.asarray(triangles)
    points = q[faces]
    face_normals = np.cross(points[:, 1]-points[:, 0], points[:, 2]-points[:, 0])
    normals = np.zeros_like(q)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals/np.maximum(lengths, 1e-12)


def bind_cloth_material(mesh, rest_positions, appearance='dry_carbon_twill', yarn_pitch=.002):
    """Bind a local MDL material plus a portable UsdPreviewSurface fallback.

    UVs are derived once from material rest positions, never current positions.
    Scaling by rest dimensions keeps the physical yarn pitch independent of grid
    resolution, sag, and the sheet's subsequent rigid/deformable motion.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    if appearance not in MATERIALS:
        raise ValueError(f'Unknown cloth appearance: {appearance}')
    if not np.isfinite(yarn_pitch) or yarn_pitch <= 0:
        raise ValueError('Yarn pitch must be finite and positive')
    prim = mesh.GetPrim()
    prim.CreateAttribute('fabric:appearance', Sdf.ValueTypeNames.Token).Set(appearance)
    prim.CreateAttribute('fabric:mechanicalCalibration', Sdf.ValueTypeNames.Token).Set('uncalibrated')
    if appearance == 'legacy_red':
        mesh.CreateDisplayColorPrimvar('constant').Set([Gf.Vec3f(.75, .15, .15)])
        return

    rest = np.asarray(rest_positions, dtype=float)
    if rest.ndim != 2 or rest.shape[1] != 3 or not np.isfinite(rest).all():
        raise ValueError('Finite (N, 3) cloth rest positions required')
    size = np.ptp(rest[:, :2], axis=0)
    if np.any(size <= 0):
        raise ValueError('Cloth rest sheet must span both material axes')
    st = ((rest[:, :2]-rest[:, :2].min(axis=0))/size).astype(np.float32)
    UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
        'st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
            Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(st)))
    mesh.CreateDisplayColorPrimvar('constant').Set([Gf.Vec3f(.012, .013, .014)])
    prim.CreateAttribute('fabric:yarnPitchM', Sdf.ValueTypeNames.Float).Set(yarn_pitch)
    prim.CreateAttribute('fabric:weave', Sdf.ValueTypeNames.Token).Set('2x2_twill')
    scale = Gf.Vec2f(*(size/(4*yarn_pitch)))

    required = [ASSETS/'DryCarbonTwill.mdl', *(ASSETS/f'carbon_twill_{name}.png'
                 for name in ('basecolor', 'normal', 'roughness', 'warp_mask'))]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f'Cloth appearance asset missing: {path}')
    stage = prim.GetStage()
    material = UsdShade.Material.Define(stage, '/World/Looks/DryCarbonTwill')
    shader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild('Mdl'))
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(Sdf.AssetPath(str(ASSETS/'DryCarbonTwill.mdl')), 'mdl')
    shader.SetSourceAssetSubIdentifier('DryCarbonTwill', 'mdl')
    shader.CreateInput('texture_scale', Sdf.ValueTypeNames.Float2).Set(scale)
    shader.CreateInput('normal_strength', Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateOutput('out', Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput('mdl').ConnectToSource(shader.ConnectableAPI(), 'out')

    # The fallback uses the same filtered maps for USD viewers without MDL.
    uv = UsdShade.Shader.Define(stage, material.GetPath().AppendChild('UV'))
    uv.CreateIdAttr('UsdPrimvarReader_float2')
    uv.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
    uv.CreateOutput('result', Sdf.ValueTypeNames.Float2)
    transform = UsdShade.Shader.Define(stage, material.GetPath().AppendChild('Scale'))
    transform.CreateIdAttr('UsdTransform2d')
    transform.CreateInput('in', Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), 'result')
    transform.CreateInput('scale', Sdf.ValueTypeNames.Float2).Set(scale)
    transform.CreateOutput('result', Sdf.ValueTypeNames.Float2)
    preview = UsdShade.Shader.Define(stage, material.GetPath().AppendChild('Preview'))
    preview.CreateIdAttr('UsdPreviewSurface')
    preview.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.)
    preview.CreateInput('useSpecularWorkflow', Sdf.ValueTypeNames.Int).Set(1)
    preview.CreateInput('specularColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.025))
    for name, target, value_type, channel in (
            ('basecolor', 'diffuseColor', Sdf.ValueTypeNames.Color3f, 'rgb'),
            ('roughness', 'roughness', Sdf.ValueTypeNames.Float, 'r'),
            ('normal', 'normal', Sdf.ValueTypeNames.Normal3f, 'rgb')):
        texture = UsdShade.Shader.Define(stage, material.GetPath().AppendChild(name.title()))
        texture.CreateIdAttr('UsdUVTexture')
        texture.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(str(ASSETS/f'carbon_twill_{name}.png')))
        texture.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set(
            'sRGB' if name == 'basecolor' else 'raw')
        for axis in ('S', 'T'):
            texture.CreateInput('wrap'+axis, Sdf.ValueTypeNames.Token).Set('repeat')
        texture.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(transform.ConnectableAPI(), 'result')
        if name == 'normal':
            texture.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2., 2., 2., 1.))
            texture.CreateInput('bias', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1., -1., -1., 0.))
        texture.CreateOutput(channel, Sdf.ValueTypeNames.Float3 if channel == 'rgb' else Sdf.ValueTypeNames.Float)
        preview.CreateInput(target, value_type).ConnectToSource(texture.ConnectableAPI(), channel)
    preview.CreateOutput('surface', Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(preview.ConnectableAPI(), 'surface')
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
