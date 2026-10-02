"""Reproducible source audit of the installed Newton Style3D implementation."""
import ast
import hashlib
import inspect
from pathlib import Path
import newton


def audit_style3d():
    root = Path(inspect.getfile(newton.solvers.SolverStyle3D)).parent
    files = ['solver_style3d.py', 'kernels.py', 'builder.py', 'cloth.py',
             'collision/collision.py', 'collision/kernels.py']
    sources = {name: (root/name).read_text() for name in files}
    reads = set()
    for name in ('solver_style3d.py', 'collision/collision.py'):
        reads.update(node.attr for node in ast.walk(ast.parse(sources[name]))
                     if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load))
    inactive = {field: field not in reads for field in ('tri_ka', 'tri_kd', 'tri_materials')}
    kernels = sources['kernels.py']
    # Audit is implementation-specific; emit evidence rather than implying that
    # this table describes all Newton solvers or future versions.
    return dict(
        newton_version=getattr(newton, '__version__', 'unknown'),
        source_directory=str(root),
        source_sha256={name: hashlib.sha256(src.encode()).hexdigest() for name, src in sources.items()},
        parameter_usage={
            'tri_ke_weft/warp/shear': 'Used: style3d.tri_aniso_ke in force and PD matrix.',
            'edge_ke_weft/warp/shear': 'Used: cloth builder converts orientation-dependent bending to edge_bending_properties[:,0].',
            'tri_ka': 'Stored by builder; not read by installed Style3D force kernels.',
            'tri_kd': 'Stored by builder; not read by installed Style3D force kernels.',
            'edge_kd': 'Stored as edge_bending_properties[:,1]; installed bending kernel reads column 0 only.',
            'contact_ke/kd/mu': 'Used by external body-particle contact; friction mixes global and shape mu geometrically.',
            'contact_margin': 'Passed explicitly to each private CollisionPipeline by ClothRuntime.',
            'contact_radius': 'Legacy soft_contact_radius assignment is not consumed by this collision path.',
            'particle_radius': 'Used by external-contact response and capsule clearance.',
            'self_contact': 'Own collision radius 0.003 m and normal VF/EE/EF factors 0.5/0.1/1.0; no tangential self-contact friction in this implementation.',
            'velocity_damping': 'update_velocity multiplies the position-difference velocity by 0.998 each substep.',
        },
        verified_signatures=dict(tri_aniso_ke_read='tri_aniso_ke' in reads,
                                 inactive_triangle_fields=inactive,
                                 fixed_velocity_damping='0.998 *' in kernels,
                                 bending_stiffness_column='edge_bending_properties[eid][0]' in kernels),
        scope='Source audit of installed Style3D, not a calibrated material damage model.')
