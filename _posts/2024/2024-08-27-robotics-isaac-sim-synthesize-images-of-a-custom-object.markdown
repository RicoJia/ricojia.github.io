---
layout: post
title: Robotics - Isaac SIM Image Synthesis for A Custom Object
date: 2024-08-25 13:19
subtitle: Isaac SIM
header-img: img/post-bg-unix
tags:
  - Robotics
comments: true
---
## Synthesize Images using Isaac Sim

Here is a minimal pipeline to synthesize images given the CAD of an object, in the YOLO dataset format

```
CAD/USD object
  → Isaac Sim headless renderer
  → randomized camera / lighting / material / background / distractors
  → RGB image
  → semantic segmentation mask from Isaac Replicator
  → YOLO bbox from that mask
  → RF-DETR-style dataset folder
```

Tools being used:

- **Isaac Sim 6.0 / SimulationApp**: This is the main Omniverse/Isaac Sim app in headless mode that we use.
- **Omniverse Replicator** `omni.replicator.core`: Creates camera, render product, **RGB annotator**, semantic segmentation annotator. RGB annotater outputs RGB images; semantic segmentation annotator gives per-pixel object / class IDs
- **USD / Pixar pxr** `Usd`, `UsdGeom`, `UsdShade`, `Gf`, `Sdf`: Creates/edits scene objects, materials, lights, meshes, transforms. USD (universal scene description) is a 3D framework for complex lightweight scenes and rendering, whereas CAD stores exact, highly accurate physical dimensions.
 	- USD stores meshes, transforms, object hierachy, materials, textures, cameras, lights, joints.. properties of multiple objects in a scene
 	- CAD stores extrudes / cuts/ fillets, precise dimensions, exact geometry like cylinders  instead of meshes
- **Isaac Sim semantics utils**: Adds semantic class labels to the motor sleeve prim/mesh
- **Warp**: GPU kernel backend used internally by Replicator; script redirects cache to `/tmp`

### Sample workflow (skipping operations that are less relevant)

```python

# Launches Isaac Sim without a GUI.We geenerate fixed size images. 
simulation_app = SimulationApp(
    launch_config={
        "headless": True,
        "width": 640,
        "height": 480,
        # this is quite photorealistic
        "renderer": "RayTracedLighting",
    }
)

# Create annotator with placeholder render_product (basically output from the camera)
rgb_annotator = rep.annotators.get("rgb")
rgb_annotator.attach(render_product)

sem_annotator = rep.annotators.get(
    "semantic_segmentation",
    init_params={"semanticTypes": ["class"], "colorize": False},
)
# the USD prim has semantic labels ALREADY.
sem_annotator.attach(render_product)

stage = omni.usd.get_context().get_stage()
# Load the object's CAD/USD reference and finds its mesh and bounding box
stage.DefinePrim("/World", "Xform")
sleeve_prim = stage.DefinePrim("/World/MotorSleeve", "Xform")
sleeve_prim.GetReferences().AddReference(USD_PATH)
mesh_prim = find_first_mesh(stage)
# calculate 3D bounding box of the object in world frame
mn, mx, center, extent = compute_bbox(mesh_prim)


# Now the fun begins: add randomizable material to the object
mat_shader = setup_material(stage, mesh_prim)
for i in NUM_FRAME: 
 # randomize diffuse color, roughness
 mat_name, mat_diffuse, mat_roughness, mat_metallic = randomize_material(mat_shader, mat_idx)
 # randmoize dome light, key light, and fill light
 dome_prim, key_prim, fill_prim = add_lights(stage, center, extent)
 # randomize camera pose 
 rep.functional.modify.pose(
     cam_prim,
     position_value=(cam_x, cam_y, cam_z),
     look_at_value=(look_x, look_y, look_z),
 )
 
 # ── Semantic segmentation -> motor_sleeve-only mask ───────────────
 seg_raw = sem_annotator.get_data()
 mask, seg_info, unique_semantic_ids = semantic_mask_for_label(seg_raw, LABEL)
```

Most important aspect is camera pose randomization. We randomize the camera pose on a hypothetical sphere ( the azimuth - elevation - radius model, a.k.a spherical coordinates).
