# pan-v2: lightweight, support-free tilting tray.
# Prints right side up with the rib grid on the bed. The top skin bridges
# between ribs, the bosses are cut flat at the bed, and the screw holes are
# teardrops, so nothing needs support.
# Interfaces match pan-v1: pivot axis along Y at z=2, boss faces at y=+-90,
# 2x M3 self-tap holes at x=+-9 on the axis for pan-hub / pan-idler.
import cadquery as cq

L, W = 180.0, 160.0       # floor footprint (x = slide direction, y = pivot axis)
CORNER_R = 6.0
TOP = 4.0                 # landing surface height, same as v1
SKIN = 1.0                # top skin: 1 bridge layer + 4 solid at 0.2 mm
BOTTOM = -4.0             # bed plane
RIB = 0.8                 # grid ribs, two 0.4 mm lines
RIM = 1.2                 # outer rim, wider for bed grip
RIBS_X = [-60.0, -30.0, 0.0, 30.0, 60.0]   # ribs running along y
RIBS_Y = [-54.0, -27.0, 0.0, 27.0, 54.0]   # ribs running along x

WALL_T, WALL_TOP, WALL_X = 1.6, 20.0, 84.0
WALL_FILLET = 3.0         # 45 deg gusset where wall meets floor

AXIS_Z = 2.0
BOSS_R, BOSS_Y0, BOSS_Y1 = 15.0, 78.0, 90.0
BOSS_TOP = 9.0            # boss only needs to surround the screw holes
HOLE_R, HOLE_X, HOLE_DEPTH = 1.4, 9.0, 11.0

skin_z = TOP - SKIN
outline = cq.Sketch().rect(L, W).vertices().fillet(CORNER_R)

# top skin
pan = cq.Workplane("XY").workplane(offset=skin_z).placeSketch(outline).extrude(SKIN)

# perimeter rim under the skin
rim = (
    cq.Workplane("XY").workplane(offset=BOTTOM).placeSketch(outline).extrude(skin_z - BOTTOM)
    .cut(
        cq.Workplane("XY").workplane(offset=BOTTOM)
        .placeSketch(cq.Sketch().rect(L - 2 * RIM, W - 2 * RIM).vertices().fillet(CORNER_R - RIM))
        .extrude(skin_z - BOTTOM)
    )
)
pan = pan.union(rim)

# rib grid
h = skin_z - BOTTOM
for x in RIBS_X:
    pan = pan.union(cq.Workplane("XY").box(RIB, W - 2 * RIM, h, centered=(True, True, False)).translate((x, 0, BOTTOM)))
for y in RIBS_Y:
    pan = pan.union(cq.Workplane("XY").box(L - 2 * RIM, RIB, h, centered=(True, True, False)).translate((0, y, BOTTOM)))

# end walls with an inside gusset
for s in (1, -1):
    wall = cq.Workplane("XY").box(2 * WALL_X, WALL_T, WALL_TOP - TOP, centered=(True, False, False))
    gusset = (
        cq.Workplane("YZ").polyline([(0, 0), (-WALL_FILLET, 0), (0, WALL_FILLET)]).close()
        .extrude(WALL_X, both=True)
    )
    wall = wall.union(gusset).translate((0, W / 2 - WALL_T, TOP))
    if s < 0:
        wall = wall.mirror("XZ")
    pan = pan.union(wall)

# pivot bosses, cut flat at the bed and flat on top
for s in (1, -1):
    boss = (
        cq.Workplane("XZ", origin=(0, s * BOSS_Y0, 0)).center(0, AXIS_Z).circle(BOSS_R)
        .extrude(-s * (BOSS_Y1 - BOSS_Y0))
    )
    boss = boss.intersect(cq.Workplane("XY").box(2 * BOSS_R, 400, BOSS_TOP - BOTTOM, centered=(True, True, False)).translate((0, 0, BOTTOM)))
    pan = pan.union(boss)

# teardrop screw holes, point up so the roof is 45 deg
apex = HOLE_R * 2 ** 0.5
k = HOLE_R / 2 ** 0.5
for s in (1, -1):
    for x in (-HOLE_X, HOLE_X):
        tear = (
            cq.Workplane("XZ", origin=(0, s * BOSS_Y1, 0)).center(x, AXIS_Z)
            .circle(HOLE_R).extrude(s * HOLE_DEPTH)
            .union(
                cq.Workplane("XZ", origin=(0, s * BOSS_Y1, 0)).center(x, AXIS_Z)
                .polyline([(-k, k), (0, apex), (k, k), (0, 0)]).close().extrude(s * HOLE_DEPTH)
            )
        )
        pan = pan.cut(tear)

if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    cq.exporters.export(pan, os.path.join(here, "pan-v2.step"))
    cq.exporters.export(pan, os.path.join(here, "pan-v2.stl"), tolerance=0.02, angularTolerance=0.1)
