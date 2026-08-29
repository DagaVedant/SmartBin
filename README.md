# SmartBin

A sorting head that drops into a public trash can and puts recyclables in the right side by itself.

[![View PCB on KiCanvas](https://hack.club/pcb-badge)](https://kicanvas.org/?repo=https://github.com/DagaVedant/SmartBin/tree/main/PCB)

![Assembly](CAD/assembly/assembly-front.svg)

## What it does

Someone drops an item in. An infrared beam across the throat notices, the item lands on a
tilting pan, a load cell weighs it, an LED ring lights it and a camera photographs it. A
small neural network on a Raspberry Pi decides what it is, a servo tilts the pan, and the
item slides into either the trash side or the recycling side.

## Why

Public recycling bins have high contamination rates. When enough non-recyclable material
gets in, a materials recovery facility can reject an entire load — so the whole bin goes to
landfill anyway. The failure happens at the moment of disposal, when someone makes a
one-second decision with no feedback and no consequence.

Almost every recycling project tells a *person* what to do. This one does the sorting
itself, so the person's behaviour is not part of the loop.

## How it works

```
item dropped
   -> IR break-beam triggers
   -> item settles on the pan, load cell reads weight
   -> LED ring on, camera captures, LED off
   -> classifier returns one of six categories plus a confidence
   -> weight adjusts the result
   -> county rules map the category to a bin
   -> below the confidence threshold, it goes to trash
   -> servo tilts, item slides, pan returns level
   -> event written to SQLite, served as JSON
```

Two ideas drive most of the design:

**Mistakes are not symmetric.** A recyclable in the trash costs one recyclable. Trash in the
recycling can cost a whole load. So every uncertain decision goes to trash, enforced in two
independent places — a confidence gate at runtime, and a config check that refuses to start
if a rules file would ever send an unidentified item to recycling.

**The Pi cannot sleep between items.** Booting takes 20-30 seconds and an item lands in
under one, so the Pi stays awake and *idle* power dominates the entire budget. That single
fact picks the board: a Zero 2 W idles at about 10 Wh/day where a 4B needs about 65.

## The board

![Board](images/board-top.png)

65 x 45 mm, two layers, 34 components, Pi Zero HAT footprint. The mounting holes sit on the
Pi Zero's 58 x 23 mm pattern; the extra 15 mm overhangs the Pi.

| Block | Parts |
|---|---|
| Load cell front end | HX711, pass transistor, feedback divider, 4 caps |
| Battery sense | MCP3208 12-bit SPI ADC, divider, filter |
| Servo rail | Isolated supply, 470 uF bulk, star ground through a single 0R link |
| LED ring | Logic-level MOSFET low-side switch |
| Break-beam | Receiver pull-up, emitter current limit |

The HX711 runs at 3.3 V rather than 5 V so both its signals are directly Pi-compatible —
at 5 V its logic threshold is 3.5 V, which a 3.3 V GPIO cannot reliably drive.

The two grounds meet at exactly one point. A stalling servo sharing a return path browns
out the Pi and corrupts its SD card, and that failure looks exactly like a software bug for
an infuriating number of hours.

The servo runs on GPIO12 and the LED ring on GPIO13. Those are deliberately on different
hardware PWM channels — GPIO12 and GPIO18 are both PWM0, so the obvious pairing would have
put two things that need independent duty cycles on one timer.

## The mechanics

![Side view](CAD/assembly/assembly-side.svg)

7 fabricated parts plus the servo and bearing, 21 placed instances, 180 x 330 x 101 mm
envelope. Every part is a single connected
solid, every joint measures a 0.00 mm gap, there are zero static interferences, and the pan
clears everything through its full ±45 deg sweep with 4.9 mm to spare.

The CAD covers the pan mechanism only — the part that actually does the sorting — mounted
flat on a base plate for demo:

| Part | Job |
|---|---|
| `base-plate` | 330 x 100 x 6 mm datum everything bolts to |
| `pan` | The tilting tray the item lands on |
| `pan-hub` | Driven side, couples to the servo horn |
| `pan-idler` | Free side, rides in the 608ZZ |
| `servo-mount` | MG996R drops into a captive pocket, flange screws through a 4 mm face plate, shaft on the pivot axis |
| `bearing-block` | Carries the 608ZZ on the idler side |
| `pi-mount` | Pi Zero on 6 mm standoffs, 58 x 23 mm pattern |

The pivot is held at two points 206 mm apart. The base plate is what keeps those two
supports coaxial — without it the servo's own reaction torque twists them out of alignment
and the pivot binds. All three bracket feet are coplanar so they land flat on it.

The pivot sits 70 mm above the plate. That is not arbitrary: the pan is 180 mm wide, so at
45 deg its edges swing 65 mm below the axis. Any lower and the pan hits the plate before it
finishes tilting.

The base plate is 320 mm long, which does not fit a 220 mm bed. It is cut from 6 mm plywood
or acrylic rather than printed. The other six parts all print.

Everything around the mechanism — the bridge that carries the camera, the throat funnel, the
sensor mounts, the chute and the enclosure — is not modelled.

`smartbin-head.step` carries the whole thing as one named, colour-coded assembly tree rather
than a flattened solid, so it opens in Onshape, Fusion or SolidWorks with every part listed
separately. In Onshape: **+ → Import**, and leave "Import as a single Part Studio" unchecked
so it lands as an assembly. Parts arrive pre-positioned but as fixed geometry — to animate
the tilt you would re-mate the pan to the two supports with a revolute mate on the pivot axis.

## Repository layout

| Path | Contents |
|---|---|
| `CAD/assembly/` | `smartbin-head.step` — full assembly, named parts. Plus front/side views |
| `CAD/parts/` | The 7 parts as STEP |
| `PCB/` | KiCad schematic, board and project, plus gerbers in `PCB/fab/` |
| `images/` | Board renders |
| `firmware/` | On-device Python service: trigger, capture, classify, gate, actuate, log, JSON API |
| `ml/` | Classifier training and session-split evaluation. Never runs on the device. |
| `bill_of_materials.csv` | Every part needed to build one |

## State

| Item | Status |
|---|---|
| Schematic | 1 error — C6 pin 1 is missing a junction to the battery divider |
| Board layout | Routed, ground plane poured, **DRC clean** — 0 violations, 0 unconnected |
| Gerbers | Exported to `PCB/fab/` |
| CAD | 7 parts, zero interferences, all joints in contact |
| Firmware | Gate, config, storage and sort loop tested — 46 tests, runs with no hardware attached |
| Classifier | Not trained. No dataset collected yet. |
| Physical build | Not started |

Nothing here has been fabricated or printed yet. The numbers above come from KiCad's DRC/ERC
and from an interference check on the CAD assembly, not from measurement.

## Known unknowns

The pan tilt angle is 45 deg because that was the starting guess, and no one has confirmed
that a wet or crumpled piece of paper actually slides off a printed PETG pan at that angle.
Every part downstream of the pan depends on it. Print the pan and test it before building
anything else.
