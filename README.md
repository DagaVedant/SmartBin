# smartbin

a sorting head that drops into a normal trash can and puts recyclables in the right side by itself.

[![View PCB on KiCanvas](https://hack.club/pcb-badge)](https://kicanvas.org/?repo=https://github.com/DagaVedant/SmartBin/tree/main/PCB)

![board](images/board-top.png)

## what it does

you throw something in. an ir beam notices, the item lands on a tilting pan, a load cell weighs it, an led ring lights it and a camera photographs it. a small neural net on a pi decides what it is, a servo tilts the pan, and the item slides into either the trash side or the recycling side.

## why

- public recycling bins get contaminated really easily
- if enough non-recyclable stuff ends up in one, the facility can reject the whole load, so all of it goes to landfill anyway
- basically every recycling project i've seen is an app that tells *you* which bin to use
- i wanted the bin to just do it, so the person doesn't have to care

## how it works

```
item dropped
   -> ir break-beam triggers
   -> item settles on the pan, load cell reads weight
   -> led ring on, camera captures, led off
   -> classifier returns one of six categories plus a confidence
   -> weight adjusts the result
   -> county rules map the category to a bin
   -> below the confidence threshold it goes to trash
   -> servo tilts, item slides, pan returns level
   -> event written to sqlite, served as json
```

## the board

65 × 45 mm, 2 layers, 34 components, pi zero hat footprint. mounting holes on the 58 × 23 pattern; the extra 15 mm overhangs the pi.

| block | parts |
|---|---|
| load cell front end | hx711, pass transistor, feedback divider, 4 caps |
| battery sense | mcp3208 12-bit spi adc, divider, filter |
| servo rail | isolated supply, 470 uf bulk, star ground through one 0r link |
| led ring | logic-level mosfet low-side switch |
| break-beam | receiver pull-up, emitter current limit |

- **hx711 runs at 3.3 v, not 5 v.** at 5 v its logic threshold is 3.5 v and a 3.3 v gpio can't reliably hit that. running it at 3.3 v deletes 4 level-shifting parts
- **the two grounds meet at exactly one point.** a stalling servo sharing a return browns out the pi and corrupts the sd card, and it looks exactly like a software bug for hours
- **servo on gpio12, led on gpio13.** deliberately different pwm channels -- gpio12 and gpio18 are both pwm0, so the obvious pairing would've put two things needing independent duty cycles on one timer

fully routed, ground plane poured, **drc clean -- 0 violations, 0 unconnected.** gerbers in `PCB/gerber_drl_files/`.

## the mechanics

![assembly](images/assembly.png)

| part | job |
|---|---|
| `base-plate` | 330 × 100 × 6 mm datum everything bolts to |
| `pan` | the tilting tray the item lands on |
| `pan-hub` | driven side, couples to the servo horn |
| `pan-idler` | free side, rides in the 608zz |
| `servo-mount` | mg996r drops into a captive pocket, shaft on the pivot axis |
| `bearing-block` | carries the 608zz on the idler side |
| `pi-mount` | pi on 6 mm standoffs, 58 × 23 pattern |

## the classifier

six classes: pet bottle, metal can, clean paper, organics, soiled, unknown. mobilenetv2 with a frozen backbone, exported as int8 tflite because a zero 2 w has 512 mb and no accelerator.

```
python build_manifest.py     scan session folders -> manifest.csv
python train.py              train on the session split
python evaluate.py           honest vs naive accuracy, confusion, recycle precision
python export.py             int8 tflite for the pi
```
