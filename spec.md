# SmartBin update spec

Agreed scope from the requirements interview.

## Hardware
Pi Zero W, Camera Module 3, MG90S servo on GPIO12 (hardware PWM), HX711 + 5 kg bar load cell (DT GPIO5, SCK GPIO6). No break-beam, no LED ring.

## Behaviour
- Trigger: a sustained weight change on the load cell.
- Classes (7): rigid_plastic, soft_plastic_foam, cardboard_paper, metal, glass, ewaste_hazard, no_object.
- Recycle: rigid_plastic, metal, glass, cardboard_paper. Trash: soft_plastic_foam, ewaste_hazard, no_object.
- no_object means the pan does not tilt. Low confidence goes to trash. `unknown` never maps to recycle.
- Scale is calibrated with a 47 g reference mass.

## Servo
Hardware PWM through sysfs (`dtoverlay=pwm,pin=12,func=4`), run as root, with a calibration file (center, us per degree, invert).

## Model
- Runtime: small CNN in plain NumPy, weights in `.npz`. Nothing extra to install on the Pi.
- Training: distill the MobileNetV3Large teacher (DumpGuard run3) into a ~128 px depthwise-separable student on the RTX 5080, using the DumpGuard manifest.

## Repo
- `firmware/smartbin` is the single code tree. Old flat scripts, old `ml/`, `production/` and `pi-setup.sh` are removed.
- `ml/` holds the distillation pipeline.
- CAD parts are named without versions: servo-mount, pan-hub, bearing-block, print-plate. Hub horn dimensions are placeholders until measured.
- README rewritten for the real hardware. Commit and push to main.
