import time

from gpiozero import AngularServo, DigitalInputDevice, DigitalOutputDevice, PWMLED
from picamera2 import Picamera2


class Hardware:
    def __init__(self, config):
        self.config = config
        pins = config["pins"]

        self.beam = DigitalInputDevice(pins["beam"], pull_up=True)
        self.led = PWMLED(pins["led"])
        self.servo = AngularServo(pins["servo"], min_angle=-90, max_angle=90)
        self.data = DigitalInputDevice(pins["hx711_data"])
        self.clock = DigitalOutputDevice(pins["hx711_clock"])

        self.camera = Picamera2()
        self.camera.configure(self.camera.create_still_configuration())
        self.camera.start()

        self.level()

    def wait_for_item(self):
        self.beam.wait_for_inactive()
        self.beam.wait_for_active()

    def read_raw(self):
        while self.data.value:
            time.sleep(0.001)

        value = 0
        for _ in range(24):
            self.clock.on()
            value = (value << 1) | self.data.value
            self.clock.off()

        self.clock.on()
        self.clock.off()

        if value & 0x800000:
            value -= 0x1000000
        return value

    def read_weight(self):
        samples = sorted(self.read_raw() for _ in range(11))
        median = samples[len(samples) // 2]
        cell = self.config["load_cell"]
        return (median - cell["offset"]) / cell["scale"]

    def light(self, on):
        self.led.value = 1.0 if on else 0.0

    def capture(self):
        return self.camera.capture_array()

    def tilt(self, bin_name):
        angle = self.config["tilt_angle_deg"]
        self.servo.angle = angle if bin_name == "recycle" else -angle
        time.sleep(self.config["tilt_hold_seconds"])

    def level(self):
        self.servo.angle = 0
        time.sleep(0.4)

    def close(self):
        self.light(False)
        self.level()
        self.camera.stop()
