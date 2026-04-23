# modules/rpi_controller.py
# Raspberry Pi 4B Hardware Controller — GPIO, IR LED, Status LEDs
#
# Pin Assignments (config.py):
#   GPIO 18 (PWM0) → IR LED brightness via MOSFET gate (1kHz PWM)
#   GPIO 17        → IR LED enable (active HIGH)
#   GPIO 27        → Status LED green (system OK)
#   GPIO 22        → Alert LED red (low RA warning)
#
# Circuit:  RPi GPIO 18 ──[1kΩ]──→ Gate(IRLZ34N)
#           Source → GND, Drain → IR LED array → 12V supply
#           [1N4007 flyback diode across LED array]

import time
import threading
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import GPIO_IR_PWM, GPIO_IR_ENABLE, GPIO_STATUS_LED, GPIO_ALERT_LED


def _is_raspberry_pi() -> bool:
    """Detect if running on Raspberry Pi hardware."""
    try:
        with open("/proc/device-tree/model", "r") as f:
            return "raspberry pi" in f.read().lower()
    except (FileNotFoundError, PermissionError):
        return False


class RPiController:
    """
    Raspberry Pi GPIO controller for RetroFusion hardware.

    In production (on RPi):
        - Controls IR LED array via PWM (1kHz, 0-100% duty cycle)
        - Manages status/alert indicator LEDs
        - Auto-adjusts IR brightness based on ambient lux

    In simulation (non-RPi):
        - All GPIO calls are no-ops with logging
        - State is tracked internally for dashboard display
    """

    def __init__(self, simulation: bool = None):
        # Auto-detect simulation mode if not specified
        if simulation is None:
            self.simulation = not _is_raspberry_pi()
        else:
            self.simulation = simulation

        self._ir_enabled = False
        self._ir_duty = 0
        self._status_led = False
        self._alert_led = False
        self._pwm = None
        self._lock = threading.Lock()

        if not self.simulation:
            self._init_gpio()
        else:
            print("[RPi] Running in SIMULATION mode (no GPIO hardware)")

    def _init_gpio(self):
        """Initialize GPIO pins on real Raspberry Pi."""
        try:
            import RPi.GPIO as GPIO
            self._GPIO = GPIO

            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)

            # IR LED control
            GPIO.setup(GPIO_IR_ENABLE, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(GPIO_IR_PWM, GPIO.OUT)
            self._pwm = GPIO.PWM(GPIO_IR_PWM, 1000)  # 1kHz PWM
            self._pwm.start(0)

            # Status LEDs
            GPIO.setup(GPIO_STATUS_LED, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(GPIO_ALERT_LED, GPIO.OUT, initial=GPIO.LOW)

            # System OK indicator
            GPIO.output(GPIO_STATUS_LED, GPIO.HIGH)
            self._status_led = True

            print("[RPi] GPIO initialized successfully")
            print(f"      IR PWM: GPIO {GPIO_IR_PWM}")
            print(f"      IR Enable: GPIO {GPIO_IR_ENABLE}")
            print(f"      Status LED: GPIO {GPIO_STATUS_LED}")
            print(f"      Alert LED: GPIO {GPIO_ALERT_LED}")

        except ImportError:
            print("[RPi] RPi.GPIO not available. Falling back to simulation.")
            self.simulation = True
        except Exception as e:
            print(f"[RPi] GPIO init failed: {e}. Falling back to simulation.")
            self.simulation = True

    def set_ir_brightness(self, ambient_lux: float):
        """
        Auto-adjust IR LED brightness based on ambient light level.

        Logic (from design doc Section 3.2):
            ambient < 5 lux    → Night  → 80% duty cycle (full IR)
            ambient < 50 lux   → Dusk   → 40% duty cycle (partial IR)
            ambient >= 50 lux  → Day    → 0% duty cycle (IR off, use daylight)
        """
        with self._lock:
            if ambient_lux < 5:
                duty = 80     # Night — full IR illumination
            elif ambient_lux < 50:
                duty = 40     # Dusk — partial IR
            else:
                duty = 0      # Day — IR off, use daylight

            self._ir_duty = duty
            enable = duty > 0

            if not self.simulation:
                self._GPIO.output(GPIO_IR_ENABLE, enable)
                self._pwm.ChangeDutyCycle(duty)
            
            self._ir_enabled = enable

    def set_ir_manual(self, duty_cycle: int):
        """Manually set IR LED duty cycle (0-100%)."""
        with self._lock:
            duty = max(0, min(100, duty_cycle))
            self._ir_duty = duty
            self._ir_enabled = duty > 0

            if not self.simulation:
                self._GPIO.output(GPIO_IR_ENABLE, self._ir_enabled)
                self._pwm.ChangeDutyCycle(duty)

    def set_status_led(self, on: bool):
        """Set green status LED (system OK indicator)."""
        with self._lock:
            self._status_led = on
            if not self.simulation:
                self._GPIO.output(GPIO_STATUS_LED, on)

    def set_alert_led(self, on: bool):
        """Set red alert LED (low RA warning)."""
        with self._lock:
            self._alert_led = on
            if not self.simulation:
                self._GPIO.output(GPIO_ALERT_LED, on)

    def alert_on_fail(self, status: str):
        """
        Automatically control alert LED based on measurement status.

        FAIL → red LED ON (steady)
        MARGINAL → red LED BLINK (handled by caller via threading)
        PASS → red LED OFF, green LED ON
        """
        if status == "FAIL":
            self.set_alert_led(True)
            self.set_status_led(False)
        elif status == "MARGINAL":
            self.set_alert_led(True)
            self.set_status_led(True)
        else:  # PASS
            self.set_alert_led(False)
            self.set_status_led(True)

    def get_state(self) -> dict:
        """Return current GPIO state for dashboard display."""
        with self._lock:
            return {
                "simulation": self.simulation,
                "ir_enabled": self._ir_enabled,
                "ir_duty_cycle": self._ir_duty,
                "status_led": self._status_led,
                "alert_led": self._alert_led,
            }

    def cleanup(self):
        """Clean up GPIO pins on shutdown."""
        if not self.simulation:
            try:
                self._pwm.stop()
                self._GPIO.output(GPIO_IR_ENABLE, False)
                self._GPIO.output(GPIO_STATUS_LED, False)
                self._GPIO.output(GPIO_ALERT_LED, False)
                self._GPIO.cleanup()
                print("[RPi] GPIO cleaned up")
            except Exception as e:
                print(f"[RPi] Cleanup error: {e}")

    def __del__(self):
        self.cleanup()
