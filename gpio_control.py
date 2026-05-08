from gpiozero import Button, LED
import signal

# GPIO 17 as input (button)
button = Button(17)

# GPIO 24 as output (LED)
led = LED(24)

# When button is pressed (GPIO 17 high), turn on LED (GPIO 24 high)
button.when_pressed = led.on

# When button is released (GPIO 17 low), turn off LED (GPIO 24 low)
button.when_released = led.off

# Keep the script running
signal.pause()