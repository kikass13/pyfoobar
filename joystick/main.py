
### pip install devices
from inputs import devices

def calib(connected_devices):
    print("==================================")
    print("press every key do get a dict of all possible keys and quit with start button")
    print("==================================")
    d = {}
    # Iterate over events from the gamepad
    running = True
    for device in connected_devices:
        print(f"Listening to {device.name}")
        while running:
            events = device.read()
            for event in events:
                print(event.ev_type, event.code, event.state)
                d[event.code] = event.code
                if event.code == "BTN_START":
                    running=False
    print("==================================")
    for k,v in d.items():
        print(k)

def run(connected_devices):
    keyset = {
        "BTN_SOUTH" : lambda x: print("S"),
        "SYN_REPORT": lambda x: None,
        "BTN_WEST": lambda x: print("W"),
        "BTN_NORTH": lambda x: print("N"),
        "BTN_EAST": lambda x: print("E"),
        "BTN_SELECT": lambda x: print("SELECT"),
        "BTN_MODE": lambda x: print("MODE"),
        "BTN_THUMBR": lambda x: print("THUMBR"),
        "ABS_RX": lambda x: print("rx %s" % x),
        "ABS_RY": lambda x: print("ry %s" % x),
        "BTN_THUMBL": lambda x: print("THUMBL"),
        "ABS_X": lambda x: print("lx %s" % x),
        "ABS_Y": lambda x: print("ly %s" % x),
        "ABS_HAT0Y": lambda x: print("_"),
        "ABS_HAT0X": lambda x: print("_"),
        "BTN_TL": lambda x: print("_"),
        "BTN_TR": lambda x: print("_"),
        "ABS_Z": lambda x: print("_"),
        "BTN_TL2": lambda x: print("_"),
        "ABS_RZ": lambda x: print("_"),
        "BTN_TR2": lambda x: print("_"),
        "BTN_START": lambda x: print("START"),
    }
    for device in connected_devices:
        print(f"Listening to {device.name}")
        while True:
            events = device.read()
            for event in events:
                key = event.code
                if key in keyset:
                    val = (event.state - 128) * -1
                    f = keyset[event.code]
                    f(val)
                else:
                    print("unknown key: %s | %s | %s" % (event.ev_type, event.code, event.state))

def main():
    # Get a list of connected input devices
    connected_devices = devices.gamepads

    # Print out the connected devices
    print(connected_devices)

    #calib(connected_devices)
    run(connected_devices)

if __name__ == '__main__':
    main()
    