# Keyboard robot · DTU 34753

Problem 11a asks the real arm to recognise the keys of a keyboard placed in front of it and type `hello world`. This folder is a standalone implementation for the four AX-12A motors, wrist camera and stylus used by Group 3. Start with `type_hello.py`.

## Folder layout

```
Project/
├── type_hello.py        main program: photo, plan, press the keys
├── robot/               robot code (imported, not run)
├── tests/               camera_test.py (live view), calibrate_camera.py (keyboard + camera)
├── requirements.txt
└── README.md
```

## Setup (once)

```
brew install tesseract
python -m pip install -r requirements.txt
```

## Run

Run everything from the `Project` folder. The port (`/dev/tty.usbmodem*`) is found
automatically and camera 0 is used; override with `--port` and `--camera`.

1. Power the arm, connect the camera and keyboard, and clear the area the links will sweep. Keep the same rigid camera mount and the keyboard on the calibrated desk. The board can translate and rotate between runs. Keep it still during a run.
2. Plan once. The robot moves to the saved camera pose, takes a fresh photo, prints key coordinates in base millimetres and checks the whole route. It saves the latest exposure as `last-camera.png`. This mode moves the arm but does not press keys.

   ```
   python type_hello.py plan
   ```

3. Run. Leave the terminal window focused, use lowercase Latin input and keep Caps Lock off. A character received in the terminal is the feedback signal for a physical press, so the keyboard the robot types on must be connected to this computer. The program takes a fresh photo even if a plan was run earlier.

   ```
   python type_hello.py run
   ```

To return without photographing or typing, run `python type_hello.py home`. At startup the controller reads the encoders and sets the hold targets before enabling the motors. When low, it first plans a lift at the current XY. It checks a direct path home and can use an upright waypoint if the direct path fails the height check. Unreachable lifts, unmodelled obstructions, a slipping horn and a loss of communication require local attention. Ctrl+C requests a stop and hold. Support the arm before removing power.

## How it works

| File | Responsibility |
| --- | --- |
| `type_hello.py` (root) | Acquire a frame and encoders, plan the phrase, read physical keystrokes and return home. |
| `robot/vision.py` | Rotate candidate views, read bright legends, discard OCR outliers using the staggered letter rows. |
| `robot/keyboard.py` | Project observed pixels into base XY. Fit a rotation and translation to the keyboard grid. Calculate all target centres from that fit. |
| `robot/kinematics.py` | Course DH forward and analytic inverse kinematics, in millimetres and degrees. |
| `robot/motion.py` | Select tool pitch and compute hover, press and lift waypoints. Check estimated tip, housing and link heights. |
| `robot/arm.py` | Send one synchronised position-and-speed packet per motion segment and read the encoders. |
| `robot/setup.json` | Robot-specific camera calibration, home pose, keyboard dimensions and contact heights. |
| `tests/calibrate_camera.py` | Optional, for a new camera mount, photo pose or replacement robot. |

**Vision.** One good photo is enough. If the first view fails, the program tries the other calibrated camera poses. It uses at least seven reliable letters across two or more rows. I and L are excluded from matching because the camera frequently confuses them. The fitted row geometry locates L. Cropped keys and the point inside the space bar are inferred from the same live fit. A failed fit stops the run. There is no stored table of base XY coordinates for the letters.

**Movement.** Travel to a hover about 20 mm above the estimated local key surface, descend until the keyboard reports a character or the bounded endpoint is reached, then lift. The next letter normally starts directly from that hover. The bus sends positions and proportional joint speeds together. Reading the encoders does not interrupt the servo movement. The default transfer setting is 60 degrees/s for the joint with the largest travel. `--speed` changes that setting. The near-contact descent is capped at 14 degrees/s. Physical speed depends on the load. The recorded joint diagnostics already showed a torque limit of 1023 on all four motors.

**Calibration.** Base axes and camera motion come from assignment problems 1 and 2. The two supplied camera projections were derived from the archived reference images, measured H contact and home-row direction. Column pitch is 18.44 mm from E–O and H–L. Row pitch is 16.8 mm from the ruler displacement. Five recorded surface heights and the measured 2.10 mm H travel set pressing depth. Other heights are interpolated on the live grid. The camera housing clearance uses the supplied approximate 14 mm protrusion and 43 mm reach. The calculation checks the modelled geometry, so an obstacle or a different bracket matters. The method assumes a fixed keyboard height and camera resolution of 640 × 480.

**Course material.** The assignment, Part 5, specifies the result and the use of a wrist-mounted camera. Lecture 6 covers paths, joint-space waypoints and piecewise trajectories. The fifth-order polynomial exercise in Part 2 applies to a prescribed circle. For the keyboard, the waypoint speeds are coordinated by the servo driver. Week 8 vision exercises distinguish image coordinates and world measurements.

## New keyboard

```
python tests/calibrate_camera.py                 # standard keyboard, 19.05 mm keys
python tests/calibrate_camera.py --height -45    # with the measured height of the key tops
```

The arm takes one photo, works out the keyboard height (or, with `--height`, checks that the
camera sees a normal key spacing), shows the new values and asks whether to overwrite
`robot/setup.json`. Only the keyboard values change; the camera calibration is reused.

If the robot then presses next to the keys, correct the positions for this camera:

```
python tests/calibrate_camera.py --aim
```

The arm hovers over each letter of "hello world"; nudge the tip with `w/a/s/d` until it is
above the centre of the key, press Enter. It fits a rotation, shift and stretch and saves it
in `robot/setup.json` (`xy_correction`). Running it again refines it.
Moving or rotating the keyboard on the same table needs no new calibration, only a fresh photo.
