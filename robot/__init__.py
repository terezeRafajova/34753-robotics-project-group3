"""Robot code for Problem 11a (imported by type_hello.py and the scripts in tests/).

  arm.py         AX-12A motors: synchronised moves, encoder reading
  kinematics.py  forward and inverse kinematics (Problems 1 and 2)
  vision.py      reads the letters on the keyboard (OpenCV + Tesseract)
  keyboard.py    camera pixels -> key positions in mm from the robot base
  motion.py      one key press: hover, press, lift, with clearance checks
  setup.json     calibration: photo pose, camera projection, key heights
"""
