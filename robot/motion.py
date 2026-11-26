"""A keystroke is three moves: travel above the key, lower, lift."""
import numpy as np

from .arm import check_angles
from .kinematics import D1, A2, A4, fk, ik


def geometry(q, setup):
    tool = fk(q)[0]
    tip = tool[:3, 3]
    downward = float(np.clip(-tool[2, 0], -1, 1))
    # Lowest edge of a cylinder bounding the housing behind the tip.
    back = setup["camera_gap"] if downward >= 0 else A4
    edge = tip[2] + back * downward - setup["camera_radius"] * np.sqrt(1 - downward**2)
    elbow = D1 + A2 * np.cos(np.radians(q[1]))
    wrist = tip[2] + A4 * downward
    return tip, min(edge, wrist, elbow)


def key_poses(xy, z, setup):
    for pitch in (-180, -175, -170):
        try:
            hover = ik(*xy, z + 20, pitch_deg=pitch)
            bottom = ik(*xy, z - setup["travel"] - 1.5, pitch_deg=pitch)
            check_angles(hover)
            check_angles(bottom)
            if geometry(bottom, setup)[1] >= setup["surface_z"] + 2:
                return hover, bottom
        except ValueError:
            continue
    rho = float(np.hypot(*xy))
    hint = "farther away from" if rho < 100 else "closer to"
    raise ValueError(f"Cannot reach this key at {xy.round(1)} mm ({rho:.0f} mm from the base axis). "
                     f"Move the keyboard {hint} the robot.")


LIFT_SAG = 3.0   # mm the tip may sink when a press turns into a lift (the loaded arm sags)


def check_segment(start, goal, kind, setup):
    tip, housing = geometry(start, setup)
    last_z = tip[2]
    if kind == "travel":
        floor = setup["surface_z"] + 5
    elif kind == "press":
        floor = geometry(goal, setup)[0][2] - 1.0
    else:
        floor = min(tip[2], setup["surface_z"] + 5) - LIFT_SAG
    body_floor = setup["surface_z"] + 2
    if kind == "lift":
        body_floor = min(housing - LIFT_SAG, body_floor)

    def check(q):
        point, body = geometry(q, setup)
        if point[2] < floor or body < body_floor:
            raise ValueError(f"{kind}: insufficient clearance; tip Z {point[2]:.1f}, housing/link Z {body:.1f}.")
        return point

    # Check the joint-space segment before sending it, then check measured poses in flight.
    check_angles(goal)
    steps = max(2, int(np.ceil(np.max(np.abs(goal - start)))) + 1)
    for q in np.linspace(start, goal, steps):
        point = check(q)
        if kind == "lift" and point[2] < last_z - LIFT_SAG:
            raise ValueError("Initial lift would first dip towards the keyboard.")
        last_z = max(last_z, point[2])

    return check


class Motion:
    def __init__(self, arm, setup, speed):
        self.arm = arm
        self.setup = setup
        self.speed = speed
        self.surface = setup["surface_z"]

    def move(self, goal, kind="travel", event=None):
        self.arm.validate(goal)
        check = check_segment(self.arm.angles(), goal, kind, self.setup)
        target_tip = geometry(goal, self.setup)[0]

        def monitor(q):
            point = check(q)
            if kind == "press" and point[2] < self.surface + 2 and np.linalg.norm(point[:2] - target_tip[:2]) > 5.5:
                raise ValueError("Tip moved sideways during the press.")
            if event is not None and event():
                # Cancel the descending goal as soon as Windows receives a character.
                self.arm.command(q, [40] * 4)
                return True
            error = np.max(np.abs(q - goal))
            if kind == "press":
                return error <= 0.45
            if kind == "lift":
                return error <= 2 and point[2] >= self.surface + 9
            # Travel to the hover point 20 mm above a key. Sideways accuracy matters for hitting
            # the key; height does not, and the loaded shoulder sags a few mm (the clearance
            # check above still guards against going too low).
            sideways = np.linalg.norm(point[:2] - target_tip[:2])
            return error <= 2 and sideways <= 5 and abs(point[2] - target_tip[2]) <= 8

        velocity = min(self.speed, 14) if kind == "press" else self.speed
        return self.arm.move(goal, velocity, monitor)

    def home(self):
        goal = np.asarray(self.setup["home"])
        q = self.arm.angles()
        tip, body = geometry(q, self.setup)
        if tip[2] < self.surface + 12 or body < self.surface + 5:
            # Raise at the same XY and pitch before swinging the base.
            raised = ik(*tip[:2], max(tip[2]+12, self.surface+20), pitch_deg=sum(q[1:]),
                        elbow="up" if q[2] <= 0 else "down")
            if np.max(np.abs(raised - q)) > 40:
                raise ValueError("Cannot lift vertically from this folded configuration.")
            self.move(raised, "lift")
        q = self.arm.angles()
        try:
            check_segment(q, goal, "travel", self.setup)
        except ValueError:
            upright = np.array([q[0], 0., 0., 0.])
            check_segment(q, upright, "travel", self.setup)
            check_segment(upright, goal, "travel", self.setup)
            self.move(upright)
        self.move(goal)
