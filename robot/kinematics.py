"""Forward and inverse kinematics from Problems 1 and 2 of the report.

Units: mm and degrees at the interface, radians inside.
q = [q1, q2, q3, q4] are the DH joint variables (q = 0 is the arm straight up).
"""
import numpy as np

D1, A2, A3, A4 = 50.0, 93.0, 93.0, 50.0        # link dimensions [mm]
T45 = np.array([[1, 0, 0, -15],                 # stylus -> camera, eq. (13)
                [0, 1, 0, 45],
                [0, 0, 1, 0],
                [0, 0, 0, 1]], dtype=float)


def dh(theta, d, a, alpha):
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st * ca,  st * sa, a * ct],
                     [st,  ct * ca, -ct * sa, a * st],
                     [0,        sa,       ca,      d],
                     [0,         0,        0,      1]])


def fk(q_deg):
    """Return (T04, T05): base->stylus and base->camera transforms."""
    q1, q2, q3, q4 = np.radians(q_deg)
    T = (dh(q1, D1, 0, np.pi / 2)
         @ dh(q2 + np.pi / 2, 0, A2, 0)
         @ dh(q3, 0, A3, 0)
         @ dh(q4, 0, A4, 0))
    return T, T @ T45


def ik(x, y, z, w=0.0, elbow="up"):
    """Joint angles [deg] that put the stylus tip at (x, y, z) mm with (x4)_z = w.

    w = 0  -> stylus horizontal, pointing outwards
    w = -1 -> stylus pointing straight down (for pressing keys)
    Uses the positive radial branch and gamma = -arccos(w).
    """
    if not -1.0 <= w <= 1.0:
        raise ValueError("w must be in [-1, 1]")
    rho = np.hypot(x, y)
    q1 = np.arctan2(y, x)
    g = -np.arccos(w)                        # gamma = q2 + q3 + q4

    r = rho + A4 * np.sin(g)                 # wrist, eq. (33)
    s = z - D1 - A4 * np.cos(g)              # eq. (34)

    c3 = (r**2 + s**2 - A2**2 - A3**2) / (2 * A2 * A3)
    if abs(c3) > 1:
        raise ValueError(f"Target ({x}, {y}, {z}) is out of reach")
    s3 = -np.sqrt(1 - c3**2) if elbow == "up" else np.sqrt(1 - c3**2)
    q3 = np.arctan2(s3, c3)
    q2 = np.arctan2(s, r) - np.arctan2(A3 * s3, A2 + A3 * c3) - np.pi / 2
    q4 = g - q2 - q3

    q = np.degrees([q1, q2, q3, q4])
    return (q + 180) % 360 - 180             # wrap to [-180, 180)


if __name__ == "__main__":
    # Checks against the report
    T04, T05 = fk([0, 0, 0, 0])
    print("FK q=0  stylus:", T04[:3, 3].round(3), " (expect [0, 0, 286])")
    print("        camera:", T05[:3, 3].round(3), " (expect [-45, 0, 271])")

    q = ik(150, 0, 120, w=0)
    print("IK (150,0,120) elbow up:", q.round(2), " (expect [0, -6.02, -97.97, 13.99])")
    print("  back through FK:", fk(q)[0][:3, 3].round(4))

    q = ik(150, 0, 120, w=0, elbow="down")
    print("IK elbow down:          ", q.round(2), " (expect [0, -103.99, 97.97, -83.98])")

    q = ik(150, 30, 40, w=-1)
    T04, _ = fk(q)
    print("Stylus down at (150,30,40):", q.round(2),
          " FK ->", T04[:3, 3].round(3), " x4_z =", round(T04[2, 0], 3))
