import numpy as np
import math
# The following visualization imports have been removed:
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# ====================================================================
# CORE CONVERSION FUNCTIONS
# ====================================================================

def euler_to_quaternion(yaw, pitch, roll):
    """
    Converts Z-Y-X Euler angles (radians) to a Quaternion [w, x, y, z].
    
    Args:
        yaw (float): Rotation around Z-axis (psi).
        pitch (float): Rotation around Y-axis (theta).
        roll (float): Rotation around X-axis (phi).
        
    Returns:
        np.array: A 4-element array [w, x, y, z] representing the quaternion.
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    # Quaternion formula for Z-Y-X sequence: q = q_roll * q_pitch * q_yaw
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


def quaternion_to_euler(q):
    """
    Converts a Quaternion [w, x, y, z] to Z-Y-X Euler angles (radians).
    Handles Gimbal Lock (Pitch +/- 90 degrees).
    
    Args:
        q (np.array): A 4-element array [w, x, y, z] representing the quaternion.
        
    Returns:
        dict: Dictionary containing 'yaw', 'pitch', and 'roll' in radians.
    """
    # Ensure quaternion is a unit quaternion
    q = q / np.linalg.norm(q) 
    w, x, y, z = q[0], q[1], q[2], q[3]

    ysqr = y * y
    sinp = 2 * (w * y - z * x)
    
    GIMBAL_LOCK_THRESHOLD = 0.99999
    
    if abs(sinp) >= GIMBAL_LOCK_THRESHOLD:
        # Gimbal Lock condition
        pitch = math.pi / 2 if sinp > 0 else -math.pi / 2
        roll = math.atan2(2 * (x * y + w * z), 1 - 2 * (x * x + ysqr))
        yaw = 0.0 # Yaw is set to 0 and combined with roll
        print(f"  WARNING: Gimbal Lock detected at pitch ~ {math.degrees(pitch):.2f} degrees!")
        
    else:
        # Normal Case
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + ysqr))
        pitch = math.asin(sinp)
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (ysqr + z * z))

    return {'yaw': yaw, 'pitch': pitch, 'roll': roll}

# ====================================================================
# MAIN EXECUTION BLOCK
# ====================================================================

def run_demonstration(yaw_deg, pitch_deg, roll_deg, title):
    """Runs a single conversion demonstration."""
    print(f"\n--- {title} ---")
    
    yaw_rad = math.radians(yaw_deg)
    pitch_rad = math.radians(pitch_deg)
    roll_rad = math.radians(roll_deg)

    # EULER -> QUATERNION
    q = euler_to_quaternion(yaw_rad, pitch_rad, roll_rad)
    print(f"Input Euler ({yaw_deg:.0f},{pitch_deg:.0f},{roll_deg:.0f} deg) -> Quaternion: {q}")

    # QUATERNION -> EULER
    e = quaternion_to_euler(q)
    print(f"Quaternion -> Output Euler (deg): Yaw={math.degrees(e['yaw']):.2f}, Pitch={math.degrees(e['pitch']):.2f}, Roll={math.degrees(e['roll']):.2f}")
    
    # VISUALIZATION CODE REMOVED

def main():
    """Organizes and runs all demonstration cases."""
    
    # 1. Simple Yaw Rotation (Proof of concept)
    run_demonstration(yaw_deg=90, pitch_deg=0, roll_deg=0, 
                      title="Example 1: Simple Yaw Rotation (90 deg around Z)")

    # 2. Complex Rotation (General case)
    run_demonstration(yaw_deg=45, pitch_deg=30, roll_deg=60, 
                      title="Example 2: Complex Rotation (45 Yaw, 30 Pitch, 60 Roll)")

    # 3. GIMBAL LOCK (Edge Case Handling)
    run_demonstration(yaw_deg=45, pitch_deg=90, roll_deg=30, 
                      title="Example 3: GIMBAL LOCK (Critical Pitch = 90 deg)")


if __name__ == "__main__":
    main()