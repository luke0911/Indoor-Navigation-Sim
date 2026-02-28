#!/usr/bin/env python3
"""
Generate realistic dummy IMU floor-detection simulation data.

Output: Tab-separated file with columns:
  time(s)  globalX  globalY  globalZ  magX  magY  magZ  gyroAngle  pitch  floorLabel

Scenario timeline (50 Hz, ~80 s total):
  0-30 s  : ground  (flat walking)
  30-40 s : up      (climbing stairs)
  40-60 s : ground  (flat walking, upper floor)
  60-70 s : down    (descending stairs)
  70-80 s : ground  (flat walking)
"""

import numpy as np
import os

# ── Parameters ──────────────────────────────────────────────────────────────
SAMPLING_RATE = 50          # Hz
DT = 1.0 / SAMPLING_RATE   # 0.02 s
DURATION = 80.0             # seconds
N_SAMPLES = int(DURATION * SAMPLING_RATE)

SEED = 42
rng = np.random.default_rng(SEED)

# Walking cadence ~ 1.8 Hz
WALK_FREQ = 1.8

# ── Time vector ─────────────────────────────────────────────────────────────
t = np.arange(N_SAMPLES) * DT

# ── Scenario labels ─────────────────────────────────────────────────────────
labels = np.empty(N_SAMPLES, dtype=object)
for i, ti in enumerate(t):
    if ti < 30.0:
        labels[i] = "ground"
    elif ti < 40.0:
        labels[i] = "up"
    elif ti < 60.0:
        labels[i] = "ground"
    elif ti < 70.0:
        labels[i] = "down"
    else:
        labels[i] = "ground"

# ── globalZ  (vertical acceleration) ────────────────────────────────────────
# Walking produces a roughly sinusoidal bounce at ~1.8 Hz.
# Amplitude is slightly larger on stairs.
globalZ = np.zeros(N_SAMPLES)
for i in range(N_SAMPLES):
    if labels[i] == "ground":
        amp = 2.5
    else:
        amp = 3.5
    globalZ[i] = amp * np.sin(2.0 * np.pi * WALK_FREQ * t[i])

globalZ += rng.normal(0.0, 0.15, N_SAMPLES)

# ── globalX, globalY  (lateral / forward accel, mostly small) ───────────────
globalX = rng.normal(0.0, 0.3, N_SAMPLES)
globalY = rng.normal(0.0, 0.3, N_SAMPLES)

# ── Magnetometer ────────────────────────────────────────────────────────────
# Earth's magnetic field baseline + small sensor noise
MAG_BASE = np.array([20.0, -5.0, -40.0])
magX = MAG_BASE[0] + rng.normal(0.0, 1.0, N_SAMPLES)
magY = MAG_BASE[1] + rng.normal(0.0, 1.0, N_SAMPLES)
magZ = MAG_BASE[2] + rng.normal(0.0, 1.0, N_SAMPLES)

# ── gyroAngle  (heading, slow drift 45 -> ~90 deg) ─────────────────────────
# Linear trend + small random walk
gyro_trend = np.linspace(45.0, 90.0, N_SAMPLES)
gyro_walk = np.cumsum(rng.normal(0.0, 0.05, N_SAMPLES))
# Keep the walk centred so it doesn't swamp the trend
gyro_walk -= np.linspace(gyro_walk[0], gyro_walk[-1], N_SAMPLES)
gyroAngle = gyro_trend + gyro_walk

# ── pitch ───────────────────────────────────────────────────────────────────
pitch = np.zeros(N_SAMPLES)
for i in range(N_SAMPLES):
    if labels[i] == "ground":
        pitch[i] = rng.normal(0.0, 3.0)
    elif labels[i] == "up":
        pitch[i] = rng.normal(8.0, 3.0)
    elif labels[i] == "down":
        pitch[i] = rng.normal(-7.0, 3.0)

# ── Write output ────────────────────────────────────────────────────────────
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "Floor_sim_data_sample.txt")

with open(OUT_PATH, "w") as f:
    # Header
    f.write("# time(s)\tglobalX\tglobalY\tglobalZ\t"
            "magX\tmagY\tmagZ\tgyroAngle\tpitch\tfloorLabel\n")

    for i in range(N_SAMPLES):
        line = (
            f"{t[i]:.2f}\t"
            f"{globalX[i]:.4f}\t"
            f"{globalY[i]:.4f}\t"
            f"{globalZ[i]:.4f}\t"
            f"{magX[i]:.2f}\t"
            f"{magY[i]:.2f}\t"
            f"{magZ[i]:.2f}\t"
            f"{gyroAngle[i]:.2f}\t"
            f"{pitch[i]:.2f}\t"
            f"{labels[i]}"
        )
        f.write(line + "\n")

print(f"Written {N_SAMPLES} samples to {OUT_PATH}")
