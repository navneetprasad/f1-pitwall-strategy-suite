"""
Python ctypes Bridge to F1 Embedded C Shared Library (c_bridge.py)
Provides binary interfacing to compiled f1_ecu.dll / libf1_ecu.so,
with pure-Python bitwise emulation fallback for CAN 2.0B decoding and CRC-16 checks.
"""

import os
import sys
import ctypes
from pathlib import Path
import numpy as np

# CRC-16-CCITT implementation in pure Python matching C f1_compute_crc16
def compute_crc16_py(byte_data: bytes) -> int:
    crc = 0xFFFF
    for b in byte_data:
        crc ^= (b << 8)
        for _ in range(8):
            if (crc & 0x8000) != 0:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc

# Bit-level emulation of C f1_can_pack_chassis
def pack_chassis_frame_c_emulation(speed_kmh: float, throttle_pct: float, brake_pct: float, steering_deg: float, gear: int, drs: bool) -> bytes:
    raw_speed = int(speed_kmh * 100) & 0xFFFF
    raw_thr = int(throttle_pct * 2.55) & 0xFF
    raw_brk = int(brake_pct * 2.55) & 0xFF
    raw_steer = int((steering_deg + 180.0) * 100) & 0xFFFF
    raw_flags = (gear & 0x0F) | (0x10 if drs else 0x00)
    
    payload = bytearray(8)
    payload[0] = raw_speed & 0xFF
    payload[1] = (raw_speed >> 8) & 0xFF
    payload[2] = raw_thr
    payload[3] = raw_brk
    payload[4] = raw_steer & 0xFF
    payload[5] = (raw_steer >> 8) & 0xFF
    payload[6] = raw_flags
    payload[7] = compute_crc16_py(bytes(payload[:7])) & 0xFF
    return bytes(payload)

# Bit-level emulation of C f1_can_unpack_chassis
def unpack_chassis_frame_c_emulation(frame_bytes: bytes) -> dict:
    if len(frame_bytes) != 8:
        return {"error": "Invalid DLC length"}
        
    expected_crc = compute_crc16_py(frame_bytes[:7]) & 0xFF
    actual_crc = frame_bytes[7]
    crc_valid = (expected_crc == actual_crc)
    
    raw_speed = frame_bytes[0] | (frame_bytes[1] << 8)
    speed_kmh = raw_speed / 100.0
    
    throttle_pct = frame_bytes[2] / 2.55
    brake_pct = frame_bytes[3] / 2.55
    
    raw_steer = frame_bytes[4] | (frame_bytes[5] << 8)
    steering_deg = (raw_steer / 100.0) - 180.0
    
    gear = frame_bytes[6] & 0x0F
    drs_open = bool(frame_bytes[6] & 0x10)
    
    return {
        "speed_kmh": round(speed_kmh, 2),
        "throttle_pct": round(throttle_pct, 1),
        "brake_pct": round(brake_pct, 1),
        "steering_deg": round(steering_deg, 2),
        "gear": gear,
        "drs_open": drs_open,
        "crc_valid": crc_valid,
        "raw_hex": " ".join(f"{b:02X}" for b in frame_bytes)
    }
