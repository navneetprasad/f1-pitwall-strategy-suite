/**
 * @file f1_can_decoder.h
 * @brief High-Rate CAN 2.0B / CAN-FD Telemetry Frame Decoder & DBC Signal Unpacker
 * @note Designed for 1 kHz deterministic ECU loops (e.g. McLaren Applied TAG-320 / Bosch MS 6)
 */

#ifndef F1_CAN_DECODER_H
#define F1_CAN_DECODER_H

#include "f1_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Standard F1 CAN Arbitration Message IDs */
#define CAN_ID_CHASSIS_DYNAMICS  (0x120U)  /* Speed, Throttle, Brake, Steering */
#define CAN_ID_POWERTRAIN_HEALTH (0x130U)  /* Gearbox Temp, Hyd Pressure, MGU-K */
#define CAN_ID_TYRE_TEMPERATURES (0x140U)  /* FL, FR, RL, RR Surface & Carcass */

/* Raw Standard CAN 2.0B Frame Representation */
typedef struct {
    uint32_t id;                            /* 11-bit or 29-bit CAN arbitration identifier */
    uint8_t  dlc;                           /* Data Length Code (typically 8 bytes) */
    uint8_t  data[F1_MAX_CAN_PAYLOAD_LEN];  /* Raw byte payload */
    uint32_t timestamp_us;                  /* Microsecond hardware timestamp */
} f1_can_frame_t;

/* Decoded Chassis Dynamics Telemetry (Message ID 0x120) */
typedef struct {
    float    speed_kmh;         /* Range: 0.0 to 380.0 km/h, scale: 0.058 km/h/LSB */
    float    throttle_pct;      /* Range: 0.0 to 100.0 %, scale: 0.1 %/LSB */
    float    brake_pct;         /* Range: 0.0 to 100.0 %, scale: 0.1 %/LSB */
    float    steering_deg;      /* Range: -180.0 to +180.0 deg, scale: 0.05 deg/LSB */
    uint8_t  gear;              /* Range: 1 to 8 (0 = N, 255 = R) */
    bool     drs_open;          /* True if DRS flap is actuated */
} f1_chassis_telemetry_t;

/* Decoded Powertrain & Hydraulic Health Telemetry (Message ID 0x130) */
typedef struct {
    float    gearbox_vibration_rms; /* mm/s (0.0 to 50.0, scale 0.01) */
    float    gearbox_oil_temp_c;    /* °C (0.0 to 180.0, scale 0.1) */
    float    hydraulic_press_bar;   /* bar (0.0 to 250.0, scale 0.1) */
    float    mguk_core_temp_c;      /* °C (0.0 to 220.0, scale 0.1) */
    uint8_t  fault_code;            /* 0 = None, 1 = Caution, 2 = Critical DNF Warning */
} f1_powertrain_telemetry_t;

/* Circular Ring Buffer for High-Frequency Frame Buffering */
typedef struct {
    f1_can_frame_t frames[F1_RING_BUFFER_CAPACITY];
    uint32_t head;
    uint32_t tail;
    uint32_t count;
    uint32_t dropped_frames;
} f1_can_ring_buffer_t;

/* API Function Prototypes */
f1_status_t f1_can_init_ring_buffer(f1_can_ring_buffer_t *rb);
f1_status_t f1_can_push_frame(f1_can_ring_buffer_t *rb, const f1_can_frame_t *frame);
f1_status_t f1_can_pop_frame(f1_can_ring_buffer_t *rb, f1_can_frame_t *frame);

f1_status_t f1_can_pack_chassis(const f1_chassis_telemetry_t *telemetry, f1_can_frame_t *frame);
f1_status_t f1_can_unpack_chassis(const f1_can_frame_t *frame, f1_chassis_telemetry_t *telemetry);

f1_status_t f1_can_pack_powertrain(const f1_powertrain_telemetry_t *telemetry, f1_can_frame_t *frame);
f1_status_t f1_can_unpack_powertrain(const f1_can_frame_t *frame, f1_powertrain_telemetry_t *telemetry);

uint16_t f1_compute_crc16(const uint8_t *data, uint8_t length);

#ifdef __cplusplus
}
#endif

#endif /* F1_CAN_DECODER_H */
