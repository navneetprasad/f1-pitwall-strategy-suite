/**
 * @file test_f1_ecu.c
 * @brief Standalone Embedded C Test Suite for F1 CAN Decoder and EKF Estimator
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "f1_can_decoder.h"
#include "f1_ekf_core.h"

static int g_tests_run = 0;
static int g_tests_passed = 0;

#define ASSERT_TRUE(cond, msg) do { \
    g_tests_run++; \
    if (cond) { \
        g_tests_passed++; \
        printf("  [PASS] %s\n", msg); \
    } else { \
        printf("  [FAIL] %s (Line %d)\n", msg, __LINE__); \
    } \
} while(0)

static void test_can_chassis_encoding(void) {
    printf("\n=== RUNNING: CAN Chassis (0x120) Pack & Unpack Test ===\n");
    f1_chassis_telemetry_t original = {
        .speed_kmh    = 312.45f,
        .throttle_pct = 98.5f,
        .brake_pct    = 0.0f,
        .steering_deg = -12.4f,
        .gear         = 7U,
        .drs_open     = true
    };

    f1_can_frame_t frame;
    f1_status_t status = f1_can_pack_chassis(&original, &frame);
    ASSERT_TRUE(status == F1_STATUS_OK, "f1_can_pack_chassis returns OK");
    ASSERT_TRUE(frame.id == CAN_ID_CHASSIS_DYNAMICS, "CAN ID matches 0x120");
    ASSERT_TRUE(frame.dlc == 8U, "Frame DLC is 8 bytes");

    f1_chassis_telemetry_t decoded;
    status = f1_can_unpack_chassis(&frame, &decoded);
    ASSERT_TRUE(status == F1_STATUS_OK, "f1_can_unpack_chassis returns OK");
    ASSERT_TRUE(fabsf(decoded.speed_kmh - original.speed_kmh) < 0.05f, "Speed precision within 0.05 km/h");
    ASSERT_TRUE(fabsf(decoded.throttle_pct - original.throttle_pct) < 0.5f, "Throttle precision within 0.5%");
    ASSERT_TRUE(fabsf(decoded.steering_deg - original.steering_deg) < 0.05f, "Steering angle precision within 0.05 deg");
    ASSERT_TRUE(decoded.gear == 7U, "Gear matches Gear 7");
    ASSERT_TRUE(decoded.drs_open == true, "DRS state matches active");
}

static void test_can_crc_tampering(void) {
    printf("\n=== RUNNING: CAN Frame CRC-16 Integrity Test ===\n");
    f1_chassis_telemetry_t original = {
        .speed_kmh = 240.0f, .throttle_pct = 50.0f, .brake_pct = 0.0f,
        .steering_deg = 0.0f, .gear = 5U, .drs_open = false
    };

    f1_can_frame_t frame;
    (void)f1_can_pack_chassis(&original, &frame);

    /* Deliberately corrupt byte 2 (Throttle) */
    frame.data[2] ^= 0xFFU;

    f1_chassis_telemetry_t decoded;
    f1_status_t status = f1_can_unpack_chassis(&frame, &decoded);
    ASSERT_TRUE(status == F1_STATUS_ERR_CRC_MISMATCH, "Tampered payload caught by CRC checksum");
}

static void test_ekf_sideslip_estimation(void) {
    printf("\n=== RUNNING: Vehicle Dynamics EKF State Estimator Test ===\n");
    f1_ekf_filter_t ekf;
    f1_status_t status = f1_ekf_init(&ekf, NULL);
    ASSERT_TRUE(status == F1_STATUS_OK, "EKF initialized with nominal parameters");

    /* Simulate a 100-step cornering entry */
    float delta_steer = 0.12f;  /* ~6.8 degrees steer input */
    float dt = 0.02f;           /* 50 Hz loop */

    for (int i = 0; i < 50; ++i) {
        (void)f1_ekf_predict(&ekf, delta_steer, -4.5f, dt);
        (void)f1_ekf_update(&ekf, 52.0f, 18.5f, 0.35f, delta_steer);
    }

    float sideslip_deg = f1_ekf_get_sideslip_deg(&ekf);
    printf("  [INFO] Estimated Cornering Sideslip Angle (beta): %.3f deg\n", sideslip_deg);
    ASSERT_TRUE(sideslip_deg != 0.0f, "EKF dynamic sideslip state propagated");
    ASSERT_TRUE(fabsf(sideslip_deg) < 12.0f, "Sideslip angle bounded within physical tire grip limits");
}

int main(void) {
    printf("=================================================================\n");
    printf("🏎️  F1 ECU EMBEDDED C TELEMETRY & STATE ESTIMATOR TEST HARNESS\n");
    printf("=================================================================\n");

    test_can_chassis_encoding();
    test_can_crc_tampering();
    test_ekf_sideslip_estimation();

    printf("\n-----------------------------------------------------------------\n");
    printf("TEST RESULTS: %d / %d Tests Passed (%.1f%%)\n",
           g_tests_passed, g_tests_run,
           (float)g_tests_passed / (float)g_tests_run * 100.0f);
    printf("=================================================================\n");

    return (g_tests_passed == g_tests_run) ? EXIT_SUCCESS : EXIT_FAILURE;
}
