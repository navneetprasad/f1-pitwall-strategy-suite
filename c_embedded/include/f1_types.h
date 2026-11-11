/**
 * @file f1_types.h
 * @brief Common Fixed-Width Types, Constants, and Status Codes for F1 ECU Telemetry
 * @note MISRA-C:2012 Compliant Definitions
 */

#ifndef F1_TYPES_H
#define F1_TYPES_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* System Return Status Codes */
typedef enum {
    F1_STATUS_OK                  =  0,
    F1_STATUS_ERR_NULL_PTR        = -1,
    F1_STATUS_ERR_INVALID_DLC     = -2,
    F1_STATUS_ERR_CRC_MISMATCH    = -3,
    F1_STATUS_ERR_BUFFER_OVERFLOW = -4,
    F1_STATUS_ERR_OUT_OF_BOUNDS   = -5
} f1_status_t;

/* Vehicle Operating State Flags */
typedef enum {
    F1_STATE_NOMINAL     = 0x01,
    F1_STATE_VSC_ACTIVE  = 0x02,
    F1_STATE_SC_ACTIVE   = 0x04,
    F1_STATE_PIT_LANE    = 0x08,
    F1_STATE_FAULT_ALERT = 0x10
} f1_vehicle_state_t;

/* Physical Constants */
#define F1_GRAVITY_MSS          (9.80665f)
#define F1_MAX_CAN_PAYLOAD_LEN  (8U)
#define F1_RING_BUFFER_CAPACITY (128U)

#ifdef __cplusplus
}
#endif

#endif /* F1_TYPES_H */
