/**
 * @file f1_ekf_core.h
 * @brief Real-Time Extended Kalman Filter (EKF) Core for Vehicle Sideslip Angle Estimation
 * @note Implemented in pure C99 without dynamic heap allocation for hard real-time execution.
 */

#ifndef F1_EKF_CORE_H
#define F1_EKF_CORE_H

#include "f1_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define EKF_STATE_DIM       (4U)  /* [v_x, v_y, r, psi] */
#define EKF_MEASUREMENT_DIM (3U)  /* [v_gps, a_y_imu, r_gyro] */

/* F1 Chassis Physical Geometry & Inertial Constants */
typedef struct {
    float mass_kg;          /* Total vehicle mass (798.0 kg) */
    float inertia_z_kgm2;   /* Yaw moment of inertia (1250.0 kg*m^2) */
    float dist_cg_front_m;  /* Distance from CG to front axle a (1.80 m) */
    float dist_cg_rear_m;   /* Distance from CG to rear axle b (1.60 m) */
    float cornering_stiff_f;/* Front cornering stiffness Cf (160000.0 N/rad) */
    float cornering_stiff_r;/* Rear cornering stiffness Cr (185000.0 N/rad) */
} f1_vehicle_params_t;

/* EKF Internal Filter State & Covariance Matrices */
typedef struct {
    float x[EKF_STATE_DIM];                             /* State vector: [vx, vy, r, psi] */
    float P[EKF_STATE_DIM][EKF_STATE_DIM];             /* Error covariance matrix */
    float Q[EKF_STATE_DIM][EKF_STATE_DIM];             /* Process noise covariance */
    float R[EKF_MEASUREMENT_DIM][EKF_MEASUREMENT_DIM]; /* Measurement noise covariance */
    f1_vehicle_params_t params;                        /* Vehicle parameters */
    uint32_t step_count;
} f1_ekf_filter_t;

/* EKF Public API */
f1_status_t f1_ekf_init(f1_ekf_filter_t *ekf, const f1_vehicle_params_t *params);

f1_status_t f1_ekf_predict(
    f1_ekf_filter_t *ekf,
    float delta_steer_rad,
    float ax_accel_mss,
    float dt_sec
);

f1_status_t f1_ekf_update(
    f1_ekf_filter_t *ekf,
    float v_gps_ms,
    float ay_imu_mss,
    float r_gyro_rads,
    float delta_steer_rad
);

float f1_ekf_get_sideslip_rad(const f1_ekf_filter_t *ekf);
float f1_ekf_get_sideslip_deg(const f1_ekf_filter_t *ekf);

#ifdef __cplusplus
}
#endif

#endif /* F1_EKF_CORE_H */
