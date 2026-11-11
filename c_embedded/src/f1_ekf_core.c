/**
 * @file f1_ekf_core.c
 * @brief Real-Time Extended Kalman Filter (EKF) Core Implementation in C99
 */

#include "f1_ekf_core.h"
#include <math.h>
#include <string.h>

#define PI_CONST (3.141592653589793f)

/* Internal Helper: Invert a 3x3 symmetric positive-definite matrix analytically */
static bool mat_inv_3x3(const float A[3][3], float A_inv[3][3]) {
    float det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);

    if (fabsf(det) < 1e-7f) {
        return false;
    }

    float inv_det = 1.0f / det;

    A_inv[0][0] = (A[1][1] * A[2][2] - A[1][2] * A[2][1]) * inv_det;
    A_inv[0][1] = (A[0][2] * A[2][1] - A[0][1] * A[2][2]) * inv_det;
    A_inv[0][2] = (A[0][1] * A[1][2] - A[0][2] * A[1][1]) * inv_det;

    A_inv[1][0] = (A[1][2] * A[2][0] - A[1][0] * A[2][2]) * inv_det;
    A_inv[1][1] = (A[0][0] * A[2][2] - A[0][2] * A[2][0]) * inv_det;
    A_inv[1][2] = (A[0][2] * A[1][0] - A[0][0] * A[1][2]) * inv_det;

    A_inv[2][0] = (A[1][0] * A[2][1] - A[1][1] * A[2][0]) * inv_det;
    A_inv[2][1] = (A[0][1] * A[2][0] - A[0][0] * A[2][1]) * inv_det;
    A_inv[2][2] = (A[0][0] * A[1][1] - A[0][1] * A[1][0]) * inv_det;

    return true;
}

f1_status_t f1_ekf_init(f1_ekf_filter_t *ekf, const f1_vehicle_params_t *params) {
    if (ekf == NULL) {
        return F1_STATUS_ERR_NULL_PTR;
    }
    (void)memset(ekf, 0, sizeof(f1_ekf_filter_t));

    /* Default F1 vehicle parameters if not supplied */
    if (params != NULL) {
        ekf->params = *params;
    } else {
        ekf->params.mass_kg           = 798.0f;
        ekf->params.inertia_z_kgm2     = 1250.0f;
        ekf->params.dist_cg_front_m    = 1.80f;
        ekf->params.dist_cg_rear_m     = 1.60f;
        ekf->params.cornering_stiff_f  = 160000.0f;
        ekf->params.cornering_stiff_r  = 185000.0f;
    }

    /* State Initialization [vx=50.0 m/s, vy=0.0, r=0.0, psi=0.0] */
    ekf->x[0] = 50.0f;
    ekf->x[1] = 0.0f;
    ekf->x[2] = 0.0f;
    ekf->x[3] = 0.0f;

    /* Covariance P */
    ekf->P[0][0] = 1.0f;
    ekf->P[1][1] = 0.5f;
    ekf->P[2][2] = 0.05f;
    ekf->P[3][3] = 0.05f;

    /* Process Noise Q */
    ekf->Q[0][0] = 0.15f;
    ekf->Q[1][1] = 0.10f;
    ekf->Q[2][2] = 0.01f;
    ekf->Q[3][3] = 0.005f;

    /* Measurement Noise R */
    ekf->R[0][0] = 0.40f;   /* GPS Velocity error */
    ekf->R[1][1] = 0.60f;   /* Lateral Accelerometer error */
    ekf->R[2][2] = 0.015f;  /* Yaw Gyro error */

    ekf->step_count = 0U;
    return F1_STATUS_OK;
}

f1_status_t f1_ekf_predict(
    f1_ekf_filter_t *ekf,
    float delta_steer_rad,
    float ax_accel_mss,
    float dt_sec
) {
    if (ekf == NULL) {
        return F1_STATUS_ERR_NULL_PTR;
    }

    float vx  = (ekf->x[0] < 5.0f) ? 5.0f : ekf->x[0];
    float vy  = ekf->x[1];
    float r   = ekf->x[2];
    float cos_d = cosf(delta_steer_rad);

    /* Tire slip angles */
    float alpha_f = delta_steer_rad - ((vy + ekf->params.dist_cg_front_m * r) / vx);
    float alpha_r = - ((vy - ekf->params.dist_cg_rear_m * r) / vx);

    /* Lateral cornering forces with saturation limits */
    float Fyf = ekf->params.cornering_stiff_f * alpha_f;
    if (Fyf > 18000.0f) Fyf = 18000.0f;
    if (Fyf < -18000.0f) Fyf = -18000.0f;

    float Fyr = ekf->params.cornering_stiff_r * alpha_r;
    if (Fyr > 22000.0f) Fyr = 22000.0f;
    if (Fyr < -22000.0f) Fyr = -22000.0f;

    /* Non-linear equations of motion */
    float vx_dot = ax_accel_mss + (vy * r);
    float vy_dot = ((Fyf * cos_d + Fyr) / ekf->params.mass_kg) - (vx * r);
    float r_dot  = ((ekf->params.dist_cg_front_m * Fyf * cos_d - 
                     ekf->params.dist_cg_rear_m * Fyr) / ekf->params.inertia_z_kgm2);

    /* State Euler Propagation */
    ekf->x[0] += vx_dot * dt_sec;
    ekf->x[1] += vy_dot * dt_sec;
    ekf->x[2] += r_dot * dt_sec;
    ekf->x[3] += r * dt_sec;

    /* Analytical State Jacobian F */
    float F[EKF_STATE_DIM][EKF_STATE_DIM];
    (void)memset(F, 0, sizeof(F));
    for (uint8_t i = 0U; i < EKF_STATE_DIM; ++i) {
        F[i][i] = 1.0f;
    }

    F[0][1] = r * dt_sec;
    F[0][2] = vy * dt_sec;

    float dFyf_dvy = -ekf->params.cornering_stiff_f / vx;
    float dFyf_dr  = -ekf->params.dist_cg_front_m * ekf->params.cornering_stiff_f / vx;
    float dFyr_dvy = -ekf->params.cornering_stiff_r / vx;
    float dFyr_dr  =  ekf->params.dist_cg_rear_m * ekf->params.cornering_stiff_r / vx;

    F[1][0] = -r * dt_sec;
    F[1][1] = 1.0f + (((dFyf_dvy * cos_d + dFyr_dvy) / ekf->params.mass_kg) * dt_sec);
    F[1][2] = (-vx + ((dFyf_dr * cos_d + dFyr_dr) / ekf->params.mass_kg)) * dt_sec;

    F[2][1] = (((ekf->params.dist_cg_front_m * dFyf_dvy * cos_d - 
                 ekf->params.dist_cg_rear_m * dFyr_dvy) / ekf->params.inertia_z_kgm2) * dt_sec);
    F[2][2] = 1.0f + (((ekf->params.dist_cg_front_m * dFyf_dr * cos_d - 
                        ekf->params.dist_cg_rear_m * dFyr_dr) / ekf->params.inertia_z_kgm2) * dt_sec);
    F[3][2] = dt_sec;

    /* P = F * P * F^T + Q */
    float FP[EKF_STATE_DIM][EKF_STATE_DIM] = {{0}};
    for (uint8_t i = 0U; i < 4U; ++i) {
        for (uint8_t j = 0U; j < 4U; ++j) {
            for (uint8_t k = 0U; k < 4U; ++k) {
                FP[i][j] += F[i][k] * ekf->P[k][j];
            }
        }
    }

    for (uint8_t i = 0U; i < 4U; ++i) {
        for (uint8_t j = 0U; j < 4U; ++j) {
            float val = 0.0f;
            for (uint8_t k = 0U; k < 4U; ++k) {
                val += FP[i][k] * F[j][k];
            }
            ekf->P[i][j] = val + ekf->Q[i][j];
        }
    }

    ekf->step_count++;
    return F1_STATUS_OK;
}

f1_status_t f1_ekf_update(
    f1_ekf_filter_t *ekf,
    float v_gps_ms,
    float ay_imu_mss,
    float r_gyro_rads,
    float delta_steer_rad
) {
    if (ekf == NULL) {
        return F1_STATUS_ERR_NULL_PTR;
    }

    float vx  = (ekf->x[0] < 5.0f) ? 5.0f : ekf->x[0];
    float vy  = ekf->x[1];
    float r   = ekf->x[2];
    float cos_d = cosf(delta_steer_rad);

    float alpha_f = delta_steer_rad - ((vy + ekf->params.dist_cg_front_m * r) / vx);
    float alpha_r = - ((vy - ekf->params.dist_cg_rear_m * r) / vx);

    float Fyf = ekf->params.cornering_stiff_f * alpha_f;
    float Fyr = ekf->params.cornering_stiff_r * alpha_r;

    /* Model measurement prediction h(x) */
    float h_x[3];
    h_x[0] = vx;
    h_x[1] = (Fyf * cos_d + Fyr) / ekf->params.mass_kg;
    h_x[2] = r;

    /* Measurement Jacobian H (3x4) */
    float H[3][4] = {{0}};
    H[0][0] = 1.0f;

    float dFyf_dvy = -ekf->params.cornering_stiff_f / vx;
    float dFyf_dr  = -ekf->params.dist_cg_front_m * ekf->params.cornering_stiff_f / vx;
    float dFyr_dvy = -ekf->params.cornering_stiff_r / vx;
    float dFyr_dr  =  ekf->params.dist_cg_rear_m * ekf->params.cornering_stiff_r / vx;

    H[1][1] = (dFyf_dvy * cos_d + dFyr_dvy) / ekf->params.mass_kg;
    H[1][2] = (dFyf_dr * cos_d + dFyr_dr) / ekf->params.mass_kg;
    H[2][2] = 1.0f;

    /* S = H * P * H^T + R (3x3) */
    float HP[3][4] = {{0}};
    for (uint8_t i = 0U; i < 3U; ++i) {
        for (uint8_t j = 0U; j < 4U; ++j) {
            for (uint8_t k = 0U; k < 4U; ++k) {
                HP[i][j] += H[i][k] * ekf->P[k][j];
            }
        }
    }

    float S[3][3] = {{0}};
    for (uint8_t i = 0U; i < 3U; ++i) {
        for (uint8_t j = 0U; j < 3U; ++j) {
            float val = 0.0f;
            for (uint8_t k = 0U; k < 4U; ++k) {
                val += HP[i][k] * H[j][k];
            }
            S[i][j] = val + ekf->R[i][j];
        }
    }

    float S_inv[3][3];
    if (!mat_inv_3x3(S, S_inv)) {
        return F1_STATUS_ERR_OUT_OF_BOUNDS;
    }

    /* Kalman Gain K = P * H^T * S^-1 (4x3) */
    float PHT[4][3] = {{0}};
    for (uint8_t i = 0U; i < 4U; ++i) {
        for (uint8_t j = 0U; j < 3U; ++j) {
            for (uint8_t k = 0U; k < 4U; ++k) {
                PHT[i][j] += ekf->P[i][k] * H[j][k];
            }
        }
    }

    float K[4][3] = {{0}};
    for (uint8_t i = 0U; i < 4U; ++i) {
        for (uint8_t j = 0U; j < 3U; ++j) {
            for (uint8_t k = 0U; k < 3U; ++k) {
                K[i][j] += PHT[i][k] * S_inv[k][j];
            }
        }
    }

    /* Measurement Innovation residual y = z - h(x) */
    float y[3];
    y[0] = v_gps_ms - h_x[0];
    y[1] = ay_imu_mss - h_x[1];
    y[2] = r_gyro_rads - h_x[2];

    /* State Update x = x + K * y */
    for (uint8_t i = 0U; i < 4U; ++i) {
        for (uint8_t j = 0U; j < 3U; ++j) {
            ekf->x[i] += K[i][j] * y[j];
        }
    }

    /* Covariance Update P = (I - K * H) * P */
    float KH[4][4] = {{0}};
    for (uint8_t i = 0U; i < 4U; ++i) {
        for (uint8_t j = 0U; j < 4U; ++j) {
            for (uint8_t k = 0U; k < 3U; ++k) {
                KH[i][j] += K[i][k] * H[k][j];
            }
        }
    }

    float I_KH[4][4];
    for (uint8_t i = 0U; i < 4U; ++i) {
        for (uint8_t j = 0U; j < 4U; ++j) {
            I_KH[i][j] = ((i == j) ? 1.0f : 0.0f) - KH[i][j];
        }
    }

    float P_new[4][4] = {{0}};
    for (uint8_t i = 0U; i < 4U; ++i) {
        for (uint8_t j = 0U; j < 4U; ++j) {
            for (uint8_t k = 0U; k < 4U; ++k) {
                P_new[i][j] += I_KH[i][k] * ekf->P[k][j];
            }
        }
    }

    (void)memcpy(ekf->P, P_new, sizeof(P_new));
    return F1_STATUS_OK;
}

float f1_ekf_get_sideslip_rad(const f1_ekf_filter_t *ekf) {
    if (ekf == NULL) return 0.0f;
    float vx = (ekf->x[0] < 5.0f) ? 5.0f : ekf->x[0];
    return atan2f(ekf->x[1], vx);
}

float f1_ekf_get_sideslip_deg(const f1_ekf_filter_t *ekf) {
    return f1_ekf_get_sideslip_rad(ekf) * (180.0f / PI_CONST);
}
