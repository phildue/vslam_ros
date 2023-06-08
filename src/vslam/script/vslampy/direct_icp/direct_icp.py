import cv2 as cv
import numpy as np
from sophus.sophuspy import SE3

from vslampy.direct_icp.overlay import Log
from vslampy.direct_icp.weights import TDistributionMultivariateWeights
from vslampy.utils.utils import statsstr
from vslampy.camera import Camera


class DirectIcp:
    def __init__(
        self,
        cam: Camera,
        nLevels: int,
        weight_prior=0.0,
        min_gradient_intensity=10 * 8,
        min_gradient_depth=np.inf,
        max_gradient_depth=0.5,
        max_z=5.0,
        max_iterations=100,
        min_parameter_update=1e-4,
        max_error_increase=1.1,
        weight_function=TDistributionMultivariateWeights(5, np.identity(2)),
        log=Log(),
    ):
        self.nLevels = nLevels
        self.cam = [cam.resize(1 / (2**l)) for l in range(nLevels)]
        self.f_no = 0
        self.I0 = None
        self.Z0 = None
        self.t0 = None

        self.min_dI = min_gradient_intensity
        self.min_dZ = min_gradient_depth
        self.max_dZ = max_gradient_depth
        self.max_z = max_z

        self.max_iterations = max_iterations
        self.min_parameter_update = min_parameter_update
        self.max_error_increase = max_error_increase

        self.weight_prior = weight_prior
        self.weight_function = weight_function
        self.image_log = log
        self.border_dist = 0.01

    def compute_egomotion(
        self, t1: float, I1: np.array, Z1: np.array, guess=SE3()
    ) -> SE3:
        self.f_no += 1
        I1, Z1 = self.compute_pyramid(I1, Z1)

        if self.I0 is None:
            self.I0 = I1
            self.Z0 = Z1
            self.t0 = t1
            return guess

        prior = guess
        motion = prior
        for l_ in range(self.nLevels):
            level = self.nLevels - (l_ + 1)
            self.image_log.log_level(level)

            dI0 = self.compute_jacobian_image(self.I0[level])
            dZ0 = self.compute_jacobian_depth(self.Z0[level])

            mask_selected = self.select_constraints(
                self.Z0[level].reshape(-1, 1), dI0, dZ0
            )

            pcl0 = self.cam[level].reconstruct(
                self.cam[level].image_coordinates()[mask_selected],
                self.Z0[level].reshape(-1, 1)[mask_selected],
            )

            Jwx, Jwy = self.compute_jacobian_warp_xy(motion * pcl0, self.cam[level])

            JIJw = dI0[:, :1][mask_selected] * Jwx + dI0[:, 1:][mask_selected] * Jwy
            JZJw = dZ0[:, :1][mask_selected] * Jwx + dZ0[:, 1:][mask_selected] * Jwy

            error = np.zeros((self.max_iterations,))
            dx = np.zeros((6,))
            reason = "Max iterations exceeded"
            self.weight_function.scale = np.identity(2)
            for i in range(self.max_iterations):
                pcl0t = motion * pcl0
                uv0t = self.cam[level].project(pcl0t)
                mask_visible = self.cam[level].select_visible(uv0t, pcl0t[:, 2])
                uv0t = uv0t[mask_visible]

                i1wxp, z1wxp, mask_valid = self.interpolate(I1[level], Z1[level], uv0t)

                if i1wxp.shape[0] < 6:
                    reason = "Not enough constraints"
                    motion = SE3()
                    break

                pcl1t = motion.inverse() * (
                    self.cam[level].reconstruct(uv0t[mask_valid], z1wxp)
                )
                uv0 = (
                    self.cam[level]
                    .image_coordinates()[mask_selected][mask_visible][mask_valid]
                    .astype(int)
                )

                i0x = self.I0[level][uv0[:, 1], uv0[:, 0]].reshape((-1,))
                z0x = self.Z0[level][uv0[:, 1], uv0[:, 0]].reshape((-1,))

                r = np.vstack(((i1wxp - i0x), pcl1t[:, 2] - z0x)).T

                weights = self.weight_function.compute_weight_matrices(
                    r, np.ones((r.shape[0],))
                )

                error[i] = np.sum(
                    r[:, np.newaxis] @ (weights @ r[:, :, np.newaxis])
                ) + self.weight_prior * np.linalg.norm((motion * prior.inverse()).log())

                if i > 0 and error[i] / error[i - 1] > self.max_error_increase:
                    reason = f"Error increased: {error[i]/error[i-1]:.2f}/{self.max_error_increase:.2f}"
                    motion = SE3.exp(dx) * motion
                    break

                JZJw_Jtz = JZJw[mask_visible][mask_valid] - self.compute_jacobian_se3_z(
                    pcl1t
                )
                J = np.hstack(
                    [
                        JIJw[mask_visible][mask_valid][:, np.newaxis],
                        JZJw_Jtz[:, np.newaxis],
                    ]
                )
                dx = self.solve_linear_equations(r, J, weights, prior, motion)
                motion = SE3.exp(-dx) * motion

                self.image_log.log_iteration(
                    i,
                    uv0,
                    (self.I0, self.Z0),
                    (i1wxp, z1wxp),
                    r,
                    weights,
                    error,
                    dx,
                )
                if np.linalg.norm(dx) < self.min_parameter_update:
                    reason = f"Min Step Size reached: {np.linalg.norm(dx):.6f}/{self.min_parameter_update:.6f}"
                    break

            self.image_log.log_converged(reason, motion)
        self.I0 = I1
        self.Z0 = Z1
        self.t0 = t1
        return motion

    def compute_pyramid(self, I, Z):
        I = [I]
        Z = [Z]
        for l in range(1, self.nLevels):
            I += [cv.pyrDown(I[l - 1])]
            Z += [
                cv.resize(
                    Z[l - 1], (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_NEAREST
                )
            ]
        return I, Z

    def compute_jacobian_image(self, I):
        dIdx = cv.Sobel(I, cv.CV_64F, dx=1, dy=0, scale=1 / 8)
        dIdy = cv.Sobel(I, cv.CV_64F, dx=0, dy=1, scale=1 / 8)

        return np.reshape(np.dstack([dIdx, dIdy]), (-1, 2))

    def compute_jacobian_depth(self, Z):
        dZ = np.gradient(Z)

        return np.reshape(np.dstack([dZ[1], dZ[0]]), (-1, 2))

    def select_constraints(self, Z, dI, dZ):
        return (
            (np.isfinite(Z[:, 0]))
            & (np.isfinite(dZ[:, 0]))
            & (np.isfinite(dZ[:, 1]))
            & (Z[:, 0] > 0)
            & (Z[:, 0] < self.max_z)
            & (np.abs(dZ[:, 0]) < self.max_dZ)
            & (np.abs(dZ[:, 1]) < self.max_dZ)
            & (
                (np.abs(dI[:, 0]) > self.min_dI)
                | (np.abs(dI[:, 1]) > self.min_dI)
                | (np.abs(dZ[:, 0]) > self.min_dZ)
                | (np.abs(dZ[:, 1]) > self.min_dZ)
            )
        )

    def compute_jacobian_warp_xy(self, pcl: np.array, cam: Camera) -> np.array:
        x = pcl[:, 0]
        y = pcl[:, 1]
        z_inv = 1.0 / pcl[:, 2]
        z_inv_2 = z_inv * z_inv

        Jx = np.zeros((pcl.shape[0], 6))
        Jx[:, 0] = z_inv
        Jx[:, 2] = -x * z_inv_2
        Jx[:, 3] = y * Jx[:, 2]
        Jx[:, 4] = 1.0 - x * Jx[:, 2]
        Jx[:, 5] = -y * z_inv
        Jx *= cam.fx

        Jy = np.zeros((pcl.shape[0], 6))
        Jy[:, 1] = z_inv
        Jy[:, 2] = -y * z_inv_2
        Jy[:, 3] = -1.0 + y * Jy[:, 2]
        Jy[:, 4] = -Jy[:, 3]
        Jy[:, 5] = x * z_inv
        Jy *= cam.fy

        return Jx, Jy

    def compute_jacobian_se3_z(self, pcl: np.array) -> np.array:
        J = np.zeros((pcl.shape[0], 6))
        J[:, 2] = 1.0
        J[:, 3] = pcl[:, 1]
        J[:, 4] = -pcl[:, 0]
        return J

    def interpolate(self, I: np.array, Z, uv: np.array) -> np.array:
        u = uv[:, 0]
        v = uv[:, 1]
        u0 = np.floor(u).astype(int)
        u1 = np.ceil(u).astype(int)
        v0 = np.floor(v).astype(int)
        v1 = np.ceil(v).astype(int)

        w_u1 = u - u0
        w_u0 = 1.0 - w_u1
        w_v1 = v - v0
        w_v0 = 1.0 - w_v1
        u_v_wu_wv = (
            (u0, v0, w_u0, w_v0),
            (u1, v0, w_u1, w_v0),
            (u0, v1, w_u0, w_v1),
            (u1, v1, w_u1, w_v1),
        )
        w = []
        for u_, v_, w_u, w_v in u_v_wu_wv:
            w_ = np.reshape(w_u * w_v, (-1, 1))
            w_[~np.isfinite(Z[v_, u_].reshape((-1, 1)))] = 0
            w_[Z[v_, u_].reshape((-1, 1)) <= 0] = 0
            w += [w_]

        IZ = np.dstack([I, Z])
        IZvu = (
            w[0] * IZ[v0, u0]
            + w[1] * IZ[v0, u1]
            + w[2] * IZ[v1, u0]
            + w[3] * IZ[v1, u1]
        )
        IZvu /= np.reshape(w[0] + w[1] + w[2] + w[3], (-1, 1))

        mask_valid = (
            np.isfinite(IZvu[:, 0].reshape((-1,)))
            & np.isfinite(IZvu[:, 1].reshape((-1,)))
            & (IZvu[:, 1].reshape((-1,)) > 0)
        )

        return (
            IZvu[:, 0].reshape((-1,))[mask_valid],
            IZvu[:, 1].reshape((-1,))[mask_valid],
            mask_valid,
        )

    def solve_linear_equations(self, r, J, weights, prior, motion):
        A = self.weight_prior * np.identity(6)
        b = self.weight_prior * (motion * prior.inverse()).log()

        JT = np.transpose(J, (0, 2, 1))
        A += np.sum(JT @ weights @ J, axis=0).reshape((6, 6))
        b += np.sum(JT @ weights @ r[:, :, np.newaxis], axis=0).reshape((6,))

        dx = np.linalg.solve(A, b)
        return dx
