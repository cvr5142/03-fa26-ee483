"""
Lane pose (d, phi) from line segments with a Kalman filter. No ROS in this file: the node
lane_kalman_node.py feeds it.

    segments (pixels) --GroundProjector--> segments (meters, robot frame)
                      --LaneSegmentFit--> one measurement (d, phi) per frame
    wheel odometry (ds, dtheta) + measurement --LaneKalmanFilter--> (d, phi, sigma_d, sigma_phi)

Conventions (same as dt-core lane_filter / LanePose):
    robot frame: x forward, y left, meters, origin under the wheel axle
    d   > 0  robot is LEFT of the lane center
    phi > 0  robot is turned LEFT of the lane direction

Copied from aux_code/lane_pose_aux/src/lane_pose_lib (ground_projection.py, segment_fit.py,
lane_kalman.py). Keep them in sync.
"""
import cv2
import numpy as np

WHITE, YELLOW = 0, 1  # duckietown_msgs/Segment colors


# --------------------------------------------------------------------------------------------
# Ground projection (dt-core ground_projection_node, without ROS)
# --------------------------------------------------------------------------------------------
class GroundProjector:
    """Normalized image coordinates (like Segment.pixels_normalized) -> ground meters.
        1. normalized [0,1] -> pixel                  (multiply by the camera width/height)
        2. rectify the pixel (undo lens distortion)   (cv2.undistortPoints with K, D, R, P)
        3. [x, y, w] = H @ [u, v, 1]  -> (x/w, y/w)   (H: extrinsic calibration homography)
    """

    def __init__(self, K, D, R, P, width, height, H):
        self.K = np.asarray(K, float).reshape(3, 3)
        self.D = np.asarray(D, float).ravel()
        self.R = np.asarray(R, float).reshape(3, 3)
        self.P = np.asarray(P, float).reshape(3, 4)
        self.width, self.height = int(width), int(height)
        self.H = np.asarray(H, float).reshape(3, 3)

    def normalized_to_ground(self, pts):
        pts = np.asarray(pts, float).reshape(-1, 2)
        if len(pts) == 0:
            return np.zeros((0, 2))
        px = pts * [self.width, self.height]
        rect = cv2.undistortPoints(px.reshape(-1, 1, 2), self.K, self.D, R=self.R, P=self.P).reshape(-1, 2)
        homog = np.hstack([rect, np.ones((len(rect), 1))]) @ self.H.T
        return homog[:, :2] / homog[:, 2:3]


# --------------------------------------------------------------------------------------------
# One frame: robust straight-line fit of the two stripes
# --------------------------------------------------------------------------------------------
class LaneSegmentFit:
    """(d, phi) fitted to one frame's ground segments. Endpoint order does not matter.

    The white and yellow stripes are parallel lines at angle phi to the robot, each at its own
    lateral position a_color (free, so phi comes only from the stripe directions). In the lane
    frame a point's lateral coordinate is l(p) = sin(phi) x + cos(phi) y; its residual is
    l(p) - a_color. Points are sampled along each segment (long segments count more), both
    colors weigh the same, and outliers (other lane, crossings) barely count: a grid over phi
    with a truncated cost picks the start, damped Newton steps with a Cauchy loss refine it
    (numpy only).
    d comes from the midpoint of the two stripes (or one stripe and the lane width).
    """

    def __init__(self, lanewidth, linewidth_white, linewidth_yellow, range_min, range_max,
                 sample_step, outlier_dist, min_segments, balance_colors, phi_step, **_):
        self.lanewidth, self.range_min, self.range_max = lanewidth, range_min, range_max
        self.sample_step, self.outlier_dist = sample_step, outlier_dist
        self.min_segments, self.balance_colors = min_segments, balance_colors
        # lateral position of each stripe's center relative to the lane center
        self.offset = {WHITE: -(lanewidth / 2 + linewidth_white / 2), YELLOW: lanewidth / 2 + linewidth_yellow / 2}
        self.phis = np.arange(-1.4, 1.4 + 1e-9, phi_step)

    def select(self, points, colors):
        mid = points.mean(axis=1)
        dist = np.hypot(mid[:, 0], mid[:, 1])
        return (np.isin(colors, (WHITE, YELLOW)) & (points[:, 0, 0] > 0) & (points[:, 1, 0] > 0)
                & (dist >= self.range_min) & (dist < self.range_max) & np.isfinite(points).all(axis=(1, 2)))

    def _samples(self, points, colors, keep):
        xs, ys, cs = [], [], []
        for k in np.flatnonzero(keep):
            p0, p1 = points[k]
            n = max(2, int(np.ceil(np.linalg.norm(p1 - p0) / self.sample_step)) + 1)
            q = p0 + np.linspace(0, 1, n)[:, None] * (p1 - p0)
            xs.append(q[:, 0])
            ys.append(q[:, 1])
            cs.append(np.full(n, colors[k]))
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(cs)

    def _refine(self, x, y, masks, w0, phi, a, iters=50):
        """Minimize sum rho(r) over theta = (phi, a_color...), rho(r) = C^2 log(1 + w0 r^2 / C^2)
        (Cauchy loss, C = outlier_dist/2 in sqrt(w0) units), with damped Newton steps (numpy only).
        r = l(p) - a_color, so dr/dphi = t(p) = cos(phi) x - sin(phi) y, d2r/dphi2 = -l(p), dr/da = -1.
        Where the Hessian is not positive definite (points in the flat tail of the loss), the step
        uses the Gauss-Newton one (rho'' replaced by its reweighted least squares value)."""
        c2 = (0.5 * self.outlier_dist * np.sqrt(w0).mean()) ** 2
        M = np.stack(masks).astype(float)  # (colors, points)
        th = np.array([phi] + list(a), float)

        def evaluate(th):
            l = self.lateral(x, y, th[0])
            r = l - th[1:] @ M
            return l, r, float(np.sum(c2 * np.log1p(w0 * r ** 2 / c2)))

        l, r, f = evaluate(th)
        for _ in range(iters):
            t = np.cos(th[0]) * x - np.sin(th[0]) * y
            u = w0 * r ** 2 / c2
            d1 = 2 * w0 * r / (1 + u)                # rho'
            g = np.concatenate([[d1 @ t], -(M @ d1)])
            step = None
            for newton in (True, False):
                d2 = 2 * w0 * (1 - u) / (1 + u) ** 2 if newton else 2 * w0 / (1 + u)  # rho'' or Gauss-Newton
                H = np.diag(np.concatenate([[d2 @ (t * t) - (d1 @ l if newton else 0.0)], M @ d2]))
                H[0, 1:] = H[1:, 0] = -(M @ (d2 * t))
                try:
                    np.linalg.cholesky(H)  # positive definite?
                except np.linalg.LinAlgError:
                    continue
                step = -np.linalg.solve(H, g)
                break
            if step is None:
                break
            s = 1.0
            while s > 1e-4:  # backtracking: accept the first step that lowers the cost enough
                new = th + s * step
                new[0] = np.clip(new[0], -1.5, 1.5)
                l_new, r_new, f_new = evaluate(new)
                if f_new <= f + 1e-4 * s * (g @ step):
                    break
                s *= 0.5
            if f_new > f:
                break
            done = np.max(np.abs(new - th)) < 1e-9
            th, l, r, f = new, l_new, r_new, f_new
            if done:
                break
        return float(th[0]), [float(v) for v in th[1:]]

    @staticmethod
    def lateral(x, y, phi):
        return np.sin(phi) * x + np.cos(phi) * y

    def _robust_offsets(self, L, masks):
        out = []
        for m in masks:
            l = L[..., m]
            a = np.median(l, axis=-1, keepdims=True)
            near = np.abs(l - a) <= self.outlier_dist
            cnt = near.sum(axis=-1, keepdims=True)
            a = np.where(cnt > 0, (l * near).sum(axis=-1, keepdims=True) / np.maximum(cnt, 1), a)
            out.append(a[..., 0])
        return out

    def fit(self, points, colors):
        """points: (N, 2, 2) ground meters, colors: (N,). Returns dict with d, phi (nan if no
        fit), inliers (weight fraction within outlier_dist), n (segments used), n_colors."""
        points, colors = np.asarray(points, float).reshape(-1, 2, 2), np.asarray(colors, int)
        keep = self.select(points, colors)
        out = {"d": float("nan"), "phi": float("nan"), "inliers": 0.0, "n": int(keep.sum()), "n_colors": 0}
        if keep.sum() < self.min_segments:
            return out
        x, y, col = self._samples(points, colors, keep)
        cols = [c for c in (WHITE, YELLOW) if np.any(col == c)]
        masks = [col == c for c in cols]
        w0 = np.ones(len(x))
        if self.balance_colors and len(cols) == 2:
            for m in masks:
                w0[m] = 0.5 / m.sum()
        w0 /= w0.sum()

        def resid(l, a):
            r = l.copy()
            for m, ai in zip(masks, a):
                r[..., m] -= np.asarray(ai)[..., None] if np.ndim(ai) else ai
            return r

        L = self.lateral(x, y, self.phis[:, None])
        cost = np.sum(w0 * np.minimum(resid(L, self._robust_offsets(L, masks)) ** 2, self.outlier_dist ** 2), axis=-1)
        phi0 = self.phis[int(np.argmin(cost))]
        a0 = [float(a) for a in self._robust_offsets(self.lateral(x, y, phi0), masks)]
        phi, offsets = self._refine(x, y, masks, w0, phi0, a0)
        a = dict(zip(cols, offsets))
        if len(cols) == 2:
            d = 0.5 * (self.offset[WHITE] + self.offset[YELLOW]) - 0.5 * (a[WHITE] + a[YELLOW])
        else:
            d = self.offset[cols[0]] - a[cols[0]]
        r = resid(self.lateral(x, y, phi), offsets)
        out.update(d=float(d), phi=float(phi), n_colors=len(cols),
                   inliers=float(np.sum(w0[np.abs(r) <= self.outlier_dist])))
        return out


# --------------------------------------------------------------------------------------------
# Kalman filter on (d, phi)
# --------------------------------------------------------------------------------------------
class LaneKalmanFilter:
    """x = [d, phi].
    Predict (robot moved ds, turned dtheta; lane assumed straight):
        phi' = phi + dtheta,  d' = d + ds * sin(phi + dtheta/2)
        Q = diag(q_d_time*dt + q_d_dist*|ds|, q_phi_time*dt + q_phi_rot*|dtheta|)
    Update with z = (d_meas, phi_meas), H = I, R = diag(r_d^2, r_phi^2) / max(quality, floor);
    r_d doubles (single_color_d_scale) when only one stripe was seen.
    A measurement with y^T S^-1 y > gate is rejected; after max_rejects in a row, restart from it.
    """

    def __init__(self, sigma_d_0, sigma_phi_0, q_d_time, q_d_dist, q_phi_time, q_phi_rot, r_d, r_phi,
                 quality_floor, single_color_d_scale, gate, max_rejects, **_):
        self.sigma_d_0, self.sigma_phi_0 = sigma_d_0, sigma_phi_0
        self.q = (q_d_time, q_d_dist, q_phi_time, q_phi_rot)
        self.r_d, self.r_phi, self.quality_floor = r_d, r_phi, quality_floor
        self.single_color_d_scale, self.gate, self.max_rejects = single_color_d_scale, gate, int(max_rejects)
        self.x, self.P, self.rejects = None, None, 0

    @property
    def initialized(self):
        return self.x is not None

    def reset(self, z=None):
        self.x = None if z is None else np.array(z, float)
        self.P = None if z is None else np.diag([self.sigma_d_0 ** 2, self.sigma_phi_0 ** 2])
        self.rejects = 0

    def predict(self, ds, dtheta, dt):
        if not self.initialized:
            return
        q_d_time, q_d_dist, q_phi_time, q_phi_rot = self.q
        d, phi = self.x
        mid = phi + dtheta / 2
        self.x = np.array([d + ds * np.sin(mid), phi + dtheta])
        F = np.array([[1.0, ds * np.cos(mid)], [0.0, 1.0]])
        Q = np.diag([q_d_time * dt + q_d_dist * abs(ds), q_phi_time * dt + q_phi_rot * abs(dtheta)])
        self.P = F @ self.P @ F.T + Q

    def update(self, z, quality=1.0, both_colors=True):
        """Returns 'none', 'init', 'accepted', 'rejected' or 'restart'."""
        if z is None or not np.all(np.isfinite(z)):
            return "none"
        z = np.asarray(z, float)
        if not self.initialized:
            self.reset(z)
            return "init"
        r_d = self.r_d * (1.0 if both_colors else self.single_color_d_scale)
        S = self.P + np.diag([r_d ** 2, self.r_phi ** 2]) / max(quality, self.quality_floor)
        y = z - self.x
        if float(y @ np.linalg.solve(S, y)) > self.gate:
            self.rejects += 1
            if self.rejects > self.max_rejects:
                self.reset(z)
                return "restart"
            return "rejected"
        K = self.P @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K) @ self.P
        self.P = (self.P + self.P.T) / 2
        self.rejects = 0
        return "accepted"


# --------------------------------------------------------------------------------------------
# Everything together, one call per segment list
# --------------------------------------------------------------------------------------------
class LanePoseKF:
    def __init__(self, params, projector=None):
        """params: dict with the LaneSegmentFit and LaneKalmanFilter parameters.
        projector: GroundProjector, or None if the segments are already on the ground."""
        self.fit = LaneSegmentFit(**params)
        self.kf = LaneKalmanFilter(**params)
        self.projector = projector

    def step(self, points, colors, ds, dtheta, dt, on_ground=False):
        """points: (N, 2, 2) normalized pixels (or ground meters if on_ground). Odometry since the
        last call: ds [m], dtheta [rad], dt [s]. Returns a dict (see keys below)."""
        points = np.asarray(points, float).reshape(-1, 2, 2)
        if not on_ground:
            points = self.projector.normalized_to_ground(points.reshape(-1, 2)).reshape(-1, 2, 2)
        if dt > 0:
            self.kf.predict(ds, dtheta, dt)
        meas = self.fit.fit(points, colors)
        status = self.kf.update((meas["d"], meas["phi"]), quality=meas["inliers"],
                                both_colors=meas["n_colors"] == 2)
        if self.kf.initialized:
            d, phi = self.kf.x
            P = self.kf.P
        else:
            d = phi = float("nan")
            P = np.full((2, 2), np.nan)
        return {"d": float(d), "phi": float(phi), "P": P, "status": status,
                "d_meas": meas["d"], "phi_meas": meas["phi"], "inliers": meas["inliers"], "n": meas["n"]}
