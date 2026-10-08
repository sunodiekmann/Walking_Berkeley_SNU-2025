import numpy as np
import pickle


class PolyReferenceMotion:
    def __init__(self, polynomial_coefficients: str):
        data = pickle.load(open(polynomial_coefficients, "rb"))

        self.dxs = []
        self.dys = []
        self.dthetas = []

        self.dx_range = None
        self.dy_range = None
        self.dtheta_range = None

        self.period = None
        self.fps = None
        self.nb_steps_in_period = None
        self.degree = None
        self.y_dim = None

        # coeff tensor: (nx, ny, nth, y_dim, degree+1) in low->high storage (as in PKL)
        self.data_array = None

        self.process(data)

    def _parse_key(self, name: str):
        # expected: "{dx}_{dy}_{dtheta}"
        split = name.split("_")
        if len(split) < 3:
            raise ValueError(f"Bad motion key '{name}', expected 'dx_dy_dtheta'")
        dx = float(split[0])
        dy = float(split[1])
        dtheta = float(split[2])
        return dx, dy, dtheta

    def process(self, data: dict):
        print("[Poly ref data] Processing ...")

        # pass 1: gather grids + metadata
        for name, entry in data.items():
            dx, dy, dtheta = self._parse_key(name)
            self.dxs.append(dx)
            self.dys.append(dy)
            self.dthetas.append(dtheta)

            if self.period is None:
                self.period = float(entry["period"])
                self.fps = int(entry["fps"])
                self.nb_steps_in_period = int(entry.get("nb_steps_in_period", round(self.period * self.fps)))
                self.degree = int(entry.get("degree", 0))
                self.y_dim = int(entry.get("y_dim", len(entry["coefficients"])))

        self.dxs = sorted(set(self.dxs))
        self.dys = sorted(set(self.dys))
        self.dthetas = sorted(set(self.dthetas))

        self.dx_range = (float(min(self.dxs)), float(max(self.dxs)))
        self.dy_range = (float(min(self.dys)), float(max(self.dys)))
        self.dtheta_range = (float(min(self.dthetas)), float(max(self.dthetas)))

        nx, ny, nth = len(self.dxs), len(self.dys), len(self.dthetas)

        # infer y_dim/degree from first entry if not provided
        if self.y_dim is None or self.degree is None or self.degree == 0:
            any_entry = next(iter(data.values()))
            coeff0 = any_entry["coefficients"]["dim_0"]
            self.degree = len(coeff0) - 1
            self.y_dim = len(any_entry["coefficients"])

        # allocate tensor (nx, ny, nth, y_dim, deg+1)
        self.data_array = np.zeros((nx, ny, nth, self.y_dim, self.degree + 1), dtype=np.float32)

        # pass 2: fill tensor
        for name, entry in data.items():
            dx, dy, dtheta = self._parse_key(name)

            ix = self.dxs.index(dx)
            iy = self.dys.index(dy)
            it = self.dthetas.index(dtheta)

            coeffs_dict = entry["coefficients"]  # dim_k -> list(low->high)
            for dim in range(self.y_dim):
                self.data_array[ix, iy, it, dim, :] = np.asarray(coeffs_dict[f"dim_{dim}"], dtype=np.float32)

        print("dx grid:", self.dxs)
        print("dy grid:", self.dys)
        print("dtheta grid:", self.dthetas)
        print("coeff tensor shape:", self.data_array.shape)
        print("[Poly ref data] Done processing")

    def vel_to_index(self, dx, dy, dtheta):
        dx = float(np.clip(dx, self.dx_range[0], self.dx_range[1]))
        dy = float(np.clip(dy, self.dy_range[0], self.dy_range[1]))
        dtheta = float(np.clip(dtheta, self.dtheta_range[0], self.dtheta_range[1]))

        ix = int(np.argmin(np.abs(np.asarray(self.dxs, dtype=np.float32) - dx)))
        iy = int(np.argmin(np.abs(np.asarray(self.dys, dtype=np.float32) - dy)))
        it = int(np.argmin(np.abs(np.asarray(self.dthetas, dtype=np.float32) - dtheta)))
        return ix, iy, it

    def sample_polynomial(self, t: float, coeffs_low_to_high: np.ndarray) -> np.ndarray:
        """
        coeffs_low_to_high: (y_dim, deg+1) low->high order
        returns: (y_dim,)
        """
        # np.polyval expects high->low, so flip last axis
        c_hi_to_low = coeffs_low_to_high[..., ::-1]
        # vectorized polyval over dims
        return np.array([np.polyval(c_hi_to_low[d], t) for d in range(c_hi_to_low.shape[0])], dtype=np.float32)

    def get_reference_motion(self, dx, dy, dtheta, i: int):
        ix, iy, itheta = self.vel_to_index(dx, dy, dtheta)
        t = (i % self.nb_steps_in_period) / float(self.nb_steps_in_period)
        t = float(np.clip(t, 0.0, 1.0))
        coeffs = self.data_array[ix, iy, itheta]  # (y_dim, deg+1) low->high
        return self.sample_polynomial(t, coeffs)


if __name__ == "__main__":
    PRM = PolyReferenceMotion("berkeley_humanoid_lite/data/polynomial_coefficients.pkl")

    vals = []
    select_dim = -1
    for i in range(PRM.nb_steps_in_period):
        vals.append(PRM.get_reference_motion(0.0, -0.05, -0.1, i)[select_dim])

    import matplotlib.pyplot as plt

    ts = np.arange(0, PRM.nb_steps_in_period)
    plt.plot(ts, vals)
    plt.show()
