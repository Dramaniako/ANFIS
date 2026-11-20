"""Implementasi Adaptive Neuro-Fuzzy Inference System (ANFIS) 5-Layer dari nol.

Mengikuti arsitektur jaringan adaptif Jang (1993) dengan Sistem Inferensi Fuzzy
Takagi-Sugeno Orde-1 dan Algoritma Hybrid Learning:
- Forward Pass : Ridge Least Squares Estimator (LSE) untuk parameter konsekuen.
- Backward Pass: Backpropagation dengan Normalisasi Langkah Adaptif (Jang Step Size)
                 untuk parameter premis fungsi keanggotaan.
"""

from __future__ import annotations

import copy
import itertools
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

MAX_GRID_RULES = 128

_MF_LINGUISTIC_LABELS: dict[int, list[str]] = {
    2: ["Rendah", "Tinggi"],
    3: ["Rendah", "Sedang", "Tinggi"],
    4: ["Sangat Rendah", "Rendah", "Sedang", "Tinggi"],
}


def get_mf_label(n_mfs: int, mf_idx: int) -> str:
    """Mengembalikan label linguistik yang mudah dibaca untuk indeks fungsi keanggotaan."""
    labels = _MF_LINGUISTIC_LABELS.get(n_mfs)
    if labels and 0 <= mf_idx < len(labels):
        return labels[mf_idx]
    return f"MF-{mf_idx + 1}"


class ANFIS:
    """Model Adaptive Neuro-Fuzzy Inference System (ANFIS) 5-Lapisan untuk Regresi."""

    def __init__(
        self,
        n_inputs: int,
        partition_type: Literal["grid", "scatter"] = "grid",
        n_mfs: int = 2,
        n_rules: int = 12,
        mf_type: Literal["gaussian", "gbell"] = "gaussian",
        learning_rate: float = 0.02,
        l2_reg: float = 1e-3,
        epochs: int = 50,
        random_state: int = 42,
        feature_names: list[str] | None = None,
    ) -> None:
        if n_inputs < 1:
            raise ValueError("Jumlah fitur input (n_inputs) minimal harus 1.")
        if partition_type not in ("grid", "scatter"):
            raise ValueError(f"partition_type harus 'grid' atau 'scatter', diterima: {partition_type!r}")
        if mf_type not in ("gaussian", "gbell"):
            raise ValueError(f"mf_type harus 'gaussian' atau 'gbell', diterima: {mf_type!r}")

        if partition_type == "grid":
            if n_mfs < 2 or n_mfs > 4:
                raise ValueError("Untuk Grid Partitioning, jumlah MF per input harus antara 2 hingga 4.")
            total_rules = n_mfs ** n_inputs
            if total_rules > MAX_GRID_RULES:
                raise ValueError(
                    f"Kombinasi {n_inputs} fitur dengan {n_mfs} MF menghasilkan {total_rules} aturan "
                    f"(melebihi batas aman {MAX_GRID_RULES} aturan). Kurangi jumlah fitur, turunkan jumlah MF menjadi 2, "
                    f"atau beralih ke metode Scatter Partitioning (K-Means)."
                )
            resolved_rules = total_rules
        else:
            if n_rules < 2 or n_rules > 64:
                raise ValueError("Untuk Scatter Partitioning, jumlah aturan (cluster) harus antara 2 hingga 64.")
            resolved_rules = int(n_rules)

        self.n_inputs = int(n_inputs)
        self.partition_type: Literal["grid", "scatter"] = partition_type
        self.n_mfs = int(n_mfs)
        self.n_rules = int(resolved_rules)
        self.mf_type: Literal["gaussian", "gbell"] = mf_type
        self.learning_rate = float(learning_rate)
        self.l2_reg = float(l2_reg)
        self.epochs = int(epochs)
        self.random_state = int(random_state)
        self.feature_names = (
            list(feature_names)
            if feature_names is not None
            else [f"x{j + 1}" for j in range(self.n_inputs)]
        )

        self.premise_params_: dict[str, np.ndarray] = {}
        self.initial_premise_params_: dict[str, np.ndarray] = {}
        self.consequent_params_: np.ndarray | None = None
        self.rule_mf_indices_: np.ndarray | None = None
        self.history_: list[dict[str, float]] = []
        self.is_fitted_: bool = False

    def _init_premise_params(self, X: np.ndarray) -> None:
        """Menginisialisasi parameter premis fungsi keanggotaan untuk Grid atau Scatter."""
        D = self.n_inputs
        if self.partition_type == "grid":
            M = self.n_mfs
            centers_1d = np.linspace(0.1, 0.9, M)
            centers = np.tile(centers_1d, (D, 1))
            width_val = 0.48 / max(M - 1, 1)

            if self.mf_type == "gaussian":
                self.premise_params_ = {
                    "c": centers.copy(),
                    "sigma": np.full((D, M), width_val, dtype=np.float64),
                }
            else:
                self.premise_params_ = {
                    "c": centers.copy(),
                    "a": np.full((D, M), width_val, dtype=np.float64),
                    "b": np.full((D, M), 2.0, dtype=np.float64),
                }

            combos = list(itertools.product(range(M), repeat=D))
            self.rule_mf_indices_ = np.asarray(combos, dtype=np.int64)
        else:
            R = min(self.n_rules, len(X))
            self.n_rules = R
            kmeans = KMeans(n_clusters=R, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            centers = kmeans.cluster_centers_.astype(np.float64)

            global_std = np.maximum(np.std(X, axis=0), 0.15)
            sigmas = np.zeros((R, D), dtype=np.float64)
            for k in range(R):
                cluster_pts = X[labels == k]
                if len(cluster_pts) >= 2:
                    c_std = np.std(cluster_pts, axis=0)
                    sigmas[k] = np.where(c_std < 0.08, global_std * 0.65, c_std * 1.15)
                else:
                    sigmas[k] = global_std * 0.75
            sigmas = np.clip(sigmas, 0.18, 0.55)

            if self.mf_type == "gaussian":
                self.premise_params_ = {"c": centers, "sigma": sigmas}
            else:
                self.premise_params_ = {
                    "c": centers,
                    "a": sigmas.copy(),
                    "b": np.full((R, D), 2.0, dtype=np.float64),
                }
            self.rule_mf_indices_ = None

        self.initial_premise_params_ = copy.deepcopy(self.premise_params_)
        self.consequent_params_ = np.zeros((self.n_rules, D + 1), dtype=np.float64)

    def _eval_membership(
        self,
        x_vals: np.ndarray,
        params: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Menghitung nilai derajat keanggotaan untuk tensor input & parameter selaras."""
        if self.mf_type == "gaussian":
            diff = x_vals - params["c"]
            sigma = params["sigma"]
            return np.exp(-0.5 * (diff / sigma) ** 2)

        diff = x_vals - params["c"]
        a = params["a"]
        b = params["b"]
        z = (diff / a) ** 2 + 1e-12
        return 1.0 / (1.0 + np.power(z, b))

    def _forward_layers(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Menjalankan propagasi maju lengkap Lapisan 1 hingga Lapisan 5 ANFIS."""
        X_arr = np.asarray(X, dtype=np.float64)
        N, D = X_arr.shape
        if D != self.n_inputs:
            raise ValueError(f"Dimensi fitur input tidak sesuai: diharapkan {self.n_inputs}, diterima {D}.")

        # --- Layer 1: Fuzzifikasi ---
        if self.partition_type == "grid":
            assert self.rule_mf_indices_ is not None
            x_exp = X_arr[:, :, None]  # (N, D, 1)
            params_exp = {k: v[None, :, :] for k, v in self.premise_params_.items()}
            mu_grid = self._eval_membership(x_exp, params_exp)  # (N, D, M)
            col_idx = np.arange(D)[None, :]  # (1, D)
            mu_rules = mu_grid[:, col_idx, self.rule_mf_indices_]  # (N, R, D)
        else:
            mu_grid = np.empty((0,), dtype=np.float64)
            x_exp = X_arr[:, None, :]  # (N, 1, D)
            params_exp = {k: v[None, :, :] for k, v in self.premise_params_.items()}
            mu_rules = self._eval_membership(x_exp, params_exp)  # (N, R, D)

        # --- Layer 2: Product T-Norm Firing Strength ---
        w = np.prod(mu_rules, axis=2)  # (N, R)

        # --- Layer 3: Normalisasi Firing Strength ---
        w_sum = np.maximum(np.sum(w, axis=1, keepdims=True), 1e-12)  # (N, 1)
        w_bar = w / w_sum  # (N, R)

        # --- Layer 4: Node Konsekuen Takagi-Sugeno Orde-1 ---
        X_aug = np.hstack([X_arr, np.ones((N, 1), dtype=np.float64)])  # (N, D + 1)
        assert self.consequent_params_ is not None
        f = X_aug @ self.consequent_params_.T  # (N, R)
        weighted_f = w_bar * f  # (N, R)

        # --- Layer 5: Agregasi Output ---
        y_pred = np.sum(weighted_f, axis=1)  # (N,)

        return {
            "X": X_arr,
            "X_aug": X_aug,
            "mu_grid": mu_grid,
            "mu_rules": mu_rules,
            "w": w,
            "w_sum": w_sum,
            "w_bar": w_bar,
            "f": f,
            "weighted_f": weighted_f,
            "y_pred": y_pred,
        }

    def _solve_consequents_lse(
        self,
        w_bar: np.ndarray,
        X_aug: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Menghitung parameter konsekuen optimal menggunakan Ridge Least Squares Estimator (LSE)."""
        N, R = w_bar.shape
        D_plus_1 = X_aug.shape[1]

        # Matriks desain A berukuran (N, R * (D + 1))
        A = (w_bar[:, :, None] * X_aug[:, None, :]).reshape(N, R * D_plus_1)
        n_params = R * D_plus_1

        ATA = A.T @ A
        ATA[np.diag_indices(n_params)] += self.l2_reg
        ATy = A.T @ y

        try:
            theta = np.linalg.solve(ATA, ATy)
        except np.linalg.LinAlgError:
            theta, _, _, _ = np.linalg.lstsq(ATA, ATy, rcond=None)

        return theta.reshape(R, D_plus_1)

    def _compute_premise_gradients(
        self,
        cache: dict[str, np.ndarray],
        y: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Menghitung gradien analitik eksak fungsi rugi L = 0.5 * mean((y_pred - y)^2) terhadap premis."""
        X_arr = cache["X"]
        mu_rules = cache["mu_rules"]
        w_sum = cache["w_sum"]
        f = cache["f"]
        y_pred = cache["y_pred"]

        N, R, D = mu_rules.shape

        # L = (1 / 2N) * sum((y_pred - y)^2)
        dL_dy = (y_pred - y) / float(N)  # (N,)

        # d(y_pred)/d(w_k) = (f_k - y_pred) / sum_m(w_m)
        dL_dw = dL_dy[:, None] * (f - y_pred[:, None]) / w_sum  # (N, R)

        # Produk seluruh mu kecuali dimensi j menggunakan prefix & suffix product (aman terhadap nilai 0)
        prefix = np.ones_like(mu_rules)
        suffix = np.ones_like(mu_rules)
        for j in range(1, D):
            prefix[:, :, j] = prefix[:, :, j - 1] * mu_rules[:, :, j - 1]
        for j in range(D - 2, -1, -1):
            suffix[:, :, j] = suffix[:, :, j + 1] * mu_rules[:, :, j + 1]
        prod_except_j = prefix * suffix  # (N, R, D)

        dL_dmu_rules = dL_dw[:, :, None] * prod_except_j  # (N, R, D)

        # Kumpulkan parameter yang selaras dengan shape (1, R, D)
        if self.partition_type == "grid":
            assert self.rule_mf_indices_ is not None
            col_idx = np.arange(D)[None, :]
            rule_params = {
                k: v[col_idx, self.rule_mf_indices_][None, :, :]
                for k, v in self.premise_params_.items()
            }
        else:
            rule_params = {k: v[None, :, :] for k, v in self.premise_params_.items()}

        x_exp = X_arr[:, None, :]  # (N, 1, D)
        diff = x_exp - rule_params["c"]

        local_rule_grads: dict[str, np.ndarray] = {}
        if self.mf_type == "gaussian":
            sigma = rule_params["sigma"]
            dmu_dc = mu_rules * (diff / (sigma ** 2))
            dmu_dsigma = mu_rules * ((diff ** 2) / (sigma ** 3))
            local_rule_grads["c"] = np.sum(dL_dmu_rules * dmu_dc, axis=0)
            local_rule_grads["sigma"] = np.sum(dL_dmu_rules * dmu_dsigma, axis=0)
        else:
            a = rule_params["a"]
            b = rule_params["b"]
            z = (diff / a) ** 2 + 1e-12
            mu_one_minus_mu = mu_rules * (1.0 - mu_rules)
            dmu_dc = mu_one_minus_mu * (2.0 * b * diff) / ((a ** 2) * z)
            dmu_da = mu_one_minus_mu * (2.0 * b * (diff ** 2)) / ((a ** 3) * z)
            dmu_db = -mu_one_minus_mu * np.log(z)
            local_rule_grads["c"] = np.sum(dL_dmu_rules * dmu_dc, axis=0)
            local_rule_grads["a"] = np.sum(dL_dmu_rules * dmu_da, axis=0)
            local_rule_grads["b"] = np.sum(dL_dmu_rules * dmu_db, axis=0)

        if self.partition_type == "scatter":
            return local_rule_grads

        # Agregasi gradien dari aturan (R, D) kembali ke tabel MF (D, M)
        assert self.rule_mf_indices_ is not None
        grads: dict[str, np.ndarray] = {}
        for pname, rgrad in local_rule_grads.items():
            agg = np.zeros((D, self.n_mfs), dtype=np.float64)
            for j in range(D):
                np.add.at(agg[j], self.rule_mf_indices_[:, j], rgrad[:, j])
            grads[pname] = agg
        return grads

    def _apply_premise_step(self, grads: dict[str, np.ndarray], kappa: float) -> None:
        """Memperbarui parameter premis memakai pembaruan gradien ternormalisasi ala Jang (1993)."""
        flat_norms_sq = sum(float(np.sum(g ** 2)) for g in grads.values())
        grad_norm = float(np.sqrt(flat_norms_sq))
        if grad_norm < 1e-12:
            return

        step = kappa / grad_norm
        for pname, grad in grads.items():
            self.premise_params_[pname] = self.premise_params_[pname] - step * grad

        # Jaga parameter tetap pada rentang numerik yang stabil dan bermakna secara fisik
        self.premise_params_["c"] = np.clip(self.premise_params_["c"], -0.25, 1.25)
        if self.mf_type == "gaussian":
            self.premise_params_["sigma"] = np.clip(self.premise_params_["sigma"], 0.06, 1.50)
        else:
            self.premise_params_["a"] = np.clip(self.premise_params_["a"], 0.06, 1.50)
            self.premise_params_["b"] = np.clip(self.premise_params_["b"], 0.50, 6.00)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        progress_callback: Callable[[int, int, dict[str, float]], None] | None = None,
    ) -> ANFIS:
        """Melatih ANFIS dengan Hybrid Learning (Forward LSE + Backward Gradient Descent)."""
        X_tr = np.asarray(X_train, dtype=np.float64)
        y_tr = np.asarray(y_train, dtype=np.float64).ravel()
        if len(X_tr) != len(y_tr):
            raise ValueError("Jumlah sampel X_train dan y_train harus sama.")

        has_val = X_val is not None and y_val is not None
        X_v = np.asarray(X_val, dtype=np.float64) if has_val else None
        y_v = np.asarray(y_val, dtype=np.float64).ravel() if has_val else None

        self._init_premise_params(X_tr)
        self.history_ = []

        kappa = self.learning_rate
        best_loss = float("inf")
        best_premise = copy.deepcopy(self.premise_params_)
        best_consequent = self.consequent_params_.copy()

        recent_rmses: list[float] = []

        for epoch in range(1, self.epochs + 1):
            # 1. Forward Pass bagian awal: hitung derajat keanggotaan & firing strength ternormalisasi
            pre_cache = self._forward_layers(X_tr)

            # 2. Estimasi konsekuen optimal menggunakan Ridge LSE
            self.consequent_params_ = self._solve_consequents_lse(
                pre_cache["w_bar"],
                pre_cache["X_aug"],
                y_tr,
            )

            # 3. Evaluasi error setelah LSE
            post_cache = self._forward_layers(X_tr)
            train_mse = float(np.mean((post_cache["y_pred"] - y_tr) ** 2))
            train_rmse = float(np.sqrt(train_mse))

            val_rmse = float("nan")
            if has_val and X_v is not None and y_v is not None:
                val_pred = self._forward_layers(X_v)["y_pred"]
                val_rmse = float(np.sqrt(np.mean((val_pred - y_v) ** 2)))

            if train_rmse < best_loss:
                best_loss = train_rmse
                best_premise = copy.deepcopy(self.premise_params_)
                best_consequent = self.consequent_params_.copy()

            entry = {
                "epoch": float(epoch),
                "train_rmse": train_rmse,
                "train_mse": train_mse,
                "val_rmse": val_rmse,
                "kappa": kappa,
            }
            self.history_.append(entry)

            if progress_callback is not None:
                progress_callback(epoch, self.epochs, entry)

            # 4. Backward Pass: hitung gradien premis dan perbarui dengan langkah adaptif Jang
            if epoch < self.epochs:
                grads = self._compute_premise_gradients(post_cache, y_tr)
                self._apply_premise_step(grads, kappa)

                # Jadwal adaptasi step-size ala Jang (1993)
                recent_rmses.append(train_rmse)
                if len(recent_rmses) >= 5:
                    d1 = recent_rmses[-4] - recent_rmses[-5]
                    d2 = recent_rmses[-3] - recent_rmses[-4]
                    d3 = recent_rmses[-2] - recent_rmses[-3]
                    d4 = recent_rmses[-1] - recent_rmses[-2]
                    if d1 < 0 and d2 < 0 and d3 < 0 and d4 < 0:
                        kappa = min(kappa * 1.1, 0.20)
                        recent_rmses = recent_rmses[-1:]
                    elif (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0):
                        kappa = max(kappa * 0.9, 1e-4)

        # Pulihkan parameter terbaik selama proses iterasi
        self.premise_params_ = best_premise
        self.consequent_params_ = best_consequent
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Memprediksi skor ISPU untuk sampel masukan ternormalisasi X."""
        if not self.is_fitted_ and self.consequent_params_ is None:
            raise RuntimeError("Model ANFIS belum dilatih. Panggil fit() terlebih dahulu.")
        X_arr = np.asarray(X, dtype=np.float64)
        single = X_arr.ndim == 1
        if single:
            X_arr = X_arr.reshape(1, -1)
        preds = self._forward_layers(X_arr)["y_pred"]
        return preds[0] if single else preds

    def evaluate_mf_curve(
        self,
        feature_idx: int,
        x_grid: np.ndarray,
        use_initial: bool = False,
    ) -> list[tuple[str, np.ndarray]]:
        """Menghasilkan kurva derajat keanggotaan pada grid [0, 1] untuk visualisasi GUI."""
        params_src = self.initial_premise_params_ if use_initial else self.premise_params_
        x_vals = np.asarray(x_grid, dtype=np.float64)
        curves: list[tuple[str, np.ndarray]] = []

        if self.partition_type == "grid":
            for m in range(self.n_mfs):
                sub_p = {k: v[feature_idx, m] for k, v in params_src.items()}
                mu = self._eval_membership(x_vals, sub_p)
                curves.append((get_mf_label(self.n_mfs, m), mu))
        else:
            for k in range(self.n_rules):
                sub_p = {pname: val[k, feature_idx] for pname, val in params_src.items()}
                mu = self._eval_membership(x_vals, sub_p)
                curves.append((f"Aturan-{k + 1}", mu))

        return curves

    def format_rule_antecedent(self, rule_idx: int) -> str:
        """Membentuk representasi teks bagian IF dari suatu aturan fuzzy."""
        if self.partition_type == "grid" and self.rule_mf_indices_ is not None:
            clauses = [
                f"{self.feature_names[j]} = {get_mf_label(self.n_mfs, int(self.rule_mf_indices_[rule_idx, j]))}"
                for j in range(self.n_inputs)
            ]
            return "IF " + " AND ".join(clauses)

        centers = self.premise_params_["c"][rule_idx]
        clauses = [f"{self.feature_names[j]} ≈ {centers[j]:.2f}" for j in range(self.n_inputs)]
        return "IF " + " AND ".join(clauses)

    def forward_trace(
        self,
        x_scaled_1d: np.ndarray,
        x_raw_1d: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Menghasilkan rincian lengkap komputasi Lapisan 1–5 untuk satu sampel masukan."""
        x_sc = np.asarray(x_scaled_1d, dtype=np.float64).ravel()
        if len(x_sc) != self.n_inputs:
            raise ValueError(f"Panjang vektor x_scaled harus {self.n_inputs}, diterima {len(x_sc)}.")

        cache = self._forward_layers(x_sc.reshape(1, -1))
        x_raw = np.asarray(x_raw_1d, dtype=np.float64).ravel() if x_raw_1d is not None else x_sc

        # Tabel Layer 1
        mf_rows: list[dict[str, Any]] = []
        if self.partition_type == "grid":
            mu_grid = cache["mu_grid"][0]  # (D, M)
            for j in range(self.n_inputs):
                for m in range(self.n_mfs):
                    mf_rows.append(
                        {
                            "Fitur": self.feature_names[j],
                            "Input_Mentah": float(x_raw[j]),
                            "Input_Normal": float(x_sc[j]),
                            "MF": f"MF-{m + 1}",
                            "Label_Linguistik": get_mf_label(self.n_mfs, m),
                            "Derajat_Keanggotaan": float(mu_grid[j, m]),
                        }
                    )
        else:
            mu_rules = cache["mu_rules"][0]  # (R, D)
            for k in range(self.n_rules):
                for j in range(self.n_inputs):
                    mf_rows.append(
                        {
                            "Fitur": self.feature_names[j],
                            "Input_Mentah": float(x_raw[j]),
                            "Input_Normal": float(x_sc[j]),
                            "MF": f"Cluster-{k + 1}",
                            "Label_Linguistik": f"Aturan-{k + 1}",
                            "Derajat_Keanggotaan": float(mu_rules[k, j]),
                        }
                    )

        # Tabel Layer 2 - 5 per aturan
        w = cache["w"][0]
        w_sum = float(cache["w_sum"][0, 0])
        w_bar = cache["w_bar"][0]
        f = cache["f"][0]
        weighted_f = cache["weighted_f"][0]
        y_pred = float(cache["y_pred"][0])

        rule_rows: list[dict[str, Any]] = []
        for k in range(self.n_rules):
            rule_rows.append(
                {
                    "Aturan": f"R{k + 1}",
                    "Antecedent": self.format_rule_antecedent(k),
                    "w (Layer 2)": float(w[k]),
                    "w_bar (Layer 3)": float(w_bar[k]),
                    "f_k (Layer 4)": float(f[k]),
                    "w_bar * f_k": float(weighted_f[k]),
                }
            )

        rule_df = pd.DataFrame(rule_rows).sort_values("w_bar (Layer 3)", ascending=False).reset_index(drop=True)
        mf_df = pd.DataFrame(mf_rows)

        return {
            "x_raw": x_raw,
            "x_scaled": x_sc,
            "mf_table": mf_df,
            "rule_table": rule_df,
            "w_sum": w_sum,
            "y_pred": y_pred,
        }

    def get_premise_params_dataframe(self) -> pd.DataFrame:
        """Mengembalikan tabel perbandingan parameter premis sebelum dan sesudah pelatihan."""
        rows: list[dict[str, Any]] = []
        if self.partition_type == "grid":
            for j in range(self.n_inputs):
                for m in range(self.n_mfs):
                    row: dict[str, Any] = {
                        "Fitur": self.feature_names[j],
                        "MF": get_mf_label(self.n_mfs, m),
                        "c_awal": float(self.initial_premise_params_["c"][j, m]),
                        "c_akhir": float(self.premise_params_["c"][j, m]),
                    }
                    if self.mf_type == "gaussian":
                        row["sigma_awal"] = float(self.initial_premise_params_["sigma"][j, m])
                        row["sigma_akhir"] = float(self.premise_params_["sigma"][j, m])
                    else:
                        row["a_awal"] = float(self.initial_premise_params_["a"][j, m])
                        row["a_akhir"] = float(self.premise_params_["a"][j, m])
                        row["b_awal"] = float(self.initial_premise_params_["b"][j, m])
                        row["b_akhir"] = float(self.premise_params_["b"][j, m])
                    rows.append(row)
        else:
            for k in range(self.n_rules):
                for j in range(self.n_inputs):
                    row = {
                        "Aturan": f"R{k + 1}",
                        "Fitur": self.feature_names[j],
                        "c_awal": float(self.initial_premise_params_["c"][k, j]),
                        "c_akhir": float(self.premise_params_["c"][k, j]),
                    }
                    if self.mf_type == "gaussian":
                        row["sigma_awal"] = float(self.initial_premise_params_["sigma"][k, j])
                        row["sigma_akhir"] = float(self.premise_params_["sigma"][k, j])
                    else:
                        row["a_awal"] = float(self.initial_premise_params_["a"][k, j])
                        row["a_akhir"] = float(self.premise_params_["a"][k, j])
                        row["b_awal"] = float(self.initial_premise_params_["b"][k, j])
                        row["b_akhir"] = float(self.premise_params_["b"][k, j])
                    rows.append(row)
        return pd.DataFrame(rows)

    def get_rules_dataframe(self) -> pd.DataFrame:
        """Mengembalikan tabel basis aturan fuzzy lengkap beserta persamaan konsekuen Takagi-Sugeno."""
        assert self.consequent_params_ is not None
        rows: list[dict[str, Any]] = []
        for k in range(self.n_rules):
            coeffs = self.consequent_params_[k]
            terms = [f"{coeffs[j]:+.3f}·{self.feature_names[j]}" for j in range(self.n_inputs)]
            eq = " ".join(terms) + f" {coeffs[-1]:+.3f}"
            row: dict[str, Any] = {
                "Aturan": f"R{k + 1}",
                "Premis (IF)": self.format_rule_antecedent(k),
                "Konsekuen (THEN f_k)": eq.lstrip("+"),
            }
            for j in range(self.n_inputs):
                row[f"p_{self.feature_names[j]}"] = float(coeffs[j])
            row["bias_r"] = float(coeffs[-1])
            rows.append(row)
        return pd.DataFrame(rows)

    def summary(self) -> dict[str, Any]:
        """Mengembalikan ringkasan arsitektur dan jumlah parameter model ANFIS."""
        n_premise = int(sum(v.size for v in self.premise_params_.values()))
        n_consequent = int(self.consequent_params_.size) if self.consequent_params_ is not None else 0
        return {
            "n_inputs": self.n_inputs,
            "partition_type": self.partition_type,
            "n_mfs": self.n_mfs if self.partition_type == "grid" else None,
            "n_rules": self.n_rules,
            "mf_type": self.mf_type,
            "n_premise_params": n_premise,
            "n_consequent_params": n_consequent,
            "total_params": n_premise + n_consequent,
            "epochs_run": len(self.history_),
            "final_train_rmse": self.history_[-1]["train_rmse"] if self.history_ else None,
        }
