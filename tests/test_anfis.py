"""Pengujian otomatis unit, integrasi, verifikasi gradien analitik, dan smoke test GUI."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anfis import ANFIS
from src.evaluation import compute_classification_metrics, compute_regression_metrics
from src.ispu_standard import POLLUTANT_ORDER, classify_ispu, compute_ispu_official, compute_sub_index, validate_manual_input
from src.manual_example import compute_worked_manual_example
from src.preprocessing import MinMaxScalerCustom, prepare_train_test_data, validate_and_clean_dataframe


class TestISPUStandardAndPreprocessing(unittest.TestCase):
    def test_ispu_breakpoints_and_categories(self) -> None:
        self.assertAlmostEqual(compute_sub_index(50.0, "pm10"), 50.0)
        self.assertAlmostEqual(compute_sub_index(15.5, "pm25"), 50.0)
        self.assertAlmostEqual(compute_sub_index(55.4, "pm25"), 100.0)
        self.assertAlmostEqual(compute_sub_index(400.0, "so2"), 200.0)
        self.assertAlmostEqual(compute_sub_index(30000.0, "co"), 300.0)
        self.assertAlmostEqual(compute_sub_index(1000.0, "o3"), 500.0)

        self.assertEqual(classify_ispu(25.0)["name"], "Baik")
        self.assertEqual(classify_ispu(75.0)["name"], "Sedang")
        self.assertEqual(classify_ispu(150.0)["name"], "Tidak Sehat")
        self.assertEqual(classify_ispu(250.0)["name"], "Sangat Tidak Sehat")
        self.assertEqual(classify_ispu(350.0)["name"], "Berbahaya")

        off = compute_ispu_official({"pm10": 50.0, "pm25": 55.4, "so2": 20.0, "co": 1000.0, "o3": 60.0, "no2": 30.0})
        self.assertAlmostEqual(off["ispu"], 100.0)
        self.assertEqual(off["polutan_kritis"], "PM2.5")

    def test_manual_input_validation_rules(self) -> None:
        errs, warns = validate_manual_input(
            {"pm10": -5.0, "pm25": 25.0},
            feature_bounds={"pm10": (10.0, 400.0), "pm25": (5.0, 300.0)},
        )
        self.assertTrue(any("negatif" in e.lower() for e in errs))
        self.assertEqual(len(warns), 0)

        errs_ok, warns_extrap = validate_manual_input(
            {"pm10": 480.0, "pm25": 50.0},
            feature_bounds={"pm10": (10.0, 400.0), "pm25": (5.0, 300.0)},
        )
        self.assertEqual(len(errs_ok), 0)
        self.assertEqual(len(warns_extrap), 1)

    def test_dataset_exists_and_valid(self) -> None:
        csv_path = PROJECT_ROOT / "data" / "ispu_dki_all.csv"
        self.assertTrue(csv_path.exists())
        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 1200)
        self.assertFalse(df[POLLUTANT_ORDER + ["ispu"]].isna().any().any())
        self.assertEqual(df["kategori"].nunique(), 5)

    def test_preprocessing_cleaning_and_scaler(self) -> None:
        base_df = pd.read_csv(PROJECT_ROOT / "data" / "ispu_dki_all.csv").head(60).copy()
        base_df = base_df.astype({"pm10": object})
        base_df.loc[0, "pm10"] = "-"
        base_df.loc[1, "pm25"] = -12.0

        clean_df, rep = validate_and_clean_dataframe(
            base_df,
            feature_cols=POLLUTANT_ORDER,
            target_col="ispu",
            missing_strategy="median",
            clip_negative=True,
        )
        self.assertEqual(len(clean_df), 60)
        self.assertEqual(rep["imputed_cells"], 1)
        self.assertEqual(rep["negative_fixed"], 1)
        self.assertGreaterEqual(float(clean_df["pm25"].min()), 0.0)

        split = prepare_train_test_data(clean_df, POLLUTANT_ORDER, "ispu", test_size=0.2, random_state=42)
        scaler: MinMaxScalerCustom = split["scaler"]
        self.assertTrue(np.allclose(split["X_train"].min(axis=0), 0.0))
        self.assertTrue(np.allclose(split["X_train"].max(axis=0), 1.0))
        restored = scaler.inverse_transform(split["X_test"])
        self.assertTrue(np.allclose(restored, split["X_test_raw"]))


class TestANFISEngine(unittest.TestCase):
    def test_analytical_vs_numerical_gradients(self) -> None:
        rng = np.random.default_rng(7)
        X_sm = rng.uniform(0.05, 0.95, size=(12, 3))
        y_sm = 100.0 * X_sm[:, 0] ** 2 + 50.0 * X_sm[:, 1] - 20.0 * X_sm[:, 2] + 10.0

        for ptype in ("grid", "scatter"):
            for mtype in ("gaussian", "gbell"):
                model = ANFIS(
                    n_inputs=3,
                    partition_type=ptype,
                    n_mfs=2,
                    n_rules=5,
                    mf_type=mtype,
                    epochs=2,
                    l2_reg=1e-3,
                    random_state=7,
                )
                model._init_premise_params(X_sm)
                pre = model._forward_layers(X_sm)
                model.consequent_params_ = model._solve_consequents_lse(pre["w_bar"], pre["X_aug"], y_sm)
                post = model._forward_layers(X_sm)
                grads = model._compute_premise_gradients(post, y_sm)

                eps = 1e-6
                max_rel_err = 0.0
                for pname, g_arr in grads.items():
                    orig = model.premise_params_[pname].copy()
                    for idx in np.ndindex(*orig.shape):
                        model.premise_params_[pname][idx] = orig[idx] + eps
                        loss_plus = 0.5 * float(np.mean((model._forward_layers(X_sm)["y_pred"] - y_sm) ** 2))
                        model.premise_params_[pname][idx] = orig[idx] - eps
                        loss_minus = 0.5 * float(np.mean((model._forward_layers(X_sm)["y_pred"] - y_sm) ** 2))
                        model.premise_params_[pname][idx] = orig[idx]

                        num_g = (loss_plus - loss_minus) / (2.0 * eps)
                        ana_g = float(g_arr[idx])
                        rel_err = abs(num_g - ana_g) / max(1.0, abs(num_g), abs(ana_g))
                        max_rel_err = max(max_rel_err, rel_err)

                self.assertLess(max_rel_err, 1e-4, msg=f"Gradien tidak cocok untuk ({ptype}, {mtype})")

    def test_anfis_hybrid_learning_on_ispu_dataset(self) -> None:
        df = pd.read_csv(PROJECT_ROOT / "data" / "ispu_dki_all.csv")
        split = prepare_train_test_data(df, POLLUTANT_ORDER, "ispu", test_size=0.2, random_state=42)

        lr = LinearRegression().fit(split["X_train"], split["y_train"])
        lr_metrics = compute_regression_metrics(split["y_test"], lr.predict(split["X_test"]))

        for ptype, kwargs in [("grid", {"n_mfs": 2, "epochs": 35}), ("scatter", {"n_rules": 12, "epochs": 35})]:
            model = ANFIS(
                n_inputs=6,
                partition_type=ptype,
                mf_type="gaussian",
                learning_rate=0.02,
                l2_reg=1e-3,
                random_state=42,
                feature_names=POLLUTANT_ORDER,
                **kwargs,
            )
            model.fit(split["X_train"], split["y_train"], split["X_test"], split["y_test"])

            self.assertLessEqual(model.history_[-1]["train_rmse"], model.history_[0]["train_rmse"])

            yp = model.predict(split["X_test"])
            reg = compute_regression_metrics(split["y_test"], yp)
            cls = compute_classification_metrics(split["y_test"], yp)

            self.assertGreater(reg["R2"], 0.94)
            self.assertLess(reg["RMSE"], 15.0)
            self.assertLess(reg["RMSE"], lr_metrics["RMSE"])
            self.assertGreater(cls["accuracy"], 0.85)

    def test_forward_trace_exact_consistency(self) -> None:
        df = pd.read_csv(PROJECT_ROOT / "data" / "ispu_dki_all.csv")
        split = prepare_train_test_data(df, POLLUTANT_ORDER, "ispu", test_size=0.2, random_state=42)
        model = ANFIS(
            n_inputs=6,
            partition_type="scatter",
            n_rules=10,
            mf_type="gbell",
            epochs=10,
            feature_names=POLLUTANT_ORDER,
        ).fit(split["X_train"], split["y_train"])

        trace = model.forward_trace(split["X_test"][0], split["X_test_raw"][0])
        rule_df = trace["rule_table"]
        self.assertAlmostEqual(float(rule_df["w_bar (Layer 3)"].sum()), 1.0, places=9)
        self.assertAlmostEqual(float(rule_df["w_bar * f_k"].sum()), trace["y_pred"], places=9)
        self.assertAlmostEqual(trace["y_pred"], float(model.predict(split["X_test"][0])), places=9)

    def test_manual_example_matches_anfis_engine(self) -> None:
        ex = compute_worked_manual_example()
        self.assertAlmostEqual(ex["y_manual"], ex["y_engine"], places=9)
        self.assertAlmostEqual(float(np.sum(ex["w_bar"])), 1.0, places=9)


class TestStreamlitGUISmoke(unittest.TestCase):
    def test_streamlit_all_pages_smoke(self) -> None:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30)
        at.run()
        self.assertEqual(len(at.exception), 0)

        page_names = [
            "1. Beranda",
            "2. Input & Eksplorasi Data",
            "3. Pelatihan ANFIS 5-Layer",
            "4. Hasil & Trace Layer 1–5",
            "5. Evaluasi & Komparasi",
            "6. Informasi Sistem & Teori",
        ]
        for pname in page_names:
            at.sidebar.radio(key="active_page").set_value(pname).run()
            self.assertEqual(len(at.exception), 0, msg=f"Exception di halaman {pname}: {at.exception}")

        # Uji interaktif Halaman 2: validasi input negatif & prediksi valid
        at.sidebar.radio(key="active_page").set_value("2. Input & Eksplorasi Data").run()
        at.number_input(key="manual_field_pm10").set_value(-15.0).run()
        predict_btn = next(b for b in at.button if "Hitung Prediksi ISPU" in b.label)
        predict_btn.click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(len(at.error) > 0, msg="Input negatif seharusnya memunculkan pesan error.")

        at.number_input(key="manual_field_pm10").set_value(88.0).run()
        predict_btn = next(b for b in at.button if "Hitung Prediksi ISPU" in b.label)
        predict_btn.click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.error), 0)

        # Uji interaktif Halaman 4: Trace setelah prediksi manual
        at.sidebar.radio(key="active_page").set_value("4. Hasil & Trace Layer 1–5").run()
        at.radio[0].set_value("Input Manual Terakhir").run()
        self.assertEqual(len(at.exception), 0)


if __name__ == "__main__":
    unittest.main()
