from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_DIR = PROJECT_ROOT / "code" / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from anxiety_math_hater_ols_utils import (  # noqa: E402
    fit_math_hater_gender_ols,
    fit_simple_ols,
    load_ml_dataset,
    multi_regression_summary_frame,
    prepare_regression_frame,
    prepare_gender_regression_frame,
    regression_summary_frame,
)


class TestMlRegression(unittest.TestCase):
    def test_prepare_regression_frame_builds_expected_columns(self):
        df = pd.DataFrame(
            {
                "amas_score": [10, 20],
                "mseaq_anx": [5, 7],
                "math_hater_flg": [False, True],
            }
        )

        frame = prepare_regression_frame(df)

        self.assertEqual(list(frame.columns), ["amas_score", "mseaq_anx", "math_hater_flg", "anxiety_score"])
        self.assertEqual(frame["anxiety_score"].tolist(), [15.0, 27.0])
        self.assertEqual(frame["math_hater_flg"].tolist(), [0, 1])

    def test_fit_simple_ols_matches_closed_form_binary_regression(self):
        df = pd.DataFrame(
            {
                "amas_score": [0, 0, 0, 0],
                "mseaq_anx": [10, 12, 20, 22],
                "math_hater_flg": [0, 0, 1, 1],
            }
        )

        result = fit_simple_ols(df)

        self.assertTrue(np.isclose(result.intercept, 11.0))
        self.assertTrue(np.isclose(result.slope, 10.0))
        self.assertTrue(np.isclose(result.r2, 25.0 / 26.0))
        self.assertEqual(result.n_obs, 4)
        self.assertLess(result.slope_pvalue, 0.1)

    def test_real_dataset_regression_runs_and_returns_summary(self):
        df = load_ml_dataset(PROJECT_ROOT / "code" / "ml" / "ml_dataset.csv")
        frame = prepare_regression_frame(df)
        result = fit_simple_ols(frame)
        summary = regression_summary_frame(result)

        self.assertEqual(result.n_obs, len(df))
        self.assertEqual(summary.shape, (2, 7))
        self.assertTrue(np.isfinite(result.intercept))
        self.assertTrue(np.isfinite(result.slope))
        self.assertGreaterEqual(result.r2, 0.0)
        self.assertLessEqual(result.r2, 1.0)

    def test_prepare_gender_regression_frame_groups_gender(self):
        df = pd.DataFrame(
            {
                "amas_score": [1, 2, 3],
                "mseaq_anx": [4, 5, 6],
                "math_hater_flg": [0, 1, 0],
                "gender": ["woman", "man", "genderqueer"],
            }
        )

        frame = prepare_gender_regression_frame(df)

        self.assertEqual(frame["gender_group"].tolist(), ["woman", "man", "other"])
        self.assertEqual(frame["anxiety_score"].tolist(), [5.0, 7.0, 9.0])

    def test_fit_math_hater_gender_ols_recovers_known_coefficients(self):
        genders = ["woman", "woman", "man", "man", "other", "other", "woman", "other"]
        math = [0, 1, 0, 1, 0, 1, 0, 1]
        y = [10 + 2 * m + (5 if g == "man" else 0) + (-3 if g == "other" else 0) for g, m in zip(genders, math)]
        df = pd.DataFrame(
            {
                "amas_score": [0] * 8,
                "mseaq_anx": y,
                "math_hater_flg": math,
                "gender": genders,
            }
        )
        frame = prepare_gender_regression_frame(df)
        result = fit_math_hater_gender_ols(frame)

        self.assertTrue(np.isclose(result.coefficients[0], 10.0))
        self.assertTrue(np.isclose(result.coefficients[1], 2.0))
        self.assertTrue(np.isclose(result.coefficients[2], 5.0))
        self.assertTrue(np.isclose(result.coefficients[3], -3.0))
        self.assertEqual(result.n_obs, 8)

    def test_real_dataset_gender_model_runs_and_returns_summary(self):
        df = load_ml_dataset(PROJECT_ROOT / "code" / "ml" / "ml_dataset.csv")
        frame = prepare_gender_regression_frame(df)
        result = fit_math_hater_gender_ols(frame)
        summary = multi_regression_summary_frame(result)

        self.assertEqual(result.n_obs, len(df))
        self.assertEqual(summary.shape[0], 4)
        self.assertEqual(summary.shape[1], 7)
        self.assertTrue(np.isfinite(result.r2))
        self.assertTrue(np.isfinite(result.f_stat))


if __name__ == "__main__":
    unittest.main()
