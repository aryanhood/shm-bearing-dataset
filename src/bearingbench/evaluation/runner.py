"""Leak-free multi-seed dual-dataset benchmark, ablations, ONNX INT8 profiling, and replay."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, f1_score

from ..baselines.unsupervised import evaluate_zero_label_cwru, evaluate_zero_label_uottawa
from ..data.cwru import (
    CWRU_CLASSES,
    CWRU_PROTOCOLS,
    CWRUDataset,
    cwru_fold_splits,
    prepare_cwru_dataset,
)
from ..data.encoder import audit_sensors, decode_encoder
from ..data.signal_ops import (
    extract_multiband_cnn_input,
    multiband_order_envelopes,
    slice_order_windows,
)
from ..data.uottawa import (
    UOTTAWA_CLASSES,
    UOTTAWA_PROTOCOLS,
    UottawaDataset,
    load_uottawa_recording,
    prepare_uottawa_dataset,
    uottawa_fold_splits,
)
from ..edge.onnx_edge import EdgeClassifier, export_onnx, quantize_onnx_int8
from ..edge.supervisor import SupervisoryController
from ..models.classical import predict_proba_sklearn, train_mlp, train_random_forest
from ..models.cnn import (
    NESTED_DIMS,
    MatryoshkaOrderCNN,
    SlicedMatryoshkaCNN,
    TrainConfig,
    predict_logits_torch,
    predict_nested_logits_torch,
    train_matryoshka_cnn,
    train_order_cnn,
)
from ..uncertainty.calibration import (
    TemperatureScaler,
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
    reliability_bins,
    selective_risk_coverage,
    softmax,
)
from ..uncertainty.conformal import (
    SplitAPSConformal,
    run_adaptive_conformal_inference,
)
from .stats import mcnemar_exact, mean_std, paired_bootstrap_acc_diff

SEEDS: tuple[int, ...] = (7, 19, 42, 73, 101)
UOTTAWA_EPOCHS = 35
UOTTAWA_BATCH_SIZE = 64
CWRU_EPOCHS = 24
CWRU_BATCH_SIZE = 128
CAL_EPOCHS = 20


def _eval_probs_metrics(probs: NDArray[np.float64], y: NDArray[np.int64]) -> dict[str, float]:
    pred = probs.argmax(axis=1)
    rc = selective_risk_coverage(probs, y)
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "ece": round(expected_calibration_error(probs, y), 4),
        "brier": round(brier_score(probs, y), 4),
        "nll": round(negative_log_likelihood(probs, y), 4),
        "aurc": rc.aurc,
        "sel_acc_80": rc.acc_at_80_cov,
        "sel_acc_90": rc.acc_at_90_cov,
    }


def _aggregate_seed_dicts(rows: list[dict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in rows[0]:
        st = mean_std([r[key] for r in rows])
        out[f"{key}_mean"] = st.mean
        out[f"{key}_std"] = st.std
    return out


def _per_class_recall(
    y_true: NDArray[np.int64], y_pred: NDArray[np.int64], classes: tuple[str, ...]
) -> dict[str, float]:
    out: dict[str, float] = {}
    for c_idx, c_name in enumerate(classes):
        mask = y_true == c_idx
        out[c_name] = round(float(np.mean(y_pred[mask] == c_idx)), 4) if np.any(mask) else 0.0
    return out


def _internal_oof_mrl_calibration(
    x: NDArray[np.float32],
    y: NDArray[np.int64],
    tr_mask: NDArray[np.bool_],
    group_ids: NDArray[np.int64] | NDArray[np.str_],
    *,
    n_classes: int,
    seed: int,
    batch_size: int = 64,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Compute strictly leak-free 2-fold recording-grouped OOF logits inside tr_mask."""
    tr_idx = np.flatnonzero(tr_mask)
    tr_groups = group_ids[tr_idx]
    uniq = sorted({g.item() if hasattr(g, "item") else str(g) for g in tr_groups})
    if len(uniq) < 2:
        half = np.zeros(len(tr_idx), dtype=int)
        for cls in np.unique(y[tr_idx]):
            cls_pos = np.flatnonzero(y[tr_idx] == cls)
            half[cls_pos[1::2]] = 1
        sub_groups = half
    else:
        even_groups = set(uniq[::2])
        raw_vals = [g.item() if hasattr(g, "item") else str(g) for g in tr_groups]
        sub_groups = np.array([0 if v in even_groups else 1 for v in raw_vals], dtype=int)

    oof_logits = np.zeros((len(tr_idx), n_classes), dtype=np.float64)
    for half_id in (0, 1):
        cal_local = sub_groups == half_id
        fit_local = ~cal_local
        m = train_matryoshka_cnn(
            x[tr_idx[fit_local]],
            y[tr_idx[fit_local]],
            n_classes=n_classes,
            cfg=TrainConfig(epochs=CAL_EPOCHS, batch_size=batch_size, seed=seed),
        )
        oof_logits[cal_local] = predict_logits_torch(m, x[tr_idx[cal_local]])
    return oof_logits, y[tr_idx]


def _evaluate_uncertainty_and_conformal(
    x_all: NDArray[np.float32],
    y_all: NDArray[np.int64],
    group_ids: NDArray[np.int64] | NDArray[np.str_],
    splits: list[tuple[str, NDArray[np.bool_], NDArray[np.bool_]]],
    *,
    n_classes: int,
    batch_size: int,
    seed: int,
    fold_models: dict[str, MatryoshkaOrderCNN],
    chronological_sort_keys: NDArray[np.str_] | None = None,
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, object],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    list[int],
]:
    """Evaluate raw, temperature-scaled, Split APS, and Online ACI on a protocol for one seed."""
    raw_logits_list: list[NDArray[np.float64]] = []
    ts_probs_list: list[NDArray[np.float64]] = []
    y_test_list: list[NDArray[np.int64]] = []
    aps_sets_all: list[tuple[int, ...]] = []
    aci_sets_all: list[tuple[int, ...]] = []
    aci_covered_all: list[int] = []
    ts_temps: list[float] = []

    for fold_name, tr_m, te_m in splits:
        model = fold_models[fold_name]
        te_logits = predict_logits_torch(model, x_all[te_m])
        te_y = y_all[te_m]

        # Strictly recording-grouped internal OOF calibration inside tr_m
        cal_logits, cal_y = _internal_oof_mrl_calibration(
            x_all,
            y_all,
            tr_m,
            group_ids,
            n_classes=n_classes,
            seed=seed + 100,
            batch_size=batch_size,
        )
        scaler = TemperatureScaler.fit(cal_logits, cal_y)
        cal_probs_ts = scaler.predict_proba(cal_logits)
        te_probs_ts = scaler.predict_proba(te_logits)
        ts_temps.append(round(scaler.temperature, 4))

        # Order test samples chronologically if stems are available
        te_idx = np.flatnonzero(te_m)
        if chronological_sort_keys is not None:
            order = np.argsort(chronological_sort_keys[te_idx], kind="mergesort")
        else:
            order = np.arange(len(te_idx))

        aps = SplitAPSConformal.fit(cal_probs_ts, cal_y, alpha=0.10, seed=seed)
        aps_sets = aps.predict_sets(te_probs_ts, seed=seed + 1)
        aci_res = run_adaptive_conformal_inference(
            aps.cal_scores,
            te_probs_ts[order],
            te_y[order],
            alpha_target=0.10,
            gamma=0.05,
            rolling_window=40,
            seed=seed + 2,
        )

        raw_logits_list.append(te_logits)
        ts_probs_list.append(te_probs_ts)
        y_test_list.append(te_y)
        aps_sets_all.extend(aps_sets)
        aci_sets_all.extend(aci_res.sets)
        aci_covered_all.extend(aci_res.covered_stream)

    raw_logits = np.concatenate(raw_logits_list, axis=0)
    raw_probs = softmax(raw_logits)
    ts_probs = np.concatenate(ts_probs_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    from ..uncertainty.conformal import evaluate_prediction_sets

    aps_m = evaluate_prediction_sets(aps_sets_all, y_test, target_coverage=0.90)
    aci_m = evaluate_prediction_sets(
        aci_sets_all,
        np.concatenate(
            [
                y_all[te_m][
                    np.argsort(chronological_sort_keys[np.flatnonzero(te_m)], kind="mergesort")
                    if chronological_sort_keys is not None
                    else np.arange(int(te_m.sum()))
                ]
                for _, _, te_m in splits
            ]
        ),
        target_coverage=0.90,
    )

    cal_summary = {
        "raw_single": _eval_probs_metrics(raw_probs, y_test),
        "temp_scaled_single": _eval_probs_metrics(ts_probs, y_test),
    }
    conformal_summary: dict[str, object] = {
        "mean_temperature": round(float(np.mean(ts_temps)), 4),
        "split_aps": {
            "coverage": aps_m.coverage,
            "avg_set_size": aps_m.avg_set_size,
            "singleton_rate": aps_m.singleton_rate,
            "singleton_accuracy": aps_m.singleton_accuracy,
        },
        "online_aci": {
            "coverage": aci_m.coverage,
            "avg_set_size": aci_m.avg_set_size,
            "singleton_rate": aci_m.singleton_rate,
            "singleton_accuracy": aci_m.singleton_accuracy,
        },
    }
    return cal_summary, conformal_summary, raw_probs, ts_probs, y_test, aci_covered_all


def run_cwru_suite(cw: CWRUDataset) -> dict[str, object]:
    """Run 5-seed multi-protocol CWRU benchmark."""
    n_classes = len(CWRU_CLASSES)
    protocols_out: dict[str, object] = {}
    per_class_out: dict[str, object] = {}
    diam_breakdown: dict[str, dict[str, list[float]]] = {}
    mrl_nested_out: dict[str, object] = {}
    uncertainty_out: dict[str, object] = {}

    for proto in CWRU_PROTOCOLS:
        print(f"  [CWRU] protocol={proto}", flush=True)
        splits = cwru_fold_splits(cw, proto)
        model_seed_rows: dict[str, list[dict[str, float]]] = {
            "gain_sensitive_mlp": [],
            "gain_invariant_rf": [],
            "cwru_env_cnn": [],
            "cwru_mrl_cnn": [],
        }
        recall_acc: dict[str, list[dict[str, float]]] = {m: [] for m in model_seed_rows}
        nested_rows: dict[int, list[dict[str, float]]] = {d: [] for d in NESTED_DIMS}
        raw_probs_by_seed: list[NDArray[np.float64]] = []
        ts_cal_seed_rows: list[dict[str, float]] = []
        raw_cal_seed_rows: list[dict[str, float]] = []
        aps_cov_list: list[float] = []
        aps_sz_list: list[float] = []
        aci_cov_list: list[float] = []
        aci_sz_list: list[float] = []
        y_ref: NDArray[np.int64] = np.empty(0, dtype=np.int64)

        for s_idx, seed in enumerate(SEEDS):
            y_all_map: dict[str, list[int]] = {m: [] for m in model_seed_rows}
            p_all_map: dict[str, list[NDArray[np.float64]]] = {m: [] for m in model_seed_rows}
            nested_logits_map: dict[int, list[NDArray[np.float64]]] = {d: [] for d in NESTED_DIMS}
            fold_mrl_models: dict[str, MatryoshkaOrderCNN] = {}

            for fold_name, tr_m, te_m in splits:
                y_tr, y_te = cw.y[tr_m], cw.y[te_m]

                mlp = train_mlp(cw.gain16_x[tr_m], y_tr, seed=seed)
                p_mlp = predict_proba_sklearn(mlp, cw.gain16_x[te_m], n_classes=n_classes)

                rf = train_random_forest(cw.phys27_x[tr_m], y_tr, seed=seed)
                p_rf = predict_proba_sklearn(rf, cw.phys27_x[te_m], n_classes=n_classes)

                cnn = train_order_cnn(
                    cw.cnn_x[tr_m],
                    y_tr,
                    n_classes=n_classes,
                    cfg=TrainConfig(epochs=CWRU_EPOCHS, batch_size=CWRU_BATCH_SIZE, seed=seed),
                )
                p_cnn = softmax(predict_logits_torch(cnn, cw.cnn_x[te_m]))

                mrl = train_matryoshka_cnn(
                    cw.cnn_x[tr_m],
                    y_tr,
                    n_classes=n_classes,
                    cfg=TrainConfig(epochs=CWRU_EPOCHS, batch_size=CWRU_BATCH_SIZE, seed=seed),
                )
                fold_mrl_models[fold_name] = mrl
                nested_lg = predict_nested_logits_torch(mrl, cw.cnn_x[te_m])
                p_mrl = softmax(nested_lg[128])

                for name, probs in (
                    ("gain_sensitive_mlp", p_mlp),
                    ("gain_invariant_rf", p_rf),
                    ("cwru_env_cnn", p_cnn),
                    ("cwru_mrl_cnn", p_mrl),
                ):
                    y_all_map[name].extend(y_te.tolist())
                    p_all_map[name].append(probs)
                    if proto == "unseen_fault_size":
                        diam_breakdown.setdefault(name, {}).setdefault(fold_name, []).append(
                            float(accuracy_score(y_te, probs.argmax(axis=1)))
                        )

                for d in NESTED_DIMS:
                    nested_logits_map[d].append(nested_lg[d])

            for name in model_seed_rows:
                yt = np.asarray(y_all_map[name], dtype=np.int64)
                pr = np.concatenate(p_all_map[name], axis=0)
                model_seed_rows[name].append(_eval_probs_metrics(pr, yt))
                recall_acc[name].append(_per_class_recall(yt, pr.argmax(axis=1), CWRU_CLASSES))
                if name == "cwru_mrl_cnn":
                    y_ref = yt
                    raw_probs_by_seed.append(pr)

            for d in NESTED_DIMS:
                pr_d = softmax(np.concatenate(nested_logits_map[d], axis=0))
                nested_rows[d].append(_eval_probs_metrics(pr_d, y_ref))

            # Internal leak-free calibration + Split APS + ACI across first 3 seeds for speed
            if s_idx < 3:
                cal_s, conf_s, _, _, _, _ = _evaluate_uncertainty_and_conformal(
                    cw.cnn_x,
                    cw.y,
                    cw.load,
                    splits,
                    n_classes=n_classes,
                    batch_size=CWRU_BATCH_SIZE,
                    seed=seed,
                    fold_models=fold_mrl_models,
                )
                raw_cal_seed_rows.append(cal_s["raw_single"])
                ts_cal_seed_rows.append(cal_s["temp_scaled_single"])
                sp = cast(dict[str, float], conf_s["split_aps"])
                ac = cast(dict[str, float], conf_s["online_aci"])
                aps_cov_list.append(sp["coverage"])
                aps_sz_list.append(sp["avg_set_size"])
                aci_cov_list.append(ac["coverage"])
                aci_sz_list.append(ac["avg_set_size"])

        protocols_out[proto] = {k: _aggregate_seed_dicts(v) for k, v in model_seed_rows.items()}
        per_class_out[proto] = {
            name: {c: round(float(np.mean([r[c] for r in rows])), 4) for c in CWRU_CLASSES}
            for name, rows in recall_acc.items()
        }
        mrl_nested_out[proto] = {str(d): _aggregate_seed_dicts(nested_rows[d]) for d in NESTED_DIMS}

        ens_probs = np.mean(np.stack(raw_probs_by_seed, axis=0), axis=0)
        uncertainty_out[proto] = {
            "raw_single": _aggregate_seed_dicts(raw_cal_seed_rows),
            "temp_scaled_single": _aggregate_seed_dicts(ts_cal_seed_rows),
            "ensemble_5seed": _eval_probs_metrics(ens_probs, y_ref),
            "split_aps": {
                "coverage_mean": mean_std(aps_cov_list).mean,
                "coverage_std": mean_std(aps_cov_list).std,
                "set_size_mean": mean_std(aps_sz_list).mean,
            },
            "online_aci": {
                "coverage_mean": mean_std(aci_cov_list).mean,
                "coverage_std": mean_std(aci_cov_list).std,
                "set_size_mean": mean_std(aci_sz_list).mean,
            },
        }

    diam_summary = {
        m: {
            f: {"acc_mean": mean_std(vals).mean, "acc_std": mean_std(vals).std}
            for f, vals in folds.items()
        }
        for m, folds in diam_breakdown.items()
    }
    return {
        "protocols": protocols_out,
        "per_class_recall": per_class_out,
        "unseen_fault_size_per_diameter": diam_summary,
        "mrl_nested": mrl_nested_out,
        "uncertainty_shift": uncertainty_out,
    }


def run_uottawa_suite(uo: UottawaDataset, raw_dir: Path, artifacts_dir: Path) -> dict[str, object]:
    """Run complete 5-seed uOttawa benchmark, ablations, ONNX INT8 Pareto, uncertainty & replay."""
    n_classes = len(UOTTAWA_CLASSES)
    protocols_out: dict[str, object] = {}
    per_class_out: dict[str, object] = {}
    uncertainty_out: dict[str, object] = {}
    mrl_nested_out: dict[str, object] = {}

    pooled_true: dict[str, list[int]] = {}
    pooled_preds: dict[str, dict[str, list[int]]] = {}
    reliability_plot_data: dict[str, object] = {}
    aci_plot_data: dict[str, object] = {}
    clean_models: dict[tuple[int, str], MatryoshkaOrderCNN] = {}

    for proto in UOTTAWA_PROTOCOLS:
        print(f"  [uOttawa] protocol={proto}", flush=True)
        splits = uottawa_fold_splits(uo, proto)
        model_seed_rows: dict[str, list[dict[str, float]]] = {
            "time_env_rf": [],
            "time_env_cnn": [],
            "order_physics_rf": [],
            "order_cnn": [],
            "order_mrl_cnn": [],
        }
        recall_acc: dict[str, list[dict[str, float]]] = {m: [] for m in model_seed_rows}
        nested_rows: dict[int, list[dict[str, float]]] = {d: [] for d in NESTED_DIMS}
        raw_probs_by_seed: list[NDArray[np.float64]] = []
        raw_cal_seed_rows: list[dict[str, float]] = []
        ts_cal_seed_rows: list[dict[str, float]] = []
        aps_cov_list: list[float] = []
        aps_sz_list: list[float] = []
        aps_sing_acc_list: list[float] = []
        aci_cov_list: list[float] = []
        aci_sz_list: list[float] = []
        aci_sing_acc_list: list[float] = []
        y_ref: NDArray[np.int64] = np.empty(0, dtype=np.int64)

        pooled_true[proto] = []
        pooled_preds[proto] = {m: [] for m in model_seed_rows}

        for s_idx, seed in enumerate(SEEDS):
            y_all_map: dict[str, list[int]] = {m: [] for m in model_seed_rows}
            p_all_map: dict[str, list[NDArray[np.float64]]] = {m: [] for m in model_seed_rows}
            nested_logits_map: dict[int, list[NDArray[np.float64]]] = {d: [] for d in NESTED_DIMS}
            fold_mrl_models: dict[str, MatryoshkaOrderCNN] = {}

            for fold_name, tr_m, te_m in splits:
                y_tr, y_te = uo.y[tr_m], uo.y[te_m]

                rf_t = train_random_forest(uo.time_x[tr_m], y_tr, seed=seed)
                p_rf_t = predict_proba_sklearn(rf_t, uo.time_x[te_m], n_classes=n_classes)

                cnn_t = train_order_cnn(
                    uo.time_cnn_x[tr_m],
                    y_tr,
                    n_classes=n_classes,
                    cfg=TrainConfig(epochs=UOTTAWA_EPOCHS, seed=seed),
                )
                p_cnn_t = softmax(predict_logits_torch(cnn_t, uo.time_cnn_x[te_m]))

                rf_o = train_random_forest(uo.order_x[tr_m], y_tr, seed=seed)
                p_rf_o = predict_proba_sklearn(rf_o, uo.order_x[te_m], n_classes=n_classes)

                cnn_o = train_order_cnn(
                    uo.cnn_x[tr_m],
                    y_tr,
                    n_classes=n_classes,
                    cfg=TrainConfig(epochs=UOTTAWA_EPOCHS, seed=seed),
                )
                p_cnn_o = softmax(predict_logits_torch(cnn_o, uo.cnn_x[te_m]))

                mrl_o = train_matryoshka_cnn(
                    uo.cnn_x[tr_m],
                    y_tr,
                    n_classes=n_classes,
                    cfg=TrainConfig(epochs=UOTTAWA_EPOCHS, seed=seed),
                )
                fold_mrl_models[fold_name] = mrl_o
                if proto == "unseen_speed_profile":
                    clean_models[(seed, fold_name)] = mrl_o
                nested_lg = predict_nested_logits_torch(mrl_o, uo.cnn_x[te_m])
                p_mrl_o = softmax(nested_lg[128])

                for name, probs in (
                    ("time_env_rf", p_rf_t),
                    ("time_env_cnn", p_cnn_t),
                    ("order_physics_rf", p_rf_o),
                    ("order_cnn", p_cnn_o),
                    ("order_mrl_cnn", p_mrl_o),
                ):
                    y_all_map[name].extend(y_te.tolist())
                    p_all_map[name].append(probs)

                for d in NESTED_DIMS:
                    nested_logits_map[d].append(nested_lg[d])

            for name in model_seed_rows:
                yt = np.asarray(y_all_map[name], dtype=np.int64)
                pr = np.concatenate(p_all_map[name], axis=0)
                pred_cls = pr.argmax(axis=1)
                model_seed_rows[name].append(_eval_probs_metrics(pr, yt))
                recall_acc[name].append(_per_class_recall(yt, pred_cls, UOTTAWA_CLASSES))
                pooled_preds[proto][name].extend(pred_cls.tolist())
                if name == "order_mrl_cnn":
                    y_ref = yt
                    raw_probs_by_seed.append(pr)
                    pooled_true[proto].extend(yt.tolist())

            for d in NESTED_DIMS:
                pr_d = softmax(np.concatenate(nested_logits_map[d], axis=0))
                nested_rows[d].append(_eval_probs_metrics(pr_d, y_ref))

            # Calibration + Conformal across all 5 seeds on OOD protocols (3 seeds on random_window)
            if proto != "random_window" or s_idx < 3:
                cal_s, conf_s, raw_p, ts_p, y_cal_te, aci_cov_stream = (
                    _evaluate_uncertainty_and_conformal(
                        uo.cnn_x,
                        uo.y,
                        uo.trial if proto == "unseen_speed_profile" else uo.profile_code,
                        splits,
                        n_classes=n_classes,
                        batch_size=UOTTAWA_BATCH_SIZE,
                        seed=seed,
                        fold_models=fold_mrl_models,
                        chronological_sort_keys=uo.window_stem,
                    )
                )
                raw_cal_seed_rows.append(cal_s["raw_single"])
                ts_cal_seed_rows.append(cal_s["temp_scaled_single"])
                sp = cast(dict[str, float], conf_s["split_aps"])
                ac = cast(dict[str, float], conf_s["online_aci"])
                aps_cov_list.append(sp["coverage"])
                aps_sz_list.append(sp["avg_set_size"])
                aps_sing_acc_list.append(sp["singleton_accuracy"])
                aci_cov_list.append(ac["coverage"])
                aci_sz_list.append(ac["avg_set_size"])
                aci_sing_acc_list.append(ac["singleton_accuracy"])

                if seed == 42 and proto in {"random_window", "unseen_speed_profile"}:
                    reliability_plot_data[proto] = {
                        "raw": reliability_bins(raw_p, y_cal_te),
                        "temp_scaled": reliability_bins(ts_p, y_cal_te),
                    }
                    if proto == "unseen_speed_profile":
                        # Rolling 40-step coverage comparison (Split APS vs ACI)
                        rc_curve = selective_risk_coverage(ts_p, y_cal_te)
                        roll_aci = [
                            round(float(np.mean(aci_cov_stream[max(0, i - 39) : i + 1])), 4)
                            for i in range(len(aci_cov_stream))
                        ]
                        aci_plot_data = {
                            "rolling_aci_coverage": roll_aci,
                            "split_aps_cov": sp["coverage"],
                            "aci_cov": ac["coverage"],
                            "risk_coverage_cov": rc_curve.coverages,
                            "risk_coverage_risk": rc_curve.risks,
                        }

        protocols_out[proto] = {k: _aggregate_seed_dicts(v) for k, v in model_seed_rows.items()}
        per_class_out[proto] = {
            name: {c: round(float(np.mean([r[c] for r in rows])), 4) for c in UOTTAWA_CLASSES}
            for name, rows in recall_acc.items()
        }
        mrl_nested_out[proto] = {str(d): _aggregate_seed_dicts(nested_rows[d]) for d in NESTED_DIMS}

        ens_probs = np.mean(np.stack(raw_probs_by_seed, axis=0), axis=0)
        if proto in {"random_window", "unseen_speed_profile"}:
            reliability_plot_data[f"{proto}_ensemble"] = reliability_bins(ens_probs, y_ref)

        uncertainty_out[proto] = {
            "raw_single": _aggregate_seed_dicts(raw_cal_seed_rows),
            "temp_scaled_single": _aggregate_seed_dicts(ts_cal_seed_rows),
            "ensemble_5seed": _eval_probs_metrics(ens_probs, y_ref),
            "split_aps": {
                "coverage_mean": mean_std(aps_cov_list).mean,
                "coverage_std": mean_std(aps_cov_list).std,
                "set_size_mean": mean_std(aps_sz_list).mean,
                "singleton_acc_mean": mean_std(aps_sing_acc_list).mean,
            },
            "online_aci": {
                "coverage_mean": mean_std(aci_cov_list).mean,
                "coverage_std": mean_std(aci_cov_list).std,
                "set_size_mean": mean_std(aci_sz_list).mean,
                "singleton_acc_mean": mean_std(aci_sing_acc_list).mean,
            },
        }

    # Factorial Ablations on unseen_speed_profile (5 seeds, strictly 1 factor varied)
    print("  [uOttawa] running 5-seed factorial ablations on unseen_speed_profile", flush=True)
    sp_splits = uottawa_fold_splits(uo, "unseen_speed_profile")
    ablation_variants: dict[str, tuple[NDArray[np.float32], NDArray[np.float32]]] = {
        "envelope_first_4band": (uo.cnn_x, uo.cnn_x),
        "order_first_4band": (uo.order_first_x, uo.order_first_x),
        "cpw_enabled_4band": (uo.cpw_cnn_x, uo.cpw_cnn_x),
        "single_band_0_5_5khz": (uo.cnn_x[:, 0:1, :], uo.cnn_x[:, 0:1, :]),
        "single_band_5_15khz": (uo.cnn_x[:, 1:2, :], uo.cnn_x[:, 1:2, :]),
        "single_band_15_35khz": (uo.cnn_x[:, 2:3, :], uo.cnn_x[:, 2:3, :]),
        "tach_clean_1024ppr": (uo.cnn_x, uo.cnn_x),
        "tach_256ppr": (uo.cnn_x, uo.tach_256_x),
        "tach_64ppr": (uo.cnn_x, uo.tach_64_x),
        "tach_32ppr": (uo.cnn_x, uo.tach_32_x),
        "tach_1024ppr_2pct_jitter": (uo.cnn_x, uo.tach_jitter_x),
    }

    ablation_out: dict[str, object] = {}
    pooled_ablation_preds: dict[str, list[int]] = {
        k: [] for k in ("envelope_first_4band", "order_first_4band")
    }
    pooled_ablation_true: list[int] = []

    for var_name, (x_tr_src, x_te_src) in ablation_variants.items():
        var_rows: list[dict[str, float]] = []
        ball_recalls: list[float] = []
        for seed in SEEDS:
            yt_list: list[int] = []
            pr_list: list[NDArray[np.float64]] = []
            for fold_name, tr_m, te_m in sp_splits:
                if x_tr_src is uo.cnn_x:
                    m = clean_models[(seed, fold_name)]
                else:
                    m = train_matryoshka_cnn(
                        x_tr_src[tr_m],
                        uo.y[tr_m],
                        n_classes=n_classes,
                        cfg=TrainConfig(epochs=UOTTAWA_EPOCHS, seed=seed),
                    )
                probs = softmax(predict_logits_torch(m, x_te_src[te_m]))
                yt_list.extend(uo.y[te_m].tolist())
                pr_list.append(probs)
            yt_arr = np.asarray(yt_list, dtype=np.int64)
            pr_arr = np.concatenate(pr_list, axis=0)
            pred_arr = pr_arr.argmax(axis=1)
            var_rows.append(_eval_probs_metrics(pr_arr, yt_arr))
            ball_recalls.append(_per_class_recall(yt_arr, pred_arr, UOTTAWA_CLASSES)["ball"])
            if var_name in pooled_ablation_preds:
                pooled_ablation_preds[var_name].extend(pred_arr.tolist())
                if var_name == "envelope_first_4band":
                    pooled_ablation_true.extend(yt_arr.tolist())

        agg = _aggregate_seed_dicts(var_rows)
        agg["ball_recall_mean"] = mean_std(ball_recalls).mean
        agg["ball_recall_std"] = mean_std(ball_recalls).std
        ablation_out[var_name] = agg

    # Paired bootstrap & exact McNemar statistical comparisons
    y_pool_sp = np.asarray(pooled_true["unseen_speed_profile"], dtype=np.int64)
    p_order_cnn = np.asarray(pooled_preds["unseen_speed_profile"]["order_cnn"], dtype=np.int64)
    p_time_cnn = np.asarray(pooled_preds["unseen_speed_profile"]["time_env_cnn"], dtype=np.int64)
    p_mrl_cnn = np.asarray(pooled_preds["unseen_speed_profile"]["order_mrl_cnn"], dtype=np.int64)
    p_order_rf = np.asarray(
        pooled_preds["unseen_speed_profile"]["order_physics_rf"], dtype=np.int64
    )
    y_pool_abl = np.asarray(pooled_ablation_true, dtype=np.int64)
    p_env_first = np.asarray(pooled_ablation_preds["envelope_first_4band"], dtype=np.int64)
    p_ord_first = np.asarray(pooled_ablation_preds["order_first_4band"], dtype=np.int64)

    paired_comparisons = {
        "order_cnn_vs_time_env_cnn": {
            "bootstrap": paired_bootstrap_acc_diff(y_pool_sp, p_order_cnn, p_time_cnn),
            "mcnemar": mcnemar_exact(y_pool_sp, p_order_cnn, p_time_cnn),
        },
        "order_mrl_cnn_vs_order_physics_rf": {
            "bootstrap": paired_bootstrap_acc_diff(y_pool_sp, p_mrl_cnn, p_order_rf),
            "mcnemar": mcnemar_exact(y_pool_sp, p_mrl_cnn, p_order_rf),
        },
        "envelope_first_vs_order_first": {
            "bootstrap": paired_bootstrap_acc_diff(y_pool_abl, p_env_first, p_ord_first),
            "mcnemar": mcnemar_exact(y_pool_abl, p_env_first, p_ord_first),
        },
    }

    # ONNX FP32 vs Static QDQ INT8 Edge Pareto across Matryoshka widths d in {8..128}
    print("  [uOttawa] profiling ONNX FP32 vs INT8 across Matryoshka widths", flush=True)
    pareto_out: dict[str, object] = {}
    for d in NESTED_DIMS:
        fp32_rows: list[dict[str, float]] = []
        int8_rows: list[dict[str, float]] = []
        max_pt_diff = 0.0
        fp32_lat: dict[str, float] = {}
        int8_lat: dict[str, float] = {}

        for seed in SEEDS:
            yt_list_d: list[int] = []
            fp32_probs_list: list[NDArray[np.float64]] = []
            int8_probs_list: list[NDArray[np.float64]] = []

            for fold_name, tr_m, te_m in sp_splits:
                parent = clean_models[(seed, fold_name)]
                sliced = SlicedMatryoshkaCNN(parent, d)
                with tempfile.TemporaryDirectory() as td:
                    p_fp32 = Path(td) / f"d{d}_fp32.onnx"
                    p_int8 = Path(td) / f"d{d}_int8.onnx"
                    export_onnx(sliced, p_fp32)
                    quantize_onnx_int8(p_fp32, p_int8, uo.cnn_x[tr_m])
                    e32 = EdgeClassifier(p_fp32)
                    e8 = EdgeClassifier(p_int8)
                    pt_lg = predict_logits_torch(sliced, uo.cnn_x[te_m])
                    ort32_lg = e32.predict_logits(uo.cnn_x[te_m])
                    max_pt_diff = max(max_pt_diff, float(np.max(np.abs(pt_lg - ort32_lg))))
                    fp32_probs_list.append(softmax(ort32_lg))
                    int8_probs_list.append(e8.predict_proba(uo.cnn_x[te_m]))
                    yt_list_d.extend(uo.y[te_m].tolist())
                    if seed == 42 and fold_name == "train_AB_test_CD":
                        fp32_lat = e32.benchmark_latency(n_runs=500, warmup=50)
                        int8_lat = e8.benchmark_latency(n_runs=500, warmup=50)
                        if d == 128:
                            # Persist canonical ONNX artifacts for CI edge-runtime job
                            export_onnx(
                                sliced,
                                artifacts_dir / "order_mrl_fp32.onnx",
                                artifacts_dir / "order_mrl_meta.json",
                            )
                            quantize_onnx_int8(
                                artifacts_dir / "order_mrl_fp32.onnx",
                                artifacts_dir / "order_mrl_int8.onnx",
                                uo.cnn_x[tr_m],
                            )

            yt_d = np.asarray(yt_list_d, dtype=np.int64)
            fp32_rows.append(_eval_probs_metrics(np.concatenate(fp32_probs_list, axis=0), yt_d))
            int8_rows.append(_eval_probs_metrics(np.concatenate(int8_probs_list, axis=0), yt_d))

        pareto_out[str(d)] = {
            "fp32": {**_aggregate_seed_dicts(fp32_rows), **fp32_lat},
            "int8": {**_aggregate_seed_dicts(int8_rows), **int8_lat},
            "max_torch_onnx_diff": round(max_pt_diff, 8),
        }

    # Closed-loop supervisory replay on real Trial-3 (H-C-3 healthy -> C-C-3 combined fault)
    replay_trace = _run_closed_loop_replay(
        raw_dir=raw_dir,
        onnx_path=artifacts_dir / "order_mrl_int8.onnx",
        meta_path=artifacts_dir / "order_mrl_meta.json",
        uo=uo,
        artifacts_dir=artifacts_dir,
    )

    return {
        "protocols": protocols_out,
        "per_class_recall": per_class_out,
        "mrl_nested": mrl_nested_out,
        "paired_comparisons": paired_comparisons,
        "ablations_unseen_speed_profile": ablation_out,
        "mrl_onnx_pareto": pareto_out,
        "uncertainty_and_conformal": uncertainty_out,
        "plot_data": {
            "reliability": reliability_plot_data,
            "aci": aci_plot_data,
        },
        "closed_loop_replay_summary": replay_trace["summary"],
    }


def _run_closed_loop_replay(
    *,
    raw_dir: Path,
    onnx_path: Path,
    meta_path: Path,
    uo: UottawaDataset,
    artifacts_dir: Path,
) -> dict[str, object]:
    """Replay real Trial-3 reversing speed profile (H-C-3 -> C-C-3) through INT8 + Supervisor."""
    edge = EdgeClassifier(onnx_path, meta_path)
    # Fit calibration on train_AB trials using internal OOF, then deploy on Trial 3 C-profile
    tr_ab = np.isin(uo.profile_code, ["A", "B"])
    cal_lg, cal_y = _internal_oof_mrl_calibration(
        uo.cnn_x, uo.y, tr_ab, uo.trial, n_classes=len(UOTTAWA_CLASSES), seed=142
    )
    scaler = TemperatureScaler.fit(cal_lg, cal_y)
    aps = SplitAPSConformal.fit(scaler.predict_proba(cal_lg), cal_y, alpha=0.10, seed=42)

    ctrl = SupervisoryController(derate_prob=0.55, stop_prob=0.85, severe_classes=("combined",))
    timeline: list[dict[str, object]] = []
    t_offset = 0.0
    step_idx = 0

    for stem in ("H-C-3", "C-C-3"):
        rec = load_uottawa_recording(raw_dir, stem)
        track = decode_encoder(rec.encoder, rec.fs_hz)
        audit = audit_sensors(rec.vibration, track)
        _, order_sigs = multiband_order_envelopes(rec.vibration, rec.fs_hz, track)
        band_wins: list[NDArray[np.float64]] = []
        times_arr = np.empty(0, dtype=np.float64)
        rpms_arr = np.empty(0, dtype=np.float64)
        for osig in order_sigs:
            w, ts, rs = slice_order_windows(osig)
            band_wins.append(w)
            times_arr = ts
            rpms_arr = rs

        cnn_inputs = np.stack(
            [
                extract_multiband_cnn_input([bw[k] for bw in band_wins])
                for k in range(len(band_wins))
            ]
        )
        logits = edge.predict_logits(cnn_inputs)
        probs = scaler.predict_proba(logits)
        sets = aps.predict_sets(probs, seed=100 + step_idx)

        for k in range(len(cnn_inputs)):
            cmd = ctrl.update(
                probs[k].tolist(),
                audit,
                time_s=float(times_arr[k]) + t_offset,
                rpm=float(rpms_arr[k]),
                conformal_set=sets[k],
            )
            timeline.append(
                {
                    "step": cmd.step,
                    "stem": stem,
                    "true_class": rec.spec.label,
                    "time_s": cmd.time_s,
                    "rpm": cmd.rpm,
                    "mode": cmd.mode,
                    "predicted_class": cmd.predicted_class,
                    "fault_prob": cmd.fault_prob,
                    "conformal_set_size": cmd.conformal_set_size,
                    "velocity_scale": cmd.velocity_scale,
                    "torque_limit": cmd.torque_limit,
                    "brake_engage": cmd.brake_engage,
                    "reason": cmd.reason,
                }
            )
            step_idx += 1
        t_offset += float(len(rec.vibration)) / rec.fs_hz

    derate_steps = [row for row in timeline if row["mode"] == "DERATE"]
    stop_steps = [row for row in timeline if row["mode"] == "SAFE_STOP"]
    summary = {
        "stems": ["H-C-3", "C-C-3"],
        "n_windows": len(timeline),
        "false_derate_during_healthy": sum(
            1 for row in timeline if row["stem"] == "H-C-3" and row["mode"] != "NOMINAL"
        ),
        "first_derate_time_s": derate_steps[0]["time_s"] if derate_steps else None,
        "first_safe_stop_time_s": stop_steps[0]["time_s"] if stop_steps else None,
        "final_mode": timeline[-1]["mode"] if timeline else "NOMINAL",
    }
    payload: dict[str, object] = {"summary": summary, "timeline": timeline}
    (artifacts_dir / "replay_trace.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def run_all_experiments(repo_root: Path) -> dict[str, object]:
    """Run every experiment end-to-end and write artifacts/benchmark_results.json."""
    t_start = time.perf_counter()
    cwru_dir = repo_root / "data" / "raw"
    uottawa_dir = repo_root / "data" / "uottawa"
    artifacts_dir = repo_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Preparing CWRU (40 recordings) & uOttawa (60 recordings)...", flush=True)
    cw = prepare_cwru_dataset(cwru_dir, use_cache=True)
    uo = prepare_uottawa_dataset(uottawa_dir, use_cache=True)

    print("[2/5] Evaluating zero-label Median/MAD & ISO 20816-3 baselines...", flush=True)
    zl_cwru = evaluate_zero_label_cwru(cwru_dir)
    zl_uottawa = evaluate_zero_label_uottawa(uottawa_dir)

    print("[3/5] Running 5-seed CWRU benchmark suite...", flush=True)
    cwru_res = run_cwru_suite(cw)

    print("[4/5] Running 5-seed uOttawa benchmark, ablations, ONNX INT8 & ACI...", flush=True)
    uottawa_res = run_uottawa_suite(uo, uottawa_dir, artifacts_dir)

    elapsed = round(time.perf_counter() - t_start, 2)
    report: dict[str, object] = {
        "meta": {
            "seeds": list(SEEDS),
            "nested_dims": list(NESTED_DIMS),
            "cwru_recordings": 40,
            "cwru_windows": len(cw.y),
            "uottawa_recordings": len(uo.stems),
            "uottawa_windows": len(uo.y),
            "uottawa_epochs": UOTTAWA_EPOCHS,
            "cwru_epochs": CWRU_EPOCHS,
            "elapsed_s": elapsed,
        },
        "zero_label_baselines": {
            "cwru": zl_cwru,
            "uottawa": zl_uottawa,
        },
        "cwru": cwru_res,
        "uottawa": uottawa_res,
    }

    out_path = artifacts_dir / "benchmark_results.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[5/5] Saved complete benchmark artifact to {out_path} in {elapsed}s", flush=True)
    return report
