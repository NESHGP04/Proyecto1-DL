"""
Predicción para el día de la competencia (Proyecto 1 - MLP).

Carga el preprocesador ajustado y el ensemble de MLPs guardados, aplica el MISMO
preprocesamiento al dataset de prueba, promedia las predicciones de los 5 modelos y
escribe un CSV con las predicciones. Si el CSV de prueba incluye la columna SalePrice,
calcula además el RMSE (en log y en dólares).

Uso:
    python predict.py --test ruta/al/test.csv --out predicciones.csv

Requisitos: haber ejecutado antes notebooks/02_MLP_modelo.ipynb para generar
    models/preprocessor.joblib  y  models/mlp_ensemble.pt
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import AmesPreprocessor, TARGET  # noqa: E402
from model import MLP, rmse, get_device  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# Preferimos el modelo v2 (feature engineering: target encoding + ensemble ampliado);
# si no existe, caemos al v1.
V2_PP = os.path.join(HERE, "models", "preprocessor_v2.joblib")
V2_BUNDLE = os.path.join(HERE, "models", "mlp_ensemble_v2.pt")
V1_PP = os.path.join(HERE, "models", "preprocessor.joblib")
V1_BUNDLE = os.path.join(HERE, "models", "mlp_ensemble.pt")


def _apply_target_encoding(df, bundle):
    """Aplica los mapas de target encoding guardados (v2). Devuelve matriz (n, n_te_cols)."""
    cols = bundle["te_cols"]
    gm = bundle["te_global_mean"]
    maps = bundle["te_maps"]
    return np.column_stack(
        [df[c].map(maps[c]).fillna(gm).values for c in cols]
    ).astype("float32")


def load_ensemble(device):
    if os.path.exists(V2_BUNDLE) and os.path.exists(V2_PP):
        pp_path, bundle_path, version = V2_PP, V2_BUNDLE, "v2 (target encoding + ensemble)"
    else:
        pp_path, bundle_path, version = V1_PP, V1_BUNDLE, "v1"
    pp = joblib.load(pp_path)
    bundle = torch.load(bundle_path, map_location=device, weights_only=False)
    bundle["_version"] = version
    cfg = bundle["final_cfg"]
    models = []
    for sd in bundle["state_dicts"]:
        m = MLP(bundle["in_features"], cfg["hidden"], cfg.get("activation", "relu"),
                cfg["dropout"], cfg.get("batchnorm", False)).to(device)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
    return pp, models, bundle


@torch.no_grad()
def predict(df: pd.DataFrame, pp: AmesPreprocessor, models, bundle, device) -> np.ndarray:
    """Devuelve predicciones en DÓLARES (promedio del ensemble)."""
    X = pp.transform(df)
    if "te_maps" in bundle:  # modelo v2: añade columnas de target encoding
        X = np.hstack([X, _apply_target_encoding(df, bundle)])
    X = torch.tensor(X, dtype=torch.float32).to(device)
    preds_log = np.mean([m(X).cpu().numpy() for m in models], axis=0)
    return AmesPreprocessor.target_from_log(preds_log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, help="CSV del dataset de prueba")
    ap.add_argument("--out", default="predicciones.csv", help="CSV de salida")
    args = ap.parse_args()

    device = get_device()
    pp, models, bundle = load_ensemble(device)
    print(f"Modelo {bundle['_version']} | {len(models)} modelos | device={device}")
    print(f"(OOF en entrenamiento: RMSE log={bundle['oof_rmse_log']:.4f}, "
          f"${bundle['oof_rmse_usd']:,.0f})")

    df = pd.read_csv(args.test)
    print(f"Test: {df.shape[0]} filas")

    y_pred = predict(df, pp, models, bundle, device)

    ids = df["Id"] if "Id" in df.columns else np.arange(len(df))
    out = pd.DataFrame({"Id": ids, "SalePrice_pred": np.round(y_pred, 2)})
    out.to_csv(args.out, index=False)
    print(f"Predicciones escritas en: {args.out}")

    # Si el test trae el valor real, reportamos RMSE (métrica de la competencia)
    if TARGET in df.columns:
        y_true = df[TARGET].values
        r_usd = rmse(y_true, y_pred)
        r_log = rmse(np.log1p(y_true), np.log1p(y_pred))
        print("\n=== RMSE en el set de prueba ===")
        print(f"RMSE(log)     = {r_log:.4f}")
        print(f"RMSE(dólares) = ${r_usd:,.0f}")


if __name__ == "__main__":
    main()
