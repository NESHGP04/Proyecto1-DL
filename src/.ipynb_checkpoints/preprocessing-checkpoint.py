"""
Pipeline de preprocesamiento para el dataset Ames Housing (Proyecto 1 - MLP).

Diseñado para ajustarse SOLO con datos de entrenamiento (`fit`) y luego aplicarse
de forma idéntica al set de prueba de la competencia (`transform`). El objeto se
puede guardar/cargar con joblib para reproducir el pipeline el día de la presentación.

Decisiones derivadas del EDA (ver notebooks/01_EDA.ipynb):
  1. Target -> log1p(SalePrice).
  2. NaN "de ausencia" -> categoría "None" / 0.
  3. LotFrontage -> mediana por Neighborhood.
  4. Variables de calidad -> codificación ORDINAL (Po<Fa<TA<Gd<Ex, etc.).
  5. MSSubClass y MoSold -> tratadas como categóricas (son códigos, no magnitudes).
  6. Numéricas muy sesgadas -> log1p.
  7. Numéricas -> StandardScaler.  Nominales -> One-Hot (handle_unknown='ignore').
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

TARGET = "SalePrice"
ID = "Id"

# --- Codificaciones ordinales (orden natural de peor -> mejor) ---
QUAL_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
QUAL_COLS = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
             "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]

ORD_MAPS: dict[str, dict] = {c: QUAL_MAP for c in QUAL_COLS}
ORD_MAPS.update({
    "BsmtExposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "Functional":   {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7},
    "LandSlope":    {"Sev": 0, "Mod": 1, "Gtl": 2},
    "LotShape":     {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3},
    "PavedDrive":   {"N": 0, "P": 1, "Y": 2},
    "CentralAir":   {"N": 0, "Y": 1},
    "Street":       {"Grvl": 0, "Pave": 1},
    "Alley":        {"None": 0, "Grvl": 1, "Pave": 2},
})

# Columnas categóricas cuyo NaN significa "no tiene la característica".
NONE_COLS = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageType",
             "GarageFinish", "GarageQual", "GarageCond", "BsmtQual", "BsmtCond",
             "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "MasVnrType"]

# Se tratan como categóricas nominales aunque vengan como enteros.
AS_CATEGORICAL = ["MSSubClass", "MoSold"]


class AmesPreprocessor:
    """Fit con train, transform idéntico para test. Serializable con joblib."""

    def __init__(self, skew_threshold: float = 0.75, min_frequency: int = 10,
                 clip: float = 8.0):
        self.skew_threshold = skew_threshold      # |skew| para aplicar log1p
        self.min_frequency = min_frequency        # colapsa categorías raras en one-hot
        self.clip = clip                          # recorta z-scores extremos (estabilidad)
        self.fitted = False

    # ---------- Limpieza común (no aprende parámetros) ----------
    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if TARGET in df.columns:
            df = df.drop(columns=[TARGET])
        if ID in df.columns:
            df = df.drop(columns=[ID])

        for c in AS_CATEGORICAL:
            df[c] = df[c].astype("Int64").astype(str)

        for c in NONE_COLS:
            if c in df.columns:
                df[c] = df[c].fillna("None")

        # Ordinales -> enteros
        for c, m in ORD_MAPS.items():
            if c in df.columns:
                if c not in NONE_COLS:
                    df[c] = df[c].fillna("None") if "None" in m else df[c]
                df[c] = df[c].map(m)
        return df

    # ---------- Fit ----------
    def fit(self, df: pd.DataFrame):
        df = self._clean(df)

        # 1) LotFrontage: mediana por Neighborhood (aprendida de train)
        self.lotfrontage_by_nb_ = (
            df.groupby("Neighborhood")["LotFrontage"].median().to_dict()
        )
        self.lotfrontage_global_ = df["LotFrontage"].median()
        df = self._impute_lotfrontage(df)

        # 2) Imputaciones puntuales
        self.garage_yr_fill_ = df["GarageYrBlt"].median()
        self.modes_ = {}
        df = self._impute_misc(df, fitting=True)

        # 3) Separar numéricas (incl. ordinales) vs categóricas nominales
        self.num_cols_ = df.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols_ = df.select_dtypes(include=["object"]).columns.tolist()

        # 4) Features sesgadas -> log1p (solo numéricas verdaderas, excluye ordinales)
        ordinal_names = set(ORD_MAPS) | {"OverallQual", "OverallCond"}
        cand = [c for c in self.num_cols_ if c not in ordinal_names]
        skews = df[cand].skew()
        self.skew_cols_ = skews[abs(skews) > self.skew_threshold].index.tolist()
        df[self.skew_cols_] = np.log1p(df[self.skew_cols_].clip(lower=0))

        # 5) Escalado de numéricas
        self.scaler_ = StandardScaler().fit(df[self.num_cols_])

        # 6) One-hot de nominales (colapsa categorías raras -> menos ruido/overfitting)
        self.ohe_ = OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  min_frequency=self.min_frequency, sparse_output=False)
        self.ohe_.fit(df[self.cat_cols_].astype(str))
        self.ohe_names_ = self.ohe_.get_feature_names_out(self.cat_cols_).tolist()

        self.feature_names_ = self.num_cols_ + self.ohe_names_
        self.fitted = True
        return self

    # ---------- Transform ----------
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        assert self.fitted, "Llama a fit() primero."
        df = self._clean(df)
        df = self._impute_lotfrontage(df)
        df = self._impute_misc(df, fitting=False)

        # columnas que pudieran faltar en test
        for c in self.num_cols_:
            if c not in df:
                df[c] = 0
        for c in self.cat_cols_:
            if c not in df:
                df[c] = "None"

        df[self.skew_cols_] = np.log1p(df[self.skew_cols_].clip(lower=0))
        # cualquier NaN residual en numéricas -> 0 tras escalar quedará neutro
        df[self.num_cols_] = df[self.num_cols_].fillna(0)
        num = self.scaler_.transform(df[self.num_cols_])
        cat = self.ohe_.transform(df[self.cat_cols_].astype(str))
        X = np.hstack([num, cat]).astype(np.float32)
        if self.clip:
            X = np.clip(X, -self.clip, self.clip)
        return X

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    # ---------- Target helpers ----------
    @staticmethod
    def target_to_log(y: pd.Series | np.ndarray) -> np.ndarray:
        return np.log1p(np.asarray(y, dtype=np.float64))

    @staticmethod
    def target_from_log(y_log: np.ndarray) -> np.ndarray:
        return np.expm1(np.asarray(y_log, dtype=np.float64))

    # ---------- helpers privados ----------
    def _impute_lotfrontage(self, df):
        if "LotFrontage" in df:
            df["LotFrontage"] = df.apply(
                lambda r: (self.lotfrontage_by_nb_.get(r["Neighborhood"], self.lotfrontage_global_)
                           if pd.isnull(r["LotFrontage"]) else r["LotFrontage"]),
                axis=1,
            )
        return df

    def _impute_misc(self, df, fitting: bool):
        if "GarageYrBlt" in df:
            df["GarageYrBlt"] = df["GarageYrBlt"].fillna(self.garage_yr_fill_)
        if "MasVnrArea" in df:
            df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
        # numéricas restantes -> 0 ; categóricas restantes -> moda (aprendida)
        for c in df.columns:
            if df[c].isnull().any():
                if df[c].dtype == object:
                    if fitting:
                        self.modes_[c] = df[c].mode(dropna=True)
                        self.modes_[c] = self.modes_[c].iloc[0] if len(self.modes_[c]) else "None"
                    df[c] = df[c].fillna(self.modes_.get(c, "None"))
                else:
                    df[c] = df[c].fillna(0)
        return df
