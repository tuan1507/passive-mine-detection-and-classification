# -*- coding: utf-8 -*-
"""
core.py — Toàn bộ logic xử lý, tách khỏi giao diện.
Các trang Streamlit đều import từ đây để dùng chung một nguồn logic đã kiểm thử.

Mô hình: HistGradientBoosting (nhanh, ổn định cho sản phẩm). Bài toán bảng nhỏ
(338 mẫu, 3 đặc trưng) chạy trên CPU trong tích tắc, không cần GPU.

QUAN TRỌNG (nhất quán với phần phân tích Giai đoạn 6):
Ngưỡng ưu tiên được ước lượng bằng cross-validation (chọn trên validation, đo
trên test tách biệt), KHÔNG phải trên một lần chia. Và nó KHÔNG đảm bảo bắt được
100% mìn trên dữ liệu mới — chỉ dùng để sắp thứ tự đào, không bao giờ để bỏ qua.
"""
import os
import numpy as np
import pandas as pd
import joblib

FEATURES = ["V", "H", "S"]
CLASS_NAMES = {1: "Không mìn", 2: "Mìn chống tăng", 3: "Mìn chống bộ binh",
               4: "Mìn chống bộ binh bẫy", 5: "Mìn M14"}
# Bảng nhãn loại đất theo Bảng 1 bài báo: S = (loại-1)/5, xếp theo ĐỘ ẨM rồi VẬT LIỆU.
# 1 Khô&cát, 2 Khô&mùn, 3 Khô&vôi, 4 Ẩm&cát, 5 Ẩm&mùn, 6 Ẩm&vôi.
SOIL_NAMES = {0.0: "Khô & cát", 0.2: "Khô & mùn", 0.4: "Khô & vôi",
              0.6: "Ẩm & cát", 0.8: "Ẩm & mùn", 1.0: "Ẩm & vôi"}
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "model.pkl")

# Biên an toàn: hạ ngưỡng thêm 20% so với ước lượng để thận trọng hơn.
SAFETY_MARGIN = 0.8


# ------------------------------------------------- ước lượng ngưỡng (CV, trung thực)
def estimate_threshold_cv(X, y, n_splits=5, n_repeats=4, random_state=42):
    """
    Nhất quán Giai đoạn 6: mỗi fold fit train, chọn ngưỡng recall=100% trên
    VALIDATION, đo trên TEST tách biệt. Trả về ngưỡng thận trọng + thống kê
    trung thực (recall thực trên test, tỷ lệ fold giữ được 100%).
    """
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import precision_recall_curve, confusion_matrix

    yb = (y != 1).astype(int)   # 1 = có mìn
    thr_list, test_recall, hold100, elim = [], [], [], []
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=random_state + rep)
        for tr, te in skf.split(X, yb):
            Xtr, Xte, ytr, yte = X[tr], X[te], yb[tr], yb[te]
            Xt, Xv, yt, yv = train_test_split(Xtr, ytr, test_size=0.25,
                                              stratify=ytr, random_state=random_state)
            s = StandardScaler().fit(Xt)
            m = HistGradientBoostingClassifier(random_state=random_state).fit(s.transform(Xt), yt)
            pv = m.predict_proba(s.transform(Xv))[:, list(m.classes_).index(1)]
            _, rec, thr = precision_recall_curve(yv, pv)
            idx = [i for i in range(len(thr)) if rec[i] >= 1.0]
            t = float(thr[max(idx)]) if idx else 0.0
            thr_list.append(t)
            pte = m.predict_proba(s.transform(Xte))[:, list(m.classes_).index(1)]
            pred = (pte >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
            test_recall.append(tp / (tp + fn) if (tp + fn) else np.nan)
            hold100.append(1.0 if fn == 0 else 0.0)
            elim.append(tn / (tn + fp) if (tn + fp) else np.nan)
    safe_thr = float(np.nanmean(thr_list) * SAFETY_MARGIN)   # trung bình, hạ biên an toàn
    return {
        "safe_threshold": round(safe_thr, 4),
        "test_recall_mean": round(float(np.nanmean(test_recall)) * 100, 1),
        "test_recall_std": round(float(np.nanstd(test_recall)) * 100, 1),
        "hold_100_rate": round(float(np.mean(hold100)) * 100, 0),
        "deprioritizable_pct": round(float(np.nanmean(elim)) * 100, 1),
    }


# ------------------------------------------------------------------------ train
def train_and_save(data_path, model_path=MODEL_PATH):
    """Huấn luyện pipeline 5 lớp + ước lượng ngưỡng ưu tiên (CV). Lưu vào model_path."""
    from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import recall_score

    df = pd.read_excel(data_path) if data_path.lower().endswith((".xls", ".xlsx")) \
        else pd.read_csv(data_path)
    X = df[FEATURES].values.astype(float)
    y = df["M"].values.astype(int)

    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", HistGradientBoostingClassifier(random_state=42))])
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(pipe, X, y, cv=cv).mean()

    # recall phát hiện (mìn vs không) qua CV — chỉ số hướng an toàn, đúng trọng tâm
    yb = (y != 1).astype(int)
    oof = cross_val_predict(
        Pipeline([("scaler", StandardScaler()),
                  ("clf", HistGradientBoostingClassifier(random_state=42))]),
        X, yb, cv=cv)
    detect_recall = recall_score(yb, oof, pos_label=1) * 100

    thr_stats = estimate_threshold_cv(X, y)

    pipe.fit(X, y)
    bundle = {
        "pipeline": pipe,
        "meta": {
            "features": FEATURES,
            "classes": CLASS_NAMES,
            "cv_accuracy": round(float(cv_acc) * 100, 1),      # 5 lớp
            "detect_recall_cv": round(float(detect_recall), 1),  # phát hiện mìn (CV)
            **thr_stats,
            "note": ("Hệ thống hỗ trợ sắp thứ tự ưu tiên. Ngưỡng KHÔNG đảm bảo "
                     "bắt 100% mìn trên dữ liệu mới — mọi tín hiệu VẪN phải đào "
                     "kiểm tra đầy đủ. Không dùng để bỏ qua bất kỳ tín hiệu nào."),
        },
    }
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(bundle, model_path)
    return bundle["meta"]


# ------------------------------------------------------------------------- load
_cache = {}
DEFAULT_DATA = os.path.join(os.path.dirname(__file__), "data", "raw", "Mine_Dataset.xlsx")

def _bundle_ok(b):
    """pkl hợp lệ nếu nạp được và có đủ các trường meta trung thực mới."""
    need = {"safe_threshold", "cv_accuracy", "detect_recall_cv", "hold_100_rate"}
    return isinstance(b, dict) and "pipeline" in b and need.issubset(b.get("meta", {}))

def load_bundle(model_path=MODEL_PATH):
    """Nạp mô hình; TỰ PHỤC HỒI nếu pkl thiếu/hỏng/lệch phiên bản sklearn.
    Bình thường KHÔNG train lại — chỉ train một lần khi pkl không dùng được."""
    if "bundle" in _cache:
        return _cache["bundle"]
    b = None
    try:
        b = joblib.load(model_path)
    except Exception as e:
        print("[core] Không nạp được model.pkl (%s). Tự huấn luyện lại một lần..." % e)
    if not _bundle_ok(b):
        if os.path.exists(DEFAULT_DATA):
            train_and_save(DEFAULT_DATA, model_path)
            b = joblib.load(model_path)
        elif b is None:
            raise RuntimeError("Không có model.pkl và cũng không có dữ liệu để huấn luyện.")
    _cache["bundle"] = b
    return b


# ---------------------------------------------------------------------- predict
def predict_batch(df_in, model_path=MODEL_PATH):
    """DataFrame có V,H,S (tùy chọn id,x,y) -> thêm cột dự đoán + hạng ưu tiên."""
    bundle = load_bundle(model_path)
    pipe = bundle["pipeline"]; meta = bundle["meta"]
    safe_thr = meta["safe_threshold"]

    df = df_in.copy()
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError("Thiếu cột: %s. Cần có V, H, S." % missing)

    X = df[FEATURES].values.astype(float)
    classes = list(pipe.named_steps["clf"].classes_)
    proba = pipe.predict_proba(X); pred = pipe.predict(X)
    p_mine = 1 - proba[:, classes.index(1)]

    df["loai_du_doan"] = [meta["classes"][int(c)] for c in pred]
    df["xac_suat_co_min"] = np.round(p_mine, 4)

    def level(p):
        if p >= 0.5:            return "Rất cao"
        if p >= safe_thr:       return "Cao"
        if p >= safe_thr * 0.5: return "Trung bình"
        return "Thấp"
    df["muc_uu_tien"] = [level(p) for p in p_mine]
    # cờ ưu tiên thấp: xếp CUỐI hàng đợi — KHÔNG có nghĩa bỏ qua (vẫn phải đào)
    df["uu_tien_thap"] = p_mine < safe_thr
    df["hang_uu_tien"] = df["xac_suat_co_min"].rank(
        ascending=False, method="first").astype(int)
    df = df.sort_values("hang_uu_tien").reset_index(drop=True)
    return df, meta


def summarize(df_pred):
    n = len(df_pred)
    by_level = df_pred["muc_uu_tien"].value_counts().to_dict()
    n_high = by_level.get("Rất cao", 0) + by_level.get("Cao", 0)
    n_low = int(df_pred["uu_tien_thap"].sum())
    return {
        "tong_tin_hieu": n,
        "uu_tien_cao": n_high,
        "uu_tien_thap": n_low,
        "ty_le_uu_tien_thap": round(100 * n_low / n, 1) if n else 0,
        "phan_bo_muc": by_level,
    }
