# -*- coding: utf-8 -*-
"""
core.py — Toàn bộ logic xử lý, tách khỏi giao diện.
Các trang Streamlit đều import từ đây để dùng chung một nguồn logic đã kiểm thử.
Bản notebook từng bước để đọc/debug: Notebook/08_san_pham_core.ipynb.

HAI mô hình cho HAI bài toán con (Notebook/09_cai_thien_do_chinh_xac.ipynb):
  1. PHÂN LOẠI 5 lớp — MLP (16-8) trên đặc trưng theo miền:
     - log(V): dị thường từ suy giảm theo luật lũy thừa với khoảng cách -> log tuyến tính hóa.
     - S tách thành độ ẩm (khô/ẩm) + vật liệu (cát/mùn/vôi) one-hot, thay vì một số thứ tự giả.
     CV lặp 5×3: ~72% (so với ~58% của HistGradientBoosting trên V, H, S thô), top-2 ~88%.
  2. PHÁT HIỆN / XẾP HẠNG — HistGradientBoosting nhị phân (có mìn / không) trên V, H, S.
     MLP cho xác suất quá tự tin (nhiều giá trị = 1.0) nên xếp hạng kém (ROC-AUC ~0.88);
     boosting nhị phân xếp hạng tốt hơn (ROC-AUC ~0.95) -> dùng cho P(có mìn) và ngưỡng.
Bài toán bảng nhỏ (338 mẫu) chạy trên CPU trong vài chục giây, không cần GPU.

QUAN TRỌNG (nhất quán với phần phân tích Giai đoạn 6):
Ngưỡng ưu tiên được ước lượng bằng cross-validation (chọn trên validation, đo
trên test tách biệt) TRÊN CHÍNH mô hình phát hiện dùng để tính P(có mìn).
Nó KHÔNG đảm bảo bắt được 100% mìn trên dữ liệu mới — chỉ dùng để sắp thứ tự
đào, không bao giờ để bỏ qua.
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
# Dữ liệu huấn luyện kèm theo (định dạng .xls cũ — cần thư viện xlrd).
DEFAULT_DATA = os.path.join(os.path.dirname(__file__), "data", "raw", "Mine_Dataset.xls")
MODEL_NAME = "Phân loại: MLP(16,8) + log(V) + one-hot loại đất · Phát hiện: HistGradientBoosting nhị phân"
RANDOM_STATE = 42

# Biên an toàn: hạ ngưỡng thêm 20% so với ước lượng để thận trọng hơn.
SAFETY_MARGIN = 0.8


# ------------------------------------------------------------------ mô hình
def make_features(X):
    """(V, H, S) -> [V, H, log V, khô/ẩm, cát, mùn, vôi]. Biến đổi từng dòng, không học
    tham số từ dữ liệu nên không gây rò rỉ."""
    X = np.asarray(X, dtype=float)
    V, H, S = X[:, 0], X[:, 1], X[:, 2]
    idx = np.round(S * 5).astype(int)          # 0..5 = loại đất 1..6
    mat = idx % 3                              # 0 cát, 1 mùn, 2 vôi
    return np.column_stack([V, H, np.log(V + 1e-3), (idx >= 3).astype(float),
                            mat == 0, mat == 1, mat == 2]).astype(float)


def make_pipeline():
    """Mô hình PHÂN LOẠI 5 lớp (loại mìn)."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, FunctionTransformer
    from sklearn.neural_network import MLPClassifier
    return Pipeline([
        ("features", FunctionTransformer(make_features)),
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=5000,
                              random_state=RANDOM_STATE)),
    ])


def make_detector():
    """Mô hình PHÁT HIỆN nhị phân (1 = có mìn) — nguồn của P(có mìn), xếp hạng và ngưỡng."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingClassifier
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE))])


def p_mine_of(detector, X):
    """P(có mìn) từ mô hình phát hiện nhị phân."""
    return detector.predict_proba(X)[:, list(detector.classes_).index(1)]


# ------------------------------------------------- ước lượng ngưỡng (CV, trung thực)
def estimate_threshold_cv(X, y, n_splits=5, n_repeats=4, random_state=RANDOM_STATE):
    """
    Nhất quán Giai đoạn 6: mỗi fold fit mô hình phát hiện trên train, chọn ngưỡng
    recall=100% trên VALIDATION, rồi áp ĐÚNG ngưỡng sản phẩm (× SAFETY_MARGIN) lên
    TEST tách biệt. Trả về ngưỡng thận trọng + thống kê trung thực (recall thực trên
    test, tỷ lệ fold giữ được 100%, % không-mìn được xếp cuối, ROC-AUC).
    """
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.metrics import precision_recall_curve, confusion_matrix, roc_auc_score

    yb = (y != 1).astype(int)   # 1 = có mìn
    thr_list, test_recall, hold100, elim, auc = [], [], [], [], []
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=random_state + rep)
        for tr, te in skf.split(X, y):
            Xt, Xv, yt, yv = train_test_split(X[tr], yb[tr], test_size=0.25,
                                              stratify=y[tr], random_state=random_state)
            m = make_detector().fit(Xt, yt)
            _, rec, thr = precision_recall_curve(yv, p_mine_of(m, Xv))
            idx = [i for i in range(len(thr)) if rec[i] >= 1.0]
            t = float(thr[max(idx)]) if idx else 0.0
            thr_list.append(t)
            p_te = p_mine_of(m, X[te])
            pred = (p_te >= t * SAFETY_MARGIN).astype(int)
            tn, fp, fn, tp = confusion_matrix(yb[te], pred, labels=[0, 1]).ravel()
            test_recall.append(tp / (tp + fn) if (tp + fn) else np.nan)
            hold100.append(1.0 if fn == 0 else 0.0)
            elim.append(tn / (tn + fp) if (tn + fp) else np.nan)
            auc.append(roc_auc_score(yb[te], p_te))
    safe_thr = float(np.nanmean(thr_list) * SAFETY_MARGIN)   # trung bình, hạ biên an toàn
    return {
        "safe_threshold": round(safe_thr, 4),
        "test_recall_mean": round(float(np.nanmean(test_recall)) * 100, 1),
        "test_recall_std": round(float(np.nanstd(test_recall)) * 100, 1),
        "hold_100_rate": round(float(np.mean(hold100)) * 100, 0),
        "deprioritizable_pct": round(float(np.nanmean(elim)) * 100, 1),
        "detect_roc_auc_cv": round(float(np.mean(auc)), 3),
    }


# ------------------------------------------------------------------ đánh giá CV
def evaluate_cv(X, y, n_splits=5, n_repeats=3, random_state=RANDOM_STATE):
    """CV lặp 5×3 trên đúng hai mô hình sản phẩm: accuracy, macro-F1, top-2 của mô hình
    phân loại 5 lớp; recall của mô hình phát hiện tại P(có mìn) ≥ 0.5."""
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score, recall_score

    yb = (y != 1).astype(int)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=random_state)
    acc, f1, top2, rec = [], [], [], []
    for tr, te in cv.split(X, y):
        m = make_pipeline().fit(X[tr], y[tr])
        P = m.predict_proba(X[te]); classes = list(m.classes_)
        pred = np.array(classes)[P.argmax(1)]
        acc.append(accuracy_score(y[te], pred))
        f1.append(f1_score(y[te], pred, average="macro"))
        top2.append(top_k_accuracy_score(y[te], P, k=2, labels=classes))
        d = make_detector().fit(X[tr], yb[tr])
        rec.append(recall_score(yb[te], (p_mine_of(d, X[te]) >= 0.5).astype(int)))
    pct = lambda v: round(float(np.mean(v)) * 100, 1)
    return {"cv_accuracy": pct(acc), "cv_accuracy_std": round(float(np.std(acc)) * 100, 1),
            "macro_f1_cv": pct(f1), "top2_accuracy_cv": pct(top2), "detect_recall_cv": pct(rec)}


# ------------------------------------------------------------------------ train
def load_training_data(data_path=DEFAULT_DATA):
    df = pd.read_excel(data_path) if data_path.lower().endswith((".xls", ".xlsx")) \
        else pd.read_csv(data_path)
    return df[FEATURES].values.astype(float), df["M"].values.astype(int)


def train_and_save(data_path=DEFAULT_DATA, model_path=MODEL_PATH):
    """Đánh giá CV + ước lượng ngưỡng ưu tiên (CV) + fit trên toàn bộ dữ liệu. Lưu vào model_path."""
    X, y = load_training_data(data_path)
    cv_stats = evaluate_cv(X, y)
    thr_stats = estimate_threshold_cv(X, y)

    pipe = make_pipeline().fit(X, y)
    detector = make_detector().fit(X, (y != 1).astype(int))
    bundle = {
        "pipeline": pipe,        # phân loại 5 lớp
        "detector": detector,    # phát hiện nhị phân -> P(có mìn), xếp hạng, ngưỡng
        "meta": {
            "model_name": MODEL_NAME,
            "sklearn_version": _sklearn_version(),
            "features": FEATURES,
            "classes": CLASS_NAMES,
            **cv_stats,
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

def _sklearn_version():
    import sklearn
    return sklearn.__version__

def _bundle_ok(b):
    """pkl hợp lệ nếu có đủ các trường meta của phiên bản hiện tại, đúng mô hình, và được
    tạo bằng ĐÚNG phiên bản scikit-learn đang chạy (lệch phiên bản -> sklearn chỉ cảnh báo
    'có thể sai kết quả', nên coi như không hợp lệ và tự train lại)."""
    need = {"safe_threshold", "cv_accuracy", "detect_recall_cv", "hold_100_rate",
            "model_name", "top2_accuracy_cv", "sklearn_version"}
    return (isinstance(b, dict) and "pipeline" in b and "detector" in b
            and need.issubset(b.get("meta", {}))
            and b["meta"]["model_name"] == MODEL_NAME
            and b["meta"]["sklearn_version"] == _sklearn_version())

def load_bundle(model_path=MODEL_PATH):
    """Nạp mô hình; TỰ PHỤC HỒI nếu pkl thiếu/hỏng/lệch phiên bản sklearn/mô hình cũ.
    Bình thường KHÔNG train lại — chỉ train một lần khi pkl không dùng được."""
    if "bundle" in _cache:
        return _cache["bundle"]
    import warnings
    b = None
    try:
        with warnings.catch_warnings():       # lệch phiên bản được xử lý bởi _bundle_ok
            warnings.simplefilter("ignore")
            b = joblib.load(model_path)
    except Exception as e:
        print("[core] Không nạp được model.pkl (%s)." % e)
    if not _bundle_ok(b):
        print("[core] model.pkl thiếu / mô hình cũ / lệch phiên bản scikit-learn "
              "(cần %s) -> tự huấn luyện lại một lần..." % _sklearn_version())
        if not os.path.exists(DEFAULT_DATA):
            raise RuntimeError("model.pkl thiếu/không hợp lệ và không tìm thấy dữ liệu "
                               "huấn luyện tại %s." % DEFAULT_DATA)
        train_and_save(DEFAULT_DATA, model_path)
        b = joblib.load(model_path)
    _cache["bundle"] = b
    return b


# ---------------------------------------------------------------------- predict
def validate_input(df_in):
    """Chuẩn hóa tên cột và kiểm tra V/H/S. Trả về DataFrame sạch hoặc ném ValueError
    kèm số dòng (theo Excel, dòng 1 là tiêu đề)."""
    df = df_in.copy()
    # chấp nhận tên cột có khoảng trắng / chữ thường: " v " -> "V"
    df = df.rename(columns={c: str(c).strip().upper() for c in df.columns
                            if str(c).strip().upper() in FEATURES})
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError("Thiếu cột: %s. Cần có V, H, S." % missing)
    if len(df) == 0:
        raise ValueError("File không có dòng dữ liệu nào.")

    Xdf = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    rows = lambda mask: ", ".join(str(i + 2) for i in np.flatnonzero(mask)[:10])
    bad = Xdf.isna().any(axis=1).values
    if bad.any():
        raise ValueError("Có %d dòng V/H/S trống hoặc không phải số (dòng %s)." % (bad.sum(), rows(bad)))
    out = ((Xdf < 0) | (Xdf > 1)).any(axis=1).values
    if out.any():
        raise ValueError("Có %d dòng V/H/S nằm ngoài [0, 1] (dòng %s). Dữ liệu phải được chuẩn hoá: "
                         "V = V_raw/10.6, H = H_raw/20, S ∈ {0.0, 0.2, …, 1.0}." % (out.sum(), rows(out)))
    off = (np.abs(Xdf["S"].values * 5 - np.round(Xdf["S"].values * 5)) > 1e-3)
    if off.any():
        raise ValueError("Có %d dòng S không thuộc 6 loại đất {0.0, 0.2, 0.4, 0.6, 0.8, 1.0} (dòng %s)."
                         % (off.sum(), rows(off)))
    df[FEATURES] = Xdf
    return df


def priority_level(p, safe_thr):
    """Mức ưu tiên đào theo P(có mìn)."""
    if p >= 0.5:            return "Rất cao"
    if p >= safe_thr:       return "Cao"
    if p >= safe_thr * 0.5: return "Trung bình"
    return "Thấp"


def predict_batch(df_in, model_path=MODEL_PATH):
    """DataFrame có V,H,S (tùy chọn id,x,y) -> thêm cột dự đoán + hạng ưu tiên."""
    bundle = load_bundle(model_path)
    pipe, detector, meta = bundle["pipeline"], bundle["detector"], bundle["meta"]
    safe_thr = meta["safe_threshold"]

    df = validate_input(df_in)
    X = df[FEATURES].values.astype(float)
    p_mine = p_mine_of(detector, X)                       # xếp hạng: mô hình phát hiện
    proba = pipe.predict_proba(X); classes = np.array(pipe.classes_)
    top = np.argsort(-proba, axis=1)                      # loại mìn: mô hình phân loại

    df["loai_du_doan"] = [meta["classes"][int(c)] for c in classes[top[:, 0]]]
    # top-2 đúng ~88% (CV): loại thứ hai giúp kỹ thuật viên chuẩn bị cho cả hai khả năng
    df["loai_kha_nang_2"] = [meta["classes"][int(c)] for c in classes[top[:, 1]]]
    df["xac_suat_co_min"] = np.round(p_mine, 4)
    df["muc_uu_tien"] = [priority_level(p, safe_thr) for p in p_mine]
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
