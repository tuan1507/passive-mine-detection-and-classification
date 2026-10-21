# -*- coding: utf-8 -*-
"""
train_final.py — Huấn luyện và lưu mô hình sản phẩm (xem core.MODEL_NAME).
Chạy: python train_final.py   (chạy được từ bất kỳ thư mục nào)
Bản notebook: Notebook/08_san_pham_core.ipynb
"""
import core

if __name__ == "__main__":
    meta = core.train_and_save(core.DEFAULT_DATA)
    print("Đã huấn luyện và lưu", core.MODEL_PATH)
    print("  Mô hình:", meta["model_name"])
    print("  Phân loại 5 lớp (CV 5×3): %s%% ± %s | macro-F1 %s%% | top-2 %s%%"
          % (meta["cv_accuracy"], meta["cv_accuracy_std"], meta["macro_f1_cv"], meta["top2_accuracy_cv"]))
    print("  Phát hiện: recall tại P≥0.5 (CV) %s%% | ROC-AUC %s"
          % (meta["detect_recall_cv"], meta["detect_roc_auc_cv"]))
    print("  Ngưỡng ưu tiên (CV):", meta["safe_threshold"],
          "| recall thực trên test %s%% ± %s" % (meta["test_recall_mean"], meta["test_recall_std"]),
          "| giữ recall 100%% ở %s%% lần thử (KHÔNG đảm bảo 0%% bỏ sót)" % meta["hold_100_rate"],
          "| %s%% tín hiệu không-mìn được xếp cuối" % meta["deprioritizable_pct"])
