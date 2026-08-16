# -*- coding: utf-8 -*-
"""
train_final.py — Huấn luyện và lưu mô hình chính (cây boosting).
Chạy: python train_final.py
"""
import core

if __name__ == "__main__":
    meta = core.train_and_save("data/raw/Mine_Dataset.xls")
    print("Đã huấn luyện và lưu models/model.pkl")
    print("  Độ chính xác CV:", meta["cv_accuracy"], "%")
    print("  Recall phát hiện mìn (CV):", meta["detect_recall_cv"], "%")
    print("  Ngưỡng ưu tiên (CV):", meta["safe_threshold"],
          "| giữ recall 100% ở", meta["hold_100_rate"], "% lần thử (KHÔNG đảm bảo 0% bỏ sót)")
