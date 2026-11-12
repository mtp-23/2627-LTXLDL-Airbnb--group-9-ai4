import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")                      # vẽ ra file, không cần màn hình
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

ROOT = Path(__file__).resolve().parent.parent
SENTIMENTS = ["positive", "mixed", "negative"]

MAU_CHINH, MAU_PHU, MAU_TIM = "#C2410C", "#0F766E", "#6D28D9"
MAU_NHAN = {"positive": "#15803D", "mixed": "#CA8A04", "negative": "#B91C1C"}

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 12, "axes.titleweight": "bold"})

# Hàm hỗ trợ
def doc_config(path):
    path = Path(path) if path else sorted((ROOT / "configs").glob("*.yaml"))[0]
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["snapshots"] = [str(d) for d in cfg["snapshots"]]
    return cfg

def thu_muc(cfg, khoa, mac_dinh):
    p = Path(cfg.get(khoa) or mac_dinh)
    return p if p.is_absolute() else ROOT / p

def doc_du_lieu(viz_dir, ten):
    """Đọc một file dữ liệu vẽ hình do kpi.py tạo; None nếu chưa có."""
    path = viz_dir / ten
    if not path.exists():
        return None
    return pd.read_csv(path, encoding="utf-8-sig", dtype={"snapshot": str})

def luu(fig, thu_muc_hinh, ten):
    fig.tight_layout()
    fig.savefig(thu_muc_hinh / ten, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ghi {thu_muc_hinh.name}/{ten}")

def cac_vong_ngoai(geometry):
    """Vòng ngoài của Polygon / MultiPolygon (đủ để tô màu bản đồ)."""
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"][0]]
    if geometry["type"] == "MultiPolygon":
        return [poly[0] for poly in geometry["coordinates"]]
    return []

def ve_ban_do(ax, features, gia_tri, tieu_de, cmap, nhan_thanh_mau):
    """Tô màu từng khu vực theo `gia_tri` (dict: tên khu vực -> số). Khu vực không có số liệu: màu xám."""
    patches, values = [], []
    for ft in features:
        ten = ft["properties"].get("neighbourhood")
        for vong in cac_vong_ngoai(ft["geometry"]):
            patches.append(Polygon(np.array(vong), closed=True))
            values.append(gia_tri.get(ten, np.nan))
    bang_mau = plt.get_cmap(cmap).copy()
    bang_mau.set_bad("#E5E7EB")
    pc = PatchCollection(patches, cmap=bang_mau, edgecolor="white", linewidths=0.6)
    pc.set_array(np.ma.masked_invalid(np.array(values, dtype=float)))
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(tieu_de)
    plt.colorbar(pc, ax=ax, shrink=0.75).set_label(nhan_thanh_mau)

# Ba biểu đồ
def fig1_ban_do(du_lieu, out, raw_dir):
    """Không gian: bản đồ số listing và tỷ lệ ngày còn trống theo khu vực (sau QA)."""
    if du_lieu is None:
        return print("  [bỏ qua] fig1: chưa có data/processed/viz/fig1_khu_vuc.csv")
    snap = str(du_lieu["snapshot"].iloc[0])
    geo = raw_dir / snap / "neighbourhoods.geojson"
    if not geo.exists():
        return print(f"  [bỏ qua] fig1: không thấy {geo} (hãy chạy download.py)")
    with open(geo, encoding="utf-8") as f:
        features = json.load(f)["features"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.5))
    ve_ban_do(a1, features, dict(zip(du_lieu["khu_vuc"], du_lieu["so_listing"])),
              f"Số listing theo khu vực ({snap})", "Oranges", "Số listing (phòng)")
    ve_ban_do(a2, features, dict(zip(du_lieu["khu_vuc"], du_lieu["ty_le_ngay_con_trong_pct"])),
              f"Tỷ lệ ngày còn trống theo khu vực ({snap})", "GnBu", "Ngày available = 't' (%)")
    luu(fig, out, "fig1_spatial_map.png")

def fig2_review_va_lich(du_lieu, out):
    """Thời gian: tỷ lệ listing chưa có review, review trung bình, tỷ lệ ngày còn trống qua các snapshot."""
    if du_lieu is None:
        return print("  [bỏ qua] fig2: chưa có data/processed/viz/fig2_theo_thoi_gian.csv")
    du_lieu = du_lieu.sort_values("snapshot")
    cac_hinh = [("ty_le_chua_review_pct", "Listing chưa có review (%)", MAU_CHINH, "{:.1f}%"),
                ("review_tb_moi_listing", "Review trung bình / listing", MAU_PHU, "{:.1f}"),
                ("ty_le_ngay_con_trong_pct", "Ngày còn trống (%)", MAU_TIM, "{:.1f}%")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (cot, tieu_de, mau, dinh_dang) in zip(axes, cac_hinh):
        ax.plot(du_lieu["snapshot"], du_lieu[cot], marker="o", color=mau, linewidth=2)
        for x, y in zip(du_lieu["snapshot"], du_lieu[cot]):
            ax.annotate(dinh_dang.format(y), (x, y), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=9)
        ax.set_title(tieu_de)
        ax.margins(y=0.25)
        ax.tick_params(axis="x", rotation=25)
    luu(fig, out, "fig2_reviews_availability_over_time.png")

def fig3_llm(du_lieu, out):
    """LLM: ma trận nhầm lẫn (nhãn tay vs LLM) và F1 từng lớp."""
    if du_lieu is None:
        return print("  [bỏ qua] fig3: chưa có kết quả LLM")
    mat = np.zeros((3, 3))
    for _, r in du_lieu[du_lieu["loai"] == "confusion"].iterrows():
        mat[SENTIMENTS.index(r["nhan_that"]), SENTIMENTS.index(r["nhan_llm"])] = r["gia_tri"]
    f1 = (du_lieu[(du_lieu["loai"] == "lop") & (du_lieu["chi_so"] == "f1")]
          .set_index("nhan_that").reindex(SENTIMENTS))
    tong = du_lieu[du_lieu["loai"] == "tong"].set_index("chi_so")["gia_tri"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    a1.imshow(mat, cmap="Oranges")
    a1.set_xticks(range(3), SENTIMENTS)
    a1.set_yticks(range(3), SENTIMENTS)
    a1.set_xlabel("Nhãn LLM dự đoán")
    a1.set_ylabel("Nhãn thủ công (thật)")
    a1.set_title("Ma trận nhầm lẫn")
    for i in range(3):
        for j in range(3):
            a1.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=12,
                    color="white" if mat[i, j] > mat.max() / 2 else "black")
    a1.spines[:].set_visible(False)
    a2.bar(SENTIMENTS, f1["gia_tri"], color=[MAU_NHAN[c] for c in SENTIMENTS])
    for x, (v, n) in enumerate(zip(f1["gia_tri"], f1["so_mau"])):
        a2.text(x, v + 0.02, f"{v:.2f}\n(n={int(n)})", ha="center", fontsize=9)
    a2.axhline(tong["macro_f1"], color="black", linestyle="--", linewidth=1, label=f"macro F1 = {tong['macro_f1']:.2f}")
    a2.legend(loc="upper right", frameon=False, fontsize=9)
    a2.set_ylim(0, 1.2)
    a2.set_ylabel("F1-score")
    a2.set_title(f"F1 từng lớp (accuracy = {tong['accuracy']:.2f})")
    luu(fig, out, "fig3_llm_evaluation.png")

# Chạy
def run(config_path=None):
    cfg = doc_config(config_path)
    proc, raw_dir = thu_muc(cfg, "processed_dir", "data/processed"), thu_muc(cfg, "raw_dir", "data/raw")
    viz_dir = proc / "viz"
    if not viz_dir.exists():
        raise SystemExit("Chưa có data/processed/viz/ - hãy chạy kpi.py trước:\n"
                         "    python src/kpi.py --config configs/santiago.yaml")
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    print("[viz] sinh hình vào", out)
    fig1_ban_do(doc_du_lieu(viz_dir, "fig1_khu_vuc.csv"), out, raw_dir)
    fig2_review_va_lich(doc_du_lieu(viz_dir, "fig2_theo_thoi_gian.csv"), out)
    fig3_llm(doc_du_lieu(viz_dir, "fig3_llm.csv"), out)

def main():
    ap = argparse.ArgumentParser(description="Vẽ 3 biểu đồ từ dữ liệu do kpi.py tạo")
    ap.add_argument("--config")
    run(ap.parse_args().config)

if __name__ == "__main__":
    main()