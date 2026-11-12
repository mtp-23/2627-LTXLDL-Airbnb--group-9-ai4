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

COLOR_MAIN, COLOR_SECONDARY, COLOR_PURPLE = "#C2410C", "#0F766E", "#6D28D9"
LABEL_COLORS = {"positive": "#15803D", "mixed": "#CA8A04", "negative": "#B91C1C"}

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 12, "axes.titleweight": "bold"})

# Hàm hỗ trợ
def read_config(path):
    path = Path(path) if path else sorted((ROOT / "configs").glob("*.yaml"))[0]
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["snapshots"] = [str(d) for d in cfg["snapshots"]]
    return cfg

def resolve_dir(cfg, key, default):
    p = Path(cfg.get(key) or default)
    return p if p.is_absolute() else ROOT / p

def read_viz_data(viz_dir, filename):
    """Đọc một file dữ liệu vẽ hình do kpi.py tạo; None nếu chưa có."""
    path = viz_dir / filename
    if not path.exists():
        return None
    return pd.read_csv(path, encoding="utf-8-sig", dtype={"snapshot": str})

def save_figure(fig, figures_dir, filename):
    fig.tight_layout()
    fig.savefig(figures_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ghi {figures_dir.name}/{filename}")

def outer_rings(geometry):
    """Vòng ngoài của Polygon / MultiPolygon (đủ để tô màu bản đồ)."""
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"][0]]
    if geometry["type"] == "MultiPolygon":
        return [poly[0] for poly in geometry["coordinates"]]
    return []

def draw_map(ax, features, values_by_area, title, cmap, colorbar_label):
    """Tô màu từng khu vực theo `gia_tri` (dict: tên khu vực -> số). Khu vực không có số liệu: màu xám."""
    patches, values = [], []
    for ft in features:
        area_name = ft["properties"].get("neighbourhood")
        for ring in outer_rings(ft["geometry"]):
            patches.append(Polygon(np.array(ring), closed=True))
            values.append(values_by_area.get(area_name, np.nan))
    color_map = plt.get_cmap(cmap).copy()
    color_map.set_bad("#E5E7EB")
    pc = PatchCollection(patches, cmap=color_map, edgecolor="white", linewidths=0.6)
    pc.set_array(np.ma.masked_invalid(np.array(values, dtype=float)))
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title)
    plt.colorbar(pc, ax=ax, shrink=0.75).set_label(colorbar_label)

# Ba biểu đồ
def fig1_spatial_map(data, out, raw_dir):
    """Không gian: bản đồ số listing và tỷ lệ ngày còn trống theo khu vực (sau QA)."""
    if data is None:
        return print("  [bỏ qua] fig1: chưa có data/processed/viz/fig1_areas.csv")
    snap = str(data["snapshot"].iloc[0])
    geo = raw_dir / snap / "neighbourhoods.geojson"
    if not geo.exists():
        return print(f"  [bỏ qua] fig1: không thấy {geo} (hãy chạy download.py)")
    with open(geo, encoding="utf-8") as f:
        features = json.load(f)["features"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.5))
    draw_map(a1, features, dict(zip(data["area"], data["listing_count"])),
              f"Số listing theo khu vực ({snap})", "Oranges", "Số listing (phòng)")
    draw_map(a2, features, dict(zip(data["area"], data["available_day_rate_pct"])),
              f"Tỷ lệ ngày còn trống theo khu vực ({snap})", "GnBu", "Ngày available = 't' (%)")
    save_figure(fig, out, "fig1_spatial_map.png")

def fig2_reviews_availability(data, out):
    """Thời gian: tỷ lệ listing chưa có review, review trung bình, tỷ lệ ngày còn trống qua các snapshot."""
    if data is None:
        return print("  [bỏ qua] fig2: chưa có data/processed/viz/fig2_over_time.csv")
    data = data.sort_values("snapshot")
    panels = [("no_review_rate_pct", "Listing chưa có review (%)", COLOR_MAIN, "{:.1f}%"),
                ("avg_reviews_per_listing", "Review trung bình / listing", COLOR_SECONDARY, "{:.1f}"),
                ("available_day_rate_pct", "Ngày còn trống (%)", COLOR_PURPLE, "{:.1f}%")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (column, title, color, number_format) in zip(axes, panels):
        ax.plot(data["snapshot"], data[column], marker="o", color=color, linewidth=2)
        for x, y in zip(data["snapshot"], data[column]):
            ax.annotate(number_format.format(y), (x, y), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=9)
        ax.set_title(title)
        ax.margins(y=0.25)
        ax.tick_params(axis="x", rotation=25)
    save_figure(fig, out, "fig2_reviews_availability_over_time.png")

def fig3_llm(data, out):
    """LLM: ma trận nhầm lẫn (nhãn tay vs LLM) và F1 từng lớp."""
    if data is None:
        return print("  [bỏ qua] fig3: chưa có kết quả LLM")
    matrix = np.zeros((3, 3))
    for _, r in data[data["kind"] == "confusion"].iterrows():
        matrix[SENTIMENTS.index(r["true_label"]), SENTIMENTS.index(r["llm_label"])] = r["value"]
    f1 = (data[(data["kind"] == "class") & (data["metric"] == "f1")]
          .set_index("true_label").reindex(SENTIMENTS))
    totals = data[data["kind"] == "total"].set_index("metric")["value"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    a1.imshow(matrix, cmap="Oranges")
    a1.set_xticks(range(3), SENTIMENTS)
    a1.set_yticks(range(3), SENTIMENTS)
    a1.set_xlabel("Nhãn LLM dự đoán")
    a1.set_ylabel("Nhãn thủ công (thật)")
    a1.set_title("Ma trận nhầm lẫn")
    for i in range(3):
        for j in range(3):
            a1.text(j, i, int(matrix[i, j]), ha="center", va="center", fontsize=12,
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    a1.spines[:].set_visible(False)
    a2.bar(SENTIMENTS, f1["value"], color=[LABEL_COLORS[c] for c in SENTIMENTS])
    for x, (v, n) in enumerate(zip(f1["value"], f1["sample_size"])):
        a2.text(x, v + 0.02, f"{v:.2f}\n(n={int(n)})", ha="center", fontsize=9)
    a2.axhline(totals["macro_f1"], color="black", linestyle="--", linewidth=1, label=f"macro F1 = {totals['macro_f1']:.2f}")
    a2.legend(loc="upper right", frameon=False, fontsize=9)
    a2.set_ylim(0, 1.2)
    a2.set_ylabel("F1-score")
    a2.set_title(f"F1 từng lớp (accuracy = {totals['accuracy']:.2f})")
    save_figure(fig, out, "fig3_llm_evaluation.png")

# Chạy
def run(config_path=None):
    cfg = read_config(config_path)
    proc, raw_dir = resolve_dir(cfg, "processed_dir", "data/processed"), resolve_dir(cfg, "raw_dir", "data/raw")
    viz_dir = proc / "viz"
    if not viz_dir.exists():
        raise SystemExit("Chưa có data/processed/viz/ - hãy chạy kpi.py trước:\n"
                         "    python src/kpi.py --config configs/santiago.yaml")
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    print("[viz] sinh hình vào", out)
    fig1_spatial_map(read_viz_data(viz_dir, "fig1_areas.csv"), out, raw_dir)
    fig2_reviews_availability(read_viz_data(viz_dir, "fig2_over_time.csv"), out)
    fig3_llm(read_viz_data(viz_dir, "fig3_llm.csv"), out)

def main():
    ap = argparse.ArgumentParser(description="Vẽ 3 biểu đồ từ dữ liệu do kpi.py tạo")
    ap.add_argument("--config")
    run(ap.parse_args().config)

if __name__ == "__main__":
    main()