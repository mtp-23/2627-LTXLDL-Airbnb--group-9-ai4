"""Vẽ 3 biểu đồ (không gian, thời gian, LLM) từ bảng đã tính sẵn.

Nhiệm vụ duy nhất: vẽ và lưu hình PNG. Không đọc tệp dữ liệu/cấu hình, không
tính KPI hay F1 (nhận bảng từ kpi.py / evaluate.py qua run_pipeline.py).
"""
import matplotlib
matplotlib.use("Agg")                      # vẽ ra file, không cần màn hình
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

COLOR_MAIN, COLOR_SECONDARY, COLOR_PURPLE = "#C2410C", "#0F766E", "#6D28D9"
LABEL_COLORS = {"positive": "#15803D", "mixed": "#CA8A04", "negative": "#B91C1C"}

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 12, "axes.titleweight": "bold"})

# Hàm hỗ trợ
def save_figure(fig, figures_dir, filename):
    fig.tight_layout()
    path = figures_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

def outer_rings(geometry):
    """Vòng ngoài của Polygon / MultiPolygon (đủ để tô màu bản đồ)."""
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"][0]]
    if geometry["type"] == "MultiPolygon":
        return [poly[0] for poly in geometry["coordinates"]]
    return []

def _ring_centroid(ring) -> tuple[float, float]:
    """Trọng tâm hình học của một đa giác đóng; dùng trung bình đỉnh nếu diện tích bằng 0."""
    pts = np.asarray(ring, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y2 - x2 * y
    area2 = cross.sum()
    if abs(area2) < 1e-12:
        return float(x.mean()), float(y.mean())
    cx = ((x + x2) * cross).sum() / (3.0 * area2)
    cy = ((y + y2) * cross).sum() / (3.0 * area2)
    return float(cx), float(cy)


def _polygon_area(ring) -> float:
    pts = np.asarray(ring, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    return float(abs((x * np.roll(y, -1) - np.roll(x, -1) * y).sum()) * 0.5)


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
def fig1_spatial_map(data, features, out):
    """Không gian: bản đồ số listing và tỷ lệ ngày còn trống theo khu vực (sau QA). None nếu thiếu dữ liệu."""
    if data is None or features is None:
        return None
    snap = str(data["snapshot"].iloc[0])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.5))
    draw_map(a1, features, dict(zip(data["area"], data["listing_count"])),
              f"Số listing theo khu vực ({snap})", "Oranges", "Số listing (phòng)")
    top3 = set(data.nlargest(3, "listing_count")["area"].astype(str))
    for ft in features:
        area_name = str(ft["properties"].get("neighbourhood", ""))
        if area_name in top3:
            rings = outer_rings(ft["geometry"])
            if rings:
                cx, cy = _ring_centroid(max(rings, key=_polygon_area))
                a1.text(cx, cy, area_name, ha="center", va="center", fontsize=7.5,
                        fontweight="bold", color="#111827", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.8))
    draw_map(a2, features, dict(zip(data["area"], data["available_day_rate_pct"])),
              f"Tỷ lệ ngày còn trống theo khu vực ({snap})", "GnBu", "Ngày available = 't' (%)")
    return save_figure(fig, out, "fig1_spatial_map.png")

def fig2_reviews_availability(data, out):
    """Thời gian: tỷ lệ listing không có review 12 tháng qua, review trung bình, tỷ lệ ngày còn trống qua các snapshot."""
    if data is None:
        return None
    data = data.sort_values("snapshot")
    panels = [("no_review_rate_pct", "Không có review 12T qua (%)", COLOR_MAIN, "{:.1f}%"),
                ("avg_reviews_per_listing", "Review trung bình / listing", COLOR_SECONDARY, "{:.1f}"),
                ("available_day_rate_pct", "Ngày còn trống (%)", COLOR_PURPLE, "{:.1f}%")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    xs = list(data["snapshot"])
    for ax, (column, title, color, number_format) in zip(axes, panels):
        ys = list(data[column])
        ax.plot(xs, ys, marker="o", color=color, linewidth=2)
        for idx, (x, y) in enumerate(zip(xs, ys)):
            neighbors = [ys[j] for j in (idx - 1, idx + 1) if 0 <= j < len(ys)]
            below = len(ys) > 1 and (y < min(neighbors) or (y == min(ys) and y < max(ys)))
            offset_y, va = (-14, "top") if below else (8, "bottom")
            ax.annotate(number_format.format(y), (x, y), textcoords="offset points",
                        xytext=(0, offset_y), ha="center", va=va, fontsize=9)
        ax.set_title(title)
        ax.margins(x=0.12, y=0.30)
        ax.tick_params(axis="x", rotation=25)
    return save_figure(fig, out, "fig2_reviews_availability_over_time.png")

def fig3_llm(data, out, sentiments):
    """LLM: ma trận nhầm lẫn (nhãn tay vs LLM) và F1 từng lớp. None nếu chưa có kết quả LLM."""
    if data is None:
        return None
    SENTIMENTS = list(sentiments)
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
    a2.bar(SENTIMENTS, f1["value"], color=[LABEL_COLORS[c] for c in SENTIMENTS], zorder=3)
    a2.axhline(totals["macro_f1"], color="black", linestyle="--", linewidth=1, zorder=2,
               label=f"macro F1 = {totals['macro_f1']:.2f}")
    for x, (v, n) in enumerate(zip(f1["value"], f1["sample_size"])):
        a2.text(x, v + 0.02, f"{v:.2f}\n(n={int(n)})", ha="center", va="bottom", fontsize=9, zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=1.0))
    a2.legend(loc="upper right", frameon=False, fontsize=9)
    a2.set_ylim(0, 1.2)
    a2.set_ylabel("F1-score")
    a2.set_title(f"F1 từng lớp (accuracy = {totals['accuracy']:.2f})")
    return save_figure(fig, out, "fig3_llm_evaluation.png")

# Chạy
def draw_all(fig1, fig2, fig3, features, out_dir, sentiments):
    """Vẽ cả 3 hình vào out_dir. Trả về {tên hình: đường dẫn | None nếu thiếu dữ liệu}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    return {"fig1": fig1_spatial_map(fig1, features, out_dir),
            "fig2": fig2_reviews_availability(fig2, out_dir),
            "fig3": fig3_llm(fig3, out_dir, sentiments)}
