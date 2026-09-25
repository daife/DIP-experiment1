"""Render a redistributable 28-point index diagram from schematic coordinates."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "datasets" / "annotations" / "landmark28_schema.json"
OUTPUT = ROOT / "results" / "landmark28_index.png"

# Original schematic coordinates, in image axes (0..1); never traced from a licensed image.
POINTS = [
    (.15, .54), (.20, .32), (.50, .14), (.80, .32), (.85, .54),
    (.19, .76), (.30, .80), (.41, .82), (.59, .82), (.70, .80), (.81, .76),
    (.22, .58), (.29, .62), (.39, .61), (.24, .53), (.31, .51), (.40, .54),
    (.61, .61), (.71, .62), (.78, .58), (.60, .54), (.69, .51), (.76, .53),
    (.50, .46), (.43, .30), (.50, .32), (.57, .30), (.50, .27),
]
COLORS = {
    "face_contour": "#345f79",
    "eyebrows": "#b36b24",
    "image_left_eye": "#2f8065",
    "image_right_eye": "#7355a3",
    "nose": "#bd4e54",
    "mouth": "#b0447d",
}


def main() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    indices = [index for group in schema["groups"] for index in group["indices"]]
    if schema["count"] != 28 or sorted(indices) != list(range(28)) or len(POINTS) != 28:
        raise ValueError("Schema must cover every index from 0 through 27 exactly once")
    if len({i for pair in schema["flip_pairs"] for i in pair}) != 24:
        raise ValueError("Flip pairs must contain 24 unique indices")

    fig, ax = plt.subplots(figsize=(9, 9), dpi=180)
    fig.patch.set_facecolor("#f7f4ed")
    ax.set_facecolor("#f7f4ed")
    ax.add_patch(Ellipse((.5, .52), .72, .79, facecolor="#f3dfcc", edgecolor="#b39e8e", linewidth=2))
    for x in (.31, .69):
        ax.add_patch(Ellipse((x, .555), .19, .085, facecolor="white", edgecolor="#5f514c", linewidth=1.7))
        ax.add_patch(Ellipse((x, .555), .045, .07, facecolor="#62565b", edgecolor="none"))
    ax.plot([.5, .47, .53], [.58, .43, .41], color="#aa8073", lw=1.5)
    ax.plot([.42, .5, .58], [.30, .275, .30], color="#a75b62", lw=2)

    for group in schema["groups"]:
        color = COLORS[group["name"]]
        for index in group["indices"]:
            x, y = POINTS[index]
            ax.scatter([x], [y], s=140, c=color, edgecolors="white", linewidths=1.5, zorder=4)
            dx = -.018 if x > .72 else .018
            dy = .024 if index not in (23, 25, 27) else -.04
            if index == 25:
                dy = .045
            if index == 27:
                dy = -.055
            ax.text(x + dx, y + dy, str(index), color=color, fontsize=12, weight="bold",
                    ha="center", va="center", zorder=5,
                    bbox={"facecolor": "#f7f4ed", "edgecolor": "none", "alpha": .83, "pad": .6})

    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
    ax.axis("off")
    fig.suptitle("28 landmark indices · model array order (0–27)", fontsize=18, weight="bold", y=.97)
    legend = "Contour 0–4   •   Brows 5–10   •   Image-left eye 11–16   •   Image-right eye 17–22   •   Nose 23   •   Mouth 24–27"
    fig.text(.5, .055, legend, ha="center", fontsize=10, color="#44403c")
    fig.text(.5, .025, "Schematic only. Visibility is set during human review; model confidence is separate.",
             ha="center", fontsize=9, color="#6c625c")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
