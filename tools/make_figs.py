"""Figures for the ablation report (run: py -3 make_figs.py OUTDIR)."""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT, exist_ok=True)

CONC = [1, 4, 16, 64]
TPS = {  # aggregate tok/s, measured on Ascend 910B2 / vLLM 0.26.0 / vllm-ascend v0.26.0rc
    0: [59.4, 205.7, 701.6, 1878.8],
    1: [64.3, 218.5, 743.0, 2069.4],
    2: [75.8, 240.4, 813.6, None],
    3: [83.3, 265.9, 721.6, 1599.0],
    5: [78.7, 236.0, 600.5, 1028.0],
}
ACC = {1: 2.03, 2: 2.29, 3: 2.40, 5: 2.41}
COL = {0: "#7d8590", 1: "#4c72b0", 2: "#dd8452", 3: "#55a868", 5: "#c44e52"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# --- 1. saturation curves + envelope -----------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for k, vals in TPS.items():
    y = [v if v else np.nan for v in vals]
    ax.plot(CONC, y, marker="o", ms=5, lw=1.8, color=COL[k], label=f"K={k}" + ("  (baseline)" if k == 0 else ""))
env = [max(v for v in (TPS[k][i] for k in TPS) if v) for i in range(4)]
ax.plot(CONC, env, ls="--", lw=1.2, color="#111", zorder=1, label="envelope (best K)")
for i, (c, e) in enumerate(zip(CONC, env)):
    best = max((k for k in TPS if TPS[k][i]), key=lambda k: TPS[k][i])
    ax.annotate(f"K*={best}", (c, e), textcoords="offset points", xytext=(4, 6), fontsize=8.5)
ax.axhline(TPS[0][3], color=COL[0], lw=0.8, ls=":", alpha=0.0)
ax.set_xscale("log", base=2); ax.set_xticks(CONC); ax.set_xticklabels(CONC)
ax.set_xlabel("concurrency (requests in flight)"); ax.set_ylabel("aggregate throughput (tok/s)")
ax.set_title("MTP speculative decoding on Ascend 910B2 (Qwen3.5-9B)\nthroughput vs concurrency, one curve per draft depth K")
ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=8.5, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_saturation.png", dpi=180); plt.close(fig)

# --- 2. speedup heatmap with sign flip ---------------------------------------
ks = [1, 2, 3, 5]
sp = np.array([[TPS[k][i] / TPS[0][i] if TPS[k][i] else np.nan for i in range(4)] for k in ks])
cmap = LinearSegmentedColormap.from_list("rg", ["#c44e52", "#f7e1d7", "#55a868"])
fig, ax = plt.subplots(figsize=(5.4, 3.1))
im = ax.imshow(sp, cmap=cmap, vmin=0.55, vmax=1.45, aspect="auto")
ax.set_xticks(range(4)); ax.set_xticklabels([f"conc={c}" for c in CONC])
ax.set_yticks(range(len(ks))); ax.set_yticklabels([f"K={k}" for k in ks])
for r in range(len(ks)):
    for c in range(4):
        v = sp[r, c]
        ax.text(c, r, "n/m" if np.isnan(v) else f"{v:.2f}x", ha="center", va="center",
                fontsize=10, color="white" if (not np.isnan(v) and v > 1.25) else "#222",
                fontweight="bold" if (not np.isnan(v) and v >= 1.10) else "normal")
ax.set_title("Speedup over no-speculative baseline (K=0)\nvalues < 1.00 = speculative decoding loses throughput")
fig.colorbar(im, ax=ax, shrink=0.8, label="throughput ratio")
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_speedup_heatmap.png", dpi=180); plt.close(fig)

# --- 3. latency-throughput Pareto --------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for k in [0, 1, 2, 3, 5]:
    pts = [(1000.0 / (TPS[k][i] / CONC[i]), TPS[k][i], CONC[i]) for i in range(4) if TPS[k][i]]
    ax.plot([p[0] for p in pts], [p[1] for p in pts],
            marker="s" if k == 0 else "o", lw=1.7, ms=5, color=COL[k],
            label="K=0 (baseline)" if k == 0 else f"K={k}")
    for x, y, c in pts:
        ax.annotate(str(c), (x, y), textcoords="offset points", xytext=(5, -3), fontsize=7.5, color="#555")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("per-user latency (ms / token)   ← lower is better")
ax.set_ylabel("aggregate throughput (tok/s)   ↑ higher is better")
ax.set_title("Latency–throughput trade-off (point labels = concurrency)\nupper-left is better; K=3 / K=2 beat the baseline out to conc=16")
ax.grid(alpha=0.25, which="both"); ax.legend(frameon=False, fontsize=8.5, loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_pareto.png", dpi=180); plt.close(fig)

# --- 4. K curve at n=1 with acceptance overlay --------------------------------
fig, ax = plt.subplots(figsize=(6.0, 3.8))
Ks = [0, 1, 2, 3, 4, 5]
pt = [17.2, 15.8, 13.2, 12.0, 12.0, 12.7]
ax.plot(Ks, pt, marker="o", lw=2, color="#4c72b0", label="ms / token (left)")
ax.set_xlabel("draft depth K"); ax.set_ylabel("TPOT (ms / token)", color="#4c72b0")
ax.set_ylim(10, 18.5); ax.grid(alpha=0.25)
ax2 = ax.twinx()
ax2.plot([1, 2, 3, 5], [ACC[k] for k in (1, 2, 3, 5)], marker="^", ls="--", lw=1.4, color="#55a868", label="acceptance length (right)")
ax2.set_ylabel("mean acceptance length", color="#55a868"); ax2.set_ylim(1.5, 3.0)
ax.annotate("peak plateau K=3–4\n(+43% vs K=0)", (3.5, 11.4), textcoords="offset points", xytext=(-10, -32), fontsize=8.5,
            arrowprops=dict(arrowstyle="->", lw=0.8))
ax.text(0.12, 15.5, "acceptance grows sub-linearly while iteration cost is ~K-independent\n→ deeper drafts are nearly free at low load",
        fontsize=8, color="#444", va="top")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5, loc="lower left")
ax.set_title("Single-stream (conc=1): inverted-U in K, explained by flat iteration cost")
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_k_curve.png", dpi=180); plt.close(fig)

with open(f"{OUT}/matrix.csv", "w") as f:
    f.write("k,conc_1,conc_4,conc_16,conc_64\n")
    for k, vals in TPS.items():
        f.write(f"{k}," + ",".join("" if v is None else str(v) for v in vals) + "\n")
print("figures + matrix.csv written to", OUT)
