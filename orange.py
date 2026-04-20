"""
SPDC Phase Matching Simulation
Course: UE23EC343BB1 | Orange Problem
Ref: Yesharim et al., Appl. Phys. Rev. 12, 011323 (2025)

Physics note:
  For degenerate type-0 SPDC in LiNbO3, the actual Δk from dispersion
  is ~1e6 rad/m → coherence length Lc ~ 2-3 µm.
  QPM period is chosen to exactly cancel Δk: Λ = 2π / Δk_raw.
  Plot (a) uses µm scale to show oscillations; plots also show
  parametric Δk values (small/medium/large) for pedagogical comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Constants ──────────────────────────────────────────────────────────────────
c = 3e8

# ── LiNbO3 Sellmeier (ordinary axis, Zelmon 1997) ─────────────────────────────
def n_LiNbO3(lam_m):
    """Refractive index of LiNbO3 (ordinary) via Sellmeier equation."""
    l2 = (lam_m * 1e6) ** 2          # wavelength in µm²
    return np.sqrt(4.9048 + 0.11775 / (l2 - 0.04751) - 0.027169 * l2)

# ── Wavelengths ────────────────────────────────────────────────────────────────
lambda_p = 405e-9
lambda_s = 810e-9     # degenerate signal
lambda_i = 810e-9     # degenerate idler

n_p = float(n_LiNbO3(lambda_p))
n_s = float(n_LiNbO3(lambda_s))
n_i = float(n_LiNbO3(lambda_i))

k_p = n_p * 2 * np.pi / lambda_p
k_s = n_s * 2 * np.pi / lambda_s
k_i = n_i * 2 * np.pi / lambda_i

delta_k_raw = k_p - k_s - k_i          # rad/m  (large, ~1e6)
L_c = np.pi / abs(delta_k_raw)         # coherence length (m)

# QPM: choose Λ so that G = Δk_raw exactly → perfect compensation
Lambda_QPM = 2 * np.pi / abs(delta_k_raw)
G = 2 * np.pi / Lambda_QPM
delta_k_qpm = delta_k_raw - G          # ≈ 0

print("=" * 58)
print("  SPDC Phase Matching — Key Parameters")
print("=" * 58)
print(f"  Pump wavelength        : {lambda_p*1e9:.1f} nm")
print(f"  Signal/Idler (degen.)  : {lambda_s*1e9:.1f} nm")
print(f"  n_p / n_s / n_i        : {n_p:.4f} / {n_s:.4f} / {n_i:.4f}")
print(f"  k_p                    : {k_p:.4e} rad/m")
print(f"  k_s + k_i              : {k_s+k_i:.4e} rad/m")
print(f"  Δk (no QPM)            : {delta_k_raw:.4e} rad/m  = {delta_k_raw*1e-3:.1f} rad/mm")
print(f"  Coherence length Lc    : {L_c*1e6:.2f} µm")
print(f"  QPM period Λ (exact)   : {Lambda_QPM*1e6:.2f} µm")
print(f"  QPM vector G           : {G:.4e} rad/m")
print(f"  Δk (with QPM)          : {delta_k_qpm:.4e} rad/m  ≈ 0")
print("=" * 58)

# ── sinc² efficiency ───────────────────────────────────────────────────────────
def eta(dk, L):
    x = dk * L / 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        val = np.where(np.abs(x) < 1e-10, 1.0, (np.sin(x) / x) ** 2)
    return val

# ── Parametric Δk values for pedagogical plot (a) ─────────────────────────────
# Use small multiples of 1/Lc so oscillations are visible over mm range
dk_small  = 1e2          # rad/m  — nearly matched
dk_medium = 5e3          # rad/m  — moderate mismatch
dk_large  = delta_k_raw  # rad/m  — realistic LiNbO3 mismatch (µm scale)

# ── Colors ─────────────────────────────────────────────────────────────────────
col = {
    "pm0":   "#E24B4A",
    "small": "#EF9F27",
    "med":   "#378ADD",
    "large": "#7F77DD",
    "qpm":   "#1D9E75",
    "bw":    "#7F77DD",
}

# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 12))
fig.suptitle(
    "SPDC Phase Matching Simulation — LiNbO$_3$, $\\lambda_p = 405$ nm\n"
    r"$\eta(L)=\mathrm{sinc}^2(\Delta k \cdot L/2)$   |   "
    r"$\Delta k = k_p - k_s - k_i$   |   QPM: $\Lambda = 2\pi/|\Delta k|$",
    fontsize=12, y=0.99)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

# ── Plot (a): η vs L — pedagogical Δk values, mm scale ────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
L_mm = np.linspace(0, 20e-3, 8000)

ax1.plot(L_mm*1e3, eta(0,        L_mm), color=col["pm0"],   lw=2.5,
         label=r"$\Delta k=0$ (perfect PM)")
ax1.plot(L_mm*1e3, eta(dk_small, L_mm), color=col["small"], lw=2,
         label=rf"$\Delta k={dk_small:.0f}$ rad/m (small)")
ax1.plot(L_mm*1e3, eta(dk_medium,L_mm), color=col["med"],   lw=2, ls="--",
         label=rf"$\Delta k={dk_medium:.0f}$ rad/m (medium)")

# QPM line (Δk ≈ 0, same as perfect PM)
ax1.plot(L_mm*1e3, eta(delta_k_qpm, L_mm), color=col["qpm"], lw=1.8, ls="-.",
         label=r"$\Delta k_{\rm raw}$ with QPM $\approx 0$")

# Mark coherence length for medium mismatch
Lc_med = np.pi / dk_medium
ax1.axvline(Lc_med*1e3, color=col["med"], lw=1, ls=":", alpha=0.8)
ax1.text(Lc_med*1e3 + 0.2, 0.88, r"$L_c$", color=col["med"], fontsize=10)

ax1.set_xlabel("Crystal length $L$ (mm)", fontsize=11)
ax1.set_ylabel(r"Efficiency $\eta = \mathrm{sinc}^2(\Delta k L / 2)$", fontsize=11)
ax1.set_title("(a) $\\eta$ vs $L$ — parametric $\\Delta k$ comparison", fontsize=11)
ax1.set_ylim(0, 1.10)
ax1.legend(fontsize=8.5, loc="upper right")
ax1.grid(True, alpha=0.3)

# ── Plot (b): η vs L — realistic Δk, µm scale showing oscillations ────────────
ax2 = fig.add_subplot(gs[0, 1])
L_um = np.linspace(0, 60e-6, 10000)   # 0 – 60 µm

ax2.plot(L_um*1e6, eta(0,           L_um), color=col["pm0"],  lw=2.5,
         label=r"$\Delta k=0$")
ax2.plot(L_um*1e6, eta(delta_k_raw, L_um), color=col["large"],lw=2,
         label=fr"$\Delta k={delta_k_raw*1e-3:.0f}$ rad/mm (no QPM)")
ax2.plot(L_um*1e6, eta(delta_k_raw/2, L_um), color=col["small"], lw=1.8,
         ls="--", label=fr"$\Delta k/2$")

ax2.axvline(L_c*1e6, color=col["large"], lw=1, ls=":", alpha=0.8)
ax2.text(L_c*1e6 + 0.5, 0.88, r"$L_c$", color=col["large"], fontsize=10)

ax2.set_xlabel("Crystal length $L$ (µm)", fontsize=11)
ax2.set_ylabel(r"Efficiency $\eta$", fontsize=11)
ax2.set_title(f"(b) Realistic $\\Delta k$ oscillations ($L_c = {L_c*1e6:.2f}$ µm)", fontsize=11)
ax2.set_ylim(0, 1.10)
ax2.legend(fontsize=8.5)
ax2.grid(True, alpha=0.3)

# ── Plot (c): η vs Δk — acceptance bandwidth at 3 crystal lengths ─────────────
ax3 = fig.add_subplot(gs[1, 0])
dk_scan = np.linspace(-2000, 2000, 6000)   # rad/m

for L_val, ls, lbl in [(5e-3, "-", "L = 5 mm"),
                        (10e-3,"--","L = 10 mm"),
                        (20e-3,":", "L = 20 mm")]:
    ax3.plot(dk_scan, eta(dk_scan, L_val), lw=2, ls=ls, label=lbl)

ax3.set_xlabel(r"Phase mismatch $\Delta k$ (rad/m)", fontsize=11)
ax3.set_ylabel(r"Efficiency $\eta$", fontsize=11)
ax3.set_title("(c) Acceptance bandwidth vs $\\Delta k$\n"
              "(longer $L$ → narrower bandwidth)", fontsize=11)
ax3.set_ylim(0, 1.10)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# FWHM annotation for L=10 mm
L_ann = 10e-3
dk_bw = 2 * 2.783 / L_ann
ax3.annotate("", xy=(dk_bw/2, 0.5), xytext=(-dk_bw/2, 0.5),
             arrowprops=dict(arrowstyle="<->", color="gray", lw=1.2))
ax3.text(0, 0.54, f"FWHM ≈ {dk_bw:.0f} rad/m", ha="center", fontsize=8.5, color="gray")

# ── Plot (d): 2D map η(Δk, L) — sensible range ────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
L_2d  = np.linspace(1e-3, 20e-3, 400)    # 1 – 20 mm
dk_2d = np.linspace(-3000, 3000, 400)    # rad/m
LL, DD = np.meshgrid(L_2d, dk_2d)
ETA2D = eta(DD, LL)

im = ax4.pcolormesh(L_2d*1e3, dk_2d, ETA2D,
                    cmap="plasma", shading="auto", vmin=0, vmax=1)
cb = fig.colorbar(im, ax=ax4, label=r"$\eta$", pad=0.02)
ax4.set_xlabel("Crystal length $L$ (mm)", fontsize=11)
ax4.set_ylabel(r"Phase mismatch $\Delta k$ (rad/m)", fontsize=11)
ax4.set_title(r"(d) 2D map $\eta(\Delta k,\,L)$" + "\n(bright band = phase-matched region)",
              fontsize=11)

plt.savefig("spdc_phase_matching.png", dpi=150, bbox_inches="tight")
print("\nFigure saved → spdc_phase_matching.png")
plt.show()
