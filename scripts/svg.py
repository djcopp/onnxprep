"""Small SVG toolkit for the README figures: themed bar charts and box-and-arrow diagrams.

Every figure is written twice, ``<name>-light.svg`` and ``<name>-dark.svg``, and the README picks one with
``<picture>``. Backgrounds are transparent so the figures sit on GitHub's own page colour.
"""

from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass
from xml.sax.saxutils import escape

FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace"


@dataclass(frozen=True)
class Theme:
    name: str
    ink: str  # primary text
    muted: str  # secondary text
    grid: str  # gridlines, hairlines
    line: str  # box outlines, arrows
    panel: str  # box fill
    accent_panel: str  # highlighted box fill
    series: tuple[str, ...]  # categorical slots, fixed order
    neutral: str  # baselines


LIGHT = Theme(
    "light",
    ink="#1f2328",
    muted="#59636e",
    grid="#d8dee4",
    line="#818b98",
    panel="#f6f8fa",
    accent_panel="#ddeafb",
    series=("#2a78d6", "#eb6834", "#1baf7a"),
    neutral="#8c959f",
)
DARK = Theme(
    "dark",
    ink="#e6edf3",
    muted="#9198a1",
    grid="#30363d",
    line="#6e7681",
    panel="#161b22",
    accent_panel="#132a47",
    series=("#3987e5", "#d95926", "#199e70"),
    neutral="#6e7681",
)
THEMES = (LIGHT, DARK)


class Canvas:
    def __init__(self, width: int, height: int, theme: Theme):
        self.w, self.h, self.t = width, height, theme
        self.parts: list[str] = []

    def add(self, s: str) -> None:
        self.parts.append(s)

    def text(self, x, y, s, size=13, color=None, anchor="start", weight=400, mono=False, baseline="middle"):
        color = color or self.t.ink
        family = MONO if mono else FONT
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
            f'fill="{color}" text-anchor="{anchor}" dominant-baseline="{baseline}">{escape(str(s))}</text>'
        )

    def rect(self, x, y, w, h, fill, rx=0, stroke=None, stroke_width=1, dash=None):
        extra = f' stroke="{stroke}" stroke-width="{stroke_width}"' if stroke else ""
        extra += f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"{extra}/>'
        )

    def line(self, x1, y1, x2, y2, color=None, width=1, dash=None):
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color or self.t.grid}" '
            f'stroke-width="{width}"{dash_attr}/>'
        )

    def bar(self, x0, x1, y, h, fill, radius=4):
        """Horizontal bar anchored at x0 with a rounded data end at x1."""
        left, right = min(x0, x1), max(x0, x1)
        w = right - left
        r = min(radius, w / 2, h / 2)
        if x1 >= x0:  # square at the baseline (left), round at the end (right)
            d = (
                f"M{left:.1f},{y:.1f} H{right - r:.1f} Q{right:.1f},{y:.1f} {right:.1f},{y + r:.1f} "
                f"V{y + h - r:.1f} Q{right:.1f},{y + h:.1f} {right - r:.1f},{y + h:.1f} H{left:.1f} Z"
            )
        else:
            d = (
                f"M{right:.1f},{y:.1f} H{left + r:.1f} Q{left:.1f},{y:.1f} {left:.1f},{y + r:.1f} "
                f"V{y + h - r:.1f} Q{left:.1f},{y + h:.1f} {left + r:.1f},{y + h:.1f} H{right:.1f} Z"
            )
        self.add(f'<path d="{d}" fill="{fill}"/>')

    def arrow(self, points, color=None, width=1.5, dash=None, head=True):
        color = color or self.t.line
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in points)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{dash_attr}/>')
        if head:
            (xa, ya), (xb, yb) = points[-2], points[-1]
            ang = math.atan2(yb - ya, xb - xa)
            size = 7
            p1 = (xb - size * math.cos(ang - 0.45), yb - size * math.sin(ang - 0.45))
            p2 = (xb - size * math.cos(ang + 0.45), yb - size * math.sin(ang + 0.45))
            self.add(
                f'<path d="M{p1[0]:.1f},{p1[1]:.1f} L{xb:.1f},{yb:.1f} L{p2[0]:.1f},{p2[1]:.1f} Z" fill="{color}"/>'
            )

    def box(self, x, y, w, h, title, lines=(), accent=False, dashed=False, mono_title=False, title_size=14):
        t = self.t
        self.rect(
            x,
            y,
            w,
            h,
            t.accent_panel if accent else t.panel,
            rx=8,
            stroke=t.series[0] if accent else t.grid,
            stroke_width=1.2,
            dash="4 4" if dashed else None,
        )
        n = 1 + len(lines)
        top = y + h / 2 - (n - 1) * 9
        self.text(x + w / 2, top, title, size=title_size, anchor="middle", weight=600, mono=mono_title)
        for i, s in enumerate(lines):
            self.text(x + w / 2, top + 18 * (i + 1), s, size=12, color=t.muted, anchor="middle")

    def svg(self, title: str) -> str:
        body = "\n".join(self.parts)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
            f'viewBox="0 0 {self.w} {self.h}" role="img">\n<title>{escape(title)}</title>\n{body}\n</svg>\n'
        )


def fmt_x(v: float) -> str:
    if v >= 10:
        return f"{v:.0f}×"
    if v >= 1:
        return f"{v:.1f}×"
    return f"{v:.2f}×"


def fmt_ms(v: float) -> str:
    if v >= 1000:
        return f"{v / 1000:.1f} s"
    if v >= 10:
        return f"{v:.0f} ms"
    return f"{v:.1f} ms"


def speedup_chart(
    rows: list[tuple[str, str, dict[str, float]]],
    series: list[str],
    theme: Theme,
    axis_label: str,
    lo: float = 0.03,
    hi: float = 100.0,
    width: int = 880,
) -> Canvas:
    """Grouped horizontal bars on a log axis, anchored at 1× so faster grows right and slower grows left.

    rows: (label, sublabel, {series name: speed-up}).
    """
    t = theme
    label_w, right_pad, bar_h, gap = 230, 56, 12, 2
    group_h = len(series) * (bar_h + gap) - gap
    row_h = max(group_h + 22, 44)
    top = 44 if len(series) > 1 else 14
    height = top + row_h * len(rows) + 64
    c = Canvas(width, height, t)
    x0, x1 = label_w, width - right_pad

    def sx(v: float) -> float:
        v = min(max(v, lo), hi)
        return x0 + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (x1 - x0)

    # legend, top left over the plot; a single series is named by the axis label instead
    lx = x0
    for i, name in enumerate(series if len(series) > 1 else []):
        c.rect(lx, 12, 12, 12, t.series[i], rx=3)
        c.text(lx + 18, 18.5, name, size=12, color=t.muted)
        lx += 18 + 7.2 * len(name) + 24

    # gridlines at decades, plus the 1× reference
    ticks = [v for v in (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300) if lo <= v <= hi]
    plot_bottom = top + row_h * len(rows)
    for v in ticks:
        x = sx(v)
        c.line(x, top - 6, x, plot_bottom, t.ink if v == 1 else t.grid, width=1.2 if v == 1 else 1)
        c.text(x, plot_bottom + 14, f"{v:g}×", size=11, color=t.muted, anchor="middle")
    c.text(sx(1) - 8, plot_bottom + 34, "◀ slower", size=11, color=t.muted, anchor="end")
    c.text(sx(1) + 8, plot_bottom + 34, "faster ▶", size=11, color=t.muted, anchor="start")
    c.text(x1, plot_bottom + 54, axis_label, size=11, color=t.muted, anchor="end")

    for r, (label, sub, values) in enumerate(rows):
        y = top + r * row_h
        cy = y + row_h / 2
        c.text(0, cy - (7 if sub else 0), label, size=13, weight=600)
        if sub:
            c.text(0, cy + 10, sub, size=11, color=t.muted)
        gy = cy - group_h / 2
        for i, name in enumerate(series):
            v = values.get(name)
            by = gy + i * (bar_h + gap)
            if v is None:
                continue
            c.bar(sx(1), sx(v), by, bar_h, t.series[i])
            end = sx(v)
            if v >= 1:
                c.text(end + 6, by + bar_h / 2, fmt_x(v), size=11, color=t.ink, anchor="start")
            else:
                c.text(end - 6, by + bar_h / 2, fmt_x(v), size=11, color=t.ink, anchor="end")
        if r < len(rows) - 1:
            c.line(0, y + row_h, x1, y + row_h, t.grid, width=0.6)
    return c


def time_chart(
    rows: list[tuple[str, float, int | None]],
    theme: Theme,
    axis_label: str,
    width: int = 880,
    log: bool = False,
    label_w: int = 250,
) -> Canvas:
    """One horizontal bar per row: (label, milliseconds, colour slot or None for a neutral baseline)."""
    t = theme
    bar_h, row_h, top = 14, 30, 8
    height = top + row_h * len(rows) + 40
    c = Canvas(width, height, t)
    x0, x1 = label_w, width - 70
    vmax = max(v for _, v, _ in rows)
    vmin = min(v for _, v, _ in rows)
    if log:
        lo = 10 ** math.floor(math.log10(vmin))
        hi = 10 ** math.ceil(math.log10(vmax))

        def sx(v):
            return x0 + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (x1 - x0)

        ticks = [lo * 10**k for k in range(int(round(math.log10(hi / lo))) + 1)]
        base = lo
    else:
        step = 10 ** math.floor(math.log10(vmax))
        hi = math.ceil(vmax / step) * step

        def sx(v):
            return x0 + v / hi * (x1 - x0)

        ticks = [step * k for k in range(int(hi / step) + 1)]
        if len(ticks) > 7:
            ticks = ticks[::2]
        base = 0
    bottom = top + row_h * len(rows)
    for v in ticks:
        c.line(sx(v), top, sx(v), bottom, t.grid)
        c.text(sx(v), bottom + 14, fmt_ms(v) if v else "0", size=11, color=t.muted, anchor="middle")
    c.text(x1, bottom + 32, axis_label, size=11, color=t.muted, anchor="end")
    for r, (label, v, slot) in enumerate(rows):
        cy = top + r * row_h + row_h / 2
        c.text(label_w - 12, cy, label, size=13, anchor="end", weight=600 if slot is not None else 400)
        fill = t.neutral if slot is None else t.series[slot]
        c.bar(sx(base), sx(v), cy - bar_h / 2, bar_h, fill)
        c.text(sx(v) + 6, cy, fmt_ms(v), size=11, anchor="start")
    return c


def write(canvas_for_theme, name: str, title: str, out: pathlib.Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        (out / f"{name}-{theme.name}.svg").write_text(canvas_for_theme(theme).svg(title))
