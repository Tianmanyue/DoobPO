"""Embed Computer Modern math outlines in the policy-tilt diagrams.

Run after editing a formula or its placement. The committed SVGs contain paths, so
the website renders the same math lettering without loading a font or a script.
"""

from pathlib import Path
import re

from matplotlib import rcParams
from matplotlib.path import Path as MathPath
from matplotlib.textpath import TextPath


ROOT = Path(__file__).resolve().parents[1] / "docs" / "static" / "images"
rcParams["mathtext.fontset"] = "cm"

FORMULAS = {
    "reference": r"$\pi_{\mathrm{old}}(a\mid o)$",
    "ppo": r"$r_{\mathrm{PPO}}>0$",
    "grpo": r"$r_{\mathrm{GRPO}}>0$",
    "tilt": r"$\pi_{\mathrm{target}}(a\mid o)\propto r_{\mathrm{rule}}(o,a)\,\pi_{\mathrm{old}}(a\mid o)$",
}

# Coordinates are in each SVG's viewBox. The last field centers the whole formula.
LABELS = {
    "update-rule-tilts.svg": [
        ("reference", 64, 134, 22, "#30343b", False),
        ("ppo", 365, 130, 22, "#1b54b4", False),
        ("grpo", 365, 232, 22, "#147879", False),
        ("tilt", 480, 328, 27, "#30343b", True),
    ],
    "update-rule-tilts-mobile.svg": [
        ("reference", 96, 72, 21, "#30343b", False),
        ("ppo", 365, 236, 19, "#1b54b4", False),
        ("grpo", 356, 410, 19, "#147879", False),
        ("tilt", 240, 584, 25, "#30343b", True),
    ],
}


def number(value):
    return f"{float(value):.5f}".rstrip("0").rstrip(".") or "0"


def path_data(path):
    commands = []
    for values, code in path.iter_segments():
        points = [number(value) for value in values]
        if code == MathPath.MOVETO:
            commands.append(f"M{points[0]},{points[1]}")
        elif code == MathPath.LINETO:
            commands.append(f"L{points[0]},{points[1]}")
        elif code == MathPath.CURVE3:
            commands.append(f"Q{points[0]},{points[1]} {points[2]},{points[3]}")
        elif code == MathPath.CURVE4:
            commands.append(
                f"C{points[0]},{points[1]} {points[2]},{points[3]} {points[4]},{points[5]}"
            )
        elif code == MathPath.CLOSEPOLY:
            commands.append("Z")
        else:
            raise ValueError(f"Unsupported path segment: {code}")
    return " ".join(commands)


def render_label(label, x, y, size, color, centered, indent, newline):
    path = TextPath((0, 0), FORMULAS[label], size=1)
    bounds = path.get_extents()
    origin_x = x - (bounds.x0 + bounds.x1) * size / 2 if centered else x - bounds.x0 * size
    lines = [
        f'{indent}<!-- math-label:{label}:start -->',
        f'{indent}<g aria-label="{label} formula" fill="{color}" transform="translate({number(origin_x)} {y}) scale({size} -{size})">',
        f'{indent}  <path d="{path_data(path)}"/>',
        f"{indent}</g>",
        f'{indent}<!-- math-label:{label}:end -->',
    ]
    return newline.join(lines)


def main():
    for filename, labels in LABELS.items():
        file = ROOT / filename
        source = file.read_bytes().decode("utf-8")
        newline = "\r\n" if "\r\n" in source else "\n"
        for label, x, y, size, color, centered in labels:
            pattern = re.compile(
                rf'(?m)^(?P<indent>[ \t]*)<!-- math-label:{label}:start -->\r?\n'
                rf'.*?^[ \t]*<!-- math-label:{label}:end -->',
                re.DOTALL,
            )
            match = pattern.search(source)
            if match is None:
                raise ValueError(f"Missing {label} marker in {filename}")
            new = render_label(label, x, y, size, color, centered, match["indent"], newline)
            source = source[: match.start()] + new + source[match.end() :]
        file.write_bytes(source.encode("utf-8"))
        print(f"Rendered {filename}")


if __name__ == "__main__":
    main()
