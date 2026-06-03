import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fisher_kpp_rk4.visualization import plot_cross_section_view, solve_2d_fisher_kpp_snapshots

OUTPUT_DIR = PROJECT_ROOT / "outputs"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    result = solve_2d_fisher_kpp_snapshots()
    output_path = OUTPUT_DIR / "cross_section_view.png"
    plot_cross_section_view(result, output_path)
    print(f"Saved: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
