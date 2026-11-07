"""Render the reference Cliffwalk Figure 1 from portable, verified raw Cliffwalk data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True,
                        help='New directory; existing outputs are never replaced')
    parser.add_argument('--data-root', type=Path,
                        help='Optional verified Cliffwalk study data directory')
    args = parser.parse_args(argv)
    from scripts.reproduction_manifest import guard_output_path
    from scripts.render_main_two_panel import render_figure
    output = guard_output_path(args.output_root, ROOT)
    if output.exists():
        raise ValueError('Choose a new nonexistent output directory; existing artifacts are preserved')
    metadata = render_figure(output / 'figure_main_empirical_results_composite',
                              data_root=args.data_root)
    print(json.dumps({key: metadata[key] for key in
                     ('main_panel', 'raw_statistics_match_reference_arrays',
                      'pixel_identical_to_reference_at_300dpi', 'displayed_methods')}, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2)
