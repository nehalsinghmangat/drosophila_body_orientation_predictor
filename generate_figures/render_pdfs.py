"""
Render the publication figures (publication_figures/*.svg) to PDF with Inkscape, one PDF next to each SVG.

    python generate_figures/render_pdfs.py [SVG ...] [--crop] [--out-dir DIR]

With no SVGs given, renders every SVG in publication_figures/. The figurefirst template layer ("Layer 1") is
hidden in the SVGs, so its rectangles do not appear in the PDFs. Inkscape is looked up in $INKSCAPE, then on
the PATH, then at the macOS application path.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
FIGURE_DIR = REPO_DIR / 'publication_figures'
MACOS_INKSCAPE = Path('/Applications/Inkscape.app/Contents/MacOS/inkscape')


def find_inkscape():
    for candidate in (os.environ.get('INKSCAPE'), shutil.which('inkscape'), MACOS_INKSCAPE):
        if candidate and Path(candidate).exists():
            return str(candidate)
    sys.exit('Inkscape not found: install it, or set $INKSCAPE to its executable.')


def render(inkscape, svg, pdf, crop):
    """Export one SVG to PDF: the whole page, or with crop only the drawing."""
    cmd = [inkscape, str(svg), '--export-type=pdf', f'--export-filename={pdf}']
    if crop:
        cmd.append('--export-area-drawing')
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not pdf.exists():
        sys.exit(f'Inkscape failed on {svg}:\n{result.stderr}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('svgs', nargs='*', type=Path, help='SVGs to render (default: publication_figures/*.svg)')
    parser.add_argument('--crop', action='store_true', help='crop each PDF to its drawing instead of the letter page')
    parser.add_argument('--out-dir', type=Path, help='write the PDFs here (default: next to each SVG)')
    args = parser.parse_args()

    svgs = args.svgs or sorted(FIGURE_DIR.glob('*.svg'))
    if not svgs:
        sys.exit(f'No SVGs in {FIGURE_DIR}.')
    inkscape = find_inkscape()
    for svg in svgs:
        out_dir = args.out_dir or svg.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf = out_dir / svg.with_suffix('.pdf').name
        render(inkscape, svg, pdf, args.crop)
        print(f'{svg} -> {pdf}')


if __name__ == '__main__':
    main()
