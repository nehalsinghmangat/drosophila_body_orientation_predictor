"""
Run every trajectory in example_inference/trajectories/ through heading_inference.py with the notebook's
default settings, and tile the resulting heading plots into a single SVG grid (one Inkscape layer per
trajectory, labelled with its name).

    python example_inference/make_trajectory_grid.py [--cols 3] [--gap 4] [--label-size 6] [--out PATH]
"""
import argparse
import io
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

EXAMPLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLE_DIR))
from heading_inference import app  # noqa: E402

SVG_NS = 'http://www.w3.org/2000/svg'
INKSCAPE_NS = 'http://www.inkscape.org/namespaces/inkscape'
ET.register_namespace('', SVG_NS)
ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
ET.register_namespace('inkscape', INKSCAPE_NS)


def notebook_defaults():
    """The notebook's default model, wedge size and stride, read from its UI elements."""
    _, defs = app.run()
    return {name: SimpleNamespace(value=defs[name].value) for name in ('model_dropdown', 'wedge_size', 'stride')}


def heading_svg(name, defaults):
    """Heading plot for one trajectory, saved as the notebook's Export SVG button does."""
    pressed = SimpleNamespace(value=True)
    _, defs = app.run(defs={'traj_dropdown': SimpleNamespace(value=name), 'inputs_button': pressed,
                            'predict_button': pressed, 'export_button': SimpleNamespace(value=False),
                            **defaults})
    buf = io.BytesIO()
    defs['heading_fig'].savefig(buf, format='svg', bbox_inches='tight')
    plt.close('all')
    return ET.fromstring(buf.getvalue())


def prefix_ids(root, prefix):
    """Keep ids unique across panels: prefix every id and every reference to one."""
    for el in root.iter():
        for key, value in list(el.attrib.items()):
            if key == 'id':
                el.attrib[key] = prefix + value
            elif key.endswith('href') and value.startswith('#'):
                el.attrib[key] = '#' + prefix + value[1:]
            elif 'url(#' in value:
                el.attrib[key] = value.replace('url(#', 'url(#' + prefix)


def tile(panels, cols, gap, label_size):
    """
    Grid of (name, svg root) panels: each column as wide as its widest panel, each row as tall as its
    tallest, panels centred in their cells. Sizes are in pt (matplotlib's SVG user unit).
    """
    sizes = [tuple(float(v) for v in root.get('viewBox').split()[2:]) for _, root in panels]
    n, rows = len(panels), -(-len(panels) // cols)
    label_h = 1.5 * label_size
    col_w = [max(sizes[i][0] for i in range(c, n, cols)) for c in range(min(cols, n))]
    row_h = [max(sizes[i][1] for i in range(r * cols, min((r + 1) * cols, n))) + label_h for r in range(rows)]
    width, height = sum(col_w) + gap * (len(col_w) - 1), sum(row_h) + gap * (rows - 1)

    out = ET.Element(f'{{{SVG_NS}}}svg', {'version': '1.1', 'width': f'{width:g}pt', 'height': f'{height:g}pt',
                                         'viewBox': f'0 0 {width:g} {height:g}'})
    for i, ((name, root), (w, h)) in enumerate(zip(panels, sizes)):
        r, c = divmod(i, cols)
        x0, y0 = sum(col_w[:c]) + gap * c, sum(row_h[:r]) + gap * r
        layer = ET.SubElement(out, f'{{{SVG_NS}}}g', {'id': f'layer{i + 1}',
                                                      f'{{{INKSCAPE_NS}}}groupmode': 'layer',
                                                      f'{{{INKSCAPE_NS}}}label': name})
        if label_size:
            label = ET.SubElement(layer, f'{{{SVG_NS}}}text', {
                'x': f'{x0 + col_w[c] / 2:g}', 'y': f'{y0 + label_size:g}', 'text-anchor': 'middle',
                'style': f'font-size:{label_size:g}px;font-family:sans-serif'})
            label.text = name
        panel = ET.SubElement(layer, f'{{{SVG_NS}}}g', {
            'transform': f'translate({x0 + (col_w[c] - w) / 2:g} {y0 + label_h + (row_h[r] - label_h - h) / 2:g})'})
        prefix_ids(root, f'p{i + 1}_')
        panel.extend(child for child in root if child.tag != f'{{{SVG_NS}}}metadata')
    return ET.ElementTree(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cols', type=int, default=3, help='panels per row (default 3)')
    parser.add_argument('--gap', type=float, default=4, help='space between panels, pt (default 4)')
    parser.add_argument('--label-size', type=float, default=6,
                        help='trajectory-name font size, pt; 0 for no labels (default 6)')
    parser.add_argument('--out', type=Path, default=EXAMPLE_DIR / 'figures' / 'all_trajectories.svg',
                        help='output SVG (default example_inference/figures/all_trajectories.svg)')
    args = parser.parse_args()

    names = sorted(p.stem for p in (EXAMPLE_DIR / 'trajectories').glob('*.csv'))
    defaults = notebook_defaults()
    print('Notebook defaults:', {k: v.value for k, v in defaults.items()})
    panels = []
    for name in names:
        print(f'  {name}', flush=True)
        panels.append((name, heading_svg(name, defaults)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tree = tile(panels, args.cols, args.gap, args.label_size)
    ET.indent(tree)
    tree.write(args.out, xml_declaration=True, encoding='utf-8')
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
