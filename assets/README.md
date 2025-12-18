# PyRADE Visual Assets

This directory contains visual assets for the README and documentation.

## Required Images

### 1. `logo.png`
- **Dimensions:** 512x512 pixels
- **Format:** PNG with transparency
- **Description:** PyRADE hexagon logo for hero section
- **Usage:** README hero section

### 2. `convergence.png`
- **Dimensions:** 800x600 pixels recommended
- **Format:** PNG or SVG
- **Description:** Convergence behavior plot showing how algorithms converge over iterations
- **Usage:** Visual Results section
- **Suggestion:** Export from paper results or generate with visualization module

### 3. `comparison.png`
- **Dimensions:** 800x600 pixels recommended
- **Format:** PNG or SVG
- **Description:** Performance comparison chart across benchmark functions
- **Usage:** Visual Results section
- **Suggestion:** Export from paper results showing algorithm comparison

### 4. `architecture.png` (Optional)
- **Dimensions:** 1200x800 pixels recommended
- **Format:** PNG or SVG
- **Description:** Architecture diagram showing PyRADE's modular design
- **Usage:** Architecture section
- **Note:** Currently using docs/image.png

### 5. `banner.png` (Optional)
- **Dimensions:** 1200x630 pixels (OpenGraph standard)
- **Format:** PNG
- **Description:** Social media preview banner
- **Usage:** GitHub social preview, sharing on social media

## Generating Images from Paper Results

You can generate convergence and comparison plots using PyRADE's visualization module:

```python
from pyrade.visualization import plot_convergence, plot_comparison
import matplotlib.pyplot as plt

# Generate convergence plot
# ... your code to run algorithms and collect history ...

# Generate comparison plot
# ... your code to compare multiple algorithms ...

# Save figures
plt.savefig('assets/convergence.png', dpi=150, bbox_inches='tight')
plt.savefig('assets/comparison.png', dpi=150, bbox_inches='tight')
```

## Quick Generation Script

Run the paper results generation:

```bash
python generate_paper_results.py
```

Then export the best figures to this assets folder.

## Current Status

- [ ] logo.png
- [ ] convergence.png
- [ ] comparison.png
- [ ] architecture.png
- [ ] banner.png

## Notes

- Use high-resolution images (at least 150 DPI for print quality)
- Optimize PNG files for web (use tools like `optipng` or `tinypng`)
- Consider SVG for diagrams (better scalability)
- Ensure images are readable in both light and dark themes
