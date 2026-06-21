import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import ndimage
from torchvision.transforms import Compose


def get_display_image(pil_image, preprocess):
    """Apply the preprocess pipeline up to (but excluding) ToTensor/Normalize,
    so the result lines up spatially with the RISE saliency maps but stays viewable."""
    display_transforms = [t for t in preprocess.transforms
                           if t.__class__.__name__ not in ("ToTensor", "Normalize", "MaybeToTensor")]
    display_image = Compose(display_transforms)(pil_image)
    return np.asarray(display_image).astype(np.float32) / 255.0


def saliency_entropy(sal):
    """Normalized Shannon entropy of a saliency map treated as a probability
    distribution over pixels. Returns a value in [0, 1]; values near 1 mean the
    map spreads its energy uniformly across the image (unfocused), values near 0
    mean it concentrates on a small region (focused)."""
    p = sal - sal.min()
    total = p.sum()
    if total <= 0:
        return 1.0
    p = (p / total).flatten()
    p = p[p > 0]
    ent = -np.sum(p * np.log(p))
    max_ent = np.log(p.size) if p.size > 1 else 1.0
    return ent / max_ent


def find_bounding_boxes(sal_norm, n_boxes='auto', box_threshold=0.5, auto_rel_thresh=0.3, min_area=4):
    """Find bounding boxes around the most focused/dense regions of a normalized
    saliency map.

    sal_norm: (H, W) array, normalized to [0, 1]
    n_boxes: int -> return exactly the n_boxes most focused regions, ranked by
        the total saliency mass within each region.
        'auto' -> automatically select all regions whose mass is at least
        `auto_rel_thresh` of the most focused region's mass (no fixed count).
    box_threshold: regions are found by binarizing sal_norm at this fraction of
        its max value, then taking connected components of the result.
    min_area: connected components smaller than this many pixels are discarded
        as noise.

    Returns a list of (x0, y0, x1, y1, score) tuples in pixel coordinates,
    sorted by score descending.
    """
    binary = sal_norm >= box_threshold * sal_norm.max()
    labeled, num_features = ndimage.label(binary)
    if num_features == 0:
        return []

    regions = []
    for label_id in range(1, num_features + 1):
        mask = labeled == label_id
        if mask.sum() < min_area:
            continue
        ys, xs = np.where(mask)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        score = float(sal_norm[mask].sum())
        regions.append((x0, y0, x1, y1, score))

    regions.sort(key=lambda r: r[4], reverse=True)
    if not regions:
        return []

    if n_boxes == 'auto':
        max_score = regions[0][4]
        regions = [r for r in regions if r[4] >= auto_rel_thresh * max_score]
    else:
        regions = regions[:n_boxes]

    return regions


def save_concept_saliency_maps(display_image, saliency, concepts, concept_presence, out_dir,
                                cmap='jet', alpha=0.5, n_boxes='auto', box_threshold=0.5,
                                auto_rel_thresh=0.3, box_color='lime'):
    """Save one overlayed saliency map per concept into out_dir, titled with the concept text.

    display_image: (H, W, 3) array in [0, 1]
    saliency: (num_concepts, H, W) array
    concepts: list of concept strings, aligned with saliency's first dim
    entropy_percentile: maps whose normalized entropy falls at or above this
        percentile (i.e. the most spread-out/unfocused maps) are skipped
    n_boxes: int or 'auto', passed to find_bounding_boxes to control how many
        of the most focused/dense regions get a bounding box drawn around them
    box_threshold, auto_rel_thresh: passed through to find_bounding_boxes
    box_color: matplotlib color used to draw the bounding boxes
    """
    os.makedirs(out_dir, exist_ok=True)
    saliency = np.asarray(saliency)

    cutoff = np.percentile([s.detach().cpu().numpy() for s in concept_presence.values()], 75)
    entropies = [saliency_entropy(sal) for sal in saliency]
    entropy_cutoff = np.percentile(entropies, 90)
    for i, (concept, sal) in enumerate(zip(concepts, saliency)):
        if concept_presence[concept] < cutoff:
            print(f"[Warining] {concept} similarity is too low {concept_presence[concept]:.3f}")
    
        if entropies[i] < entropy_cutoff:
            print(f"[Warining] {concept} saliency map is too unfocused (entropy {entropies[i]:.3f})")
            
        sal_norm = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
        boxes = find_bounding_boxes(sal_norm, n_boxes=n_boxes, box_threshold=box_threshold,
                                     auto_rel_thresh=auto_rel_thresh)

        safe_name = ''.join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in concept)
        safe_name = safe_name.strip().replace(' ', '_')

        fig, ax = plt.subplots(figsize=(5, 5.5))
        ax.imshow(display_image)
        ax.imshow(sal_norm, cmap=cmap, alpha=alpha)
        ax.axis('off')
        ax.set_title(concept, fontsize=14)
        out_path = os.path.join(out_dir, f'{i:02d}_{safe_name}.png')
        fig.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 5.5))
        ax.imshow(display_image)
        for x0, y0, x1, y1, score in boxes:
            rect = Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2,
                              edgecolor=box_color, facecolor='none')
            ax.add_patch(rect)
        ax.axis('off')
        ax.set_title(concept, fontsize=14)
        out_path_boxes = os.path.join(out_dir, f'{i:02d}_{safe_name}_boxes.png')
        fig.savefig(out_path_boxes, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
