# Wasserstein Style Transfer

Proprietary implementation of Wasserstein Style Transfer (Kolkin et al., 2019) with support for Gaussian and GMM-based feature transport.

## Installation

Ensure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

(The environment is located at `/root/ENS/.venv` if running on the provided server).

## Usage

The main script is `main.py`. It takes a content image and one or more style images as input.

### Basic Command

```bash
python main.py <content_image_path> <style_image_path> [options]
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `content` | str | **Required** | Path to the content image. |
| `styles` | list[str] | **Required** | Path to one or more style images. Separate multiple paths with spaces. |
| `--out` | str | `output.jpg` | Path where the result will be saved. |
| `--alpha` | float | `0.5` | Interpolation factor (0.0 = content, 1.0 = style). |
| `--device` | str | `cuda` (if avail) | Device to run on (`cpu` or `cuda`). |
| `--style_weights` | list[float] | Equal | Weights for multiple style images (must sum to 1). |
| `--method` | str | `gaussian` | Transfer method: `gaussian` (fast, unimodal) or `gmm` (slower, multimodal). |
| `--K` | int | `5` | Number of GMM components (only used if `--method gmm`). |

## Examples

### 1. Simple Gaussian Transfer (Fast)
Standard style transfer using Gaussian statistics (mean/covariance).

```bash
python main.py images/content.jpg images/style.jpg --out result_gaussian.jpg --alpha 0.8
```

### 2. Gaussian Mixture Model (GMM) Transfer
Captures more complex style statistics (e.g. multiple distinct colors/textures) using a Mixture of Gaussians.
*Note: Slower (~5 mins) due to fitting and optimal transport calculation.*

```bash
python main.py images/content.jpg images/style.jpg --out result_gmm.jpg --method gmm --K 5
```

### 3. Multi-Style Transfer
Mixes two or more styles. The algorithm computes a specialized Wasserstein barycenter for the styles.

```bash
python main.py images/content.jpg images/style1.jpg images/style2.jpg --out result_mix.jpg --style_weights 0.7 0.3
```

## Implementation Details

- **Gaussian Method**: Uses a closed-form solution for the Monge map between Gaussians. For N=2 styles, uses a closed-form geodesic interpolation.
- **GMM Method**: Fits GMMs to features at 5 VGG layers. Uses an optimized GPU-accelerated Optimal Transport solver to map feature distributions.
