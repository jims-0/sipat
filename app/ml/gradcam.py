"""
Grad-CAM (Gradient-weighted Class Activation Mapping)
Visual explanation for CNN predictions
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def generate_gradcam(input_image_path, output_image_path, class_idx):
    """
    Generate Grad-CAM heatmap overlay on the input image

    Args:
        input_image_path: Path to original leaf image
        output_image_path: Path to save heatmap overlay
        class_idx: Predicted class index

    In production, this would:
    1. Load the full Keras/TF model (not TFLite)
    2. Get activations from the last conv layer
    3. Compute gradients w.r.t. predicted class
    4. Weight activations by gradients
    5. Generate heatmap and overlay on original image
    """
    try:
        # Load original image
        img = Image.open(input_image_path)
        img = img.convert('RGB')
        width, height = img.size

        # MOCK GRAD-CAM (replace with real implementation)
        # For now, create a simple heatmap overlay to demonstrate the concept

        # Create a mock heatmap (in production, this comes from the CNN)
        # Simulate attention on center-left region (where disease typically shows)
        heatmap = np.zeros((height, width), dtype=np.float32)

        # Create a gradient that's hot in the center-left region
        for y in range(height):
            for x in range(width):
                # Distance from center-left focus point
                focus_x = width * 0.35
                focus_y = height * 0.5
                dist = np.sqrt((x - focus_x)**2 + (y - focus_y)**2)
                max_dist = np.sqrt(width**2 + height**2) / 2
                heatmap[y, x] = max(0, 1 - (dist / max_dist * 1.5))

        # Normalize heatmap to [0, 255]
        heatmap = (heatmap * 255).astype(np.uint8)

        # Convert to RGB heatmap (red = hot, blue = cold)
        heatmap_colored = Image.new('RGB', (width, height))
        pixels = heatmap_colored.load()

        for y in range(height):
            for x in range(width):
                intensity = heatmap[y, x]
                # Red-yellow-green colormap
                if intensity > 200:
                    r, g, b = 255, 0, 0  # Red (high attention)
                elif intensity > 150:
                    r, g, b = 255, 165, 0  # Orange
                elif intensity > 100:
                    r, g, b = 255, 255, 0  # Yellow
                elif intensity > 50:
                    r, g, b = 0, 255, 0  # Green
                else:
                    r, g, b = 0, 0, 255  # Blue (low attention)

                pixels[x, y] = (r, g, b)

        # Blend heatmap with original image (60% original, 40% heatmap)
        blended = Image.blend(img, heatmap_colored, alpha=0.4)

        # Add legend
        draw = ImageDraw.Draw(blended)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

        legend_text = "Red = High attention | Blue = Low attention"
        draw.text((10, height - 30), legend_text, fill=(255, 255, 255), font=font)

        # Save
        blended.save(output_image_path, quality=85, optimize=True)

        return True

    except Exception as e:
        print(f"Grad-CAM generation error: {str(e)}")
        # Fallback: just copy the original image
        img = Image.open(input_image_path)
        img.save(output_image_path)
        return False
