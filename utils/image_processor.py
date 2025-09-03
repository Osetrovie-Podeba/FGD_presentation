import numpy as np
from PIL import Image
import io


def preprocess_image(image_data: bytes, target_width: int, target_height: int) -> np.ndarray:
    """
    Preprocess image to match your model's expected input format.
    Based on your config: 384x250 with rescaling to 0-1.
    """
    # Load image from bytes
    image = Image.open(io.BytesIO(image_data))

    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize to target dimensions (384x250)
    image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    # Convert to numpy array
    image_array = np.array(image, dtype=np.float32)

    # Rescale to 0-1 (matching your model's rescaling layer)
    image_array = image_array / 255.0

    return image_array
