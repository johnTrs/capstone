import base64
from io import BytesIO
from PIL import Image

def encode_image_to_base64(pixmap):
    return base64.b64encode(pixmap.pil_tobytes("png")).decode("utf-8")

def decode_base64_image(base64_image):
    base64_image = base64_image.split(",")[-1]
    image_data = base64.b64decode(base64_image)
    image = Image.open(BytesIO(image_data))
    return image

def quad_to_rect(quad_box):
    """
    Convert a single quadrilateral box to a rectangular bounding box.

    Args:
        quad_box (list): List of 8 coordinates [x1, y1, x2, y2, x3, y3, x4, y4] representing
                         the corners of a quadrilateral.

    Returns:
        list: List of 4 coordinates [x_min, y_min, x_max, y_max] representing the smallest enclosing rectangle.
    """
    # Extract coordinates
    x1, y1, x2, y2, x3, y3, x4, y4 = quad_box
    
    # Calculate the min and max coordinates to form the rectangular bounding box
    x_min = min(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    x_max = max(x1, x2, x3, x4)
    y_max = max(y1, y2, y3, y4)
    
    # Return the rectangular bounding box
    return [x_min, y_min, x_max, y_max]

def unnormalize_box(bbox, width, height):
     return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
     ]

def normalize_box(bbox, width, height):
    return [
        bbox[0] / width * 1000,  # x1
        bbox[1] / height * 1000, # y1
        bbox[2] / width * 1000,  # x2
        bbox[3] / height * 1000, # y2
    ]
