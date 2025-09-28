import torch
from PIL import Image
from torchvision import transforms as TF


def load_and_preprocess_images(image_path_list, mode="crop"):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mode (str, optional): Preprocessing mode:
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.
                             - "original": Maintains original resolution as much as possible, crops excess to make dimensions divisible by 14 (patch size)
                             - "gt": Keeps original resolution without any modifications related to patch size
    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)
        int: The final width of the preprocessed image.
        int: The final height of the preprocessed image.

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="original": Keeps resolution as close to original as possible. For each dimension (width/height), 
          finds the largest value ≤ original size that is divisible by 14, then center-crops to that size.
        - All modes except "gt" ensure dimensions are divisible by 14 for compatibility with model requirements (patch size=14)
        - When mode="gt": Maintains original resolution without any cropping or resizing related to patch size
        - **Added functionality:** For modes "original" and "gt", if the image width is greater than its height,
          the image will be rotated 90 degrees counter-clockwise to ensure a portrait orientation.
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad", "original", "gt"]:
        raise ValueError("Mode must be 'crop', 'pad', 'original', or 'gt'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()

    # Process all images
    for image_path in image_path_list:
        # Open image
        img = Image.open(image_path)

        # Handle alpha channel by blending with white background
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        img = img.convert("RGB")

        width, height = img.size

        # --- New logic: Rotate image for "original" or "gt" modes if width > height ---
        if mode in ["original", "gt"] and width > height:
            img = img.transpose(Image.Transpose.ROTATE_90)
            width, height = img.size
        # --- End of new logic ---

        if mode == "gt":
            # Keep original resolution without any modifications for patch size
            img_tensor = to_tensor(img)
            
        elif mode == "original":
            # Calculate target dimensions: largest values ≤ original that are divisible by 14
            target_width = (width // 14) * 14
            target_height = (height // 14) * 14

            # Center crop to target dimensions if needed
            if width != target_width or height != target_height:
                # Calculate crop coordinates
                left = (width - target_width) // 2
                top = (height - target_height) // 2
                right = left + target_width
                bottom = top + target_height
                img = img.crop((left, top, right, bottom))

            # Convert to tensor (no resizing,保持原始分辨率)
            img_tensor = to_tensor(img)

        else:  # Handle "crop" and "pad" modes using original logic
            target_size = 518
            if mode == "pad":
                # Make largest dimension 518px while maintaining aspect ratio
                if width >= height:
                    new_width = target_size
                    new_height = round(height * (new_width / width) / 14) * 14
                else:
                    new_height = target_size
                    new_width = round(width * (new_height / height) / 14) * 14
            else:  # mode == "crop"
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14

            # Resize
            img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
            img_tensor = to_tensor(img)

            # Crop in "crop" mode if needed
            if mode == "crop" and new_height > target_size:
                start_y = (new_height - target_size) // 2
                img_tensor = img_tensor[:, start_y: start_y + target_size, :]

            # Pad in "pad" mode if needed
            if mode == "pad":
                h_padding = target_size - img_tensor.shape[1]
                w_padding = target_size - img_tensor.shape[2]
                if h_padding > 0 or w_padding > 0:
                    pad_top = h_padding // 2
                    pad_bottom = h_padding - pad_top
                    pad_left = w_padding // 2
                    pad_right = w_padding - pad_left
                    img_tensor = torch.nn.functional.pad(
                        img_tensor, (pad_left, pad_right, pad_top, pad_bottom), 
                        mode="constant", value=1.0
                    )

        shapes.add((img_tensor.shape[1], img_tensor.shape[2]))
        images.append(img_tensor)

    # Handle different shapes across images (pad to max dimensions)
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        padded_images = []
        for img_tensor in images:
            h_padding = max_height - img_tensor.shape[1]
            w_padding = max_width - img_tensor.shape[2]
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                img_tensor = torch.nn.functional.pad(
                    img_tensor, (pad_left, pad_right, pad_top, pad_bottom), 
                    mode="constant", value=1.0
                )
            padded_images.append(img_tensor)
        images = padded_images

    # Stack into batch tensor
    images = torch.stack(images)

    # Ensure correct shape for single image
    if len(image_path_list) == 1 and images.dim() == 3:
        images = images.unsqueeze(0)

    return images, width, height