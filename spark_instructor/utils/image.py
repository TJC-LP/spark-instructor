"""Utilities for images."""

import base64
import re
import warnings
from typing import Literal, cast

import httpx
from anthropic.types.image_block_param import ImageBlockParam, Source
from openai.types.chat.chat_completion_content_part_image_param import ImageURL


def is_base64_encoded(url: str) -> bool:
    """Check if the given URL is already base64 encoded."""
    try:
        if url.startswith("data:image/"):
            base64_part = url.split(",")[1]
            base64.b64decode(base64_part)
            return True
    except Exception as e:
        warnings.warn(f"Base64 encoding error encountered: {e}")
    return False


def fetch_and_encode_image(url: str) -> str:
    """Fetch the image from the URL and encode it in base64."""
    response = httpx.get(url)
    response.raise_for_status()
    image_data = response.content
    encoded_image = base64.b64encode(image_data).decode("utf-8")
    return encoded_image


def get_media_type(url: str) -> Literal["image/jpeg", "image/png", "image/gif", "image/webp"]:
    """Determine the media type of the image based on its extension using regex."""
    jpeg_pattern = r".(jpe?g)"
    png_pattern = r".png"
    gif_pattern = r".gif"
    webp_pattern = r".webp"

    if re.search(jpeg_pattern, url, re.IGNORECASE):
        return "image/jpeg"
    elif re.search(png_pattern, url, re.IGNORECASE):
        return "image/png"
    elif re.search(gif_pattern, url, re.IGNORECASE):
        return "image/gif"
    elif re.search(webp_pattern, url, re.IGNORECASE):
        return "image/webp"
    else:
        raise ValueError("Unsupported image format")


def convert_image_url_to_image_block_param(image_url: ImageURL) -> ImageBlockParam:
    """Convert an ImageURL to an ImageBlockParam."""
    url = image_url["url"]

    if is_base64_encoded(url):
        # Extract the base64 part from the data URL
        base64_data = url.split(",")[1]
        media_type = cast(
            Literal["image/jpeg", "image/png", "image/gif", "image/webp"], url.split(";")[0].split(":")[1]
        )
        assert media_type in {
            "image/jpeg",
            "image/png",
            "image/gif",
            "image/webp",
        }, f"Unsupported media type {media_type}"
    else:
        base64_data = fetch_and_encode_image(url)
        media_type = get_media_type(url)

    source: Source = {"data": base64_data, "media_type": media_type, "type": "base64"}

    image_block_param: ImageBlockParam = {"source": source, "type": "image"}

    return image_block_param
