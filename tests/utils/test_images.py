import base64
from unittest.mock import Mock, patch

import pytest

# Import the functions to be tested
from spark_instructor.utils.image import (
    convert_image_url_to_image_block_param,
    fetch_and_encode_image,
    get_media_type,
    is_base64_encoded,
)


# Test is_base64_encoded function
def test_is_base64_encoded(valid_base64):
    invalid_base64 = "https://example.com/image.jpg"

    assert is_base64_encoded(valid_base64)
    assert not is_base64_encoded(invalid_base64)


# Test fetch_and_encode_image function
@patch("httpx.get")
def test_fetch_and_encode_image(mock_get):
    mock_response = Mock()
    mock_response.content = b"test image content"
    mock_get.return_value = mock_response

    result = fetch_and_encode_image("https://example.com/image.jpg")
    expected = base64.b64encode(b"test image content").decode("utf-8")

    assert result == expected
    mock_get.assert_called_once_with("https://example.com/image.jpg")


# Test get_media_type function
def test_get_media_type():
    assert get_media_type("image.jpg") == "image/jpeg"
    assert get_media_type("image.jpeg") == "image/jpeg"
    assert get_media_type("image.png") == "image/png"
    assert get_media_type("image.gif") == "image/gif"
    assert get_media_type("image.webp") == "image/webp"

    with pytest.raises(ValueError, match="Unsupported image format"):
        get_media_type("image.bmp")


# Test convert_image_url_to_image_block_param function
@patch("spark_instructor.utils.image.fetch_and_encode_image")
def test_convert_image_url_to_image_block_param(mock_fetch, valid_base64):
    # Test with a regular URL
    mock_fetch.return_value = "encoded_image_data"
    image_url = {"url": "https://example.com/image.jpg"}
    result = convert_image_url_to_image_block_param(image_url)

    assert result == {
        "source": {"data": "encoded_image_data", "media_type": "image/jpeg", "type": "base64"},
        "type": "image",
    }

    image_url = {"url": valid_base64}
    result = convert_image_url_to_image_block_param(image_url)

    assert result == {
        "source": {
            "data": valid_base64.split(",")[-1],
            "media_type": "image/jpeg",
            "type": "base64",
        },
        "type": "image",
    }

    # Test with unsupported media type
    with pytest.raises(ValueError, match="Unsupported image format"):
        convert_image_url_to_image_block_param({"url": "data:image/bmp;base64,abc123"})


# Test error handling
def test_error_handling():
    with pytest.warns(UserWarning, match="Base64 encoding error encountered"):
        is_base64_encoded("data:image/jpeg;base64,invalid_base64")

    with pytest.raises(ValueError, match="Unsupported image format"):
        get_media_type("image.tiff")
