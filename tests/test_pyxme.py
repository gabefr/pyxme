"""
Test suite for pyxme pixel art converter
Run with: pytest tests/test_pyxme.py -v
"""

import pytest
import numpy as np
from PIL import Image
import tempfile
import os
from pathlib import Path

# Import functions from main module
from pyxme.main import (
    pixelate_image,
    adjust_brightness,
    adjust_contrast,
    adjust_saturation,
    apply_sharpen,
    apply_edge_enhancement,
    quantize_colors,
    snap_to_palette,
    apply_palette,
    constrain_image_size,
    process_image,
    COLOR_PALETTES,
    PRESETS,
)


# Fixtures
@pytest.fixture
def test_image():
    """Create a simple test image (100x100 red square)"""
    img = Image.new("RGB", (100, 100), color=(250, 0, 0))
    return img


@pytest.fixture
def gradient_image():
    """Create a gradient test image"""
    width, height = 200, 200
    array = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        array[i, :] = [int(i * 255 / height), 128, 255 - int(i * 255 / height)]
    return Image.fromarray(array)


@pytest.fixture
def temp_image_file(test_image):
    """Create a temporary image file"""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        test_image.save(f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def grayscale_image():
    """Create a grayscale test image"""
    img = Image.new("L", (100, 100), color=128)
    return img


@pytest.fixture
def rgba_image():
    """Create an RGBA test image with transparency"""
    img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    return img


@pytest.fixture
def palette_image():
    """Create a palette mode image"""
    img = Image.new("P", (100, 100))
    # Set a simple palette
    palette = []
    for i in range(256):
        palette.extend([i, i, i])
    img.putpalette(palette)
    return img


@pytest.fixture
def cmyk_image():
    """Create a CMYK test image"""
    img = Image.new("CMYK", (100, 100), color=(0, 100, 100, 0))
    return img


# Unit Tests - Image Processing Functions
class TestImageProcessing:
    """Test individual image processing functions"""

    def test_pixelate_image(self, test_image):
        """Test pixelation effect"""
        result = pixelate_image(test_image, pixel_size=10)
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_pixelate_preserves_dimensions(self, gradient_image):
        """Ensure pixelation doesn't change image dimensions"""
        original_size = gradient_image.size
        result = pixelate_image(gradient_image, pixel_size=8)
        assert result.size == original_size

    def test_adjust_brightness_increase(self, test_image):
        """Test brightness increase"""
        result = adjust_brightness(test_image, factor=1.5)
        result_array = np.array(result)
        original_array = np.array(test_image)
        assert np.mean(result_array) > np.mean(original_array)

    def test_adjust_brightness_decrease(self, test_image):
        """Test brightness decrease"""
        result = adjust_brightness(test_image, factor=0.5)
        result_array = np.array(result)
        original_array = np.array(test_image)
        assert np.mean(result_array) < np.mean(original_array)

    def test_adjust_contrast(self, gradient_image):
        """Test contrast adjustment"""
        result = adjust_contrast(gradient_image, factor=1.5)
        assert result.size == gradient_image.size
        assert result.mode == "RGB"

    def test_adjust_saturation(self, test_image):
        """Test saturation adjustment"""
        result = adjust_saturation(test_image, factor=1.5)
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_apply_sharpen(self, test_image):
        """Test sharpening filter"""
        result = apply_sharpen(test_image, factor=2.0)
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_edge_enhancement(self, gradient_image):
        """Test edge enhancement filter"""
        result = apply_edge_enhancement(gradient_image)
        assert result.size == gradient_image.size

    def test_quantize_colors_without_dither(self, gradient_image):
        """Test color quantization without dithering"""
        result = quantize_colors(gradient_image, colors=16, dither=False)
        assert result.mode == "RGB"
        # Check that colors are reduced
        unique_colors = len(np.unique(np.array(result).reshape(-1, 3), axis=0))
        assert unique_colors <= 16

    def test_quantize_colors_with_dither(self, gradient_image):
        """Test color quantization with dithering"""
        result = quantize_colors(gradient_image, colors=16, dither=True)
        assert result.mode == "RGB"

    def test_snap_to_palette(self, gradient_image):
        """Test aggressive palette snapping"""
        result = snap_to_palette(gradient_image, colors=8)
        assert result.mode == "RGB"
    
    def test_snap_to_palette_reduces_colors(self, gradient_image):
        """Test that clean palette actually reduces color count"""
        result = snap_to_palette(gradient_image, colors=8)
        result_array = np.array(result)
        unique_colors = len(np.unique(result_array.reshape(-1, 3), axis=0))
        # Should have at most 8 colors (might be less due to image content)
        assert unique_colors <= 8
    

# Unit Tests - Palette Application
class TestPaletteApplication:
    """Test color palette application"""

    def test_apply_palette_nes(self, test_image):
        """Test NES palette application"""
        result = apply_palette(test_image, "nes")
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_apply_palette_gameboy(self, test_image):
        """Test Game Boy palette application"""
        result = apply_palette(test_image, "gameboy")
        assert result.size == test_image.size
        # Game Boy has 4 colors
        unique_colors = len(np.unique(np.array(result).reshape(-1, 3), axis=0))
        assert unique_colors <= 4

    def test_apply_palette_pico8(self, gradient_image):
        """Test PICO-8 palette application"""
        result = apply_palette(gradient_image, "pico8")
        assert result.size == gradient_image.size
        unique_colors = len(np.unique(np.array(result).reshape(-1, 3), axis=0))
        assert unique_colors <= 16

    @pytest.mark.parametrize("palette_name", list(COLOR_PALETTES.keys()))
    def test_all_palettes(self, test_image, palette_name):
        """Test all available palettes"""
        result = apply_palette(test_image, palette_name)
        assert result.size == test_image.size
        assert result.mode == "RGB"


# Unit Tests - Image Size Constraint
class TestImageConstraint:
    """Test image size constraint functionality"""

    def test_constrain_large_image(self):
        """Test constraining oversized image"""
        large_image = Image.new("RGB", (4000, 3000), color=(128, 128, 128))
        result = constrain_image_size(large_image, max_dimension=2048)
        assert max(result.size) == 2048
        # Check aspect ratio preserved
        original_ratio = large_image.size[0] / large_image.size[1]
        result_ratio = result.size[0] / result.size[1]
        assert abs(original_ratio - result_ratio) < 0.01

    def test_constrain_small_image_unchanged(self, test_image):
        """Test that small images are not resized"""
        result = constrain_image_size(test_image, max_dimension=2048)
        assert result.size == test_image.size

    def test_constrain_portrait_image(self):
        """Test constraining portrait orientation"""
        portrait = Image.new("RGB", (1000, 2000), color=(128, 128, 128))
        result = constrain_image_size(portrait, max_dimension=1024)
        assert max(result.size) == 1024
        assert result.size[1] == 1024  # Height should be constrained

    def test_constrain_landscape_image(self):
        """Test constraining landscape orientation"""
        landscape = Image.new("RGB", (3000, 1000), color=(128, 128, 128))
        result = constrain_image_size(landscape, max_dimension=1024)
        assert max(result.size) == 1024
        assert result.size[0] == 1024  # Width should be constrained


# Integration Tests - Full Processing Pipeline
class TestProcessingPipeline:
    """Test complete image processing pipeline"""

    def test_process_image_basic(self, temp_image_file):
        """Test basic image processing"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as out:
            output_path = out.name

        try:
            process_image(
                input_path=temp_image_file,
                output_path=output_path,
                pixel_size=10,
                colors=16,
                brightness=1.0,
                contrast=1.0,
                saturation=1.0,
                sharpen=1.5,
                dither=False,
                edge_enhance=False,
                clean_palette=False,
                palette_name=None,
                max_size=2048,
            )
            assert os.path.exists(output_path)
            result = Image.open(output_path)
            assert result.mode == "RGB"
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_process_image_with_palette(self, temp_image_file):
        """Test processing with color palette"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as out:
            output_path = out.name

        try:
            process_image(
                input_path=temp_image_file,
                output_path=output_path,
                pixel_size=12,
                colors=16,
                brightness=0.9,
                contrast=1.3,
                saturation=1.4,
                sharpen=2.0,
                dither=True,
                edge_enhance=False,
                clean_palette=False,
                palette_name="nes",
                max_size=2048,
            )
            assert os.path.exists(output_path)
            result = Image.open(output_path)
            assert result.mode == "RGB"
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_process_image_with_edge_enhance(self, temp_image_file):
        """Test processing with edge enhancement"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as out:
            output_path = out.name

        try:
            process_image(
                input_path=temp_image_file,
                output_path=output_path,
                pixel_size=8,
                colors=32,
                brightness=1.0,
                contrast=1.2,
                saturation=1.3,
                sharpen=2.5,
                dither=False,
                edge_enhance=True,
                clean_palette=False,
                palette_name=None,
                max_size=2048,
            )
            assert os.path.exists(output_path)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_process_image_with_clean_palette(self, temp_image_file):
        """Test processing with clean palette mode"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as out:
            output_path = out.name

        try:
            process_image(
                input_path=temp_image_file,
                output_path=output_path,
                pixel_size=10,
                colors=16,
                brightness=1.0,
                contrast=1.0,
                saturation=1.0,
                sharpen=1.5,
                dither=False,
                edge_enhance=False,
                clean_palette=True,
                palette_name=None,
                max_size=2048,
            )
            assert os.path.exists(output_path)
            result = Image.open(output_path)
            assert result.mode == "RGB"
            
            # Check that colors are actually reduced
            result_array = np.array(result)
            unique_colors = len(np.unique(result_array.reshape(-1, 3), axis=0))
            assert unique_colors <= 16
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


    @pytest.mark.parametrize("preset_name", list(PRESETS.keys()))
    def test_all_presets(self, temp_image_file, preset_name):
        """Test all preset configurations"""
        preset = PRESETS[preset_name]
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as out:
            output_path = out.name

        try:
            process_image(
                input_path=temp_image_file,
                output_path=output_path,
                pixel_size=preset["pixel_size"],
                colors=preset["colors"],
                brightness=preset["brightness"],
                contrast=preset["contrast"],
                saturation=preset["saturation"],
                sharpen=preset["sharpen"],
                dither=preset["dither"],
                edge_enhance=False,
                clean_palette=False,
                palette_name=preset["palette"],
                max_size=2048,
            )
            assert os.path.exists(output_path)
            result = Image.open(output_path)
            assert result.mode == "RGB"
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_process_large_image_memory_safe(self):
        """Test processing large image with size constraint"""
        large_image = Image.new("RGB", (4000, 4000), color=(100, 150, 200))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as inp:
            large_image.save(inp.name)
            input_path = inp.name

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as out:
            output_path = out.name

        try:
            process_image(
                input_path=input_path,
                output_path=output_path,
                pixel_size=10,
                colors=16,
                brightness=1.0,
                contrast=1.0,
                saturation=1.0,
                sharpen=1.5,
                dither=False,
                edge_enhance=False,
                clean_palette=False,
                palette_name=None,
                max_size=1024,
            )
            assert os.path.exists(output_path)
            result = Image.open(output_path)
            # Should be constrained to max 1024px
            assert max(result.size) <= 1024
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)


# Validation Tests
class TestValidation:
    """Test input validation and edge cases"""

    def test_invalid_pixel_size(self, test_image):
        """Test behavior with invalid pixel size"""
        # Should handle gracefully or raise appropriate error
        with pytest.raises(Exception):
            pixelate_image(test_image, pixel_size=0)

    def test_extreme_brightness(self, test_image):
        """Test extreme brightness values"""
        result = adjust_brightness(test_image, factor=10.0)
        assert result.mode == "RGB"

    def test_zero_colors(self, gradient_image):
        """Test with minimum colors"""
        result = quantize_colors(gradient_image, colors=2, dither=False)
        unique_colors = len(np.unique(np.array(result).reshape(-1, 3), axis=0))
        assert unique_colors <= 2

    def test_nonexistent_input_file(self):
        """Test processing with nonexistent file"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as out:
            output_path = out.name

        try:
            with pytest.raises(SystemExit):
                process_image(
                    input_path="nonexistent_file.jpg",
                    output_path=output_path,
                    pixel_size=10,
                    colors=16,
                    brightness=1.0,
                    contrast=1.0,
                    saturation=1.0,
                    sharpen=1.5,
                    dither=False,
                    edge_enhance=False,
                    clean_palette=False,
                    palette_name=None,
                    max_size=2048,
                )
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_corrupted_image_file(self):
        """Test processing with corrupted image file"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False, mode='wb') as inp:
            # Write invalid image data
            inp.write(b"This is not a valid image file")
            input_path = inp.name

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as out:
            output_path = out.name

        try:
            with pytest.raises(SystemExit):
                process_image(
                    input_path=input_path,
                    output_path=output_path,
                    pixel_size=10,
                    colors=16,
                    brightness=1.0,
                    contrast=1.0,
                    saturation=1.0,
                    sharpen=1.5,
                    dither=False,
                    edge_enhance=False,
                    clean_palette=False,
                    palette_name=None,
                    max_size=2048,
                )
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_empty_file(self):
        """Test processing with empty file"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False, mode='wb') as inp:
            # Create empty file
            input_path = inp.name

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as out:
            output_path = out.name

        try:
            with pytest.raises(SystemExit):
                process_image(
                    input_path=input_path,
                    output_path=output_path,
                    pixel_size=10,
                    colors=16,
                    brightness=1.0,
                    contrast=1.0,
                    saturation=1.0,
                    sharpen=1.5,
                    dither=False,
                    edge_enhance=False,
                    clean_palette=False,
                    palette_name=None,
                    max_size=2048,
                )
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)


# Non-RGB Image Mode Tests
class TestImageModes:
    """Test handling of different image modes"""

    def test_grayscale_image_conversion(self, grayscale_image):
        """Test that grayscale images are converted to RGB"""
        assert grayscale_image.mode == "L"
        result = pixelate_image(grayscale_image.convert("RGB"), pixel_size=10)
        assert result.mode == "RGB"

    def test_rgba_image_conversion(self, rgba_image):
        """Test that RGBA images are converted to RGB"""
        assert rgba_image.mode == "RGBA"
        rgb_image = rgba_image.convert("RGB")
        result = pixelate_image(rgb_image, pixel_size=10)
        assert result.mode == "RGB"

    def test_palette_image_conversion(self, palette_image):
        """Test that palette mode images are converted to RGB"""
        assert palette_image.mode == "P"
        rgb_image = palette_image.convert("RGB")
        result = pixelate_image(rgb_image, pixel_size=10)
        assert result.mode == "RGB"

    def test_cmyk_image_conversion(self, cmyk_image):
        """Test that CMYK images are converted to RGB"""
        assert cmyk_image.mode == "CMYK"
        rgb_image = cmyk_image.convert("RGB")
        result = pixelate_image(rgb_image, pixel_size=10)
        assert result.mode == "RGB"

    def test_process_grayscale_file(self, grayscale_image):
        """Test full processing pipeline with grayscale image"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as inp:
            grayscale_image.save(inp.name)
            input_path = inp.name

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as out:
            output_path = out.name

        try:
            process_image(
                input_path=input_path,
                output_path=output_path,
                pixel_size=10,
                colors=16,
                brightness=1.0,
                contrast=1.0,
                saturation=1.0,
                sharpen=1.5,
                dither=False,
                edge_enhance=False,
                clean_palette=False,
                palette_name=None,
                max_size=2048,
            )
            assert os.path.exists(output_path)
            result = Image.open(output_path)
            assert result.mode == "RGB"
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_process_rgba_file(self, rgba_image):
        """Test full processing pipeline with RGBA image"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as inp:
            rgba_image.save(inp.name)
            input_path = inp.name

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as out:
            output_path = out.name

        try:
            process_image(
                input_path=input_path,
                output_path=output_path,
                pixel_size=10,
                colors=16,
                brightness=1.0,
                contrast=1.0,
                saturation=1.0,
                sharpen=1.5,
                dither=False,
                edge_enhance=False,
                clean_palette=False,
                palette_name=None,
                max_size=2048,
            )
            assert os.path.exists(output_path)
            result = Image.open(output_path)
            assert result.mode == "RGB"
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_process_palette_file(self, palette_image):
        """Test full processing pipeline with palette mode image"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as inp:
            palette_image.save(inp.name)
            input_path = inp.name

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as out:
            output_path = out.name

        try:
            process_image(
                input_path=input_path,
                output_path=output_path,
                pixel_size=10,
                colors=16,
                brightness=1.0,
                contrast=1.0,
                saturation=1.0,
                sharpen=1.5,
                dither=False,
                edge_enhance=False,
                clean_palette=False,
                palette_name=None,
                max_size=2048,
            )
            assert os.path.exists(output_path)
            result = Image.open(output_path)
            assert result.mode == "RGB"
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    @pytest.mark.parametrize(
        "mode,color",
        [
            ("L", 128),  # Grayscale
            ("LA", (128, 255)),  # Grayscale with alpha
            ("RGB", (255, 0, 0)),  # RGB
            ("RGBA", (255, 0, 0, 128)),  # RGB with alpha
            ("P", None),  # Palette (need special handling)
        ],
    )
    def test_all_image_modes(self, mode, color):
        """Test processing with various image modes"""
        if mode == "P":
            img = Image.new("P", (50, 50))
            palette = list(range(256)) * 3
            img.putpalette(palette)
        else:
            img = Image.new(mode, (50, 50), color=color)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as inp:
            img.save(inp.name)
            input_path = inp.name

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as out:
            output_path = out.name

        try:
            process_image(
                input_path=input_path,
                output_path=output_path,
                pixel_size=5,
                colors=8,
                brightness=1.0,
                contrast=1.0,
                saturation=1.0,
                sharpen=1.0,
                dither=False,
                edge_enhance=False,
                clean_palette=False,
                palette_name=None,
                max_size=2048,
            )
            assert os.path.exists(output_path)
            result = Image.open(output_path)
            assert result.mode == "RGB"
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    @pytest.mark.parametrize(
        "mode,color,dither,edge_enhance,clean_palette",
        [
            # Test all combinations with RGB mode
            ("RGB", (255, 0, 0), False, False, False),
            ("RGB", (255, 0, 0), True, False, False),
            ("RGB", (255, 0, 0), False, True, False),
            ("RGB", (255, 0, 0), False, False, True),
            ("RGB", (255, 0, 0), True, True, False),
            ("RGB", (255, 0, 0), True, False, True),
            ("RGB", (255, 0, 0), False, True, True),
            ("RGB", (255, 0, 0), True, True, True),
            # Test combinations with grayscale
            ("L", 128, False, False, False),
            ("L", 128, True, False, False),
            ("L", 128, False, True, True),
            ("L", 128, True, True, True),
            # Test combinations with RGBA
            ("RGBA", (255, 0, 0, 128), False, False, False),
            ("RGBA", (255, 0, 0, 128), True, True, False),
            ("RGBA", (255, 0, 0, 128), False, False, True),
            ("RGBA", (255, 0, 0, 128), True, True, True),
        ],
    )
    def test_all_option_combinations(self, mode, color, dither, edge_enhance, clean_palette):
        """Test processing with all option combinations across different image modes"""
        if mode == "P":
            img = Image.new("P", (50, 50))
            palette_data = list(range(256)) * 3
            img.putpalette(palette_data)
        else:
            img = Image.new(mode, (50, 50), color=color)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as inp:
            img.save(inp.name)
            input_path = inp.name

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as out:
            output_path = out.name

        try:
            process_image(
                input_path=input_path,
                output_path=output_path,
                pixel_size=8,
                colors=8,
                brightness=1.0,
                contrast=1.0,
                saturation=1.0,
                sharpen=1.5,
                dither=dither,
                edge_enhance=edge_enhance,
                clean_palette=clean_palette,
                palette_name=None,
                max_size=2048,
            )
            assert os.path.exists(output_path)
            result = Image.open(output_path)
            assert result.mode == "RGB"
            
            # Verify color count if clean_palette is used
            if clean_palette:
                result_array = np.array(result)
                unique_colors = len(np.unique(result_array.reshape(-1, 3), axis=0))
                assert unique_colors <= 8
                
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    @pytest.mark.parametrize(
        "palette_name,dither,edge_enhance",
        [
            ("nes", False, False),
            ("nes", True, False),
            ("nes", False, True),
            ("nes", True, True),
            ("pico8", False, False),
            ("pico8", True, True),
            ("commodore64", False, False),
            ("commodore64", True, True),
            ("cga", True, False),
            ("apollo", False, True),
        ],
    )
    def test_palette_with_options(self, gradient_image, palette_name, dither, edge_enhance):
        """Test all palettes combined with different processing options"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as inp:
            gradient_image.save(inp.name)
            input_path = inp.name

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as out:
            output_path = out.name

        try:
            process_image(
                input_path=input_path,
                output_path=output_path,
                pixel_size=10,
                colors=16,  # This is ignored when palette_name is set
                brightness=1.0,
                contrast=1.2,
                saturation=1.3,
                sharpen=2.0,
                dither=dither,
                edge_enhance=edge_enhance,
                clean_palette=False,  # clean_palette not used with palette_name
                palette_name=palette_name,
                max_size=2048,
            )
            assert os.path.exists(output_path)
            result = Image.open(output_path)
            assert result.mode == "RGB"
            
            # Verify colors match palette
            result_array = np.array(result)
            unique_colors = len(np.unique(result_array.reshape(-1, 3), axis=0))
            palette_size = len(COLOR_PALETTES[palette_name])
            assert unique_colors <= palette_size
            
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)


# Performance Tests
class TestPerformance:
    """Test performance characteristics"""

    def test_chunked_palette_processing(self, gradient_image):
        """Test that chunked processing produces consistent results"""
        # Apply palette with different chunk sizes (if exposed)
        result1 = apply_palette(gradient_image, "pico8")
        result2 = apply_palette(gradient_image, "pico8")
        
        # Results should be identical
        np.testing.assert_array_equal(np.array(result1), np.array(result2))

    def test_processing_speed_baseline(self, test_image):
        """Baseline performance test"""
        import time
        start = time.time()
        result = pixelate_image(test_image, pixel_size=10)
        elapsed = time.time() - start
        # Should complete in reasonable time
        assert elapsed < 1.0  # Less than 1 second for small image


# Data Validation Tests
class TestDataValidation:
    """Test color palette and preset data structures"""

    def test_all_palettes_valid(self):
        """Ensure all palettes have valid RGB values"""
        for name, palette in COLOR_PALETTES.items():
            assert len(palette) > 0, f"Palette {name} is empty"
            for color in palette:
                assert len(color) == 3, f"Color in {name} is not RGB"
                assert all(0 <= c <= 255 for c in color), f"Invalid color value in {name}"

    def test_all_presets_valid(self):
        """Ensure all presets have required fields"""
        required_fields = [
            "pixel_size",
            "colors",
            "brightness",
            "contrast",
            "saturation",
            "sharpen",
            "dither",
            "palette",
            "description",
        ]
        for name, preset in PRESETS.items():
            for field in required_fields:
                assert field in preset, f"Preset {name} missing field {field}"

    def test_preset_palette_references_valid(self):
        """Ensure preset palette references exist"""
        for name, preset in PRESETS.items():
            palette_name = preset["palette"]
            if palette_name is not None:
                assert (
                    palette_name in COLOR_PALETTES
                ), f"Preset {name} references invalid palette {palette_name}"


