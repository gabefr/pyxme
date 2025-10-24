import sys
import argparse
from importlib.metadata import version

from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from scipy.spatial.distance import cdist

__version__ = version("pyxme")

BANNER = (
    "██████  ██    ██ ██   ██ ███    ███ ███████\n"
    "██   ██  ██  ██   ██ ██  ████  ████ ██     \n"
    "██████    ████     ███   ██ ████ ██ █████  \n"
    "██         ██     ██ ██  ██  ██  ██ ██     \n"
    "██         ██    ██   ██ ██      ██ ███████\n"
    f"version: {__version__}"
)

# Predefined retro color palettes (RGB values)
COLOR_PALETTES = {
    "nes": [
        (124, 124, 124),
        (0, 0, 252),
        (0, 0, 188),
        (68, 40, 188),
        (148, 0, 132),
        (168, 0, 32),
        (168, 16, 0),
        (136, 20, 0),
        (80, 48, 0),
        (0, 120, 0),
        (0, 104, 0),
        (0, 88, 0),
        (0, 64, 88),
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
        (188, 188, 188),
        (0, 120, 248),
        (0, 88, 248),
        (104, 68, 252),
        (216, 0, 204),
        (228, 0, 88),
        (248, 56, 0),
        (228, 92, 16),
        (172, 124, 0),
        (0, 184, 0),
        (0, 168, 0),
        (0, 168, 68),
        (0, 136, 136),
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
        (248, 248, 248),
        (60, 188, 252),
        (104, 136, 252),
        (152, 120, 248),
        (248, 120, 248),
        (248, 88, 152),
        (248, 120, 88),
        (252, 160, 68),
        (248, 184, 0),
        (184, 248, 24),
        (88, 216, 84),
        (88, 248, 152),
        (0, 232, 216),
        (120, 120, 120),
        (0, 0, 0),
        (0, 0, 0),
        (252, 252, 252),
        (164, 228, 252),
        (184, 184, 248),
        (216, 184, 248),
        (248, 184, 248),
        (248, 164, 192),
        (240, 208, 176),
        (252, 224, 168),
        (248, 216, 120),
        (216, 248, 120),
        (184, 248, 184),
        (184, 248, 216),
        (0, 252, 252),
        (216, 216, 216),
        (0, 0, 0),
        (0, 0, 0),
    ],
    "gameboy": [(15, 56, 15), (48, 98, 48), (139, 172, 15), (155, 188, 15)],
    "gameboy_pocket": [(0, 0, 0), (96, 96, 96), (160, 160, 160), (255, 255, 255)],
    "pico8": [
        (0, 0, 0),
        (29, 43, 83),
        (126, 37, 83),
        (0, 135, 81),
        (171, 82, 54),
        (95, 87, 79),
        (194, 195, 199),
        (255, 241, 232),
        (255, 0, 77),
        (255, 163, 0),
        (255, 236, 39),
        (0, 228, 54),
        (41, 173, 255),
        (131, 118, 156),
        (255, 119, 168),
        (255, 204, 170),
    ],
    "commodore64": [
        (0, 0, 0),
        (255, 255, 255),
        (136, 0, 0),
        (170, 255, 238),
        (204, 68, 204),
        (0, 204, 85),
        (0, 0, 170),
        (238, 238, 119),
        (221, 136, 85),
        (102, 68, 0),
        (255, 119, 119),
        (51, 51, 51),
        (119, 119, 119),
        (170, 255, 102),
        (0, 136, 255),
        (187, 187, 187),
    ],
    "cga": [
        (0, 0, 0),
        (0, 0, 170),
        (0, 170, 0),
        (0, 170, 170),
        (170, 0, 0),
        (170, 0, 170),
        (170, 85, 0),
        (170, 170, 170),
        (85, 85, 85),
        (85, 85, 255),
        (85, 255, 85),
        (85, 255, 255),
        (255, 85, 85),
        (255, 85, 255),
        (255, 255, 85),
        (255, 255, 255),
    ],
    "gameboy_color": [
        (0, 0, 0),
        (52, 104, 86),
        (136, 192, 112),
        (224, 248, 208),
        (134, 32, 104),
        (188, 74, 155),
        (248, 120, 248),
        (248, 180, 248),
        (248, 56, 0),
        (252, 152, 56),
        (252, 224, 168),
        (252, 252, 252),
    ],
    "apollo": [
        (26, 28, 44),
        (93, 39, 93),
        (177, 62, 83),
        (239, 125, 87),
        (255, 205, 117),
        (167, 240, 112),
        (56, 183, 100),
        (37, 113, 121),
        (41, 54, 111),
        (59, 93, 201),
        (65, 166, 246),
        (115, 239, 247),
        (244, 244, 244),
        (148, 176, 194),
        (86, 108, 134),
        (51, 60, 87),
    ],
    "ega": [
        (0, 0, 0),
        (0, 0, 170),
        (0, 170, 0),
        (0, 170, 170),
        (170, 0, 0),
        (170, 0, 170),
        (170, 85, 0),
        (170, 170, 170),
        (85, 85, 85),
        (85, 85, 255),
        (85, 255, 85),
        (85, 255, 255),
        (255, 85, 85),
        (255, 85, 255),
        (255, 255, 85),
        (255, 255, 255),
        (0, 0, 0),
        (20, 20, 20),
        (32, 32, 32),
        (44, 44, 44),
        (56, 56, 56),
        (68, 68, 68),
        (80, 80, 80),
        (96, 96, 96),
        (112, 112, 112),
        (128, 128, 128),
        (144, 144, 144),
        (160, 160, 160),
        (180, 180, 180),
        (200, 200, 200),
        (224, 224, 224),
        (252, 252, 252),
        (0, 0, 252),
        (64, 0, 252),
        (124, 0, 252),
        (188, 0, 252),
        (252, 0, 252),
        (252, 0, 188),
        (252, 0, 124),
        (252, 0, 64),
        (252, 0, 0),
        (252, 64, 0),
        (252, 124, 0),
        (252, 188, 0),
        (252, 252, 0),
        (188, 252, 0),
        (124, 252, 0),
        (64, 252, 0),
        (0, 252, 0),
        (0, 252, 64),
        (0, 252, 124),
        (0, 252, 188),
        (0, 252, 252),
        (0, 188, 252),
        (0, 124, 252),
        (0, 64, 252),
        (124, 124, 252),
        (156, 124, 252),
        (188, 124, 252),
        (220, 124, 252),
        (252, 124, 252),
        (252, 124, 220),
        (252, 124, 188),
        (252, 124, 156),
    ],
    "msx": [
        (0, 0, 0),
        (0, 0, 0),
        (33, 200, 66),
        (94, 220, 120),
        (84, 85, 237),
        (125, 118, 252),
        (212, 82, 77),
        (66, 235, 245),
        (252, 85, 84),
        (255, 121, 120),
        (212, 193, 84),
        (230, 206, 128),
        (33, 176, 59),
        (201, 91, 186),
        (204, 204, 204),
        (255, 255, 255),
    ],
}

# Preset configurations with palette support
PRESETS = {
    "cool": {
        "pixel_size": 14,
        "colors": 24,
        "brightness": 0.9,
        "contrast": 1.35,
        "saturation": 1.45,
        "sharpen": 2.0,
        "dither": False,
        "palette": None,
        "description": "Vibrant retro pixel art (default)",
    },
    "nes": {
        "pixel_size": 14,
        "colors": 54,
        "brightness": 0.95,
        "contrast": 1.35,
        "saturation": 1.5,
        "sharpen": 2.5,
        "dither": True,
        "palette": "nes",
        "description": "NES/Famicom authentic palette",
    },
    "gameboy": {
        "pixel_size": 12,
        "colors": 4,
        "brightness": 0.95,
        "contrast": 1.4,
        "saturation": 0.8,
        "sharpen": 2.0,
        "dither": True,
        "palette": "gameboy",
        "description": "Game Boy green monochrome",
    },
    "pico8": {
        "pixel_size": 12,
        "colors": 16,
        "brightness": 1.0,
        "contrast": 1.3,
        "saturation": 1.4,
        "sharpen": 2.0,
        "dither": False,
        "palette": "pico8",
        "description": "PICO-8 fantasy console",
    },
    "c64": {
        "pixel_size": 16,
        "colors": 16,
        "brightness": 0.95,
        "contrast": 1.3,
        "saturation": 1.5,
        "sharpen": 2.5,
        "dither": True,
        "palette": "commodore64",
        "description": "Commodore 64 classic",
    },
    "cga": {
        "pixel_size": 16,
        "colors": 16,
        "brightness": 1.0,
        "contrast": 1.4,
        "saturation": 1.3,
        "sharpen": 2.5,
        "dither": True,
        "palette": "cga",
        "description": "CGA 16-color graphics",
    },
    "retro": {
        "pixel_size": 16,
        "colors": 16,
        "brightness": 0.9,
        "contrast": 1.3,
        "saturation": 1.4,
        "sharpen": 2.5,
        "dither": True,
        "palette": None,
        "description": "Classic 8-bit game style",
    },
    "modern": {
        "pixel_size": 8,
        "colors": 64,
        "brightness": 0.85,
        "contrast": 1.1,
        "saturation": 1.2,
        "sharpen": 1.5,
        "dither": False,
        "palette": None,
        "description": "Modern pixel art with detail",
    },
    "extreme": {
        "pixel_size": 24,
        "colors": 8,
        "brightness": 1.0,
        "contrast": 1.5,
        "saturation": 1.5,
        "sharpen": 3.0,
        "dither": True,
        "palette": None,
        "description": "Maximum retro chunky style",
    },
    "arcade": {
        "pixel_size": 10,
        "colors": 32,
        "brightness": 1.0,
        "contrast": 1.4,
        "saturation": 1.6,
        "sharpen": 2.0,
        "dither": False,
        "palette": None,
        "description": "Bright arcade cabinet style",
    },
    "apollo": {
        "pixel_size": 10,
        "colors": 16,
        "brightness": 0.9,
        "contrast": 1.3,
        "saturation": 1.4,
        "sharpen": 2.0,
        "dither": False,
        "palette": "apollo",
        "description": "Apollo synthwave palette",
    },
}


def apply_edge_enhancement(img):
    """Enhance edges for crisper pixel art"""
    return img.filter(ImageFilter.EDGE_ENHANCE_MORE)


def apply_sharpen(img, factor):
    """Sharpen image for crisp pixel boundaries"""
    enhancer = ImageEnhance.Sharpness(img)
    return enhancer.enhance(factor)


def pixelate_image(img, pixel_size):
    """Apply pixelation effect to an image"""
    w, h = img.size
    img_small = img.resize((w // pixel_size, h // pixel_size), Image.NEAREST)
    img_pixelated = img_small.resize((w, h), Image.NEAREST)
    return img_pixelated


def adjust_brightness(img, factor):
    """Adjust image brightness"""
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def adjust_contrast(img, factor):
    """Adjust image contrast"""
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)


def adjust_saturation(img, factor):
    """Adjust color saturation"""
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(factor)


def apply_palette(img, palette_name):
    """Apply a predefined color palette to the image using optimized vectorized operations"""
    palette = np.array(COLOR_PALETTES[palette_name], dtype=np.float32)
    img_array = np.array(img, dtype=np.float32)
    h, w = img_array.shape[:2]

    # Reshape image to (h*w, 3) for vectorized processing
    pixels = img_array.reshape(-1, 3)

    # Use scipy's cdist for optimized distance calculation if available
    distances = cdist(pixels, palette, metric="sqeuclidean")
    closest_indices = np.argmin(distances, axis=1)

    # Map pixels to closest palette colors
    result = palette[closest_indices].reshape(h, w, 3)

    return Image.fromarray(result.astype("uint8"))


def quantize_colors(img, colors, dither=False):
    """Reduce colors with optional dithering"""
    if dither:
        return img.quantize(
            colors=colors,
            method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.FLOYDSTEINBERG,
        ).convert("RGB")
    else:
        return img.quantize(
            colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
        ).convert("RGB")


def snap_to_palette(img, colors):
    """More aggressive color snapping for cleaner palette"""
    img_p = img.convert("P", palette=Image.Palette.ADAPTIVE, colors=colors)
    return img_p.convert("RGB")


def process_image(
    input_path,
    output_path,
    pixel_size,
    colors,
    brightness,
    contrast,
    saturation,
    sharpen,
    dither,
    edge_enhance,
    clean_palette,
    palette_name,
):
    """Process an image to look like pixel art"""
    try:
        img = Image.open(input_path)
        print(f"[LOAD] {input_path} ({img.size[0]}x{img.size[1]})")

        if img.mode != "RGB":
            img = img.convert("RGB")

        # 1. Adjust brightness first (affects all subsequent operations)
        img = adjust_brightness(img, brightness)

        # 2. Boost contrast and saturation before reducing detail
        img = adjust_contrast(img, contrast)
        img = adjust_saturation(img, saturation)

        # 3. Optional edge enhancement (before pixelation to preserve important edges)
        if edge_enhance:
            img = apply_edge_enhancement(img)
            print("[EDGE] Applied edge enhancement")

        # 4. Pixelation (major detail reduction)
        img = pixelate_image(img, pixel_size)
        print(f"[PIXEL] Block size: {pixel_size}px")

        # 5. Color quantization/palette (after pixelation to work on final pixel structure)
        if palette_name:
            img = apply_palette(img, palette_name)
            palette_colors = len(COLOR_PALETTES[palette_name])
            print(f"[PALETTE] {palette_name.upper()} ({palette_colors} colors)")
        elif clean_palette:
            img = snap_to_palette(img, colors)
            print(f"[QUANTIZE] Clean palette: {colors} colors")
        else:
            img = quantize_colors(img, colors, dither)
            dither_status = "dithered" if dither else "flat"
            print(f"[QUANTIZE] {colors} colors ({dither_status})")

        # 6. Final sharpening (after all other operations to crisp up pixel boundaries)
        if sharpen > 1.0:
            img = apply_sharpen(img, sharpen)
            print(f"[SHARPEN] Factor: {sharpen}")

        img.save(output_path, quality=100)
        print(f"[SAVE] {output_path}")
        print(
            f"[DONE] brightness={brightness} contrast={contrast} saturation={saturation}"
        )

    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


def list_presets():
    """Display all available presets"""
    print("\nAvailable Presets:\n")
    print(f"{'NAME':<12} {'DESCRIPTION':<40} {'PALETTE':<15} {'PIXELS':<8}")
    print("-" * 80)
    for name, settings in PRESETS.items():
        pal = settings["palette"] or "adaptive"
        desc = settings["description"]
        px = f"{settings['pixel_size']}px"
        print(f"{name:<12} {desc:<40} {pal:<15} {px:<8}")
    print()


def list_palettes():
    """Display all available color palettes"""
    print("\nAvailable Color Palettes:\n")
    print(f"{'PALETTE':<20} {'COLORS':<10} {'DESCRIPTION'}")
    print("-" * 60)
    descriptions = {
        "nes": "Nintendo Entertainment System",
        "gameboy": "Nintendo Game Boy (green)",
        "gameboy_pocket": "Game Boy Pocket (grayscale)",
        "pico8": "PICO-8 fantasy console",
        "commodore64": "Commodore 64",
        "cga": "IBM CGA graphics adapter",
        "gameboy_color": "Game Boy Color",
        "apollo": "Synthwave aesthetic",
        "ega": "IBM EGA graphics adapter",
        "msx": "MSX home computer",
    }
    for name, colors in COLOR_PALETTES.items():
        desc = descriptions.get(name, "")
        print(f"{name:<20} {len(colors):<10} {desc}")
    print("\nUsage: --palette <name>\n")


def main():
    print(BANNER)
    parser = argparse.ArgumentParser(
        description="Convert images to authentic pixel art with retro palettes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s input.jpg output.jpg
  %(prog)s input.jpg output.jpg --preset nes
  %(prog)s input.jpg output.jpg --preset pico8
  %(prog)s input.jpg output.jpg --palette gameboy -p 12
  %(prog)s input.jpg output.jpg --palette commodore64 --dither
  %(prog)s input.jpg output.jpg --palette nes -p 16 --edge-enhance
  %(prog)s --list-presets
  %(prog)s --list-palettes
        """,
    )

    parser.add_argument("input", nargs="?", help="input image path")
    parser.add_argument("output", nargs="?", help="output image path")

    parser.add_argument(
        "--preset",
        choices=PRESETS.keys(),
        default="cool",
        help="preset configuration (default: cool)",
    )
    parser.add_argument(
        "--palette",
        choices=COLOR_PALETTES.keys(),
        help="predefined retro color palette",
    )
    parser.add_argument(
        "--list-presets", action="store_true", help="list all presets and exit"
    )
    parser.add_argument(
        "--list-palettes", action="store_true", help="list all color palettes and exit"
    )

    parser.add_argument("-p", "--pixel-size", type=int, help="pixel block size")
    parser.add_argument(
        "-c", "--colors", type=int, help="number of colors (without --palette)"
    )
    parser.add_argument(
        "-b", "--brightness", type=float, help="brightness factor (0.0-2.0)"
    )
    parser.add_argument("--contrast", type=float, help="contrast factor (0.0-2.0)")
    parser.add_argument(
        "-s", "--saturation", type=float, help="saturation factor (0.0-2.0)"
    )
    parser.add_argument("--sharpen", type=float, help="sharpness factor (1.0-5.0)")

    parser.add_argument(
        "--dither", action="store_true", help="enable Floyd-Steinberg dithering"
    )
    parser.add_argument(
        "--edge-enhance", action="store_true", help="enhance edges before pixelation"
    )
    parser.add_argument(
        "--clean-palette", action="store_true", help="aggressive palette snapping"
    )

    args = parser.parse_args()

    if args.list_presets:
        list_presets()
        sys.exit(0)

    if args.list_palettes:
        list_palettes()
        sys.exit(0)

    if not args.input or not args.output:
        parser.print_help()
        print("\nTip: Use --list-presets or --list-palettes for more info")
        sys.exit(1)

    # Start with preset
    preset = PRESETS[args.preset]
    print(f"[PRESET] '{args.preset}' - {preset['description']}")

    # Override with custom values if provided
    pixel_size = args.pixel_size if args.pixel_size else preset["pixel_size"]
    colors = args.colors if args.colors else preset["colors"]
    brightness = args.brightness if args.brightness else preset["brightness"]
    contrast = args.contrast if args.contrast else preset["contrast"]
    saturation = args.saturation if args.saturation else preset["saturation"]
    sharpen = args.sharpen if args.sharpen else preset["sharpen"]
    dither = args.dither if args.dither else preset["dither"]
    palette_name = args.palette if args.palette else preset["palette"]

    process_image(
        args.input,
        args.output,
        pixel_size,
        colors,
        brightness,
        contrast,
        saturation,
        sharpen,
        dither,
        args.edge_enhance,
        args.clean_palette,
        palette_name,
    )


if __name__ == "__main__":
    main()
