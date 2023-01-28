from typing import Union, Tuple, Dict, Any

from loguru import logger
from omegaconf import DictConfig


def color_to_rgba(color: Union[tuple, str], opacity: float = 1.0) -> Tuple[int, int, int, float]:
    logger.info(f'Start to cast color: {color} to rgba values.')
    # It is already in rgb
    if isinstance(color, tuple):
        if len(color) == 3:
            color = color + (opacity, )
        return color

    # Given in hex format
    if isinstance(color, str):
        if color.startswith('#'):
            hex_color = color.lstrip('#')
            if len(hex_color) == 3:
                hex_color = hex_color * 2
            return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16), opacity
        else:
            from matplotlib import colors
            return colors.to_rgba(color, alpha=opacity)
    raise ValueError(f'Unknown color: {color}')


def color_to_rgba_str(color: str, opacity: float = 1.0):
    rgba_color = color_to_rgba(color, opacity=opacity)
    rgba_color_str = f'rgba{str(rgba_color)}'
    return rgba_color_str


def make_marker(
    plotting_settings: DictConfig,
    opacity: float = 1.0,

) -> Dict[str, Any]:

    return {
        "opacity": opacity,
        "line": {"width": 0.5, "color": 'DarkSlateGrey'},
        "color": color_to_rgba_str(plotting_settings.color, opacity=opacity),
    }
