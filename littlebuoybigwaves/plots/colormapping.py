"""
@jacobrdavis

A collection of frequently used colormap tools for working with ocean wave data.

#TODO: add colorbar creation tools, specifically for line plots?
#TODO: cite colormapping

"""
__all__ = ['truncate_colormap', 'create_colorbar']

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.colorbar as colorbar
from matplotlib.axes import Axes
from typing import Tuple
import numpy as np

def truncate_colormap(
    cmap: colors.LinearSegmentedColormap,
    minval: float =0.0,
    maxval: float =1.0,
    n: int =100,
    ) -> colors.LinearSegmentedColormap:
    """
    Helper function to use a subset of a pre-defined colormap range

    Arguments:
        - cmap (colors.LinearSegmentedColormap), colormap to truncate
        - minval (float, optional), normalized min (1.0 is the bottom end of the full range); defaults to 0.0.
        - maxval (float, optional), normalized max (1.0 is the top end of the full range); defaults to 1.0.
        - n (int, optional), number of discrete colors; defaults to 100.

    Returns:
        - (colors.LinearSegmentedColormap), the truncated colormap

    Example: 

    Return a colormap equivalent to the upper %70 of the 'Blues' colormap:
    >>> cmap = plt.get_cmap('Blues')
    >>> truncate_colormap(cmap, 0.3, 1)

    """
    new_cmap = colors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval},{maxval})',
        cmap(np.linspace(minval, maxval, n)),
        N=n,
    )
    return new_cmap

def create_colorbar(
    cmap: colors.Colormap,
    norm: plt.Normalize,
    ax: Axes = None,
    **cbar_kwargs,
)-> Tuple[colorbar.Colorbar, colors.Normalize]:
    """
    Create a custom colorbar.

    Arguments:
        - cmap (colors.Colormap), color map 
        - vmin (float), minimum value on the colorbar
        - vmax (float), maximum value on the colorbar
        - ax (Axes, optional), axes to add the colorbar to; defaults to current axes.

    Returns:
        - (colorbar.Colorbar, colors.Normalize), colobar and associated norm
    """
    if ax is None:
        ax = plt.gca()

    smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    smap.set_array([])
    cbar = plt.colorbar(smap, ax=ax, **cbar_kwargs)
    return cbar