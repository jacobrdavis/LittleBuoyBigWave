"""
@jacobrdavis

A collection of frequently used base plots for working with ocean wave data.
https://towardsdatascience.com/creating-custom-plotting-functions-with-matplotlib-1f4b8eba6aa1
"""
__all__ = ['spectrogram', 'comparison_plot', 'scalar_spectrum', 'lineplot_color']

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib as mpl
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors
from matplotlib.colors import Colormap
import matplotlib.colorbar as colorbar
from .colormapping import create_colorbar
from typing import Tuple


def spectrogram(time, frequency, z, ax=None, pcolormesh_kwargs={}):

    if ax is None:
        ax = plt.gca()

    if z.ndim == 1:
        raise ValueError(f'Input `z` must have 2 dimensions. Given {z.ndim}.')

    t, f = z.shape

    if frequency.ndim == 1 and time.ndim == 1:
        frequency = np.tile(frequency, (t, 1))
        time = np.tile(time, (f, 1)).T
    elif frequency.shape == z.shape and time.ndim == 1:
        time = np.tile(time, (f, 1)).T
    elif frequency.shape == z.shape and time.shape == z.shape:
        pass
    else:
        raise ValueError('`time` and `frequency` shapes do not match `z`.')
    pcm = ax.pcolormesh(time,
                        frequency,
                        z,
                        **pcolormesh_kwargs)

    return pcm






def comparison_plot(
    X1: np.ndarray, 
    X2: np.ndarray, 
    ax: Axes =None,
    **plt_kwargs
)-> Axes:
    """
    Create a simple 1:1 scatter plot of two arrays.

    Arguments:
        - X1 (numpy.ndarray), array of shape (n,)
        - X2 (numpy.ndarray), array of shape (n,)
        - ax (matplotlib.axes.Axes, optional), axes to plot on; defaults to None.
        
        All remaining arguments are passed to `sns.scatterplot()`

    Returns:
        - (matplotlib.axes.Axes)
    """

    # Create the figure and plot a scatter of the two arrays:
    if ax is None:
        ax = plt.gca()

    sns.scatterplot(
        x = X1,
        y = X2,
        **plt_kwargs,
        ax=ax,
    )

    # Get the automated limits and round both sides; reset the limits and plot the 1:1 line:
    yl = ax.get_ylim()
    xl = ax.get_xlim()
    lims = (
        np.floor(min([xl[0], yl[0]])), 
        np.ceil(max([xl[1], yl[1]]))
    )
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.plot(lims,lims, color = 'k')
    ax.set_aspect('equal')

    return ax


def scalar_spectrum(
    E: np.ndarray, 
    f: np.ndarray,
    ax: Axes = None,
    xlim = [0.05,0.7],
    xticks = [5*10**(-2), 0.1, 0.2, 0.5],
    xlabel = 'frequency (Hz)',
    ylabel = 'energy density (m$^2$/Hz)',
    **plt_kwargs, 
)-> Axes:
    """TODO:"""

    if ax is None:
        ax = plt.gca()

    ax.plot(
        f,
        E,
        **plt_kwargs,
    )

    # """
    # Set the plot title:
    # """
    # t1_str = t1.strftime('%m-%d %Hh')
    # t2_str = t2.strftime('%m-%d %Hh')

    # ax.set_title(
    #     f'{t1_str} to {t2_str}'
    # )    

    ax.set_yscale('log')
    ax.set_xscale('log')
    # ax.set_ylim([10**(-3), 10**(2)])
    ax.set_xlim(xlim)
    ax.set_xticks(xticks)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax

def lineplot_color(
    X: np.ndarray, 
    Y: np.ndarray, 
    z: np.ndarray,
    cmap: Colormap = plt.get_cmap('inferno'),
    norm: plt.Normalize = None,
    vmin: float = None,
    vmax: float = None,
    ax: Axes = None,
    cax: Axes = None,
    plt_kwargs: dict = {},
    cbar_kwargs: dict = {},
    create_cbar: bool = True
)-> Tuple[plt.Axes, colorbar.Colorbar]:
    """
    _summary_
    #TODO: 
    - use LineCollection instead...
    - check length on input arrays and make them iterables?

    Arguments:
        - X (np.ndarray), _description_
        - Y (np.ndarray), _description_
        - z (np.ndarray), _description_
        - cmap (Colormap, optional), _description_; defaults to plt.get_cmap('inferno').
        - norm (plt.Normalize), _description_.
        - vmin (float, optional), _description_; defaults to None.
        - vmax (float, optional), _description_; defaults to None.
        - ax (Axes, optional), _description_; defaults to None.
        - plt_kwargs (dict, optional), _description_; defaults to {}.
        - cbar_kwargs (dict, optional), _description_; defaults to {}.
        - create_cbar (bool, optional), _description_; defaults to True.
        
    Returns:
        - (plt.Axes, colorbar.Colorbar), axes and colorbar
    """

    if vmin is None:
        vmin = np.min(z)
    if vmax is None:
        vmax = np.max(z)
    if norm is None:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
    if ax is None:
        ax = plt.gca()
    if cax is None:
        cax = ax

    if create_cbar:
        cbar = create_colorbar(cmap, norm, cax, **cbar_kwargs)
    else:
        cbar = None

    #TODO: vectorize using line collection
    for Xi, Yi, zi in zip(X, Y, z):
        ax.plot(
            Xi,
            Yi,
            c = cmap(norm(zi)),
            **plt_kwargs
        )

    return ax, cbar









# for f,mom in zip(SPOTs[spotID]['waves'][t1:t2]['f'], SPOTs[spotID]['waves'][t1:t2][dirMom]):
#                 ax[i].plot(
#                     f,
#                     mom,
#                     color = 'k',
#                     alpha = 0.25,
#                 )
    #   # ax.set_yscale('log')
    #     # ax[i].set_xscale('log')
    #     ax[i].set_ylim([-1, 1])
    #     ax[i].set_xlim([0.05,0.5])
    #     # ax.set_xticks([5*10**(-2), 0.1, 0.2, 0.5])
    #     # ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    #     ax[i].set_xlabel('frequency (Hz)')
    #     ax[i].set_ylabel(dirMom)

    #     if i == 0:
    #         ax[i].legend(loc='upper right')

#%%
if __name__ ==  '__main__':
    #TODO: stand-in for tests...
        fig,ax = plt.subplots(figsize =(5,5))
        X1 = np.array([0, 1, 3, 4])
        X2 = np.array([0.4, 1.1, 2.75, 4.25])
        comparison_plot(
            X1,
            X2,
            ax=ax,
            color='b',
            alpha=0.6,
        )
#%%

# from matplotlib.collections import LineCollection
# import matplotlib.animation as animation

# # Sorting and conditioning:
# srt = np.argsort(ws) # sort by ascending wind speed
# cnd = np.logical_and(wsMin < ws[srt], ws[srt] < wsMax)
# norm = plt.Normalize(vmin=np.min(ws[srt][cnd]), vmax=np.round(np.max(ws[srt][cnd])))

# # Setup colormap
# n = len(ws[cnd])
# cmap = plt.get_cmap('inferno', n)

# # Plot the spectra
# fig,ax = plt.subplots()
# ax, cbar = waveplots.lineplot_color(
#     X = [],
#     Y = [],
#     z = ws[srt][cnd],
#     vmin = np.min(ws[srt][cnd]),
#     vmax = np.round(np.max(ws[srt][cnd])),
#     plt_kwargs = dict(
#         alpha = 0.25,
#     ),
# )
# ax.plot(np.array([0.2 , 0.5]), 10**(-2)*np.array([0.2 , 0.5])**(-4), '--k')
# ax.annotate('$f^{-4}$', (0.35, 10**(-1.85)*0.35**(-4)), ha='center', va='bottom')
# ax.set_yscale('log')
# ax.set_xscale('log')
# ax.set_xlim([0.025, 0.7])
# ax.set_ylim([10**(-3), 10**(3)])
# ax.set_xticks([5*10**(-2), 0.1, 0.2, 0.5])
# ax.get_xaxis().set_major_formatter(ScalarFormatter())
# ax.set_xlabel('frequency (Hz)')
# ax.set_ylabel('energy density (m$^2$/Hz)')
# cbar.set_label('10-m wind speed, $U_{10}$ (m/s)')
# cbar.set_ticks(np.arange(0, wsMax+5, 5))


# line, = ax.plot([], [], lw = 1.5) 
# N = 117*2 # = len(E)/n_skip
# n_skip = 25
# lines = [plt.plot([], [], lw = 1.5)[0] for _ in range(N)]

# xdata, ydata = [], [] 
# def animate(i, x, y, z, line):
    
#     # x = f[s[i]
#     # y = E[srt][cnd][i]

#     # xdata.append(x) 
#     # ydata.append(y) 
#     # lst[i:i + n]
    
#     lines[i].set_data(x[i*n_skip,:], y[i*n_skip,:])
#     lines[i].set_color(cmap(norm(z[i*n_skip])))
#     return line,


# anim = animation.FuncAnimation(
#     fig, animate, frames=range(N), interval=10, blit=True, save_count=50, fargs=[f[srt][cnd], E[srt][cnd], ws[srt][cnd], line])

# from matplotlib.animation import PillowWriter
# # Save the animation as an animated GIF
# anim.save("simple_animation.gif", dpi=100,
#          writer=PillowWriter(fps=100)) 