"""
@jacobrdavis

A collection of frequently used base plots for working with ocean wave data.
https://towardsdatascience.com/creating-custom-plotting-functions-with-matplotlib-1f4b8eba6aa1
"""
__all__ = ['comparison_plot']

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib as mpl
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors

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