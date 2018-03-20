"""Class to help plot density heatmaps for bivariate distributions."""

import matplotlib.pyplot as plt
import numpy as np

class VisGrid(object):

    def __init__(self, xlim, ylim=None, steps=1):
        """Sets up a grid for visualization purposes.

        Parameters
        ----------
        xlim: tuple of size 2
            Boundary of x coordinates for the grid.
        ylim: tuple of size 2 or None
            Boundary of y coordinates for the grid. If None, the
            grid is symmetric.
        steps: float
            How big the stepsize for the girds are. Note that each
            grid is square shaped.
        """
        if ylim is None:
            ylim = xlim
        self.xlim = xlim
        self.ylim = ylim
        self.x_ticks = np.arange(xlim[0], xlim[1], steps)
        self.y_ticks = np.arange(ylim[0], ylim[1], steps)
        self.grid = np.meshgrid(self.x_ticks, self.y_ticks)       
        self.x_size = self.grid[0].shape[0]
        self.y_size = self.grid[0].shape[1]
        self.num_grids = self.x_size * self.y_size
        self.x_coord = np.reshape(self.grid[0], self.num_grids)
        self.y_coord = np.reshape(self.grid[1], self.num_grids)

    def get_flattened(self):
        """Gets the flattend coordinate points of the grid."""
        return np.array([self.x_coord, self.y_coord]).T

    def get_grid(self, f_values):
        """Transforms a function values to 2-D for heatmap."""
        return np.flipud(
            np.reshape(f_values, [self.x_size, self.y_size]))

    def heat_map(self, ax, f_values, cmap='nipy_spectral'):
        f_grid = self.get_grid(f_values)
        ax.imshow(
            f_grid,extent=[self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]],
            cmap=cmap)