from typing import Any

import matplotlib.pyplot as plt
import numpy as np

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    rank = 0
    size = 1


def Put(*args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)


class ComplexSet:

    default_details = {
        "centre": [-0.5, 0.0],
        "full_width": 3.0,
        "full_dimension": [1080, 1920],
        "zoom_factor": 1.0,
        "magnitude_threshold": 5,
        "max_iterations": 200,
        "figsize": [10, 7.5],
    }
    # all
    def __init__(self):
        for detail, value in self.default_details.items():
            setattr(self, detail, value)
        Put("Initialised set")

    def _calculate_reduced_pixel_dimension(self):
        # Ensure number of rows is divisible by size
        self.full_dimension[0] -= self.full_dimension[0] % size
        self.reduced_dimension = [
            self.full_dimension[0] // size,
            self.full_dimension[1],
        ]
        Put("Determined reduced pixel dims")

    # all - but needs to be fixed to allocate smaller arrays and properly deal with colour - done
    def allocate_arrays(self):
        self._calculate_reduced_pixel_dimension()

        if rank == 0:
            self.full_colour = np.empty(self.full_dimension, dtype=np.int16)
        else:
            self.full_colour = None
        if size > 1:
            self.colour = np.empty(self.reduced_dimension, dtype=np.int16)
        else:
            self.colour = self.full_colour

        self.z = np.empty(self.reduced_dimension, dtype=np.complex64)
        if self.require_c_array:
            self.c = np.empty(self.reduced_dimension, dtype=np.complex64)

        Put("Allocated arrays")

    # all - but needs to be fixed to produce limited extent, then full_extent added
    def _calculate_extent(self):
        half_width = self.full_width / 2 / self.zoom_factor
        aspect_ratio = self.full_dimension[1] / self.full_dimension[0]
        half_height = half_width / aspect_ratio

        self.full_extent = [
            self.centre[0] - half_width,  # Real ax min
            self.centre[0] + half_width,  # Real ax max
            self.centre[1] - half_height,  # Imag ax min
            self.centre[1] + half_height,  # Imag ax max
        ]
        reduced_width = rank * (2 * half_width) / size
        self.reduced_extent = [
            self.full_extent[0],
            self.full_extent[1],
            self.full_extent[2] + rank * (2 * half_height) / size,
            self.full_extent[3] + (rank - 1) * (2 * half_height) / size,
        ]
        Put("full:",self.full_extent)
        print("reduced:",self.reduced_extent)
        
        Put("Determined extents")

    # all - will automaticall be fixed by extent changes
    def _create_array_of_complex_coordinates(self):
        self._calculate_extent()
        real = np.tile(
            np.linspace(
                self.reduced_extent[0],
                self.reduced_extent[1],
                self.reduced_dimension[1],
            )[None, :],
            (self.reduced_dimension[0], 1),
        )
        imag = np.tile(
            np.linspace(
                self.reduced_extent[3],
                self.reduced_extent[2],
                self.reduced_dimension[0],
            )[:, None],
            (1, self.reduced_dimension[1]),
        )
        Put("Created array of complex coords")
        return real, imag

    # root only - but called internally
    def _scatter_colour_array(self):
        Put("Scattering")
        comm.Scatter([self.full_colour, MPI.INT], self.colour, root=0)
        Put("Scattered colour")

    def _gather_colour_array(self):
        Put("Gathering")
        comm.Gather(self.colour, self.full_colour, root=0)
        Put("Gathered colour")

    # all
    def _iterate(self):
        self.z *= self.z
        self.z += self.c

    # all
    def update_colour(self):
        # if rank == 0:
        #     self._scatter_colour_array()

        for i in range(self.max_iterations):
            self._iterate()
            self.colour[
                (abs(self.z) > self.magnitude_threshold) & (self.colour == 0)
            ] = (i + 1)
            self.z[self.colour != 0] = 0

        # if rank == 0:
        self._gather_colour_array()
        Put("Updated colour")

    # root only
    def initialise_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.image = self.ax.imshow(self.full_colour, extent=self.full_extent)
        self.cid = self.fig.canvas.mpl_connect("button_press_event", self._onclick)
        self.fig.canvas.flush_events()
        self.fig.canvas.draw()
        Put("Plot initialised")

    # root only
    def zoom_loop(self):
        Put("Looping for zoom")
        while True:
            self.fig.canvas.start_event_loop()

    # root only
    def _update_plot(self):
        self.image.set_data(self.full_colour)
        self.image.set_extent(self.full_extent)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        Put("Updated Plot")

    def _onclick(self, event):
        self.centre[0] = event.xdata
        self.centre[1] = event.ydata
        self.zoom_factor *= 2

        self.initialise_arrays()
        self.update_colour()
        if rank == 0:
            self._update_plot()
        Put("Detected mouse click")


class Mandelbrot(ComplexSet):

    require_c_array = True
    # all
    def initialise_arrays(self):
        self.colour.fill(0)
        self.z.fill(0.0)
        self.c.real, self.c.imag = self._create_array_of_complex_coordinates()
        Put("Initialised array values")


class Julia(ComplexSet):

    require_c_array = False
    # all
    def __init__(self, c: complex):
        super().__init__()
        self.c = c

    # all
    def initialise_arrays(self):
        self.colour.fill(0)
        self.z.real, self.z.imag = self._create_array_of_complex_coordinates()
        Put("Initialised array values")


if __name__ == "__main__":

    mandelbrot = Mandelbrot()
    mandelbrot.allocate_arrays()
    mandelbrot.initialise_arrays()

    mandelbrot.update_colour()
    if rank == 0:
        mandelbrot.initialise_plot()
        mandelbrot.zoom_loop()
