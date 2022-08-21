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


def Put(*args, rank=rank,**kwargs):
    if rank != rank:
        return
    print(f"({rank})",end=' ')
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
        reduced_height = (2 * half_height) / size
        self.reduced_extent = [
            self.full_extent[0],
            self.full_extent[1],
            self.full_extent[3] - (rank+1) * reduced_height,
            self.full_extent[3] - (rank) * reduced_height,
        ]
        Put("full:", self.full_extent,rank=0)
        Put("reduced:", self.reduced_extent)
        Put("Determined extents")

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

    def _gather_colour_array(self):
        Put("Gathering")
        comm.Gather(self.colour, self.full_colour, root=0,)
        Put("Gathered colour")

    def _iterate(self):
        self.z *= self.z
        self.z += self.c

    def update_colour(self):

        for i in range(self.max_iterations):
            self._iterate()
            self.colour[
                (abs(self.z) > self.magnitude_threshold) & (self.colour == 0)
            ] = (i + 1)
            self.z[self.colour != 0] = 0

        self._gather_colour_array()
        Put("Updated colour")

    def initialise_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.image = self.ax.imshow(self.full_colour, extent=self.full_extent)
        self.cid = self.fig.canvas.mpl_connect("button_press_event", self._onclick)
        self.fig.canvas.flush_events()
        self.fig.canvas.draw()
        Put("Plot initialised")

    def zoom_loop(self):
        Put("Looping for zoom")
        if rank == 0:
            # So this function never actually exits as far as I can tell
            # So the on_click function needs to actually run everything
            self.fig.canvas.start_event_loop()

            for dest in range(1, size):
                comm.send(obj=self.centre, dest=dest)

        else:
            while True:
                Put("Recieving in loop")
                self.centre = comm.recv(source=0)
                self.zoom_factor *= 2
                self._function_to_call_on_all()

    def _function_to_call_on_all(self):

            self.initialise_arrays()
            self.update_colour()

            # Just a flag that all processes are done updating
            if rank == 0:
                for _ in range(1, size):
                    comm.recv()
                self._update_plot()
            else:
                comm.send(obj=None, dest=0)

    def _update_plot(self):
        self.image.set_data(self.full_colour)
        self.image.set_extent(self.full_extent)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        Put("Updated Plot")


    def _onclick(self, event):
        self.centre[0] = event.xdata
        self.centre[1] = event.ydata
        Put(f"new centre: {self.centre}")
        self.zoom_factor *= 2
        Put("Detected mouse click")
        for dest in range(1, size):
            comm.send(obj=self.centre, dest=dest)
        self._function_to_call_on_all()


class Mandelbrot(ComplexSet):

    require_c_array = True
    def initialise_arrays(self):
        self.colour.fill(0)
        self.z.fill(0.0)
        self.c.real, self.c.imag = self._create_array_of_complex_coordinates()
        Put("Initialised array values")


class Julia(ComplexSet):

    require_c_array = False
    def __init__(self, c: complex):
        super().__init__()
        self.c = c

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
