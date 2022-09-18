"""
Module for visualisation of the Mandelbrot and Julia sets.

Defines a base class ComplexSet from which Julia and Mandelbrot inherit.
A ComplexSet should never be intitialised itself, only instances of Mandelbrot
and Julia should be created.

This code supports MPI, however will run without. If mpi4py is not installed,
the code simply pretends it is running only on 1 process.

Mandelbrot and Julia sets:
    
    Founded on the iterative process
        z = z^2 + c,
    
    Points on the imaginary plane are coloured based on the rate at which they
    diverge under this iterative procedure.

    Mandelbrot sets are characterised by an initial z_0 = 0 and the c value being
    each of the points in the complex plane.

    Julia sets are characterised by z being each of the points in the complex plane
    and c being a fixed complex number.

"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import yaml
from typing import Any

# Initialising the MPI
try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    # Simulating running on 1 process when mpi4py is not present
    rank = 0
    size = 1


def Put(*args, root: bool = False, **kwargs):
    """
    Function for convenient MPI printing.

    Displays ID of printing process and includes flag
    for only printing on the root process.
    Should be used exactly as print is used, but root=True
    will force only the root process to actually print
    """
    if root and rank != 0:
        return
    print(f"({rank})", end=" ")
    print(*args, **kwargs)


class ComplexSet:
    """
    General complex set class for Mandelbrot and Julia to inherit from. 
    
    Almost all behaviour is common to both classes, only array initialisation 
    differs.
    """
    default_details = {
        "centre": [-0.5, 0.0],
        "full_width": 3.0,
        "full_dimension": [1080, 1920],
        "zoom_factor": 1.0,
        "zoom_rate": 2,
        "magnitude_threshold": 5,
        "max_iterations": 100,
        "iteration_rate": 0.1,
        "figsize": [10, 7.5],
        "colourmap": "Blues_r",
    }

    def __init__(self, parameters_file: str = None):
        """
        Complex set class constructor. Still called by Mandelbrot and Julia.

        Arguments:
        parameters_file: str
            yaml file from which to read details of the plotting
        """

        # Use default_details if no parameters file provided
        if parameters_file is None:
            details = self.default_details
        else:
            details = self._load_parameters(parameters_file)

        # Pulling details into the self namespace
        for detail, value in details.items():
            setattr(self, detail, value)
        
        Put("Initialised set", root=True)


    def _load_parameters(self, parameters_file: str = None):
        """
        Load parameters from yaml file.
        
        Arguments:
        parameters_file: str
            yaml file from which to read details of the plotting

        Returns:
        parameters: dict
        """

        with open(parameters_file, "r") as f:
            parameters = yaml.safe_load(f)["parameters"]

        ## Allowing for parameters to be distinct for Mandelbrot and Julia sets
        # Below code merges the set specific details together
        # ie. {1:2,3:{4:5,6:7}} -> {1:2,4:5,6:7}
        if type(self) is Mandelbrot:
            parameters = {**parameters, **parameters.pop("mandelbrot")}
        elif type(self) is Julia:
            parameters = {**parameters, **parameters.pop("julia")}
        else:
            raise TypeError(
                "Class ComplexSet should not be constructed alone. Construct Julia or Mandelbrot instead."
            )
        return parameters


    def _calculate_reduced_pixel_dimension(self):
        """
        Determines self.reduced_dimension which is the array size when split between processes.
        """

        # Ensure number of rows is divisible by size.
        # Leave columns so that we do not lose contiguity.
        self.full_dimension[0] += self.full_dimension[0] % size
        self.reduced_dimension = [
            self.full_dimension[0] // size,
            self.full_dimension[1],
        ]
        Put("Determined reduced pixel dims", root=True)


    def allocate_arrays(self):
        """
        Allocate the z, colour and c arrays. Only done once for obvious reasons.

        Note: c is not an array in the Julia case.
        """
        
        #Note 2: I did a speed test. It is faster to use np.empty and manually initialise to 0

        # Produces self.reduced_dimension
        self._calculate_reduced_pixel_dimension()

        # full_colour is destination for all colour arrays to be gathered.
        # All plotting happens on root so only root needs this array however
        # the variable still needs to be defined on all processes
        if rank == 0:
            self.full_colour = np.empty(self.full_dimension, dtype=np.int16)
        else:
            self.full_colour = None 
        
        # All processes need their own sub-array of colour arrays to work on
        # If we only have 1 process, we can just use full_colour
        if size > 1:
            self.colour = np.empty(self.reduced_dimension, dtype=np.int16)
        else:
            self.colour = self.full_colour

        # Use 128bit complex numbers to ensure good resolution once zoomed
        # Don't need a full size array of either of these as they are simply
        # working arrays
        self.z = np.empty(self.reduced_dimension, dtype=np.complex128)
        if self.require_c_array:
            self.c = np.empty(self.reduced_dimension, dtype=np.complex128)

        Put("Allocated arrays", root=True)


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
            self.full_extent[3] - (rank + 1) * reduced_height,
            self.full_extent[3] - (rank) * reduced_height,
        ]

        Put("Determined extents", root=True)

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
        Put("Created array of complex coords", root=True)
        return real, imag

    def _gather_colour_array(self):
        if size == 1:
            return
        Put("Gathering", root=True)
        comm.Gather(self.colour, self.full_colour, root=0)


    def _iterate(self):
        self.z *= self.z
        self.z += self.c

    def update_colour(self):
        iterations = int(self.max_iterations * self.zoom_factor**self.iteration_rate)
        for i in range(iterations):
            self._iterate()
            self.colour[
                (abs(self.z) > self.magnitude_threshold) & (self.colour == 0)
            ] = (i + 1)
            # Once a point has diverged, set z=0 to stop it being iterated
            self.z[self.colour != 0] = 0
        self.colour[self.colour == 0] = iterations + 1

        self._gather_colour_array()
        Put("Updated colour", root=True)

    def initialise_plot(self):
        if rank == 0:
            set_rc_params()
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=self.figsize)
            plt.get_current_fig_manager().full_screen_toggle()
            self.image = self.ax.imshow(
                self.full_colour,
                cmap=self.colourmap,
                extent=self.full_extent,
            )
            self._plot_artifacts()
            self.cid = self.fig.canvas.mpl_connect("button_press_event", self._onclick)
            self.fig.canvas.flush_events()
            self.fig.canvas.draw()
            Put("Plot initialised", root=True)

    def _plot_artifacts(self):
        self.ax.set_xlabel(r"Re($z$)")
        self.ax.set_ylabel(r"Im($z$)")
        if self.__class__ is Mandelbrot:
            title = "Mandelbrot Set"
        elif self.__class__ is Julia:
            title = r"Julia Set: $c=$" + f"{self.c}"
        self.ax.set_title(title)
        self.fig_text = self.fig.text(0.005, 0.9, self._get_text())

    def _get_text(self):
        decimal_places = int(abs(np.log10(self.full_width / self.zoom_factor))) + 3
        text_string = f"""
        Plot centre: ({self.centre[0]:{decimal_places+3}.{decimal_places}f},{self.centre[1]:{decimal_places+3}.{decimal_places}f})
        Zoom factor: {self.zoom_factor}

        Zoom in/out: left/right click
        """
        return text_string

    def zoom_loop(self):
        Put("Looping for zoom", root=True)
        if rank == 0:
            # So this function never actually exits as far as I can tell
            # So the on_click function needs to actually run everything
            self.fig.canvas.start_event_loop()
        else:
            while True:
                Put("Receiving in loop", root=True)
                receive_obj = comm.recv(source=0)
                self.zoom_factor = receive_obj["zoom_factor"]
                self.centre = receive_obj["centre"]
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
        self.image.set_clim(vmin=self.full_colour.min(), vmax=self.full_colour.max())
        self.image.set_extent(self.full_extent)
        self.fig_text.set_text(self._get_text())
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        Put("Updated Plot", root=True)

    def _onclick(self, event):
        if None in (event.xdata, event.ydata):
            return
        self.centre[0] = event.xdata
        self.centre[1] = event.ydata
        Put(f"new centre: {self.centre}", root=True)
        if event.button == 3:
            self.zoom_factor /= self.zoom_rate
        else:
            self.zoom_factor *= self.zoom_rate
        Put("Detected mouse click", root=True)
        send_obj = {"centre": self.centre, "zoom_factor": self.zoom_factor}
        for dest in range(1, size):
            comm.send(obj=send_obj, dest=dest)
        self._function_to_call_on_all()


class Mandelbrot(ComplexSet):

    require_c_array = True

    def initialise_arrays(self):
        self.colour.fill(0)
        self.z.fill(0.0)
        self.c.real, self.c.imag = self._create_array_of_complex_coordinates()
        Put("Initialised array values", root=True)


class Julia(ComplexSet):

    require_c_array = False

    def __init__(self, c: complex, **kwargs):
        super().__init__(**kwargs)
        self.c = c

    def initialise_arrays(self):
        self.colour.fill(0)
        self.z.real, self.z.imag = self._create_array_of_complex_coordinates()
        Put("Initialised array values", root=True)


def set_rc_params():
    mpl.rcParams["xtick.direction"] = "inout"
    mpl.rcParams["ytick.direction"] = "inout"
    mpl.rcParams["font.size"] = 18
