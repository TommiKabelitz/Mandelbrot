"""
Simple script to run the Mandelbrot and Julia set plotting routines.

See README for details on running script.
"""

# Standard Library
import argparse
import random
from typing import Any


# Local
import complex_sets as cs
from utilities import Put


def command_line_interface() -> dict:
    """
    Parse command line arguments to determine set type and c value if provided.
    """

    parser = argparse.ArgumentParser(
        description="Interactively plot complex_set or Julia sets. MPI is supported through mpi4py."
    )
    parser.add_argument(
        "set_type",
        choices=("Mandelbrot", "mandelbrot", "Julia", "julia"),
        help="Type of complex set to plot.",
    )
    parser.add_argument(
        "-c",
        nargs=2,
        type=float,
        required=False,
        help="c value to use when plotting Julia sets. Specify 2 floats, ordered real, then imaginary.",
    )
    args = parser.parse_args()
    return vars(args)


def run_set(complex_set: Any):
    """
    Call the various functions of the complex set to do the plotting.
    """

    complex_set.allocate_arrays()
    complex_set.initialise_arrays()
    complex_set.update_colour()
    complex_set.initialise_plot()
    complex_set.zoom_loop()


if __name__ == "__main__":

    inputArgs = command_line_interface()

    # Setting c value
    # Include it in a tuple that get expanded in the class init
    # Will be an empty tuple for Mandelbrot, hence no argument passed
    if inputArgs["set_type"].title() == "Julia":
        if inputArgs["c"] is not None:
            args = complex(*inputArgs["c"])
        else:
            Put("c value not specified for Julia set. Randomising c.", root=True)
            random.seed()
            args = complex(random.random(), random.random())
    else:
        args = ()

    complex_set = getattr(cs, inputArgs["set_type"].title())(*args)
    run_set(complex_set)
