import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# import mpi4py

from time import perf_counter
"""
Basic mandelbrot visualisation code. Not optimised for MPI
"""

def Mandelbrot(z: complex, c: complex) -> complex:
    return z**2 + c


def InitialiseArrays(
    centre: list = [-0.5, 0.0],
    coordinate_width: float = 3,
    coordinate_height: float = 2,
    size_x: int = 900,
    size_y: int = 600,
    zoom_factor: float = 1.0,
) -> tuple:
    """
    Initialise the calculation arrays based on the desired dimensions.

    To avoid stretching the set in an unusual manner, the following equality should hold
        coordinate_width/coordinate_height = size_x/size_y.

    Arguments:
    centre: [float,float]
        [x,y] pair of coordinates for where the plot should centre
    coordinate_width: float
        Width in coordinate space of the plot area. Not to be confused with size_x.
    coordinate_height: float
        Height in coordinate space of the plot area. Not to be confused with size_y.
    size_x: int
        Number of pixels in the x direction. Also the extent of the arrays in that direction.
    size_y: int
        Number of pixels in the y direction. Also the extent of the arrays in that direction.
    zoom_factor: float
        Factor by which too zoom the plot. Is applied after the coordinate height and width are applied.
        This allows the coordinate height and widths to simply be used as starting points.
        Shrinks the axes in both the x and y directions.

    Returns:
    tuple
        (colour, z, c, details)
        colour: np.ndarray - int
            Integer array to store iteration counts. Initialised to zero
        z: np.ndarray - complex
            Complex array to store the iterated value for the set. Initialised to zero
        c: np.ndarray - complex
            Complex array initialised to contain coordinates to be utilised as the c in z^2 + c
    """
    if coordinate_width / coordinate_height != size_x / size_y:
        print(
            "WARNING: Given width and height combined with the array sizes will result in stretched output. Ensure ratios of x/y and width/height are equal to avoid this."
        )

    x_min = centre[0] - coordinate_width / 2 / zoom_factor
    x_max = centre[0] + coordinate_width / 2 / zoom_factor
    y_min = centre[1] - coordinate_height / 2 / zoom_factor
    y_max = centre[1] + coordinate_height / 2 / zoom_factor

    colour = np.zeros([size_y, size_x], dtype=np.int8)
    z = np.zeros([size_y, size_x], dtype=np.complex_)
    c = np.empty([size_y, size_x], dtype=np.complex_)
    c.real = np.tile(np.linspace(x_min, x_max, size_x)[None, :], (size_y,1))
    c.imag = np.tile(np.linspace(y_max, y_min, size_y)[:, None], (1, size_x))

    return colour, z, c


def CalculateColour(centre, zoom_factor):
    size_x = 900
    size_y = 600
    colour, z, c = InitialiseArrays(
        centre=centre,
        coordinate_width=3,
        coordinate_height=2,
        size_x=size_x,
        size_y=size_y,
        zoom_factor=zoom_factor,
    )

    x_min = c[0, 0].real
    y_max = c[0, 0].imag
    x_max = c[-1, -1].real
    y_min = c[-1, -1].imag
    extent = (x_min, x_max, y_min, y_max)

    abs_threshold = 5
    max_iterations = 200

    # ims = []
    for i in range(max_iterations):
        z = Mandelbrot(z, c)
        colour[(abs(z) > abs_threshold) & (colour == 0)] = i + 1
        z[colour != 0] = 0

        # ims.append([ax.imshow(colour.T,animated=True)])
    else:
        colour[colour == 0] = 0

    return colour, extent


def onclick(event):
    global zoom_factor,imag,fig
    centre[0] = event.xdata
    centre[1] = event.ydata
    print(centre)
    print("zooming")

    zoom_factor = zoom_factor * 2
    colour,extent = CalculateColour(centre,zoom_factor)
    
    imag.set_data(colour)
    imag.set_extent(extent)
    fig.canvas.draw()
    fig.canvas.flush_events()

    
    


if __name__ == "__main__":
    zoom_factor = 1.0
    centre = [-0.5, 0]
    plt.ion()

    fig, ax = plt.subplots(figsize=(10,7.5))

    colour,extent = CalculateColour(centre,zoom_factor)
    imag = ax.imshow(colour, extent=extent)#,cmap="autumn")
    ax.hlines(0, -2, 1, color="black", linewidth=0.1)
    ax.vlines(0, -1, 1, color="black", linewidth=0.1)

    
    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.flush_events()
    fig.canvas.draw()
    while True:
        fig.canvas.start_event_loop()
    # while True:

    #     colour, extent = CalculateColour(centre, zoom_factor)
    #     if zoom_factor == 1.0:
    #         imag = ax.imshow(colour, extent=extent)#,cmap="autumn")
    #         ax.hlines(0, -2, 1, color="black", linewidth=0.1)
    #         ax.vlines(0, -1, 1, color="black", linewidth=0.1)
    #     try:
    #         imag.set_data(colour)
    #         imag.set_extent(extent)
    #         fig.canvas.draw()
    #         fig.canvas.flush_events()

    #     except KeyboardInterrupt:
    #         break
