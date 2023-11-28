from matplotlib import collections as mc
from matplotlib.colors import colorConverter
from collections import deque
from itertools import islice
import pickle
import matplotlib.pyplot as plt
import numpy as np


def sliding_window(iterable, n):
    """
    sliding_window('ABCDEFG', 4) -> ABCD BCDE CDEF DEFG

    recipe from python docs
    """
    it = iter(iterable)
    window = deque(islice(it, n), maxlen=n)
    if len(window) == n:
        yield tuple(window)
    for x in it:
        window.append(x)
        yield tuple(window)


def color_gradient(x, y, c1, c2, **kwargs):
    """
    Creates a line collection with a gradient from colors c1 to c2,
    from data x and y.
    """
    n = len(x)
    if len(y) != n:
        raise ValueError("x and y data lengths differ")
    return mc.LineCollection(
        sliding_window(zip(x, y), 2),
        colors=np.linspace(colorConverter.to_rgb(c1), colorConverter.to_rgb(c2), n - 1),
        **kwargs,
    )


def assemble_segments(data, segments):
    final = []

    for begin, end in zip(segments, segments[1:]):
        if begin > end:
            final.append(np.concatenate([data[begin:-1], data[:end]]))
        else:
            final.append(data[begin:end])

    return np.concatenate(final)


def plot_modulation_interaction_diagram(all_value, all_modulation, phase_indices):
    fig, ax = plt.subplots()

    closed =  abs(all_value[phase_indices[0]]- all_value[phase_indices[-1]])< 1e-3
    if not closed:
        phase_indices = np.concatenate([phase_indices, [phase_indices[0]]])

    ax.plot(all_modulation, all_value, linewidth=3, color="cornflowerblue")

    modulation = assemble_segments(all_modulation, phase_indices)
    value = assemble_segments(all_value, phase_indices)

    ax.add_collection(
        color_gradient(modulation, value, "cornflowerblue", "red", linewidth=3)
    )

    phase_indices = phase_indices[:-1]
    last = np.array([np.nan, np.nan])
    
    for i, index in enumerate(phase_indices):
        ax.scatter(
            all_modulation[index], all_value[index], zorder=100, marker=".", s=200, color="black"
        )
        
        next = np.array((all_modulation[index], all_value[index]))
        
        ax.annotate(
            str(i + 1),
            xy=next,
            xytext=(5, 5)  if (np.linalg.norm(next- last)) > 1e-2 else (5, -10),
            xycoords="data",
            textcoords="offset points",
        )

        last = next

    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(abs(x1-x0)/abs(y1-y0))

    return fig, ax


def plot_diagrams(name):
    with open(f"data/pv_{name}.pickle", "rb") as file:
        diagram_data = pickle.load(file)

    for data in diagram_data:
        fig, ax = plot_modulation_interaction_diagram(*data["args"])
        ax.set_xlabel(data["xlabel"])
        ax.set_ylabel(data["ylabel"])

        plt.savefig(f"figures/{name}_{data['name']}")

plot_diagrams("baseline")

plot_diagrams("slow_shifted")
