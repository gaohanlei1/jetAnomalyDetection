import math 
import numpy as np

def plot_property_distribution(datatype1_data: np.ndarray, datatype2_data: np.ndarray, prop_name: str,
                               datatype1_label: str, datatype2_label: str, ax, bins: int = 150, is_scaled: bool = True,
                                include_zeros: bool = False, scaled_zero1: float = 0.0, scaled_zero2: float = 0.0) -> None:
    """
    Plots the distribution of a given property for datatype1 and datatype2 data.

    Parameters:
    datatype1_data (np.ndarray): Data for the datatype1 distribution.
    datatype2_data (np.ndarray): Data for the datatype2 distribution.
    prop_name (str): The name of the property being plotted (used for title and axis labels).
    datatype1_label (str): Label for the datatype1 data in the legend.
    datatype2_label (str): Label for the datatype2 data in the legend.
    bins (int, optional): Number of bins for the histogram (default is 150).
    """
    indices_label = ""
    if not include_zeros:
        datatype1_data = [item for item in datatype1_data if not math.isclose(item, scaled_zero1, rel_tol=1e-4, abs_tol=0.0) and not math.isnan(item)]
        datatype2_data = [item for item in datatype2_data if not math.isclose(item, scaled_zero2, rel_tol=1e-4, abs_tol=0.0) and not math.isnan(item)]
        indices_label = ""
    else: 
        datatype1_data = [item for item in datatype1_data if not math.isnan(item)]
        datatype2_data = [item for item in datatype2_data if not math.isnan(item)]

    ax.set_title(f"{'scaled ' if is_scaled else ''}distribution of {prop_name} {indices_label}{'' if include_zeros else ' (w/o zeroes)'}")
    
    ax.set_xlabel(prop_name)
    ax.set_ylabel('Density')

    # Calculate bin range based on min and max of both datasets
    bin_range = (min(np.min(datatype1_data), np.min(datatype2_data)),
                 max(np.max(datatype1_data), np.max(datatype2_data)))

    # Plot histograms with overlapping datatype1 and datatype2 data
    ax.hist(datatype1_data, bins=bins, range=bin_range, density=True, label=datatype1_label, alpha=0.5)
    ax.hist(datatype2_data, bins=bins, range=bin_range, density=True, label=datatype2_label, alpha=0.5)

    # Add a legend and display the plot
    ax.legend(loc='upper right')
    return ax