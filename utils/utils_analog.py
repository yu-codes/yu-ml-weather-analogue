"""
Weather Analogue Utilities Module
=================================

This module contains utility functions for finding and analyzing weather analogues.
It provides similarity metrics, search algorithms, and visualization tools for
comparing weather patterns across different time periods.
"""

import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


def generate_time_weights(time_window, leadtime):
    """
    Generate normalized weights that increase with time for analogue selection.
    
    This function creates a list of weights that emphasize more recent time points
    in the analogue matching process. The weights are normalized to sum to 1.
    
    Args:
        time_window (int): The number of time steps to consider
        leadtime (int): The lead time before the event
        
    Returns:
        list: Normalized weights for each time step
    
    Example:
        >>> generate_time_weights(3, 1)
        [0.16666666666666666, 0.25, 0.25, 0.3333333333333333]
    """
    weights = list(range(leadtime + 1, leadtime + time_window + 2))
    total = sum(weights)
    normalized_weights = [w / total for w in weights]
    return normalized_weights


def cosine_similarity(weight_matrix, data_A, data_B):
    """
    Calculate weighted cosine similarity between two data arrays.
    
    This function computes the cosine similarity between two data arrays after
    applying a weighting matrix. The result is transformed to a similarity score
    between 0 and 1, where 1 indicates identical data patterns.
    
    Args:
        weight_matrix (numpy.ndarray): Matrix of weights to apply to the data
        data_A (numpy.ndarray): First data array
        data_B (numpy.ndarray): Second data array
        
    Returns:
        float: Similarity score between 0 and 1
    """
    # Flatten the arrays for vector operations
    weight_matrix = weight_matrix.flatten()
    data_A = data_A.flatten()
    data_B = data_B.flatten()

    # Apply weights to data
    weighted_A = data_A * weight_matrix
    weighted_B = data_B * weight_matrix

    # Calculate cosine similarity
    dot_product = np.dot(weighted_A, weighted_B)
    norm_A = np.linalg.norm(weighted_A)
    norm_B = np.linalg.norm(weighted_B)

    # Handle zero-norm edge case
    if norm_A == 0 or norm_B == 0:
        return 0

    # Convert cosine similarity to angle and normalize to 0-1 range
    similarity = dot_product / (norm_A * norm_B)
    angle = np.arccos(np.clip(similarity, -1.0, 1.0))  # Clip to prevent numerical errors
    score = 1 - (angle / np.pi)
    return score


def euclidean_similarity(weight_matrix, data_A, data_B):
    """
    Calculate weighted Euclidean distance-based similarity between two data arrays.
    
    This function computes a similarity score based on the weighted Euclidean distance
    between two data arrays. The distance is converted to a similarity score using
    an exponential function, resulting in a value between 0 and 1, where 1 indicates
    identical data patterns.
    
    Args:
        weight_matrix (numpy.ndarray): Matrix of weights to apply to the data
        data_A (numpy.ndarray): First data array
        data_B (numpy.ndarray): Second data array
        
    Returns:
        float: Similarity score between 0 and 1
    """
    # Flatten the arrays for vector operations
    weight_matrix = weight_matrix.flatten()
    data_A = data_A.flatten()
    data_B = data_B.flatten()

    # Calculate weighted squared differences
    weighted_diff = (weight_matrix**2) * (data_A - data_B) ** 2
    
    # Calculate Euclidean distance
    distance = np.sqrt(np.sum(weighted_diff))
    
    # Convert distance to similarity score using exponential decay
    score = np.exp(-distance)
    return score


def ssim_similarity(weight_matrix, data_A, data_B):
    """
    Calculate Structural Similarity Index (SSIM) between two weighted data arrays.
    
    This function applies weights to the data arrays and then computes the SSIM,
    which measures similarity based on luminance, contrast, and structure. 
    The result is a value between -1 and 1, where 1 indicates perfect similarity.
    
    Args:
        weight_matrix (numpy.ndarray): Matrix of weights to apply to the data
        data_A (numpy.ndarray): First data array
        data_B (numpy.ndarray): Second data array
        
    Returns:
        float: SSIM score between -1 and 1
    """
    # Apply weights to data
    weighted_data_A = data_A * weight_matrix
    weighted_data_B = data_B * weight_matrix

    # Calculate data range for SSIM
    overall_max = max(weighted_data_A.max(), weighted_data_B.max())
    overall_min = min(weighted_data_A.min(), weighted_data_B.min())
    data_range = overall_max - overall_min

    # Calculate SSIM score
    score = ssim(weighted_data_A, weighted_data_B, data_range=data_range)
    return score


def search_analogs_atmodist(
    data, timestamps, time_window, event_start_time, analogue_number
):
    """
    Search for weather analogues using the atmospheric distance method.
    
    This function identifies similar weather patterns by computing the mean squared
    difference between a target time window and all possible time windows in the data.
    
    Args:
        data (list): List of weather data arrays
        timestamps (list): List of timestamp strings corresponding to data entries
        time_window (int): Number of time steps to consider for each window
        event_start_time (str): Start time of the target event
        analogue_number (int): Number of top analogues to return
        
    Returns:
        dict: Dictionary of top analogue events with scores and timestamps
    """
    # Find the index of the target event start time
    event_start_index = timestamps.index(event_start_time)
    
    # Convert timestamps to numpy datetime format
    timestamps = pd.to_datetime(timestamps).to_numpy()
    time_window_delta = np.timedelta64(time_window, "1h")
    
    # Extract target window data
    window_data = data[max(0, event_start_index - time_window) : event_start_index]
    
    # Calculate distance for all possible windows
    distances = []
    for i in range(time_window, len(data)):
        current_window = data[i - time_window : i]
        # Mean squared difference between windows
        distance = np.mean(np.sum((window_data - current_window) ** 2, axis=1))
        distances.append((timestamps[i - time_window], distance))
    
    # Sort by distance (smaller distance means higher similarity)
    distances.sort(key=lambda x: x[1])
    top_events = distances[:analogue_number]
    
    # Format results as a dictionary
    result = {
        id: {
            "score": score,
            "start_time": datetime,
            "end_time": datetime + time_window_delta,
        }
        for id, (datetime, score) in enumerate(top_events)
    }
    
    return result


def search_analogs(
    data,
    similarity_method,
    analogue_number,
    event_start_time,
    time_window,
    lead_time,
    variable_weights,
    grid_weights,
):
    """
    Search for weather analogues using a specified similarity method.
    
    This function identifies similar weather patterns by comparing the target event
    with potential analogues using a weighted similarity calculation across multiple
    variables and time steps.
    
    Args:
        data (list): List of tuples containing (timestamp, data_array)
        similarity_method (function): Function to calculate similarity between data arrays
        analogue_number (int): Number of top analogues to return
        event_start_time (str): Start time of the target event
        time_window (int): Number of time steps to consider for each window
        lead_time (int): Lead time before the event
        variable_weights (list): Weights for different variables
        grid_weights (numpy.ndarray): Spatial weights for the grid points
        
    Returns:
        dict: Dictionary of top analogue events with scores and timestamps
    """
    # Convert target event time to numpy datetime
    event_start_time = pd.to_datetime(event_start_time).to_numpy()
    time_window_delta = np.timedelta64(time_window, "h")

    # Define target event time range
    target_event_start = event_start_time - time_window_delta
    target_event_end = event_start_time

    # Extract target event data within the specified time range
    target_event = [
        d[1] for d in data if target_event_start <= d[0] <= target_event_end
    ]

    # Check if any events were found
    if not target_event:
        print("No events found within the specified time range.")
        return {}

    # Generate temporal weights for the time window
    time_weights = generate_time_weights(time_window, lead_time)
    
    # Calculate similarity scores for all potential analogues
    scores = []
    for data_index, (datetime, dataset) in enumerate(data):
        total_similarity = 0
        for var_index, var_weight in enumerate(variable_weights):
            for target_index, target_data in enumerate(target_event):
                # Ensure we don't go beyond the data length
                if data_index + target_index < len(data):
                    # Calculate similarity for this variable and time step
                    similarity = similarity_method(
                        grid_weights,
                        target_data[var_index],
                        data[data_index + target_index][1][var_index],
                    )
                    # Apply variable and time weights
                    weighted_similarity = (
                        similarity * var_weight * time_weights[target_index]
                    )
                    total_similarity += weighted_similarity
        scores.append((datetime, total_similarity))
    
    # Sort by similarity score (higher is better) and select top analogues
    scores.sort(key=lambda x: x[1], reverse=True)
    top_events = scores[:analogue_number]

    # Format results as a dictionary
    result = {
        id: {
            "score": score,
            "start_time": datetime,
            "end_time": datetime + time_window_delta,
        }
        for id, (datetime, score) in enumerate(top_events)
    }
    return result


def revise_analogs(
    result,
    remove_predicting_event,
    remove_overlapping_events,
    remove_extra_events,
    analog_number,
    start_time,
    end_time,
    lead_time,
):
    """
    Revise the analogue results by applying various filtering criteria.
    
    This function applies three optional filters to refine the analogue results:
    1. Remove analogues that overlap with the prediction event
    2. Remove analogues that overlap with other analogues
    3. Limit the number of analogues to the specified amount
    
    Args:
        result (dict): Dictionary of analogue events
        remove_predicting_event (bool): Whether to remove analogues overlapping with prediction event
        remove_overlapping_events (bool): Whether to remove overlapping analogues
        remove_extra_events (bool): Whether to limit the number of analogues
        analog_number (int): Maximum number of analogues to keep
        start_time (str): Start time of the prediction event
        end_time (str): End time of the prediction event
        lead_time (int): Lead time in hours
        
    Returns:
        dict: Filtered dictionary of analogue events
    """
    # Convert input times to pandas datetime objects
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    lead_time_delta = pd.to_timedelta(lead_time, unit="h")

    # Get the first key to preserve it throughout filtering
    first_key = next(iter(result))

    if remove_predicting_event:
        # Step 1: Remove analogues that overlap with the prediction event time range
        result = {
            key: val
            for key, val in result.items()
            if key == first_key  # Always keep the first analogue
            or not (
                val["start_time"] < end_time
                and val["end_time"] + lead_time_delta > start_time
            )
        }

    if remove_overlapping_events:
        # Step 2: Remove analogues that overlap with other analogues
        # Process in order of keys to preserve higher-ranked analogues
        sorted_keys = sorted(result.keys())
        to_remove = set()
        for i, key1 in enumerate(sorted_keys):
            for j in range(i + 1, len(sorted_keys)):
                key2 = sorted_keys[j]
                # Check for time overlap
                if (
                    result[key1]["end_time"] > result[key2]["start_time"]
                    and result[key1]["start_time"] < result[key2]["end_time"]
                ):
                    to_remove.add(key2)  # Remove the lower-ranked analogue
        result = {key: val for key, val in result.items() if key not in to_remove}

    if remove_extra_events:
        # Step 3: Limit the number of analogues to analog_number
        if len(result) > analog_number:
            keys_to_keep = list(result.keys())
            keys_to_keep = keys_to_keep[:analog_number]  # Keep only the top analogues
            # Renumber the keys starting from 0
            result = {
                new_key: result[old_key] for new_key, old_key in enumerate(keys_to_keep)
            }
            
    return result


def attach_data_to_analogs(data, analog_dictionary):
    """
    Attach actual data arrays to the analogue dictionary.
    
    This function enhances the analogue dictionary by adding the corresponding
    data arrays for each analogue event based on its time range.
    
    Args:
        data (list): List of tuples containing (timestamp, data_array)
        analog_dictionary (dict): Dictionary of analogue events
        
    Returns:
        dict: Enhanced analogue dictionary with data arrays
    """
    for key, value in analog_dictionary.items():
        # Convert timestamps to numpy datetime format for consistent comparison
        start_time = np.datetime64(value["start_time"])
        end_time = np.datetime64(value["end_time"])
        
        # Ensure timestamps are in numpy.datetime64 format
        value["start_time"] = start_time
        value["end_time"] = end_time
        
        # Extract data for the analogue's time range
        value["data"] = [
            d[1] for d in data if start_time <= np.datetime64(d[0]) <= end_time
        ]
        
    return analog_dictionary


def plot_analogs(
    analog_dictionary,
    specified_index,
    num_analogs_to_plot=None,
    variable_list=["d2m", "u", "v", "msl", "r"],
):
    """
    Plot analogue time series for a specific grid point.
    
    This function creates a multi-panel plot showing time series for each variable
    at a specified grid point. The original event is highlighted in blue, and analogues
    are shown in shades of red.
    
    Args:
        analog_dictionary (dict): Dictionary of analogue events with data
        specified_index (int): Index of the grid point to plot
        num_analogs_to_plot (int, optional): Number of analogues to include in the plot
        variable_list (list, optional): List of variable names for plot titles
        
    Returns:
        None: The function displays the plot directly
    """
    # Determine number of variables and time steps
    num_channels = len(variable_list)
    time_steps = len(list(analog_dictionary.values())[0]["data"])
    x_axis = np.arange(time_steps)

    # Determine how many analogues to plot
    if num_analogs_to_plot is None or num_analogs_to_plot > len(analog_dictionary):
        num_analogs_to_plot = len(analog_dictionary)

    # Create a figure with subplots for each variable
    fig, axs = plt.subplots(1, num_channels, figsize=(8, 2), sharex=True)
    titles = variable_list
    
    # Create color gradient from deep to light red for analogues
    color_palette = plt.cm.Reds(np.linspace(1, 0, num_analogs_to_plot))

    # Plot each variable
    for ch in range(num_channels):
        handles = []
        labels = []
        
        # Plot each analogue in reverse order (to put original on top)
        for i, (key, analog) in enumerate(
            list(analog_dictionary.items())[:num_analogs_to_plot][::-1]
        ):
            # Extract data for this variable at the specified grid point
            data = analog["data"]
            row, col = specified_index // 32, specified_index % 32  # Convert to grid coordinates
            time_series = [data[t][ch, row, col] for t in range(len(data))]

            # Format timestamp for label
            start_time = analog["start_time"]
            label = np.datetime_as_string(start_time, unit="s")

            # Plot original event in blue, analogues in red gradient
            if i == num_analogs_to_plot - 1:  # Original event (first in dictionary)
                (line,) = axs[ch].plot(
                    x_axis,
                    time_series,
                    label=f"observation ({label})",
                    color="blue",
                    zorder=num_analogs_to_plot,  # Place on top
                )
            else:
                (line,) = axs[ch].plot(
                    x_axis,
                    time_series,
                    label=label,
                    color=color_palette[num_analogs_to_plot - 2 - i],
                    zorder=num_analogs_to_plot - 1 - i,
                )

            handles.append(line)
            labels.append(line.get_label())

        # Set subplot title
        axs[ch].set_title(f"{titles[ch]}")
        
        # Add y-axis label only on the first subplot
        if ch == 0:
            axs[ch].set_ylabel("value")

    # Add common x-axis label
    fig.text(0.5, 0.04, "Time Steps", ha="center")

    # Reverse the handles and labels to match the plotting order
    handles = handles[::-1]
    labels = labels[::-1]

    # Add legend below the plot
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=min(num_analogs_to_plot, int(num_channels * 0.7)),
    )
    
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.show()


def plot_analogs_with_band(
    analog_dictionary, specified_index, num_analogs_to_plot=None
):
    """
    Plot analogues with mean and standard deviation bands for a specific grid point.
    
    This function creates a multi-panel plot showing time series for each variable
    at a specified grid point. The original event is shown in blue, the mean of analogues
    in red, and a red shaded band shows the standard deviation.
    
    Args:
        analog_dictionary (dict): Dictionary of analogue events with data
        specified_index (int): Index of the grid point to plot
        num_analogs_to_plot (int, optional): Number of analogues to include in the calculation
        
    Returns:
        None: The function displays the plot directly
    """
    # Number of variables and time steps
    num_channels = 5  # Default: t, u, v, z, q
    time_steps = len(list(analog_dictionary.values())[0]["data"])
    x_axis = np.arange(time_steps)
    
    # Variable names for plot titles
    titles = ["t", "u", "v", "z", "q"]

    # Determine how many analogues to plot
    if num_analogs_to_plot is None or num_analogs_to_plot > len(analog_dictionary):
        num_analogs_to_plot = len(analog_dictionary)

    # Create a figure with subplots for each variable
    fig, axs = plt.subplots(1, num_channels, figsize=(8, 2), sharex=True)

    # Plot each variable
    for ch in range(num_channels):
        all_time_series = []
        observation_series = None
        
        # Extract data for each analogue
        for i, (key, analog) in enumerate(analog_dictionary.items()):
            if i >= num_analogs_to_plot:
                break
                
            # Get data for this variable at the specified grid point
            data = analog["data"]
            row, col = specified_index // 32, specified_index % 32  # Convert to grid coordinates
            time_series = [data[t][ch, row, col] for t in range(len(data))]

            # First item is the observation, others are analogues
            if i == 0:
                observation_series = time_series
            else:
                all_time_series.append(time_series)

        # Convert to numpy array for calculations
        all_time_series = np.array(all_time_series)
        
        # Calculate mean and standard deviation across analogues
        mean_series = np.mean(all_time_series, axis=0)
        std_series = np.std(all_time_series, axis=0)

        # Plot observation (blue), mean prediction (red), and standard deviation band
        axs[ch].plot(x_axis, observation_series, label="Observation", color="blue")
        axs[ch].plot(x_axis, mean_series, label="Prediction", color="red")
        axs[ch].fill_between(
            x_axis,
            mean_series - std_series,
            mean_series + std_series,
            color="red",
            alpha=0.3,
            label="Prediction Band",
        )

        # Set subplot title
        axs[ch].set_title(f"{titles[ch]}")
        
        # Add y-axis label only on the first subplot
        if ch == 0:
            axs[ch].set_ylabel("value")

    # Add common x-axis label
    fig.text(0.5, 0.04, "Time Steps", ha="center")
    
    # Get handles and labels from the first subplot for the legend
    handles, labels = axs[0].get_legend_handles_labels()
    
    # Add legend below the plot
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        ncol=min(3, int(num_channels * 0.7)),
    )
    
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.show()
