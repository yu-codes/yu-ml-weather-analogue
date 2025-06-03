import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim  # for ssim_similarity
import matplotlib.pyplot as plt


def generate_time_weights(time_window, leadtime):
    weights = list(range(leadtime + 1, leadtime + time_window + 2))
    total = sum(weights)
    normalized_weights = [w / total for w in weights]
    return normalized_weights


def cosine_similarity(weight_matrix, data_A, data_B):
    weight_matrix = weight_matrix.flatten()
    data_A = data_A.flatten()
    data_B = data_B.flatten()

    weighted_A = data_A * weight_matrix
    weighted_B = data_B * weight_matrix

    dot_product = np.dot(weighted_A, weighted_B)
    norm_A = np.linalg.norm(weighted_A)
    norm_B = np.linalg.norm(weighted_B)

    if norm_A == 0 or norm_B == 0:
        return 0

    similarity = dot_product / (norm_A * norm_B)
    angle = np.arccos(np.clip(similarity, -1.0, 1.0))
    score = 1 - (angle / np.pi)
    return score


def euclidean_similarity(weight_matrix, data_A, data_B):
    weight_matrix = weight_matrix.flatten()
    data_A = data_A.flatten()
    data_B = data_B.flatten()

    weighted_diff = (weight_matrix**2) * (data_A - data_B) ** 2
    distance = np.sqrt(np.sum(weighted_diff))
    score = np.exp(-distance)
    return score


def ssim_similarity(weight_matrix, data_A, data_B):
    weighted_data_A = data_A * weight_matrix
    weighted_data_B = data_B * weight_matrix

    overall_max = max(weighted_data_A.max(), weighted_data_B.max())
    overall_min = min(weighted_data_A.min(), weighted_data_B.min())
    data_range = overall_max - overall_min

    score = ssim(weighted_data_A, weighted_data_B, data_range=data_range)
    return score


def search_analogs_atmodist(
    data, timestamps, time_window, event_start_time, analogue_number
):
    event_start_index = timestamps.index(event_start_time)
    timestamps = pd.to_datetime(timestamps).to_numpy()
    time_window_delta = np.timedelta64(time_window, "1h")

    window_data = data[max(0, event_start_index - time_window) : event_start_index]

    distances = []
    for i in range(time_window, len(data)):
        current_window = data[i - time_window : i]
        distance = np.mean(np.sum((window_data - current_window) ** 2, axis=1))
        distances.append((timestamps[i - time_window], distance))

    distances.sort(key=lambda x: x[1])
    top_events = distances[:analogue_number]

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
    ## events
    event_start_time = pd.to_datetime(event_start_time).to_numpy()
    time_window_delta = np.timedelta64(time_window, "h")

    target_event_start = event_start_time - time_window_delta
    target_event_end = event_start_time

    target_event = [
        d[1] for d in data if target_event_start <= d[0] <= target_event_end
    ]

    if not target_event:
        print("No events found within the specified time range.")
        return {}

    ## calculate similarity
    time_weights = generate_time_weights(time_window, lead_time)
    scores = []
    for data_index, (datetime, dataset) in enumerate(data):
        total_similarity = 0
        for var_index, var_weight in enumerate(variable_weights):
            for target_index, target_data in enumerate(target_event):
                if data_index + target_index < len(data):
                    similarity = similarity_method(
                        grid_weights,
                        target_data[var_index],
                        data[data_index + target_index][1][var_index],
                    )
                    weighted_similarity = (
                        similarity * var_weight * time_weights[target_index]
                    )
                    total_similarity += weighted_similarity
        scores.append((datetime, total_similarity))
    scores.sort(key=lambda x: x[1], reverse=True)
    top_events = scores[:analogue_number]

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
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    lead_time_delta = pd.to_timedelta(lead_time, unit="h")

    first_key = next(iter(result))

    if remove_predicting_event:
        # Step 1: Remove predicting event analogs
        result = {
            key: val
            for key, val in result.items()
            if key == first_key
            or not (
                val["start_time"] < end_time
                and val["end_time"] + lead_time_delta > start_time
            )
        }

    if remove_overlapping_events:
        # Step 2: Remove overlapping events
        sorted_keys = sorted(result.keys())
        to_remove = set()
        for i, key1 in enumerate(sorted_keys):
            for j in range(i + 1, len(sorted_keys)):
                key2 = sorted_keys[j]
                if (
                    result[key1]["end_time"] > result[key2]["start_time"]
                    and result[key1]["start_time"] < result[key2]["end_time"]
                ):
                    to_remove.add(key2)
        result = {key: val for key, val in result.items() if key not in to_remove}

    if remove_extra_events:
        # Step 3: Remove extra events
        if len(result) > analog_number:
            keys_to_keep = list(result.keys())
            keys_to_keep = keys_to_keep[:analog_number]
            result = {
                new_key: result[old_key] for new_key, old_key in enumerate(keys_to_keep)
            }
    return result


def attach_data_to_analogs(data, analog_dictionary):
    for key, value in analog_dictionary.items():
        start_time = np.datetime64(value["start_time"])
        end_time = np.datetime64(value["end_time"])
        value["start_time"] = start_time  # 确保 start_time 是 numpy.datetime64 类型
        value["end_time"] = end_time  # 确保 end_time 是 numpy.datetime64 类型
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
    num_channels = len(variable_list)
    time_steps = len(list(analog_dictionary.values())[0]["data"])
    x_axis = np.arange(time_steps)

    if num_analogs_to_plot is None or num_analogs_to_plot > len(analog_dictionary):
        num_analogs_to_plot = len(analog_dictionary)

    fig, axs = plt.subplots(1, num_channels, figsize=(8, 2), sharex=True)
    titles = variable_list
    color_palette = plt.cm.Reds(
        np.linspace(1, 0, num_analogs_to_plot)
    )  # 从深到浅分配颜色

    for ch in range(num_channels):
        handles = []
        labels = []
        for i, (key, analog) in enumerate(
            list(analog_dictionary.items())[:num_analogs_to_plot][::-1]
        ):
            data = analog["data"]
            time_series = [
                data[ch, specified_index // 32, specified_index % 32] for data in data
            ]

            # 转换 start_time 为易读格式
            start_time = analog["start_time"]
            label = np.datetime_as_string(start_time, unit="s")

            if i == num_analogs_to_plot - 1:  # Original first
                (line,) = axs[ch].plot(
                    x_axis,
                    time_series,
                    label=f"observation ({label})",
                    color="blue",
                    zorder=num_analogs_to_plot,
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

        axs[ch].set_title(f"{titles[ch]}")
        if ch == 0:
            axs[ch].set_ylabel("value")

    fig.text(0.5, 0.04, "Time Steps", ha="center")

    # Reverse the order of handles and labels to match the original plotting order
    handles = handles[::-1]
    labels = labels[::-1]

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=num_channels * 0.7,
    )
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.show()


def plot_analogs_with_band(
    analog_dictionary, specified_index, num_analogs_to_plot=None
):
    num_channels = 5
    time_steps = len(list(analog_dictionary.values())[0]["data"])
    x_axis = np.arange(time_steps)

    if num_analogs_to_plot is None or num_analogs_to_plot > len(analog_dictionary):
        num_analogs_to_plot = len(analog_dictionary)

    fig, axs = plt.subplots(1, num_channels, figsize=(8, 2), sharex=True)
    titles = ["t", "u", "v", "z", "q"]

    for ch in range(num_channels):
        all_time_series = []
        observation_series = None

        for i, (key, analog) in enumerate(analog_dictionary.items()):
            if i >= num_analogs_to_plot:
                break
            data = analog["data"]
            time_series = [
                data[ch, specified_index // 32, specified_index % 32] for data in data
            ]

            if i == 0:
                observation_series = time_series
            else:
                all_time_series.append(time_series)

        all_time_series = np.array(all_time_series)
        mean_series = np.mean(all_time_series, axis=0)
        std_series = np.std(all_time_series, axis=0)

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

        axs[ch].set_title(f"{titles[ch]}")
        if ch == 0:
            axs[ch].set_ylabel("value")

    fig.text(0.5, 0.04, "Time Steps", ha="center")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        ncol=num_channels * 0.7,
    )
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.show()
