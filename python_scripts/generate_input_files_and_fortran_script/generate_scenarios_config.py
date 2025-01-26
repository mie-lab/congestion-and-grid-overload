import datetime as dt
from datetime import datetime
import sys
import matplotlib as mpl
from matplotlib import cm
from slugify import slugify

import shared_config


def generate_revised_scenarios_with_colors():
    # Define the ranges for capacity remaining and durations
    capacity_remaining = [0.01, 0.05, 0.10, 0.20, 0.40, 0.80]  # Represented as fractions
    durations = [15, 30, 45, 60, 90]  # Durations in minutes


    # Initialize the dictionary to store scenarios
    revised_scenarios = {}


    # List of 50 colors
    colors = ['#%02x%02x%02x' % (r, g, b) for r in range(50, 255, 10) for g in range(50, 255, 10) for b in range(50, 255, 10)]
    colors = colors[:100]  # Limit to first 50 unique colors

    # Counter for color assignment
    color_index = 0


    # Start time for all accidents (9 AM in delta units)
    start_time_delta_unit = 3600 * (9 - 5)
    # Iterate over each combination of capacity remaining and duration
    for remaining in capacity_remaining:
        for duration in durations:
            # Create a unique scenario name
            scenario_name = slugify(f"{duration} mins accident ({int(remaining * 100)}% Capacity Remaining)-start-9AM")
            # Define the scenario parameters
            revised_scenarios[scenario_name] = {
                "dict_key_cell_value_fractional_capacity": dict(zip(shared_config.accident_Cell_list, [remaining] * len(shared_config.accident_Cell_list))),
                "accident_duration_start_time_delta_unit": start_time_delta_unit,
                "accident_duration_end_time_delta_unit": start_time_delta_unit + duration * 60,
                "color": colors[color_index % len(colors)],  # Cycle through the color list
                "list_of_flows": [remaining] * (duration * 60 + 1)
            }
            # Increment color index for next scenario
            color_index += 1



    # Start time for all accidents (8:30 AM in delta units)
    start_time_delta_unit = int(3600 * (8.5 - 5))
    # Iterate over each combination of capacity remaining and duration
    for remaining in capacity_remaining:
        for duration in durations:
            # Create a unique scenario name
            scenario_name = slugify(f"{duration} mins accident ({int(remaining * 100)}% Capacity Remaining)-start-830AM")
            # Define the scenario parameters
            revised_scenarios[scenario_name] = {
                "dict_key_cell_value_fractional_capacity": dict(zip(shared_config.accident_Cell_list, [remaining] * len(shared_config.accident_Cell_list))),
                "accident_duration_start_time_delta_unit": start_time_delta_unit,
                "accident_duration_end_time_delta_unit": start_time_delta_unit + duration * 60,
                "color": colors[color_index % len(colors)],  # Cycle through the color list
                "list_of_flows": [remaining] * (duration * 60 + 1)
            }
            # Increment color index for next scenario
            color_index += 1

    # Start time for all accidents (8 AM in delta units)
    start_time_delta_unit = (3600 * (8 - 5))
    # Iterate over each combination of capacity remaining and duration
    for remaining in capacity_remaining:
        for duration in durations:
            # Create a unique scenario name
            scenario_name = slugify(f"{duration} mins accident ({int(remaining * 100)}% Capacity Remaining)-start-8AM")
            # Define the scenario parameters
            revised_scenarios[scenario_name] = {
                "dict_key_cell_value_fractional_capacity": dict(zip(shared_config.accident_Cell_list, [remaining] * len(shared_config.accident_Cell_list))),
                "accident_duration_start_time_delta_unit": start_time_delta_unit,
                "accident_duration_end_time_delta_unit": start_time_delta_unit + duration * 60,
                "color": colors[color_index % len(colors)],  # Cycle through the color list
                "list_of_flows": [remaining] * (duration * 60 + 1)
            }
            # Increment color index for next scenario
            color_index += 1

    # Start time for all accidents (7:30 AM in delta units)
    start_time_delta_unit = int(3600 * (7.5 - 5))
    # Iterate over each combination of capacity remaining and duration
    for remaining in capacity_remaining:
        for duration in durations:
            # Create a unique scenario name
            scenario_name = slugify(f"{duration} mins accident ({int(remaining * 100)}% Capacity Remaining)-start-730AM")
            # Define the scenario parameters
            revised_scenarios[scenario_name] = {
                "dict_key_cell_value_fractional_capacity": dict(zip(shared_config.accident_Cell_list, [remaining] * len(shared_config.accident_Cell_list))),
                "accident_duration_start_time_delta_unit": start_time_delta_unit,
                "accident_duration_end_time_delta_unit": start_time_delta_unit + duration * 60,
                "color": colors[color_index % len(colors)],  # Cycle through the color list
                "list_of_flows": [remaining] * (duration * 60 + 1)
            }
            # Increment color index for next scenario
            color_index += 1

    # Start time for all accidents (7 AM in delta units)
    start_time_delta_unit = (3600 * (7 - 5))
    # Iterate over each combination of capacity remaining and duration
    for remaining in capacity_remaining:
        for duration in durations:
            # Create a unique scenario name
            scenario_name = slugify(f"{duration} mins accident ({int(remaining * 100)}% Capacity Remaining)-start-7AM")
            # Define the scenario parameters
            revised_scenarios[scenario_name] = {
                "dict_key_cell_value_fractional_capacity": dict(zip(shared_config.accident_Cell_list, [remaining] * len(shared_config.accident_Cell_list))),
                "accident_duration_start_time_delta_unit": start_time_delta_unit,
                "accident_duration_end_time_delta_unit": start_time_delta_unit + duration * 60,
                "color": colors[color_index % len(colors)],  # Cycle through the color list
                "list_of_flows": [remaining] * (duration * 60 + 1)
            }
            # Increment color index for next scenario
            color_index += 1

    # Start time for all accidents (6:30AM in delta units)
    start_time_delta_unit = int(3600 * (6.5 - 5))
    # Iterate over each combination of capacity remaining and duration
    for remaining in capacity_remaining:
        for duration in durations:
            # Create a unique scenario name
            scenario_name = slugify(f"{duration} mins accident ({int(remaining * 100)}% Capacity Remaining)-start-630AM")
            # Define the scenario parameters
            revised_scenarios[scenario_name] = {
                "dict_key_cell_value_fractional_capacity": dict(zip(shared_config.accident_Cell_list, [remaining] * len(shared_config.accident_Cell_list))),
                "accident_duration_start_time_delta_unit": start_time_delta_unit,
                "accident_duration_end_time_delta_unit": start_time_delta_unit + duration * 60,
                "color": colors[color_index % len(colors)],  # Cycle through the color list
                "list_of_flows": [remaining] * (duration * 60 + 1)
            }
            # Increment color index for next scenario
            color_index += 1

    # Start time for all accidents (6 AM in delta units)
    start_time_delta_unit = (3600 * (6 - 5))
    # Iterate over each combination of capacity remaining and duration
    for remaining in capacity_remaining:
        for duration in durations:
            # Create a unique scenario name
            scenario_name = slugify(f"{duration} mins accident ({int(remaining * 100)}% Capacity Remaining)-start-6AM")
            # Define the scenario parameters
            revised_scenarios[scenario_name] = {
                "dict_key_cell_value_fractional_capacity": dict(zip(shared_config.accident_Cell_list, [remaining] * len(shared_config.accident_Cell_list))),
                "accident_duration_start_time_delta_unit": start_time_delta_unit,
                "accident_duration_end_time_delta_unit": start_time_delta_unit + duration * 60,
                "color": colors[color_index % len(colors)],  # Cycle through the color list
                "list_of_flows": [remaining] * (duration * 60 + 1)
            }
            # Increment color index for next scenario
            color_index += 1

    # Start time for all accidents (930 AM in delta units)
    start_time_delta_unit = int(3600 * (9.5 - 5))
    # Iterate over each combination of capacity remaining and duration
    for remaining in capacity_remaining:
        for duration in durations:
            # Create a unique scenario name
            scenario_name = slugify(f"{duration} mins accident ({int(remaining * 100)}% Capacity Remaining)-start-930AM")
            # Define the scenario parameters
            revised_scenarios[scenario_name] = {
                "dict_key_cell_value_fractional_capacity": dict(zip(shared_config.accident_Cell_list, [remaining] * len(shared_config.accident_Cell_list))),
                "accident_duration_start_time_delta_unit": start_time_delta_unit,
                "accident_duration_end_time_delta_unit": start_time_delta_unit + duration * 60,
                "color": colors[color_index % len(colors)],  # Cycle through the color list
                "list_of_flows": [remaining] * (duration * 60 + 1)
            }
            # Increment color index for next scenario
            color_index += 1

    # Start time for all accidents (10 AM in delta units)
    start_time_delta_unit = (3600 * (10 - 5))
    # Iterate over each combination of capacity remaining and duration
    for remaining in capacity_remaining:
        for duration in durations:
            # Create a unique scenario name
            scenario_name = slugify(f"{duration} mins accident ({int(remaining * 100)}% Capacity Remaining)-start-10AM")
            # Define the scenario parameters
            revised_scenarios[scenario_name] = {
                "dict_key_cell_value_fractional_capacity": dict(zip(shared_config.accident_Cell_list, [remaining] * len(shared_config.accident_Cell_list))),
                "accident_duration_start_time_delta_unit": start_time_delta_unit,
                "accident_duration_end_time_delta_unit": start_time_delta_unit + duration * 60,
                "color": colors[color_index % len(colors)],  # Cycle through the color list
                "list_of_flows": [remaining] * (duration * 60 + 1)
            }
            # Increment color index for next scenario
            color_index += 1

    revised_scenarios[slugify("No accident")] = {
            "dict_key_cell_value_fractional_capacity": {},
            "accident_duration_start_time_delta_unit": 6100,
            "accident_duration_end_time_delta_unit": 6100,
            "color": "green",
            "list_of_flows":[]
        }
    return revised_scenarios

def default_scenario():

    return generate_revised_scenarios_with_colors()
    # 6100-6200 in the case of <no accident> is just some dummy value
    # these are not used when the accident capacity reduction is 0 → when dict of fractional capacity is empty



def sliding_window_scenario(
    stride=300,
    incident_duration=3600,
    time_dimension_length=3600 * 10,
    SECONDS_OFFSET_MIDNIGHT=3600 * 6,
    dict_incident_cells_reduced_cap=None,
    printing=False,
    scenario_end_offset=3600 * 5,
):
    """

    :param stride:  300 implies 5 minutes if we are using 1 second Δt
    :param incident_duration: same across all scenarios
    :param time_dimension_length: the total number of Δt in the day
    :param dict_incident_cells_reduced_cap: something like: {405: 0.01, 406: 0.01, 407: 0.01, 408: 0.01}
    :return:
    """
    if dict_incident_cells_reduced_cap is None:
        print("Something wrong with the capacity input")
        sys.exit(0)

    SCENARIO_PARAMS = {}

    for startTime in range(stride, scenario_end_offset, stride):  #  we skip the trivial ones; (i.e. +/- stride)
        midnight = datetime(12, 10, 30, 00, 00, 00)
        timedelta = dt.timedelta(seconds=startTime + SECONDS_OFFSET_MIDNIGHT)

        key = "Accident at: " + str((midnight + timedelta).strftime("%H:%M %p"))
        value = {}
        value["dict_key_cell_value_fractional_capacity"] = dict_incident_cells_reduced_cap
        value["accident_duration_start_time_delta_unit"] = startTime
        value["accident_duration_end_time_delta_unit"] = startTime + incident_duration
        colors = cm.get_cmap("YlOrBr")
        value["color"] = colors(startTime / time_dimension_length)

        SCENARIO_PARAMS[key] = value  #  works, no need to do value.copy(), the assignment is by value (python :) )
    # No Accident case
    SCENARIO_PARAMS["No Accident"] = {
        "dict_key_cell_value_fractional_capacity": {},
        "accident_duration_start_time_delta_unit": 3600,
        "accident_duration_end_time_delta_unit": 6300,
        "color": "green",
    }

    if printing:
        print("The generated scenarios are as follows: ")
        for key in SCENARIO_PARAMS:
            print(key, ":", SCENARIO_PARAMS[key])

    return SCENARIO_PARAMS


def get_scenario_config(scenario_type="default", SECONDS_OFFSET_MIDNIGHT=3600 * 6, frac=0.01):
    if scenario_type == "default":
        return default_scenario()
    elif scenario_type == "sliding_window":
        return sliding_window_scenario(
            stride=3600,
            incident_duration=60 * 45,
            time_dimension_length=3600 * 10,
            dict_incident_cells_reduced_cap={1040: frac, 1036: frac, 1037: frac, 1038: frac},
            SECONDS_OFFSET_MIDNIGHT=SECONDS_OFFSET_MIDNIGHT,
            scenario_end_offset=3600 * 7,
            printing=True,
        )


if __name__ == "__main__":
    # get_scenario_config("sliding_window")
    print (get_scenario_config("default").keys())

