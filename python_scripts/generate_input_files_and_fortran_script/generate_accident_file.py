import csv
import sys
import shared_config
from smartprint import smartprint as sprint
from tqdm import tqdm

def generate_single_accident(
    list_of_all_cells,
    dict_key_cell_value_fractional_capacity,
    accident_duration_start_time_delta_unit,
    accident_duration_end_time_delta_unit,
    file_path,
):
    """
    :param list_of_all_cells:
    :param dict_key_cell_value_fractional_capacity: # empty dict means no accidents
    :param accident_duration_start_time_delta_unit:
    :param accident_duration_end_time_delta_unit:
    :param file_path:
    :return:
    """

    with open(file_path, "w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow([accident_duration_start_time_delta_unit])
        csvwriter.writerow([accident_duration_end_time_delta_unit])

        for cell in list_of_all_cells:
            if cell in dict_key_cell_value_fractional_capacity:
                csvwriter.writerow([str(cell) + "	" + str(dict_key_cell_value_fractional_capacity[cell])])
            else:
                csvwriter.writerow([str(cell) + "	" + "1.0"])


def generate_single_accident_with_variable_flow(
    list_of_all_cells,
    dict_key_cell_value_fractional_capacity,
    accident_duration_start_time_delta_unit,
    accident_duration_end_time_delta_unit,
    file_path,
    list_of_flow_vals: list
):
    """

    Args:
        list_of_all_cells:
        dict_key_cell_value_fractional_capacity:
        accident_duration_start_time_delta_unit:
        accident_duration_end_time_delta_unit:
        file_path:
        list_of_flow_vals:

    Returns:

    """
    assert isinstance(list_of_flow_vals, list)
    try:
        assert len(list_of_flow_vals) == accident_duration_end_time_delta_unit - accident_duration_start_time_delta_unit + 1
    except Exception as e:
        sprint (len(list_of_flow_vals))
        sprint(accident_duration_end_time_delta_unit - accident_duration_start_time_delta_unit + 1)
        print ("Error in generate_single_accident_with_variable_flow; Checking for no-accident case ")
        try:
            assert accident_duration_end_time_delta_unit - accident_duration_start_time_delta_unit == 0
        except:
            print("Error in generate_single_accident_with_variable_flow; No accident case not found")
            raise(Exception)
            sys.exit(0)

    # with open(file_path, "w") as f:
    #     csvwriter = csv.writer(f)
    #     csvwriter.writerow([accident_duration_start_time_delta_unit])
    #     csvwriter.writerow([accident_duration_end_time_delta_unit])
    #
    #     for index in tqdm(range(len(list_of_all_cells)), "Generating accident file"):
    #         cell = list_of_all_cells[index]
    #         to_print = str(cell)
    #         for i in (range(accident_duration_start_time_delta_unit , accident_duration_end_time_delta_unit + 1)):
    #             if cell in dict_key_cell_value_fractional_capacity:
    #                 to_print = to_print + "\t" + str(list_of_flow_vals[i - accident_duration_start_time_delta_unit])
    #             else:
    #                 # to_print = to_print + "\t" + str(1.0)
    #                 pass
    #         if accident_duration_end_time_delta_unit - accident_duration_start_time_delta_unit > 0:
    #             csvwriter.writerow([to_print])

    with open(file_path, "w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow([len(shared_config.accident_Cell_list)])
        for cell in dict_key_cell_value_fractional_capacity:
            csvwriter.writerow([str(cell)
                                + "\t" + str(accident_duration_start_time_delta_unit)
                                + "\t" + str(accident_duration_end_time_delta_unit)
                                + "\t" + str(dict_key_cell_value_fractional_capacity[cell])])
