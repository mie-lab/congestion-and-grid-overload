#!/usr/bin/env python
import csv
import os
import sys

import random
import time

from scipy.stats import genextreme

print ("os.getcwd():", os.getcwd())
# os.chdir(sys.path[0])
os.chdir(os.path.dirname(__file__))
# sys.path.append("./acnportal")
# sys.path.append("./adacharge")
sys.path.append("../../")
import math
if not os.path.exists("images_from_acnsim"):
    os.mkdir("images_from_acnsim")

from copy import deepcopy
import re as regex
from acnportal.contrib.acnsim import StochasticNetwork
from adacharge import AdaptiveChargingAlgorithmOffline


import numpy as np
import pandas as pd
print (os.getcwd())

sys.path.append("./")
# from python_scripts.optimise_charging_schedule.acnportal.acnportal.acnsim import ChargingNetwork


from inspect import currentframe
import zipfile
import shared_config
from tqdm import tqdm
from slugify import slugify

os.chdir(os.path.join(shared_config.BASE_FOLDER_with_repo_name, "python_scripts/optimise_charging_schedule"))
sys.path.append(shared_config.BASE_FOLDER_with_repo_name)
# with zipfile.ZipFile("temp_data.zip", "r") as zip_ref:
#     zip_ref.extractall("temp_data/")

# os.system("mv temp_data/events events")
# for folder in ["figures", "results", "results/profit_max_results"]:
#     os.system("rm -rf " + folder)
#     os.system("mkdir " + folder)

import json
import os
import pytz
from acnportal import acnsim
from acnportal import algorithms
from acnportal.acnsim import analysis
from acnportal.acnsim.events import EventQueue
from acnportal.signals.tariffs.tou_tariff import TimeOfUseTariff
from smartprint import smartprint as sprint
import matplotlib.pyplot as plt
import argparse, sys
from adacharge.adaptive_charging_optimization import ObjectiveComponent, equal_share, quick_charge, total_energy, load_flattening, aggregate_power, tou_energy_cost, peak, demand_charge, total_energy
from adacharge.adacharge import AdaptiveSchedulingAlgorithm

parser = argparse.ArgumentParser()
parser.add_argument("--scenario", help="Scenario name within quotes")
parser.add_argument("--inputfilename", help="Input filename")

cl_args = parser.parse_args()

# os.system("cp " + os.path.join(shared_config.BASE_FOLDER, "temp_data.zip") + " ./")
# os.system("rm -rf figures")
# os.system("rm -rf events")
# os.system("rm -rf events")


API_KEY = "DEMO_TOKEN"
TIMEZONE = pytz.timezone("America/Los_Angeles")
SITE = "caltech"
PERIOD = 5  # minutes
VOLTAGE = 208  # volts
KW_TO_AMPS = 1000 / VOLTAGE
KWH_TO_AMP_PERIODS = KW_TO_AMPS * (60 / 5)
MAX_LEN = 144
EVENTS_DIR = "events/"
VERBOSE = True
caps = shared_config.capacity_list

RUNNING_ONLINE = RUNNING_OFFLINE = True
RUNNING_ONLINE_COST = RUNNING_OFFLINE_COST = False


config_path_to_EV_arrivals_file = cl_args.inputfilename

start = "9-1-2018"
end = "10-1-2018"
tariff_name = "duck_curve_stadard_from_company_buffer_removed" # "duck_curve_stadard_from_company_buffer_removed" # "duck_curve_intense_summer" #"duck_curve_with_demand_charge" #"duck_curve" # "duck_curve_with_demand_charge" # "slightly_increasing_cost"  #  "duck_curve_shifted_dip_after_9AM" # "duck_curve" # "Constant_cost" # "duck_curve" # "Constant_cost" #_with_demand_charge
revenue = 0.3
# Scenario I is the offline optimal.
scenarios = {
    "II": {
        "ideal_battery": True,
        "estimate_max_rate": False,
        "uninterrupted_charging": False,
        "quantized": False,
        "basic_evse": True,
        "offline": False,
    },
    "III": {
        "ideal_battery": True,
        "estimate_max_rate": False,
        "uninterrupted_charging": True,
        "quantized": True,
        "basic_evse": False,
        "offline":False
    },
    "Offline": {
        "ideal_battery": True,
        "estimate_max_rate": False,
        "uninterrupted_charging": False,
        "quantized": False,
        "basic_evse": True,
        "offline":True
    },
    # "IV": {
    #     "ideal_battery": False,
    #     "estimate_max_rate": True,
    #     "uninterrupted_charging": True,
    #     "quantized": False,
    #     "basic_evse": True,
    #     "offline":False
    # },
    # "V": {
    #     "ideal_battery": False,
    #     "estimate_max_rate": True,
    #     "uninterrupted_charging": True,
    #     "quantized": True,
    #     "basic_evse": False,
    # },
}
scenario_order = shared_config.scenario_order



def get_linenumber():
    cf = currentframe()
    sprint(cf.f_back.f_lineno)

def update_events_with_true_arrivals_new(network: acnsim.network.ChargingNetwork, scenario_named_string, total_runs=-1, run_number=-1):
    frac = shared_config.frac_additional_filter

    # sprint (os.getcwd()) # debug path
    # get_linenumber()
    with open(os.path.join(shared_config.CTM_input_and_output_folder_path, config_path_to_EV_arrivals_file)) as file:
        lines = file.readlines()
        assert len(lines) >= 2, "The file does not contain exactly two lines"
        if len(lines) > 2:
            print ("More than 2 lines in arrivals for optimiser file; combing DE nodes")


    arrival_list = []
    with open(os.path.join(shared_config.CTM_input_and_output_folder_path, config_path_to_EV_arrivals_file)) as f:
        skip = 0
        for row in f:
            if skip == 0:
                # skip first line
                skip = 1
                continue
            listed = row.strip().split(",")



            # TO-DO; cell and scenario not being used now
            # since we are dealing with only one scenario
            # and one cell; Need to make this generic


            cell = int(listed[0])
            scenario = listed[1]

            arrivals = [int(i) for i in listed[2:]]

            if cell in shared_config.filtered_DE:
                arrival_list.append(arrivals)

    if len(lines) > 2:
        arrivals = np.array(arrival_list).sum(axis=0).tolist()
        if shared_config.USE_mean_to_get_arrivals_at_single_EVSE_infra:
            arrivals = np.array(arrival_list).mean(axis=0).tolist()
        arrivals = [int(x) for x in arrivals]
        arrivals = [int(x / frac) for x in arrivals]

        print ("Several DEs combined")
        print ("Total arrivals: ", sum(arrivals))
        plt.clf()
        plt.plot(arrivals)
        plt.savefig("images_from_acnsim/arrivals_"+ str(np.random.rand() * 10000000000000) + ".pdf")


    if shared_config.CUTOFF_TIME_FOR_IGNORING_ARRIVALS != -1:
        arrivals = arrivals[:shared_config.CUTOFF_TIME_FOR_IGNORING_ARRIVALS - 12 * 5]  # 5 implies  5 AM
        arrivals = [int(x) for x in arrivals]

    total_num_EVS = sum(arrivals)  # //4

    # if shared_config.FAST_tryout_for_debug_single_run:
    #     assert total_num_EVS <= event_count



    # to ensure that the total number of EVs are always the same across all EVSEs
    remainder_EVS = total_num_EVS % total_runs
    if remainder_EVS > 0:
        total_num_EVS = total_num_EVS - remainder_EVS


    a = [[x] * int(total_num_EVS // total_runs) for x in range(1, total_runs + 1)]
    a = np.array(a).flatten().tolist()

    # Set a seed for repeatability
    random.seed(100)
    np.random.seed(100)

    random.shuffle(a)
    EV_to_infra_map_dict = dict(zip(range(total_num_EVS), a))

    sprint(total_num_EVS)
    expanded_arrivals = []
    for i in range(len(arrivals)): # i shows the time stamp from 5AM. so we repeat this arrival time for arrivals[i] EVs
        for j in range(arrivals[i]):
            expanded_arrivals.append(shared_config.SECONDS_OFFSET_MIDNIGHT // (PERIOD * 60) + i)

    # event_list = []
    # for i in range(event_count):
    #     event_list.append(list(events.queue)[i][1])

    # try:
    #     assert len(event_list) >= total_num_EVS
    # except:
    #     sprint(len(event_list), total_num_EVS)
    #     raise Exception("len(event_list) > total_num_EVS invalid")
    #     debug_here = True
    #     sys.exit(0)
    # get_linenumber()
    count_ignored_evs = 0

    updated_departures = []
    updated_arrivals = []

    counter = 0
    event_list_updated = []

    extra_EVS_loaded_according_to_cutoff = False


    for i in (tqdm(range(total_num_EVS))): # total_num_EVS), "Creating arrival based charging demand"):

        try:
            if EV_to_infra_map_dict[i] != run_number:
                # deterministic way to assign EVs to charging stations
                continue
        except:
            debug_breakpoint = True
            raise Exception("if EV_to_infra_map_dict[i] != run_number: is false")

        # sequential factor is used to ensure reproducibility and have the almost same demand across scenarios
        # even when running with small penetration ratios.
        # if i % sequential_factor != 0:
        #     continue

        if shared_config.freeze_departure_time:
            departure = shared_config.frozen_departure_time + shared_config.SECONDS_OFFSET_MIDNIGHT // (PERIOD * 60)
        else:
            departure = expanded_arrivals[i] + shared_config.DWELL_TIME

        # Define a simple function to generate values for Initial SoC using the Generalized Extreme Value (GEV) distribution
        def generate_initial_soc(size=1000000, loc=20, scale=15, shape=0.4):
            """
            Generate random samples for Initial SoC using the Generalized Extreme Value distribution.

            Parameters:
            - size (int): Number of samples to generate.
            - loc (float): Location parameter of the GEV distribution.
            - scale (float): Scale parameter of the GEV distribution.
            - shape (float): Shape parameter of the GEV distribution.

            Returns:
            - np.ndarray: Random samples representing the Initial SoC.
            """
            return genextreme.rvs(c=shape, loc=loc, scale=scale, size=size)

        # Generate samples for Initial SoC
        if shared_config.stochastic_SOC:
            initial_soc_samples = generate_initial_soc(size=1, loc=20, scale=15, shape=0.4)[0]
            initial_soc_samples = max(initial_soc_samples, 0)
            initial_soc_samples = min(initial_soc_samples, 100)
            initial_soc_samples /= 100
        else:
            initial_soc_samples = shared_config.SOC

        #     generate_initial_soc(size=1000000, loc=20, scale=15, shape=0.4)
        #
        # # Plot histogram for Initial SoC
        # plt.figure(figsize=(12, 6))
        # plt.hist(initial_soc_samples, bins=50, alpha=0.6, color='blue', label='Initial SoC Distribution', density=True)
        # plt.title('Histogram of Initial SoC Distribution (GEV)')
        # plt.xlabel('State of Charge (SoC)')
        # plt.ylabel('Density')
        # plt.legend()
        # plt.xlim(0, 100)
        # plt.show()

        requested_energy = shared_config.Batter_capacity * (1-initial_soc_samples)

        battery = acnsim.models.Battery(shared_config.Batter_capacity, shared_config.Batter_capacity * initial_soc_samples, shared_config.MAX_CHARGING_L2) # def __init__(self, capacity, init_charge, max_power):

        try:
            ev = acnsim.models.EV(
            arrival=expanded_arrivals[i],
            departure=departure,
            requested_energy=requested_energy,
            station_id=network.station_ids[counter],
            session_id=str(int(np.random.rand() * 1000000000000)),
            battery=battery,
            estimated_departure=departure,  # not needed, since by default, it takes the value of departure itself (inside the function)
            )
        except Exception as e:
            debug_pitstop = True
            raise Exception(e)
            # print ("One Station ID missing; boundary case, ignored; waiting for 1 minute here")
            # continue


        assert network.station_ids[counter] != None

        if shared_config.CUTOFF_TIME_FOR_RESULTS_MIXING == -1 or \
                (shared_config.CUTOFF_TIME_FOR_RESULTS_MIXING != -1 and \
                 ev.arrival >= shared_config.CUTOFF_TIME_FOR_RESULTS_MIXING):
            updated_arrivals.append(expanded_arrivals[i])
            updated_departures.append(departure)
            event = acnsim.events.PluginEvent(timestamp=expanded_arrivals[i], ev=ev)
            event_list_updated.append(event)
            event = acnsim.events.UnplugEvent(timestamp=departure, ev=ev)
            event_list_updated.append(event)

            counter += 1

        if shared_config.CUTOFF_TIME_FOR_RESULTS_MIXING != -1:
            if not extra_EVS_loaded_according_to_cutoff:
                try:
                    new_arrivals = pd.read_csv("all_evs_charging_record_" +slugify(scenario_named_string)+ "/evs_tod_" + str(shared_config.CUTOFF_TIME_FOR_RESULTS_MIXING-1) + ".csv")
                except FileNotFoundError:
                    raise FileNotFoundError(" all_evs_charging_record_ file not found for:  "+ str(scenario_named_string) + " \nIgnoring this run")


                _, accident_arrival = total_num_EVS_from_CTM_file_accident_arrivals(fname=shared_config.accident_file_name)
                _, no_accident_arrival = total_num_EVS_from_CTM_file_accident_arrivals(fname=shared_config.no_accident_file_name)

                for ev_counter in range(new_arrivals.shape[0]):




                    new_ev = new_arrivals.iloc[ev_counter]

                    if shared_config.freeze_departure_time:
                        departure = shared_config.frozen_departure_time + shared_config.SECONDS_OFFSET_MIDNIGHT // (
                                    PERIOD * 60)
                    else:
                        departure = new_ev.departure # remained unchanged for this run since we are transferring the EVs


                    original_non_accident_arrival = new_ev.arrival
                    percent_remaining = 1

                    if original_non_accident_arrival > shared_config.TRUE_ACCIDENT_START_TIME and original_non_accident_arrival < shared_config.CUTOFF_TIME_FOR_RESULTS_MIXING:
                        try:
                            percent_remaining = accident_arrival[original_non_accident_arrival- 12 * 5 ] / \
                                                no_accident_arrival[original_non_accident_arrival - 12 * 5 ]
                            print ("------------------------------------")
                            sprint (percent_remaining)
                        except:
                            raise Exception ("Error while accounting for EVs whose count is overestimated; Potential demand higher than true demand")

                    battery_capacity_unmet_for_this_ev = (new_ev.battery_capacity - new_ev.battery_current_charge) * percent_remaining
                    battery = acnsim.models.Battery(new_ev.battery_capacity, new_ev.battery_capacity - battery_capacity_unmet_for_this_ev, new_ev.battery_max_power) # capacity, init charge, max power

                    ev = acnsim.models.EV(
                        arrival=shared_config.CUTOFF_TIME_FOR_RESULTS_MIXING,  # updated from old sim
                        departure=departure,
                        requested_energy=battery_capacity_unmet_for_this_ev,
                        station_id=network.station_ids[counter],
                        session_id=str(int(np.random.rand() * 1000000000000)),
                        battery=battery,
                        estimated_departure=departure,
                        # not needed, since by default, it takes the value of departure itself (inside the function)
                    )
                    updated_arrivals.append(shared_config.CUTOFF_TIME_FOR_RESULTS_MIXING)
                    updated_departures.append(departure)

                    event = acnsim.events.PluginEvent(timestamp=shared_config.CUTOFF_TIME_FOR_RESULTS_MIXING, ev=ev)
                    event_list_updated.append(event)
                    event = acnsim.events.UnplugEvent(timestamp=departure, ev=ev)
                    event_list_updated.append(event)

                    counter += 1     #  We push all the EVs from the temporary file into today's schedule
                extra_EVS_loaded_according_to_cutoff = True

        # try: # no longer needed
        #     assert ev.departure > ev.arrival
        # except:
        #     debug_breakpoint = True

    events = EventQueue(event_list_updated)
    sprint(len(events), count_ignored_evs)

    plt.plot(updated_arrivals, label="updated_arrivals")
    plt.plot(updated_departures, label="updated_departures")
    plt.legend()
    plt.title(cl_args.scenario)
    sprint(cl_args.scenario)
    plt.savefig("images_from_acnsim/departures-and-arrivals" + cl_args.scenario + ".pdf", dpi=300)
    plt.show(block=False)
    plt.clf() # plt.show(block=False)()

    plt.clf()
    plt.hist(updated_arrivals, label="updated_arrivals", bins=40, alpha=0.4)
    nbins = 40
    if len(set(updated_departures)) == 1:
        nbins = 2
    plt.hist(updated_departures, label="updated_departures", bins=nbins, alpha=0.4)
    plt.legend()
    plt.ylim(0, 50)
    plt.title(cl_args.scenario + " hist")
    sprint(cl_args.scenario)
    with open("images_from_acnsim/Hist-departures-and-arrivals" + cl_args.scenario + "-run-num-" + str(run_number) + ".csv", "w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(updated_arrivals)
    plt.savefig("images_from_acnsim/Hist-departures-and-arrivals" + cl_args.scenario + "-run-num-" + str(run_number) + ".pdf", dpi=300)
    plt.clf() # plt.show(block=False)()

    sprint (len(updated_arrivals))
    return events  # , charging_network

def map_events_to_valid_stations(events, cn):
    """
    Only needed for offline case
    since we get keyError
    So we manually map the EVSEs and EVs
    """
    event_list = []
    for i in range(len(events)):
        event_list.append(list(events.queue)[i][1])

    counter = 0
    event_list_updated = []
    for i in range(len(events)):
        pe = event_list[i]
        dwell_time = pe.ev.departure - pe.ev.arrival

        pe.ev._station_id = cn.station_ids[counter]

        assert pe.station_id == pe.ev._station_id
        assert pe.ev.station_id == pe.ev.station_id

        event_list_updated.append(pe)
        counter += 1

    events = EventQueue(event_list_updated[: len(events)])
    return events  # , charging_network

def level_2_network(transformer_cap=200, evse_per_phase=34, is_basic_evse=True):
    """ Configurable charging network for level-2 EVSEs connected line to line
        at VOLTAGE V.

    Args:
        transformer_cap (float): Capacity of the transformer feeding the network
          [kW]
        evse_per_phase (int): Number of EVSEs on each phase. Total number of
          EVSEs will be 3 * evse_per_phase.

    Returns:
        ChargingNetwork: Configured ChargingNetwork.
    """
    network = StochasticNetwork(early_departure=True)
    voltage = VOLTAGE
    evse_type = 'AeroVironment'
    if is_basic_evse:
        evse_type = "BASIC"

    # Define the sets of EVSEs in the Caltech ACN.
    AB_ids = ['AB-{0}'.format(i) for i in range(evse_per_phase)]
    BC_ids = ['BC-{0}'.format(i) for i in range(evse_per_phase)]
    CA_ids = ['CA-{0}'.format(i) for i in range(evse_per_phase)]

    # Add Caltech EVSEs
    for evse_id in AB_ids:
        network.register_evse(acnsim.get_evse_by_type(evse_id, evse_type), voltage, 30)
    for evse_id in BC_ids:
        network.register_evse(acnsim.get_evse_by_type(evse_id, evse_type), voltage, -90)
    for evse_id in CA_ids:
        network.register_evse(acnsim.get_evse_by_type(evse_id, evse_type), voltage, 150)

    # Add Caltech Constraint Set
    AB = acnsim.Current(AB_ids)
    BC = acnsim.Current(BC_ids)
    CA = acnsim.Current(CA_ids)

    # Define intermediate currents
    I3a = AB - CA
    I3b = BC - AB
    I3c = CA - BC
    I2a = (1 / 4) * (I3a - I3c)
    I2b = (1 / 4) * (I3b - I3a)
    I2c = (1 / 4) * (I3c - I3b)

    # Build constraint set
    primary_side_constr = transformer_cap * 1000 / 3 / 277
    secondary_side_constr = transformer_cap * 1000 / 3 / 120
    network.add_constraint(I3a, secondary_side_constr, name='Secondary A')
    network.add_constraint(I3b, secondary_side_constr, name='Secondary B')
    network.add_constraint(I3c, secondary_side_constr, name='Secondary C')
    network.add_constraint(I2a, primary_side_constr, name='Primary A')
    network.add_constraint(I2b, primary_side_constr, name='Primary B')
    network.add_constraint(I2c, primary_side_constr, name='Primary C')

    return network

def level_1_network(transformer_cap=200, evse_per_phase=34, is_basic_evse=True):
    """ Configurable charging network for level-2 EVSEs connected line to line
        at VOLTAGE V.

    Args:
        transformer_cap (float): Capacity of the transformer feeding the network
          [kW]
        evse_per_phase (int): Number of EVSEs on each phase. Total number of
          EVSEs will be 3 * evse_per_phase.

    Returns:
        ChargingNetwork: Configured ChargingNetwork.
    """
    network = StochasticNetwork(early_departure=True)
    voltage = 120
    evse_type = 'AeroVironment'
    if is_basic_evse:
        evse_type = "BASIC"

    # Define the sets of EVSEs in the Caltech ACN.
    AB_ids = ['AB-{0}'.format(i) for i in range(evse_per_phase)]
    BC_ids = ['BC-{0}'.format(i) for i in range(evse_per_phase)]
    CA_ids = ['CA-{0}'.format(i) for i in range(evse_per_phase)]

    # Add Caltech EVSEs
    for evse_id in AB_ids:
        network.register_evse(acnsim.get_evse_by_type(evse_id, evse_type), voltage, 30)
    for evse_id in BC_ids:
        network.register_evse(acnsim.get_evse_by_type(evse_id, evse_type), voltage, -90)
    for evse_id in CA_ids:
        network.register_evse(acnsim.get_evse_by_type(evse_id, evse_type), voltage, 150)

    # Add Caltech Constraint Set
    AB = acnsim.Current(AB_ids)
    BC = acnsim.Current(BC_ids)
    CA = acnsim.Current(CA_ids)

    # Define intermediate currents
    I3a = AB - CA
    I3b = BC - AB
    I3c = CA - BC
    I2a = (1 / 4) * (I3a - I3c)
    I2b = (1 / 4) * (I3b - I3a)
    I2c = (1 / 4) * (I3c - I3b)

    # Build constraint set
    primary_side_constr = transformer_cap * 1000 / 3 / 277
    secondary_side_constr = transformer_cap * 1000 / 3 / 120
    network.add_constraint(I3a, secondary_side_constr, name='Secondary A')
    network.add_constraint(I3b, secondary_side_constr, name='Secondary B')
    network.add_constraint(I3c, secondary_side_constr, name='Secondary C')
    network.add_constraint(I2a, primary_side_constr, name='Primary A')
    network.add_constraint(I2b, primary_side_constr, name='Primary B')
    network.add_constraint(I2c, primary_side_constr, name='Primary C')

    return network

def level_2_network_single_phase(transformer_cap=200, evse_per_phase=34, is_basic_evse=True):
    """ Configurable charging network for level-2 EVSEs connected line to line
        at VOLTAGE V.

    Args:
        transformer_cap (float): Capacity of the transformer feeding the network
          [kW]
        evse_per_phase (int): Number of EVSEs on each phase. Total number of
          EVSEs will be 3 * evse_per_phase. Here all in the same phase.

    Returns:
        ChargingNetwork: Configured ChargingNetwork.
    """
    network = StochasticNetwork(early_departure=True)
    voltage = VOLTAGE
    evse_type = 'AeroVironment'
    if is_basic_evse:
        evse_type = "BASIC"


    # Define the sets of EVSEs in the Caltech ACN.
    AB_ids = ['AB-{0}'.format(i) for i in range(evse_per_phase)]

    # Add Caltech EVSEs
    for evse_id in AB_ids:
        network.register_evse(acnsim.get_evse_by_type(evse_id, evse_type), voltage, 0)

    # Add Caltech Constraint Set
    all_current = acnsim.Current(AB_ids)

    network.add_constraint(all_current, transformer_cap * 1000 / voltage, name='Transformer Cap')

    return network




# def level_2_network(transformer_cap=200, evse_per_phase=34, voltage=VOLTAGE, is_basic_evse=False):
#     """Configurable charging network for level-2 EVSEs connected line to line
#         at VOLTAGE V.
#
#     Args:
#         transformer_cap (float): Capacity of the transformer feeding the network
#           [kW]
#         evse_per_phase (int): Number of EVSEs on each phase. Total number of
#           EVSEs will be 3 * evse_per_phase.
#
#     Returns:
#         ChargingNetwork: Configured ChargingNetwork.
#     """
#     network = ChargingNetwork() # StochasticNetwork(early_departure=True)
#     voltage = voltage
#     evse_type = "AeroVironment"
#
#     # Define the sets of EVSEs in the Caltech ACN.
#     AB_ids = ["AB-{0}".format(i) for i in range(evse_per_phase)]
#     BC_ids = ["BC-{0}".format(i) for i in range(evse_per_phase)]
#     CA_ids = ["CA-{0}".format(i) for i in range(evse_per_phase)]
#
#     if is_basic_evse:
#         evse_type = "BASIC"  # {"AV": "BASIC", "CC": "BASIC"}
#     else:
#         evse_type = "AeroVironment"  # {"AV": "AeroVironment", "CC": "ClipperCreek"}
#
#     # Add Caltech EVSEs
#     for evse_id in AB_ids:
#         network.register_evse(evse=acnsim.get_evse_by_type(evse_id, evse_type), voltage=voltage, phase_angle=30)
#     for evse_id in BC_ids:
#         network.register_evse(acnsim.get_evse_by_type(evse_id, evse_type), voltage, -90)
#     for evse_id in CA_ids:
#         network.register_evse(acnsim.get_evse_by_type(evse_id, evse_type), voltage, 150)
#
#     # Add Caltech Constraint Set
#     AB = acnsim.Current(AB_ids)
#     BC = acnsim.Current(BC_ids)
#     CA = acnsim.Current(CA_ids)
#
#     # Define intermediate currents
#     I3a = AB - CA
#     I3b = BC - AB
#     I3c = CA - BC
#     I2a = (1 / 4) * (I3a - I3c)
#     I2b = (1 / 4) * (I3b - I3a)
#     I2c = (1 / 4) * (I3c - I3b)
#
#     # Build constraint set
#     primary_side_constr = transformer_cap * 1000 / 3 / 277
#     secondary_side_constr = transformer_cap * 1000 / 3 / 120
#     network.add_constraint(I3a, secondary_side_constr, name="Secondary A")
#     network.add_constraint(I3b, secondary_side_constr, name="Secondary B")
#     network.add_constraint(I3c, secondary_side_constr, name="Secondary C")
#     network.add_constraint(I2a, primary_side_constr, name="Primary A")
#     network.add_constraint(I2b, primary_side_constr, name="Primary B")
#     network.add_constraint(I2c, primary_side_constr, name="Primary C")
#
#     return network



# In[ ]:


def total_num_EVS_from_CTM_file_accident_arrivals(fname):
    frac = shared_config.frac_additional_filter
    """
    No input needed, filename already present in the config file
    Returns: total number of EVs according to the CTM output
    """


    with open(os.path.join(shared_config.CTM_input_and_output_folder_path, fname)) as file:
        lines = file.readlines()
        assert len(lines) >= 2, "The file does not contain exactly two lines"
        if len(lines) > 2:
            print ("More than 2 lines in arrivals for optimiser file; combing DE nodes")


    arrival_list = []


    with open(os.path.join(shared_config.CTM_input_and_output_folder_path,
                           fname)) as f:
        skip = 0
        for row in f:
            if skip == 0:
                # skip first line
                skip = 1
                continue
            listed = row.strip().split(",")

            # TO-DO; cell and scenario not being used now
            # since we are dealing with only one scenario
            # and one cell; Need to make this generic
            cell = int(listed[0])

            # sprint ((os.path.join(shared_config.CTM_input_and_output_folder_path, fname)))
            # print("Cell testing:", cell)

            arrivals = [int(i) for i in listed[2:]]
            if cell in shared_config.filtered_DE:

                arrival_list.append(arrivals)


        if len(lines) > 2:
            arrivals = np.array(arrival_list).sum(axis=0).tolist()
            if shared_config.USE_mean_to_get_arrivals_at_single_EVSE_infra:
                arrivals = np.array(arrival_list).mean(axis=0).tolist()
            # sprint(arrival_list)
            # sprint(arrivals)
            arrivals = [int(x/frac) for x in arrivals]
            print("Several DEs combined")

        # get_linenumber()

    if shared_config.CUTOFF_TIME_FOR_IGNORING_ARRIVALS != -1:
        arrivals = arrivals[:shared_config.CUTOFF_TIME_FOR_IGNORING_ARRIVALS - 12 * 5]  # 5 implies  5 AM
        arrivals = [int(x) for x in arrivals]

    total_num_EVS = sum(arrivals)
    return total_num_EVS, arrivals

def total_num_EVS_from_CTM_file():
    """
    No input needed, filename already present in the config file
    Returns: total number of EVs according to the CTM output
    """
    frac = shared_config.frac_additional_filter
    with open(os.path.join(shared_config.CTM_input_and_output_folder_path, config_path_to_EV_arrivals_file)) as file:
        lines = file.readlines()
        assert len(lines) >= 2, "The file does not contain exactly two lines"
        if len(lines) > 2:
            print ("More than 2 lines in arrivals for optimiser file; combing DE nodes")


    arrival_list = []


    with open(os.path.join(shared_config.CTM_input_and_output_folder_path,
                           config_path_to_EV_arrivals_file)) as f:
        skip = 0
        for row in f:
            if skip == 0:
                # skip first line
                skip = 1
                continue
            listed = row.strip().split(",")

            # TO-DO; cell and scenario not being used now
            # since we are dealing with only one scenario
            # and one cell; Need to make this generic
            cell = int(listed[0])

            arrivals = [int(i) for i in listed[2:]]
            if cell in shared_config.filtered_DE:
                arrival_list.append(arrivals)


        if len(lines) > 2:
            arrivals = np.array(arrival_list).sum(axis=0).tolist()
            if shared_config.USE_mean_to_get_arrivals_at_single_EVSE_infra:
                arrivals = np.array(arrival_list).mean(axis=0).tolist()
            sprint (listed)
            sprint (arrivals)
            arrivals = [int(x/frac) for x in arrivals]
            print("Several DEs combined")

        # get_linenumber()

    if shared_config.CUTOFF_TIME_FOR_IGNORING_ARRIVALS != -1:
        arrivals = arrivals[:shared_config.CUTOFF_TIME_FOR_IGNORING_ARRIVALS - 12 * 5]  # 5 implies  5 AM
        arrivals = [int(x) for x in arrivals]

    total_num_EVS = sum(arrivals)
    return total_num_EVS, arrivals


class Experiment:
    """Wrapper for ACN-Sim Experiments including caching serialized experiment to disk."""

    def __init__(self, sim):
        self.sim = sim

    def calc_metrics(self):
        """Calculate metrics from simulation."""
        if not (cl_args.scenario == "no-accident" and shared_config.CUTOFF_TIME_FOR_RESULTS_MIXING == -1): # only for no-accident case perfect info we record partial costs
            percent_remaining = [1] * 288
        else:
            _, accident_arrival = total_num_EVS_from_CTM_file_accident_arrivals(fname=shared_config.accident_file_name)
            _, no_accident_arrival = total_num_EVS_from_CTM_file_accident_arrivals(fname=shared_config.no_accident_file_name)

            percent_remaining = []
            print("Ratios:")
            print(accident_arrival, no_accident_arrival)
            try:
                for i in range(len(accident_arrival)):
                    # if both are zero; we just take the ratio as 0: implies no vehicle charging
                    if accident_arrival[i] == no_accident_arrival[i] and no_accident_arrival[i] == 0:
                        percent_remaining.append(0)
                    else:
                        percent_remaining.append (accident_arrival[i] / no_accident_arrival[i])
            except Exception as e:
                print ("Error while computing ratio inside calc_metrics; \n Exception type: ", str(e), "\n waiting for 10 minutes")
                time.sleep(600)

            percent_remaining = [1] * (12 * 5) + percent_remaining + [1] * 200 # trailing 1's to ensure that agg can be multiplied, since our arrivals
                                                                # are cutoff at 144 &  1's before 5 A.M.

            plt.clf()
            plt.plot(percent_remaining, label="percent remaining")
            plt.plot(no_accident_arrival, label="no accident arrival")
            plt.plot(accident_arrival, label="accident arrival")
            plt.legend()
            plt.title(cl_args.scenario + " cutoff " + str(shared_config.CUTOFF_TIME_FOR_RESULTS_MIXING))
            plt.savefig("images_from_acnsim/f_percentage_remaining_.pdf")
            # plt.show()
            plt.clf()

        sprint (cl_args.scenario)
        sprint (shared_config.CUTOFF_TIME_FOR_RESULTS_MIXING)
        metrics = {
            "proportion_delivered": analysis.proportion_of_energy_delivered(self.sim) * 100,
            "demands_fully_met": analysis.proportion_of_demands_met(self.sim) * 100,
            "peak_current": self.sim.peak,
            "demand_charge": analysis.demand_charge(self.sim),
            "energy_cost_all": analysis.energy_cost(self.sim, cutOfftime = -1, cap_remaining=percent_remaining),
            "energy_cost_1": analysis.energy_cost(self.sim, cutOfftime=1, cap_remaining=percent_remaining), # sanity check boundary case
            "energy_cost_" + str(shared_config.TRUE_ACCIDENT_START_TIME): analysis.energy_cost(self.sim, cutOfftime=shared_config.TRUE_ACCIDENT_START_TIME, cap_remaining=percent_remaining),
            "energy_cost_" + str(shared_config.TRUE_ACCIDENT_START_TIME + 3): analysis.energy_cost(self.sim, cutOfftime=shared_config.TRUE_ACCIDENT_START_TIME + 3, cap_remaining=percent_remaining),
            "energy_cost_" + str(shared_config.TRUE_ACCIDENT_START_TIME + 6): analysis.energy_cost(self.sim, cutOfftime=shared_config.TRUE_ACCIDENT_START_TIME + 6, cap_remaining=percent_remaining),
            "energy_cost_" + str(shared_config.TRUE_ACCIDENT_START_TIME + 9): analysis.energy_cost(self.sim, cutOfftime=shared_config.TRUE_ACCIDENT_START_TIME + 9, cap_remaining=percent_remaining),
            "total_energy_delivered": analysis.total_energy_delivered(self.sim),
            "total_energy_requested": analysis.total_energy_requested(self.sim),
            "aggregate_power_total": acnsim.analysis.aggregate_power(self.sim).sum(),
            "num_time_steps": self.sim.charging_rates.shape[1],
            "aggregate_power_timeseries": acnsim.analysis.aggregate_power(self.sim).tolist(),  # aggregate power never changes
        }
        if cl_args.scenario == "no-accident":
            assert metrics["energy_cost_" + str(shared_config.TRUE_ACCIDENT_START_TIME)] <= metrics["energy_cost_" + str(shared_config.TRUE_ACCIDENT_START_TIME + 3)] <= metrics["energy_cost_" + str(shared_config.TRUE_ACCIDENT_START_TIME + 6)] <= metrics["energy_cost_" + str(shared_config.TRUE_ACCIDENT_START_TIME + 9)]

        return metrics

    def log_local_file(self, path):
        """Write simulation, metrics and solver statistics to disk."""
        self.sim.to_json(path + "sim.json")
        with open(path + "metrics.json", "w") as outfile:
            json.dump(self.calc_metrics(), outfile)
        with open(path + "solve_stats.json", "w") as outfile:
            json.dump(self.sim.scheduler.solve_stats, outfile)

    def run_and_store(self, path, rich_progress=None, scenario_named_string=""):
        """Run experiment and store results."""
        # print(f'Starting - {path}')
        # if os.path.exists(path + "sim.json"):
        #     print(f"Already Run - {path}...")
        #     # return
        try:
            print("Simulating")
            ev_logging_folder = "all_evs_charging_record_" + slugify(scenario_named_string)
            if not os.path.exists(ev_logging_folder):
                os.mkdir(ev_logging_folder)
            os.chdir(ev_logging_folder)
            print("Current folder changed to : ",  os.getcwd())
            self.sim.run(save_ev_state=cl_args.scenario == "no-accident" and shared_config.CUTOFF_TIME_FOR_RESULTS_MIXING==-1)
            os.chdir("../")
            print("Current folder changed to : ", os.getcwd())
            # if not os.path.exists(path):
            #     os.makedirs(path)
            # self.log_local_file(path)
            # print(f'Done - {path}')
            print("Simulation complete")
        except Exception as e:
            print(f"Failed - {path}")
            raise(e)
            sys.exit(0)
        return sim


def configure_sim(
    alg,
    cap,
    start,
    events,
    basic_evse=True,
    network=None,
    estimate_max_rate=False,
    uninterrupted_charging=False,
    quantized=False,
    allow_overcharging=False,
    tariff_name=None,
    offline=False,
):
    """Configure simulation."""
    start_time = TIMEZONE.localize(pd.datetime.strptime(start, "%m-%d-%Y"))

    if estimate_max_rate:
        alg.max_rate_estimator = algorithms.SimpleRampdown()
        alg.estimate_max_rate = True
    alg.uninterrupted_charging = uninterrupted_charging
    alg.allow_overcharging = allow_overcharging

    # Some algorithms support a quantized option
    if quantized:
        try:
            alg.quantize = True
        except:
            pass
        try:
            alg.reallocate = True
        except:
            pass



    if tariff_name is not None:
        signals = {"tariff": TimeOfUseTariff(tariff_name)}
    else:
        signals = {}

    if offline:
        vbose = False
    else:
        vbose = False
    sim = acnsim.Simulator(
                            network=network,
                            scheduler=alg,
                            events=events,
                            start=start_time,
                            signals=signals,
                            period=PERIOD,
                            verbose=vbose
                    )



    if offline:
        sim.scheduler.register_events(events)
        try:
            sim.scheduler.solve()
        except:
            debug_stop = True
            print ("Error in alg.solve; Exiting execution!")
            raise ("Error in alg.solve; Exiting execution!")
            sys.exit(0)

    return sim


energy_del_base_dir = "results/infrastructure_utilization_results"


ALGS = dict()
ALGS["LLF"] = algorithms.SortedSchedulingAlgo(algorithms.least_laxity_first)
ALGS["EDF"] = algorithms.SortedSchedulingAlgo(algorithms.earliest_deadline_first)
ALGS["RR"] = algorithms.RoundRobin(algorithms.first_come_first_served, continuous_inc=1)
ALGS["Unctrl"] = algorithms.UncontrolledCharging()
# quick_charge_obj = [ ObjectiveComponent(total_energy)]  #  ObjectiveComponent(load_flattening)
# quick_charge_obj = [
#     # ObjectiveComponent(total_energy, revenue),
#     # ObjectiveComponent(tou_energy_cost),
#     # ObjectiveComponent(demand_charge),
#     ObjectiveComponent(total_energy),
#     ObjectiveComponent(equal_share, 1e-12)
# ]

for weight_obj_cost in [0.1]: #[0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]:
    if "demand_charge" in tariff_name:
        quick_charge_obj = [
            ObjectiveComponent(total_energy, 1000),
            ObjectiveComponent(tou_energy_cost),
            ObjectiveComponent(demand_charge),
            # ObjectiveComponent(total_energy)
            # ObjectiveComponent(quick_charge),
            # ObjectiveComponent(equal_share, 1e-12)
        ]
    else:
        quick_charge_obj = [
            ObjectiveComponent(total_energy),
            ObjectiveComponent(tou_energy_cost, weight_obj_cost),
            # ObjectiveComponent(quick_charge, 1e-6),
            # ObjectiveComponent(equal_share, 1e-12)

            ObjectiveComponent(equal_share, 1e-12)
        ]
        if weight_obj_cost == 0 :
            quick_charge_obj = [
                ObjectiveComponent(total_energy, 1000),
                # ObjectiveComponent(tou_energy_cost, weight_obj_cost),
                # ObjectiveComponent(quick_charge, 1e-6),
                # ObjectiveComponent(equal_share, 1e-12)

                ObjectiveComponent(equal_share, 1e-12)
            ]

    # , ObjectiveComponent(equal_share, 1e-12)
    # ObjectiveComponent(load_flattening)
    # ObjectiveComponent(quick_charge), ObjectiveComponent(tou_energy_cost), ObjectiveComponent(tou_energy_cost),


    ALGS["ASA-QC"] = AdaptiveSchedulingAlgorithm(
        quick_charge_obj, solver="MOSEK", max_recompute=shared_config.max_recompute
    )
    ALGS["ASA"] = AdaptiveSchedulingAlgorithm(quick_charge_obj, solver="MOSEK", max_recompute=shared_config.max_recompute)

    result_dict = {}
    offline_run_status = False
    if RUNNING_ONLINE:
        print("Started running scenarios!")
        from rich.progress import Progress

        # if shared_config.freeze_events_across_traffic_scenarios:
            # events_dict = {}

        with Progress() as progress:
            task1 = progress.add_task("[red]scenarios...", total=len(scenarios.items()))
            task2 = progress.add_task("[green]Capacities...", total=len(caps))
            task3 = progress.add_task("[cyan]Algorithms...", total=len(["ASA-QC", "LLF", "EDF", "RR"]))
            task4 = progress.add_task("[yellow]Simulating...", total=150)
            total = progress.add_task(
                "[yellow]Overall...", total=len(scenarios.items()) * len(caps) * len(["ASA-QC", "LLF", "EDF", "RR"])
            )

            counter1 = 0
            total_counter = 0
            # for scenario_id, scenario in scenarios.items():
            for scenario_id in scenario_order:  # ["III", "IV", "V", "II"]:

                scenario_id_ = scenario_id

                if shared_config.debug_with_few_scenarios:
                    if scenario_id_ != "II":
                        scenario_id_ = "II"

                scenario = scenarios[scenario_id_]

                counter1 += 1
                progress.update(task1, completed=counter1)
                counter2 = 0
                for cap in caps:
                    counter2 += 1
                    progress.update(task2, completed=counter2)
                    counter3 = 0

                    alg_list = shared_config.alg_list
                    if "Offline" in scenario_id:
                        alg_list = ["Offline"]
                    for alg in alg_list:
                        alg_ = alg

                        # if shared_config.debug_with_few_scenarios:
                        #     if alg_ != "ASA_QC":
                        #         alg_ = "ASA-QC"
                        # continue


                        if "Offline" not in scenario_id:
                            alg_copy = deepcopy(ALGS[alg_])
                            alg_name = alg_

                        # if "Offline" == scenario_id and offline_run_status:
                        #     continue

                        else:
                            alg_copy = deepcopy(
                                AdaptiveChargingAlgorithmOffline(
                                    quick_charge_obj, solver="MOSEK", verbose=True , constraint_type="SOC",
                                     enforce_energy_equality=False
                                )
                            )
                            alg_name = "Offline"
                            offline_run_status = True



                        total_counter += 1
                        progress.update(total, completed=total_counter)

                        counter3 += 1
                        progress.update(task3, completed=counter3)


                        assert shared_config.network_type in ["level_1", "level_2"]
                        if shared_config.network_type == "level_2":
                            cn = level_2_network(
                                transformer_cap=cap, evse_per_phase=shared_config.Total_EVS_at_One_Charging_complex // 3, is_basic_evse=True #, voltage=VOLTAGE, is_basic_evse=True
                            )
                        else:
                            cn = level_1_network(
                                transformer_cap=cap, evse_per_phase=shared_config.Total_EVS_at_One_Charging_complex // 3, is_basic_evse=True #, voltage=VOLTAGE, is_basic_evse=True
                            )

                        if shared_config.phase_angles == "single_phase":
                            cn = acnsim.sites.simple_acn(["AB-" + str(x) for x in range(1, shared_config.Total_EVS_at_One_Charging_complex)], voltage=VOLTAGE, aggregate_cap=cap)
                        # cn = acnsim.sites.caltech_acn(voltage=VOLTAGE, transformer_cap=cap, basic_evse=True)


                        total_num, _ = total_num_EVS_from_CTM_file()
                        total_num = int(total_num)
                        RUN_NUMBER = total_num // shared_config.Total_EVS_at_One_Charging_complex_for_run_num_computation + 1
                        print ("\n\nTotal run numbers: \n\n", RUN_NUMBER)
                        # if not shared_config.CLONE_CALTECH_HUBS:
                        #     RUN_NUMBER = 1
                        #     cn = level_2_single_phase(
                        #         transformer_cap=cap, evse_per_phase=len(events), voltage=VOLTAGE, basic_evse=True
                        #     )



                        # if "Offline" in scenario:
                        #     RUN_NUMBER = 1
                        for run_number in range(1, RUN_NUMBER+1):
                            sprint(cap, alg_name, scenario_id, run_number)

                            output_dir = f"{energy_del_base_dir}/{start}:{end}/{scenario_id}/{cap}/{(alg_name)}/{run_number}/"
                            sprint (run_number, "..............DEBUG")
                            sprint (cap, alg_name, scenario_id, run_number)
                            if (shared_config.CUTOFF_TIME_FOR_RESULTS_MIXING, scenario_id, cap, alg_name, run_number) in result_dict:
                                continue

                            # events = deepcopy(events_backup)
                            sprint (len(cn.station_ids))
                            try:
                                updated_events = update_events_with_true_arrivals_new(cn, scenario_named_string = str((scenario_id, cap, alg_name, run_number)),
                                                                                  total_runs=RUN_NUMBER, run_number=run_number)
                            except FileNotFoundError as e:
                                raise Exception("Run ignored due to file not found for all_evs no-accident case")
                                # print (e)
                                # continue

                            events = deepcopy(updated_events)

                            # new_queue = []
                            # # extract the first 100 elements from the original queue and add them to the new queue
                            # for i in range(100):
                            #     new_queue.append(events.get_event())
                            #
                            # events = EventQueue(new_queue)

                            # events_dict[scenario_id_, (alg_name)] = deepcopy(events)

                            sim = configure_sim(
                                cap=deepcopy(cap),
                                alg=deepcopy(alg_copy),
                                network=deepcopy(cn),
                                start=start,
                                events=deepcopy(events),
                                basic_evse=scenario["basic_evse"],
                                estimate_max_rate=scenario["estimate_max_rate"],
                                uninterrupted_charging=scenario["uninterrupted_charging"],
                                quantized=scenario["quantized"],
                                tariff_name=tariff_name,
                                offline=scenario["offline"],
                            )
                            ex = Experiment(sim)
                            print ("Simulation started")
                            sim = ex.run_and_store(output_dir, rich_progress=[progress, task4], scenario_named_string = str((scenario_id, cap, alg_name, run_number)))

                            """
                            if ex.calc_metrics()["proportion_delivered"] < 99.5:
                                for cost_weight in [1e-1, 1e-3, 1e-7, 1e-12, 1e-15]:
                                    quick_charge_obj = [
                                        ObjectiveComponent(total_energy, 1e12),
                                        ObjectiveComponent(tou_energy_cost, cost_weight),
                                    ]
                                    alg_copy = deepcopy(
                                        AdaptiveChargingAlgorithmOffline(
                                            quick_charge_obj, solver="MOSEK", verbose=True, constraint_type="SOC",
                                            # enforce_energy_equality=True
                                        )
                                    )
                                    events = deepcopy(events_backup)
                                    updated_events = update_events_with_true_arrivals_new(events, cn,
                                                                                          scenario_named_string=str((
                                                                                                                    scenario_id,
                                                                                                                    cap,
                                                                                                                    alg_name,
                                                                                                                    run_number)),
                                                                                          total_runs=RUN_NUMBER,
                                                                                          run_number=run_number)
                                    events = deepcopy(updated_events)
                                    sim = configure_sim(
                                        cap=deepcopy(cap),
                                        alg=deepcopy(alg_copy),
                                        network=deepcopy(cn),
                                        start=start,
                                        events=deepcopy(events),
                                        basic_evse=scenario["basic_evse"],
                                        estimate_max_rate=scenario["estimate_max_rate"],
                                        uninterrupted_charging=scenario["uninterrupted_charging"],
                                        quantized=scenario["quantized"],
                                        tariff_name=tariff_name,
                                        offline=scenario["offline"],
                                    )
                                    ex = Experiment(sim)
                                    print("Simulation started")
                                    sim = ex.run_and_store(output_dir, rich_progress=[progress, task4],
                                                           scenario_named_string=str(
                                                               (scenario_id, cap, alg_name, run_number)))
                                    sprint (cost_weight, ex.calc_metrics()["proportion_delivered"])
                                    with open("Offline_optimal_failure.txt", "a" ) as f:
                                        f.write(str((cost_weight, ex.calc_metrics()["proportion_delivered"])) + "\n")
                            """
                            sprint("\t\t", shared_config.CUTOFF_TIME_FOR_RESULTS_MIXING, scenario_id, cap, alg_name, run_number)
                            sprint (result_dict.keys())

                            plt.clf()
                            arr = []
                            dep = []

                            for i in range(len(list(sim.ev_history.values()))):
                                arr.append(list(sim.ev_history.values())[i].arrival)
                                dep.append(list(sim.ev_history.values())[i].departure)
                            plt.plot(arr, label="Arrivals after sim complete")
                            plt.plot(dep, label="departures after sim complete")
                            plt.title(str(
                                (scenario_id, cap, alg_name, run_number, cl_args.scenario)))
                            plt.savefig("images_from_acnsim/e_" + str(shared_config.CUTOFF_TIME_FOR_RESULTS_MIXING) + "_" + slugify("Arrivals and departure after sim complete" + str(
                                (scenario_id, cap, alg_name, run_number, cl_args.scenario))))
                            plt.clf()

                            total_demand_remaining_by_now = []
                            for key in sim.energy_demand_remaining_at_time_t:
                                total_demand_remaining_by_now.append(np.mean(sim.energy_demand_remaining_at_time_t[key]))
                            plt.clf()
                            plt.plot(total_demand_remaining_by_now, label=str((scenario_id, cap, alg_name, run_number, cl_args.scenario)))
                            plt.legend()
                            plt.savefig("images_from_acnsim/a_" + slugify("Total energy demand unmet at time t" + str((scenario_id, cap, alg_name, run_number, cl_args.scenario))))
                            plt.clf() # plt.show(block=False)()

                            total_power_demand_remaining_at_time_t = []
                            EVs_present_in_facility_at_time_t = []

                            for key in sim.power_demand_remaining_at_time_t:
                                total_power_demand_remaining_at_time_t.append(np.mean(sim.power_demand_remaining_at_time_t[key]))
                                EVs_present_in_facility_at_time_t.append(sim.EVs_present_in_facility_at_time_t[key])
                            plt.clf()
                            plt.plot(total_power_demand_remaining_at_time_t, label=str((scenario_id, cap, alg_name, run_number, cl_args.scenario)))
                            plt.title("Total power demand unmet at time t")
                            plt.savefig("images_from_acnsim/b_" + str(shared_config.CUTOFF_TIME_FOR_RESULTS_MIXING) + "_" +  slugify("Total power demand unmet at time t" + str((scenario_id, cap, alg_name, run_number, cl_args.scenario))))
                            plt.legend()
                            plt.clf() # plt.show(block=False)()


                            plt.imshow(sim.charging_rates)
                            plt.colorbar()
                            plt.savefig("images_from_acnsim/c_" + str(shared_config.CUTOFF_TIME_FOR_RESULTS_MIXING) + "_" +  cl_args.scenario + (alg_name) + "-cap-" + str(cap) + "charging_Rates.pdf",
                                        dpi=300)


                            with open("overall_power_side_results_total_energy_demand_unmet_by_now.csv", "a") as f2:
                                csvwriter = csv.writer(f2)
                                csvwriter.writerow([str((shared_config.CUTOFF_TIME_FOR_RESULTS_MIXING, cl_args.scenario, scenario_id, cap, alg_name, run_number))] + total_demand_remaining_by_now)

                            with open("overall_power_side_results_total_power_demand_unmet_at_now.csv", "a") as f2:
                                csvwriter = csv.writer(f2)
                                csvwriter.writerow([str((shared_config.CUTOFF_TIME_FOR_RESULTS_MIXING, cl_args.scenario, scenario_id, cap, alg_name, run_number))] + total_power_demand_remaining_at_time_t)

                            with open("overall_power_side_results_total_EV_count_at_now.csv", "a") as f3:
                                csvwriter = csv.writer(f3)
                                csvwriter.writerow([str((shared_config.CUTOFF_TIME_FOR_RESULTS_MIXING, cl_args.scenario, scenario_id, cap, alg_name, run_number))] + EVs_present_in_facility_at_time_t)

                            print ("Simulation complete")
                            try:
                                # sprint(ex.calc_metrics())
                                sprint(result_dict.keys())
                                result_dict[shared_config.CUTOFF_TIME_FOR_RESULTS_MIXING, scenario_id, cap, alg_name, run_number] = ex.calc_metrics()

                            except Exception as e:
                                sprint ("Error!  ", result_dict.keys(), run_number)
                                raise e

                            if shared_config.plot_individual_charging_rates:
                                plt.clf()

                                def compute_metrics_per_arrival_hour(sim):
                                    keys_ = list(sim.ev_history.keys())
                                    arrival_timestamp_list = []
                                    percentage_demand_met_list = []
                                    for i in range(len(keys_)):
                                        ev_ = sim.ev_history[keys_[i]]

                                        arrival_timestamp_list.append(ev_.arrival)
                                        percentage_demand_met_list.append(ev_.energy_delivered / ev_.requested_energy * 100)
                                    arrival_timestamp_list_sorted = [
                                        y for y, x in sorted(zip(arrival_timestamp_list, percentage_demand_met_list))
                                    ]
                                    percentage_demand_met_list_sorted = [
                                        x for y, x in sorted(zip(arrival_timestamp_list, percentage_demand_met_list))
                                    ]
                                    plt.title("% demand met against arrival time, each dot is one EV")
                                    plt.scatter(arrival_timestamp_list_sorted, percentage_demand_met_list_sorted)
                                    plt.savefig("images_from_acnsim/" + cl_args.scenario + (alg_name) + "-demand_met-" + str(cap) + ".pdf", dpi=300)


                                # percent_keys, percent_values = zip(*sorted(sim.percentage_demand_met.items()))
                                # filename_for_met_energy_demand = slugify(cl_args.scenario + "-" + str(cap) + "-" + alg_name)
                                # if not os.path.exists("percentage_demand_met.csv"):
                                #     with open("percentage_demand_met.csv", "w") as f:
                                #         csvwriter = csv.writer(f)
                                #         csvwriter.writerow(["scenario", "run_number"] + ["tod_" + str(t) for t in range(len(sim.percentage_demand_met))])
                                # else:
                                #     with open("percentage_demand_met.csv", "a") as f:
                                #         csvwriter = csv.writer(f)
                                #         csvwriter.writerow([filename_for_met_energy_demand, run_number] + list(percent_values))


                                compute_metrics_per_arrival_hour(sim)

                                plt.figure(figsize=(4.3, 2.5))
                                plt.clf()
                                plot_single_label = True
                                charging_profiles = []
                                for i in range(sim.charging_rates.shape[0]):
                                    profile = (sim.charging_rates[i, :].flatten() * VOLTAGE/1000).tolist()
                                    if not plot_single_label:
                                        plt.plot(profile, alpha=0.1)
                                    else:
                                        plt.plot(profile,
                                                 alpha=0.1)
                                        plt.plot([], [], alpha=0.2, color="black", label="Charging profiles of individual EVs (kW)")
                                        plot_single_label = False
                                    charging_profiles.append(profile)

                                # plt.plot(np.mean(sim.charging_rates, axis=0), label="mean charging current (A)", linewidth=2, color="blue")
                                # plt.plot(np.median(sim.charging_rates, axis=0), label="median", linewidth=2, color="black")
                                # cost_profile = [0.VOLTAGE, 0.202, 0.196, 0.19, 0.192, 0.195, 0.197, 0.185, 0.173, 0.162, 0.052, 0.045, 0.044, 0.043, 0.042, 0.043, 0.127, 0.156, 0.204, 0.264, 0.288, 0.26, 0.25, 0.215]
                                # cost_profile = [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.052, 0.045, 0.044, 0.043, 0.06, 0.061, 0.127, 0.156, 0.204, 0.264, 0.288, 0.26, 0.25, 0.215]
                                # cost_profile = [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.052, 0.052, 0.052, 0.052, 0.052, 0.052, 0.052, 0.127, 0.127, 0.264, 0.264, 0.264, 0.15, 0.15, 0.15]
                                # Path to the JSON file
                                file_path = '/Users/nishant/opt/anaconda3/envs/congestion-and-grid-overload/lib/python3.8/site-packages/acnportal/signals/tariffs/tariff_schedules/' + tariff_name + '.json'

                                # Open and read the JSON file
                                with open(file_path, 'r') as file:
                                    data = json.load(file)
                                cost_profile = data['schedule'][0]['tariffs']


                                cost_profile = np.array([[x] * 12 for x in cost_profile]).flatten().tolist()
                                cost_profile = np.array(cost_profile) * 50 #  shared_config.MAX_CHARGING_L2 * 1000 / VOLTAGE
                                plt.plot(cost_profile, linewidth=5, alpha=0.3, color="tab:blue", label="Cost profile")# ) str(round(shared_config.MAX_CHARGING_L2 * 1000 / VOLTAGE))) $   ($/ kWh)
                                ss = np.max(np.sum(sim.charging_rates, axis=0))
                                # aalabel = []
                                # for xx in [[f"{h}:00"] + [''] * 11 for h in range(24)]:
                                #     aalabel.extend(xx)
                                plt.xticks(range(0, 288, 12), labels=[f"{h}:00" for h in range(24) ], rotation=90, fontsize=7)
                                plt.plot(np.sum(sim.charging_rates, axis=0) * VOLTAGE /1000/ 100, label="Aggregate power delivered (100 kW)", linewidth=2, color="tab:orange")
                                # plt.title(cl_args.scenario + (alg_name))
                                plt.ylabel("Power delivered")
                                plt.xlabel("Time of Day")
                                # plt.text(4, 16, "0.15 $/kWh")
                                plt.xlim(0, 288)
                                plt.ylim(0, 20)
                                plt.grid(alpha=0.1)
                                plt.legend(loc="upper left", fontsize=6)
                                plt.tight_layout()
                                plt.savefig("images_from_acnsim/d_" + str(shared_config.CUTOFF_TIME_FOR_RESULTS_MIXING) + "_" +  cl_args.scenario + (alg_name) + "-cap-" + str(cap) + "-runnum-"+ str(run_number) + ".pdf", dpi=300)
                                # Save the figure elements
                                figure_elements = {
                                    "x_data": list(range(0, 288)),
                                    "y_data": (np.sum(sim.charging_rates, axis=0) * VOLTAGE / 1000 / 100).tolist(),
                                    "cost_profile": cost_profile.flatten().tolist(),
                                    "charging_profiles": charging_profiles,
                                    "xlabel": "Time of Day",
                                    "ylabel": "Power delivered",
                                    "legend_labels": ["Aggregate power delivered (100 kW)", "Cost profile"],
                                    "x_ticks": list(range(0, 288, 12)),
                                    "x_tick_labels": [f"{h}:00" for h in range(24)],
                                }

                                # Save as JSON file
                                with open("images_from_acnsim/d_" + str(shared_config.CUTOFF_TIME_FOR_RESULTS_MIXING) + "_" +  cl_args.scenario + (alg_name) + "-cap-" + str(cap) + "-runnum-"+ str(run_number) + ".json", "w") as json_file:
                                    json.dump(figure_elements, json_file)




                                plt.clf()





                                arrival_list_for_EV_Cost = []
                                cost_list_per_EV = []

                                energy_delivered_per_interval = []
                                for ev_num in range(sim.ev_history.__len__()):
                                    energy_per_interval_kWh = sim.charging_rates[ev_num, :] * VOLTAGE * (5 / 60 / 1000)

                                    # Path to the JSON file
                                    file_path = '/Users/nishant/opt/anaconda3/envs/congestion-and-grid-overload/lib/python3.8/site-packages/acnportal/signals/tariffs/tariff_schedules/' + tariff_name +  '.json'

                                    # Open and read the JSON file
                                    with open(file_path, 'r') as file:
                                        data = json.load(file)
                                    cost_profile = data['schedule'][0]['tariffs']

                                    rate_per_interval = np.repeat(cost_profile, 12)[:sim.charging_rates.shape[1]]
                                    cost_per_interval = rate_per_interval * energy_per_interval_kWh
                                    arrival_list_for_EV_Cost.append(sim.ev_history[list(sim.ev_history.keys())[ev_num]].arrival)
                                    cost_list_per_EV.append(np.sum(cost_per_interval))
                                    energy_delivered_per_interval.append(np.sum(energy_per_interval_kWh))
                                plt.plot(arrival_list_for_EV_Cost, cost_list_per_EV)
                                plt.xticks(range(0, 288, 12), labels=[f"{h}:00" for h in range(24) ], rotation=90, fontsize=7)
                                plt.plot(np.sum(sim.charging_rates, axis=0) * VOLTAGE /1000/ 10, label="Aggregate power delivered (10 kW)", linewidth=2, color="tab:orange")
                                plt.title(cl_args.scenario + (alg_name))
                                plt.ylabel("$ to fully charge")
                                # plt.xlim(0, 288)
                                plt.grid(alpha=0.1)
                                # plt.legend()
                                plt.tight_layout()
                                plt.savefig("images_from_acnsim/da_" + str(shared_config.CUTOFF_TIME_FOR_RESULTS_MIXING) + "_" +  cl_args.scenario + (alg_name) + "-cap-" + str(cap) + ".pdf", dpi=300)
                                with open("images_from_acnsim/" + str(shared_config.CUTOFF_TIME_FOR_RESULTS_MIXING) + "_" +  cl_args.scenario + (alg_name) + "-cap-" + str(cap) + ".csv", "w") as f:
                                    f.write( "arrival_list_for_EV_Cost" + " ," + str(arrival_list_for_EV_Cost))
                                    f.write( "\ncost_list_per_EV" + " ," + str(cost_list_per_EV))
                                    f.write( "\nenergy_delivered_per_interval" + " ," + str(energy_delivered_per_interval))







                            if shared_config.FAST_tryout_for_debug_single_run:
                                break
                            # if "Offline" in scenario_id:
                            #     # We don't need to run it again cuz the offline version has only one algorithm
                            #     break

                        print("run successful")
                        progress.reset(task4)




    start = "9-1-2018"
    end = "10-1-2018"
    # caps = list(range(20, 81, 10)) + [150]
    algs = shared_config.alg_list  # ["ASA-QC", "LLF", "EDF", "RR"]

    percent_del = dict()

    if os.path.exists("overall_power_side_results.csv"):
        mode_ = "a"
    elif not os.path.exists("overall_power_side_results.csv"):
        mode_ = "w"

    with open("overall_power_side_results.csv", mode_) as f:
        csvwriter = csv.writer(f)
        if mode_ == "w":
            csvwriter.writerow(
                [
                    "CUTOFF_TIME_FOR_RESULTS_MIXING",
                    "Traffic-scenario",
                    "Transformer-capacity",
                    "Scenario",
                    "Algorithm",
                    "Run_number",
                    "weight_obj_cost",
                    "proportion_delivered",
                    "demands_fully_met",
                    "peak_current",
                    "demand_charge",
                    "energy_cost_all",
                    "energy_cost_1",
                    "energy_cost_" + str(shared_config.TRUE_ACCIDENT_START_TIME),
                    "energy_cost_" + str(shared_config.TRUE_ACCIDENT_START_TIME + 3),
                    "energy_cost_" + str(shared_config.TRUE_ACCIDENT_START_TIME + 6),
                    "energy_cost_" + str(shared_config.TRUE_ACCIDENT_START_TIME + 9),
                    "total_energy_delivered",
                    "total_energy_requested",
                    "aggregate_power_total",
                    "num_time_steps",

                ]
                +
                ["timestamp:" + str(x) for x in range(500)] # headers for time series
            )
        for CUTOFF_TIME_FOR_RESULTS_MIXING_value in list(set([-1, 1, shared_config.CUTOFF_TIME_FOR_RESULTS_MIXING])):
            for scenario in scenario_order:
                # percent_del[scenario] = np.full((len(algs), len(caps)), np.nan)
                for col, cap in enumerate(caps):
                    run_once_already = False
                    for row, alg in enumerate(algs):

                        alg_name = alg

                        if "Offline" in scenario and run_once_already:
                            continue

                        if "Offline" in scenario and not run_once_already:
                            alg_name = "Offline"
                            run_once_already = True

                        for run_num in range(1, RUN_NUMBER+1):
                            # config = {"scenario": scenario, "start": start, "end": end, "cap": cap, "alg": alg_}
                            # percent_del[scenario][row, col] = get_metric(f"{energy_del_base_dir}/", config, "proportion_delivered")
                            if not (CUTOFF_TIME_FOR_RESULTS_MIXING_value, scenario, cap, (alg_name), run_num) in result_dict:
                                continue
                            stats = result_dict[CUTOFF_TIME_FOR_RESULTS_MIXING_value, scenario, cap, (alg_name), run_num]
                            csvwriter.writerow(
                                [
                                    str(shared_config.CUTOFF_TIME_FOR_RESULTS_MIXING),
                                    cl_args.scenario,
                                    str(cap),
                                    scenario,
                                    (alg_name),
                                    run_num,
                                    weight_obj_cost,
                                    round(stats["proportion_delivered"], 2),
                                    round(stats["demands_fully_met"], 2),
                                    round(stats["peak_current"], 2),
                                    round(stats["demand_charge"], 2),
                                    round(stats["energy_cost_all"], 2),
                                    round(stats["energy_cost_1"], 2),
                                    round(stats["energy_cost_" + str(shared_config.TRUE_ACCIDENT_START_TIME)], 2),
                                    round(stats["energy_cost_" + str(shared_config.TRUE_ACCIDENT_START_TIME + 3)], 2),
                                    round(stats["energy_cost_" + str(shared_config.TRUE_ACCIDENT_START_TIME + 6)], 2),
                                    round(stats["energy_cost_" + str(shared_config.TRUE_ACCIDENT_START_TIME + 9)], 2),
                                    round(stats["total_energy_delivered"], 2),
                                    round(stats["total_energy_requested"], 2),
                                    round(stats["aggregate_power_total"], 2),
                                    round(stats["num_time_steps"], 2),
                                ]
                                +
                                stats["aggregate_power_timeseries"]
                            )
                            if "Offline" in scenario:
                                # We don't need to run it again cuz the offline version has only one algorithm
                                pass
                                # break
                            if shared_config.FAST_tryout_for_debug_single_run:
                                break



