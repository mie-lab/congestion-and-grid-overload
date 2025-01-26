#!/usr/bin/env python
# coding: utf-8

# *If running in Colab run this first to install ACN-Portal.*

# In[ ]:


# if 'google.colab' in str(get_ipython()):
#     print('Running on CoLab')
#     subprocess.check_call([sys.executable, "-m", "pip", "install", "acnportal"])
#     subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/caltech-netlab/adacharge"])


# *Download data from figshare.*

# In[ ]:

# so that the ACN config doesn't have clash with our project config
from config import path_to_EV_arrivals_file as config_path_to_EV_arrivals_file
from config import SECONDS_OFFSET_MIDNIGHT as config_SECONDS_OFFSET_MIDNIGHT

import json
import os
import pickle
import sys
import urllib.request
import zipfile
import pandas as pd
import pytz
import seaborn as sns
import sklearn.mixture
from acnportal import acnsim
from acnportal import algorithms
from acnportal.acnsim import analysis, PluginEvent
from acnportal.acnsim.events import EventQueue
from acnportal.acnsim.events import GaussianMixtureEvents
from acnportal.contrib.acnsim import StochasticNetwork
from acnportal.signals.tariffs.tou_tariff import TimeOfUseTariff
from matplotlib import pyplot as plt
from random import shuffle
from adacharge import *

os.system("cp /Users/nishant/Downloads/temp_data.zip ./")
# os.system("rm -rf results")
# os.system("rm -rf figures")
# os.system("rm -rf events")
# os.system("rm -rf temp_data")

# Download data from figshare.
# url = 'https://ndownloader.figshare.com/files/26553950'
# urllib.request.urlretrieve(url, 'temp_data.zip')

# Extract data into temp directory
# with zipfile.ZipFile('temp_data.zip', 'r') as zip_ref:
#     zip_ref.extractall("temp_data/")
#
# # Move files from temp directory
# for dir in ["results", "figures", "events"]:
#     try:
#         os.rename(f"temp_data/{dir}", dir)
#     except FileNotFoundError:
#         pass

# Remove temporary files
# os.remove("temp_data.zip")
# os.removedirs("temp_data")

# os.system("rm -rf results")
# os.system("mkdir results")

# # Evaluating the Impact of Practical Models on Algorithm Performance
#
# In this example we use real-world operational data to evaluate (through simulations) how the Adaptive Scheduling Algorithm and other baseline algorithms handle the practical challenges in real charging systems. To do this, we consider two practical objectives, charging users quickly in highly constrained systems, and maximizing operating profits.

# In[ ]:


# # Experiment Setup

# In[ ]:


API_KEY = "DEMO_TOKEN"
TIMEZONE = pytz.timezone("America/Los_Angeles")
SITE = "caltech"
PERIOD = 5  # minutes
VOLTAGE = 208  # volts
KW_TO_AMPS = 1000 / 208
KWH_TO_AMP_PERIODS = KW_TO_AMPS * (60 / 5)
MAX_LEN = 144
FORCE_FEASIBLE = True
EVENTS_DIR = "events/"
VERBOSE = True
# Default maximum charging rate for each EV battery.
DEFAULT_BATTERY_POWER = 6.6  # kW


def level_2_network(transformer_cap=200, evse_per_phase=34, voltage=208, is_basic_evse=False):
    """Configurable charging network for level-2 EVSEs connected line to line
        at 208 V.

    Args:
        transformer_cap (float): Capacity of the transformer feeding the network
          [kW]
        evse_per_phase (int): Number of EVSEs on each phase. Total number of
          EVSEs will be 3 * evse_per_phase.

    Returns:
        ChargingNetwork: Configured ChargingNetwork.
    """
    network = StochasticNetwork(early_departure=True)
    voltage = voltage
    evse_type = "AeroVironment"

    # Define the sets of EVSEs in the Caltech ACN.
    AB_ids = ["AB-{0}".format(i) for i in range(evse_per_phase)]
    BC_ids = ["BC-{0}".format(i) for i in range(evse_per_phase)]
    CA_ids = ["CA-{0}".format(i) for i in range(evse_per_phase)]

    if is_basic_evse:
        evse_type = "BASIC"  # {"AV": "BASIC", "CC": "BASIC"}
    else:
        evse_type = "AeroVironment"  # {"AV": "AeroVironment", "CC": "ClipperCreek"}

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
    network.add_constraint(I3a, secondary_side_constr, name="Secondary A")
    network.add_constraint(I3b, secondary_side_constr, name="Secondary B")
    network.add_constraint(I3c, secondary_side_constr, name="Secondary C")
    network.add_constraint(I2a, primary_side_constr, name="Primary A")
    network.add_constraint(I2b, primary_side_constr, name="Primary B")
    network.add_constraint(I2c, primary_side_constr, name="Primary C")

    return network


# All custom unpicklers are due to SO user Pankaj Saini's answer:  https://stackoverflow.com/a/51397373/3896008
class CustomUnpicklerJPLdata(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "sklearn.mixture.gaussian_mixture":
            return sklearn.mixture.GaussianMixture
        if name == "GaussianMixture":
            return sklearn.mixture.GaussianMixture
        return super().find_class(module, name)


def get_synth_events(sessions_per_day):
    # from sklearn.mixture import _gaussian_mixture as gaussian_mixture
    # gmm = pickle.load(open('./data/jpl_weekday_40.pkl', 'rb'))
    custom_pickle_object = CustomUnpicklerJPLdata(open("./data/jpl_weekday_40.pkl", "rb"))
    gmm = custom_pickle_object.load()

    # Generate a list of the number of sessions to draw for each day.
    # This generates 30 days of charging demands.
    # number of EVs on weekends = 0; number of EVs on weekdays = 1
    # num_evs is same as the sessions per day for the function (generate events!)

    # num_evs = [0]*2 + [sessions_per_day]*5 + [0]*2 + [sessions_per_day]*5 + [0]*2 + \
    #           [sessions_per_day]*5 + [0]*2 + [sessions_per_day]*5 + [0]*2

    # Note that because we are drawing from a distribution, some sessions will be
    # invalid, we ignore these sessions and remove the corresponding plugin events.
    gen = GaussianMixtureEvents(pretrained_model=gmm, duration_min=0.08334)

    synth_events = gen.generate_events([sessions_per_day] + [0] * 29, PERIOD, VOLTAGE, DEFAULT_BATTERY_POWER)
    return synth_events


# def process_CTM_arrivals_file(fname, seconds_off_from_midnight):
#     with open(fname) as f:
#         skip = 0
#         for row in f:
#             if skip == 0:
#                 # skip first line
#                 skip = 1
#                 continue
#             listed = row.strip().split(',')
#
#             # TO-DO; cell and scenario not being used now
#             # since we are dealing with only one scenario
#             # and one cell; Need to make this generic
#             cell = listed[0]
#             scenario = listed[1]
#
#             arrivals = [int(i) for i in listed[2:]]
#     total_num_EVS = sum(arrivals)
#
#     expanded_arrivals = []
#     for i in range(len(arrivals)):
#         for j in range(arrivals[i]):
#             expanded_arrivals.append( seconds_off_from_midnight//(5*60) + i)
#
#     charging_network = level_2_network(transformer_cap=200, evse_per_phase=int(total_num_EVS//3))
#     events = get_synth_events(total_num_EVS)
#     event_list = []
#     for i in range(288):
#         event_list.append(deepcopy(events.get_current_events(i)))
#
#     counter = 0
#     event_list_updated = []
#     for i in range(len(event_list)):
#         for j in range(len(event_list[i])):
#             pe = event_list[i][j]
#             dwell_time = pe.ev.departure - pe.ev.arrival
#
#
#             # amend the arrival-departure values in the sampled data to match
#             # CTM output
#             pe.ev.arrival = expanded_arrivals[counter]
#             pe.ev.departure = dwell_time + pe.ev.arrival
#             pe.ev.estimated_departure = pe.ev.departure
#             pe.ev._estimated_departure = pe.ev.departure
#
#             # plugin as soon as vehicle comes in
#             pe.timestamp = pe.ev.arrival
#
#             pe.ev._station_id = charging_network.station_ids[counter]
#
#             # assert that all linked objects are updated.
#             assert pe.ev._estimated_departure == pe.ev._departure
#             assert pe.ev._estimated_departure == pe.ev.departure
#             assert pe.station_id == pe.ev._station_id
#             assert pe.ev.station_id == pe.ev.station_id
#
#             event_list_updated.append(pe)
#             counter += 1
#
#     events = EventQueue(event_list_updated)
#     return events, charging_network


def update_events_with_true_arrivals(events):
    with open(config_path_to_EV_arrivals_file) as f:
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
            cell = listed[0]
            scenario = listed[1]

            arrivals = [int(i) for i in listed[2:]]
    total_num_EVS = sum(arrivals) // 4

    expanded_arrivals = []
    for i in range(len(arrivals)):
        for j in range(arrivals[i]):
            expanded_arrivals.append(config_SECONDS_OFFSET_MIDNIGHT // (5 * 60) + i)

    # charging_network = level_2_network(transformer_cap=200, evse_per_phase=int(total_num_EVS//3))
    # events = get_synth_events(total_num_EVS)
    event_list = []
    for i in range(len(events)):
        event_list.append(deepcopy(events.get_current_events(i)))

    non_empty_events = []
    for i in range(len(event_list)):
        for j in range(len(event_list[i])):
            non_empty_events.append(deepcopy(event_list[i][j]))
    shuffle(non_empty_events)
    event_list = non_empty_events

    # so that we choose randomly top N EVs and modify their arrivals
    assert len(event_list) > total_num_EVS

    counter = 0
    event_list_updated = []
    for i in range(total_num_EVS):
        pe = event_list[i]
        dwell_time = pe.ev.departure - pe.ev.arrival

        # amend the arrival-departure values in the sampled data to match
        # CTM output
        try:
            pe.ev.arrival = expanded_arrivals[counter]
            pe.ev.departure = dwell_time + pe.ev.arrival
            pe.ev.estimated_departure = pe.ev.departure
            pe.ev._estimated_departure = pe.ev.departure
        except:
            debug_here = True
            sys.exit("Error in this block")

        # plugin as soon as vehicle comes in
        pe.timestamp = pe.ev.arrival

        ## no need to change station id;
        ## the algorithm allows for charging when empty
        # pe.ev._station_id = charging_network.station_ids[counter]

        # assert that all linked objects are updated.
        assert pe.ev._estimated_departure == pe.ev._departure
        assert pe.ev._estimated_departure == pe.ev.departure

        ## no need to change station id;
        ## the algorithm allows for charging when empty
        # assert pe.station_id == pe.ev._station_id
        # assert pe.ev.station_id == pe.ev.station_id

        event_list_updated.append(pe)
        counter += 1

    events = EventQueue(event_list_updated[:total_num_EVS])
    return events  # , charging_network


# if __name__ == "__main__":
# event_queue, charging_nw = process_CTM_arrivals_file(config_path_to_EV_arrivals_file, config.SECONDS_OFFSET_MIDNIGHT)


# In[ ]:


def get_events(start, end, ideal_battery, force_feasible, max_len):
    """Gather Events from ACN-Data with a local cache."""
    event_name = f"{start}:{end}:{ideal_battery}:{force_feasible}:" f"{max_len}"
    path = os.path.join(EVENTS_DIR, event_name + ".json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return acnsim.EventQueue.from_json(f)
    start_time = TIMEZONE.localize(datetime.strptime(start, "%m-%d-%Y"))
    end_time = TIMEZONE.localize(datetime.strptime(end, "%m-%d-%Y"))
    default_battery_power = 6.656
    if ideal_battery:
        battery_params = None
    else:
        battery_params = {"type": acnsim.Linear2StageBattery, "capacity_fn": acnsim.models.battery.batt_cap_fn}
    events = acnsim.acndata_events.generate_events(
        API_KEY,
        SITE,
        start_time,
        end_time,
        PERIOD,
        VOLTAGE,
        default_battery_power,
        force_feasible=force_feasible,
        max_len=max_len,
        battery_params=battery_params,
    )
    if not os.path.exists(EVENTS_DIR):
        os.mkdir(EVENTS_DIR)
    with open(path, "w") as f:
        events.to_json(f)
    return events


# In[ ]:


class Experiment:
    """Wrapper for ACN-Sim Experiments including caching serialized experiment to disk."""

    def __init__(self, sim):
        self.sim = sim

    def calc_metrics(self):
        """Calculate metrics from simulation."""
        metrics = {
            "proportion_delivered": analysis.proportion_of_energy_delivered(self.sim) * 100,
            "demands_fully_met": analysis.proportion_of_demands_met(self.sim) * 100,
            "peak_current": self.sim.peak,
            "demand_charge": analysis.demand_charge(self.sim),
            "energy_cost": analysis.energy_cost(self.sim),
            "total_energy_delivered": analysis.total_energy_delivered(self.sim),
            "total_energy_requested": analysis.total_energy_requested(self.sim),
        }
        return metrics

    def log_local_file(self, path):
        """Write simulation, metrics and solver statistics to disk."""
        self.sim.to_json(path + "sim.json")
        with open(path + "metrics.json", "w") as outfile:
            json.dump(self.calc_metrics(), outfile)
        with open(path + "solve_stats.json", "w") as outfile:
            json.dump(self.sim.scheduler.solve_stats, outfile)

    def run_and_store(self, path):
        """Run experiment and store results."""
        print(f"Starting - {path}")
        if os.path.exists(path + "sim.json"):
            print(f"Already Run - {path}...")
            return
        try:
            self.sim.run(maxCount=288)
            if not os.path.exists(path):
                os.makedirs(path)
            self.log_local_file(path)
            print(f"Done - {path}")
        except Exception as e:
            print(f"Failed - {path}")
            print(e)
            pass


# In[ ]:


def map_events_to_valid_stations(events, cn):
    """
    Only needed for offline case
    since we get keyError
    So we manually map the EVSEs and EVs
    """
    event_list = []
    for i in range(len(events)):
        event_list.append(list(events.queue)[i][1])

    # non_empty_events = []
    # for i in range(len(event_list)):
    #     for j in range(len(event_list[i])):
    #         non_empty_events.append(deepcopy(event_list[i][j]))
    #
    # event_list = non_empty_events

    counter = 0
    event_list_updated = []
    for i in range(len(events)):
        pe = event_list[i]
        dwell_time = pe.ev.departure - pe.ev.arrival

        ## align station ides for offline
        pe.ev._station_id = cn.station_ids[counter]

        assert pe.station_id == pe.ev._station_id
        assert pe.ev.station_id == pe.ev.station_id

        event_list_updated.append(pe)
        counter += 1

    events = EventQueue(event_list_updated[: len(events)])
    return events  # , charging_network


def configure_sim(
    alg,
    cap,
    start,
    events,
    basic_evse=True,
    estimate_max_rate=False,
    uninterrupted_charging=False,
    quantized=False,
    allow_overcharging=False,
    tariff_name=None,
    offline=False,
):
    """Configure simulation."""
    start_time = TIMEZONE.localize(datetime.strptime(start, "%m-%d-%Y"))

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

    # TO-DO remove eventqueue from global to local
    charging_nw = level_2_network(
        transformer_cap=cap, evse_per_phase=len(events) // 3, voltage=VOLTAGE, is_basic_evse=basic_evse
    )
    cn = charging_nw  # acnsim.sites.caltech_acn(voltage=VOLTAGE,
    # transformer_cap=cap,
    # basic_evse=basic_evse)

    # cn = acnsim.sites.caltech_acn(voltage=VOLTAGE, transformer_cap=cap, basic_evse=basic_evse)

    if offline:
        cn = acnsim.sites.caltech_acn(voltage=VOLTAGE, transformer_cap=cap, basic_evse=basic_evse)
        # events = map_events_to_valid_stations(events, cn)

    if tariff_name is not None:
        signals = {"tariff": TimeOfUseTariff(tariff_name)}
    else:
        signals = {}
    sim = acnsim.Simulator(cn, alg, events, start_time, signals=signals, period=PERIOD, verbose=False)

    if offline:
        alg.register_events(events)
        try:
            alg.solve()
        except:
            debug_stop = True
            sys.exit(0)

    return sim


# ## Experiment Configurations

# In[ ]:


start = "9-1-2018"
end = "10-1-2018"
tariff_name = "sce_tou_ev_4_march_2019"
revenue = 0.3
# Scenario I is the offline optimal.
scenarios = {
    "II": {
        "ideal_battery": True,
        "estimate_max_rate": False,
        "uninterrupted_charging": False,
        "quantized": False,
        "basic_evse": True,
    },
    "III": {
        "ideal_battery": True,
        "estimate_max_rate": False,
        "uninterrupted_charging": True,
        "quantized": True,
        "basic_evse": False,
    },
    "IV": {
        "ideal_battery": False,
        "estimate_max_rate": True,
        "uninterrupted_charging": False,
        "quantized": False,
        "basic_evse": True,
    },
    "V": {
        "ideal_battery": False,
        "estimate_max_rate": True,
        "uninterrupted_charging": True,
        "quantized": True,
        "basic_evse": False,
    },
}

# ## Energy delivered with constrained infrastructure
#
# **Note that these experiments can take a long time to run.**
#

# In[ ]:


energy_del_base_dir = "results/infrastructure_utilization_results"

# ### Running Experiments
#
# We first consider the objective of maximizing total energy delivered when infrastructure is oversubscribed. This is a common use case when electricity prices are static or when user satisfaction is the primary concern. To optimize for this operator objective, we use the Adaptive Scheduling Algorithm (ASA) with utility function
#
# \begin{equation*}
# U^\mathrm{QC}(r) := u^{QC}(r) + 10^{-12}u^{ES}(r)
# \end{equation*}
#
# Here $U^{QC}$ encourages the system to deliver energy as quickly as possible, which helps free capacity for future arrivals. We include the regularizer $u^{ES}(r)$ to promote equal sharing between similar EVs and force a unique solution. We refer to this algorithm as ASA-QC. We set the weight of the $u^{ES}(r)$ term to be small enough to ensure a strict hierarchy of terms in the objective.
#
# To control congestion in the system, we vary the capacity of transformer $t_1$ between 20 and 150 kW. For reference, the actual transformer in our system is 150 kW, and a conventional system of this size would require 362 kW of capacity. We then measure the percent of the total energy demand met using ASA-QC as well as three baseline scheduling algorithms;  least laxity first (LLF), earliest deadline first (EDF), and round-robin (RR), as implemented in ACN-Sim and are described in (Lee, ACN-Sim, 2020). These baseline algorithms are very common in the deadline scheduling literature and have been applied previously to the EV charging domain (Xu, Dynamic, 2016)(Zeballos, Proportional, 2019}. In addition, the round robin algorithm is a generalization of the equal sharing algorithm used by many charging providers today.

# In[ ]:


ALGS = dict()
ALGS["LLF"] = algorithms.SortedSchedulingAlgo(algorithms.least_laxity_first)
ALGS["EDF"] = algorithms.SortedSchedulingAlgo(algorithms.earliest_deadline_first)
ALGS["RR"] = algorithms.RoundRobin(algorithms.first_come_first_served, continuous_inc=1)

quick_charge_obj = [ObjectiveComponent(quick_charge), ObjectiveComponent(equal_share, 1e-12)]
ALGS["ASA-QC"] = AdaptiveSchedulingAlgorithm(quick_charge_obj, solver="MOSEK", max_recompute=1)

# In[ ]:


# Online Algorithms
caps = [70]  # list(range(20, 81, 10)) + [150]
for scenario_id, scenario in scenarios.items():
    for cap in caps:
        for alg in ["ASA-QC", "LLF", "EDF", "RR"]:
            output_dir = f"{energy_del_base_dir}/{start}:{end}/{scenario_id}/{cap}/{alg}/"
            events = get_events(start, end, scenario["ideal_battery"], FORCE_FEASIBLE, MAX_LEN)

            updated_events = update_events_with_true_arrivals(events)
            events = updated_events

            sim = configure_sim(
                cap=cap,
                alg=deepcopy(ALGS[alg]),
                start=start,
                events=events,
                basic_evse=scenario["basic_evse"],
                estimate_max_rate=scenario["estimate_max_rate"],
                uninterrupted_charging=scenario["uninterrupted_charging"],
                quantized=scenario["quantized"],
                tariff_name=tariff_name,
            )
            ex = Experiment(sim)
            ex.run_and_store(output_dir)

#  We also consider the maximum energy that could be delivered by solving our optimization problem with objective
#
# \begin{equation*}
#     U^\mathrm{EM\_OFF}(r) :=  \sum_{\substack{t \in \mathcal{T}\\i \in \mathcal{V}}} r_i(t)
# \end{equation*}
#
# \noindent and perfect foreknowledge of future arrivals, i.e. $\mathcal{V}$ includes all EVs, not just those present at time $k$. We also modify the constraints so that EVs cannot charge before their arrival time. We refer to this as the *Optimal* solution.

# In[ ]:


# Offline Optimal
optimal_obj = [ObjectiveComponent(total_energy)]
optimal_alg = AdaptiveChargingAlgorithmOffline(optimal_obj, solver="MOSEK")
for cap in caps:
    output_dir = f"{energy_del_base_dir}/{start}:{end}/I/{cap}/Optimal/"
    events = get_events(start, end, True, FORCE_FEASIBLE, MAX_LEN)

    updated_events = update_events_with_true_arrivals(events)
    events = updated_events

    sim = configure_sim(
        cap=cap,
        alg=deepcopy(optimal_alg),
        start=start,
        events=events,
        basic_evse=True,
        estimate_max_rate=False,
        uninterrupted_charging=False,
        quantized=False,
        tariff_name=tariff_name,
        offline=True,
    )
    ex = Experiment(sim)
    ex.run_and_store(output_dir)


# ### Analysis
#
# Results from this experiment are shown below, from which we observe the following trends.
#
# 1. In scenario II, ASA-QC performs near optimally (within 0.4\%), and significantly outperforms the baselines (by as much as 14.1\% compared to EDF with 30 kW capacity).
# 2. In almost all cases, ASA-QC performs better than baselines, especially so in highly congested settings.
# 3. Non-ideal EVSEs (scenarios III and V) have a large negative effect on ASA-QC, which we attribute to rounding of the optimal pilots and restriction of the feasible set.
# 4. Surprisingly, non-ideal EVSEs increase the performance of LLF and EDF for transformer capacities $<$60 kW. This may be because the minimum current constraint leads to better phase balancing.
# 5. Non-ideal batteries (scenarios IV and V) have relatively small effect on the performance of ASA-QC compared to baselines, indicating the robustness of the algorithm.
#
# To understand why ASA-QC performs so much better than the baselines, especially in scenario II, we must consider what information each algorithm uses. RR uses no information aside from which EVs are currently present, and as such, performs the worst. Likewise, EDF uses only information about departure time, while LLF also makes use of the EVs energy demand. Only ASA-QC actively optimizes over infrastructure constraints, allowing it to better balance phases (increasing throughput) and prioritize EVs including current and anticipated congestion. A key feature of the ASA framework is its ability to account for all available information cleanly.\footnote{When even more information is available, i.e., a model of the vehicle's battery or predictions of future EV arrivals, this information can also be accounted for in the constraint set $\mathcal{R}$ and objective $U(r)$. However, these formulations are outside the scope of this paper.

# In[ ]:


def get_metric(results_dir, config, metric_name):
    path = os.path.join(
        results_dir,
        f"{config['start']}:{config['end']}",
        config["scenario"],
        str(config["cap"]),
        config["alg"],
        "metrics.json",
    )
    if not os.path.exists(path):
        return float("nan")
    with open(path) as f:
        metrics = json.load(f)
    if metric_name is None:
        return metrics
    else:
        return metrics[metric_name]


def get_solve_stats(results_dir, config):
    path = os.path.join(
        results_dir,
        f"{config['start']}:{config['end']}",
        config["scenario"],
        str(config["cap"]),
        config["alg"],
        "solve_stats.json",
    )
    if not os.path.exists(path):
        return float("nan")
    with open(path) as f:
        return json.load(f)


def get_sim(results_dir, config):
    path = os.path.join(
        results_dir,
        f"{config['start']}:{config['end']}",
        config["scenario"],
        str(config["cap"]),
        config["alg"],
        "sim.json",
    )
    if not os.path.exists(path):
        return None
    with open(path) as f:
        try:
            return acnsim.Simulator.from_json(f)
        except:
            print(path)


# In[ ]:


start = "9-1-2018"
end = "10-1-2018"
caps = [70]  # list(range(20, 81, 10)) + [150]
algs = ["ASA-QC", "LLF", "EDF", "RR"]
scenario_order = ["II", "III", "IV", "V"]
# scenario_order = ['II', 'V']

percent_del = dict()

percent_del["Optimal"] = np.full((1, len(caps)), np.nan)
for col, cap in enumerate(caps):
    config = {"scenario": "I", "start": start, "end": end, "cap": cap, "alg": "Optimal"}
    percent_del["Optimal"][0, col] = get_metric(f"{energy_del_base_dir}/", config, "proportion_delivered")

for scenario in scenario_order:
    percent_del[scenario] = np.full((len(algs), len(caps)), np.nan)
    for row, alg in enumerate(algs):
        for col, cap in enumerate(caps):
            config = {"scenario": scenario, "start": start, "end": end, "cap": cap, "alg": alg}
            percent_del[scenario][row, col] = get_metric(f"{energy_del_base_dir}/", config, "proportion_delivered")

# In[ ]:


fig, ax = plt.subplots()
ax.set_title("Percent of Energy Demands Met")

fig.set_size_inches(5, 6)
labels = [
    "",
    "Optimal",
    "",
    "ASA-QC",
    "LLF",
    "EDF",
    "RR",
    "",
    "ASA-QC",
    "LLF",
    "EDF",
    "RR",
    "",
    "ASA-QC",
    "LLF",
    "EDF",
    "RR",
    "",
    "ASA-QC",
    "LLF",
    "EDF",
    "RR",
]
# labels = ['', 'Optimal',
#           '', 'LLF',#, 'EDF', 'RR',
#           '', 'LLF',# 'EDF', 'RR',
#           '', 'LLF',# 'EDF', 'RR',
#           '', 'LLF',]# 'EDF', 'RR']

spacer = np.full((1, len(caps)), np.nan)
stack = [spacer]
stack.append(percent_del["Optimal"])
for scenario in scenarios:
    stack.append(spacer)
    stack.append(percent_del[scenario])
heatmap = np.vstack(stack)

sns.heatmap(
    heatmap,
    annot=True,
    fmt=".1f",
    linewidth=1.5,
    ax=ax,
    cbar=False,
    annot_kws={"fontsize": 9},
    yticklabels=labels,
    xticklabels=caps,
)
title_style = {"horizontalalignment": "center", "verticalalignment": "center", "fontsize": 10}
vert_offset = 0.6
ax.text(4, 0 + vert_offset, "I. Offline Optimal", **title_style)
ax.text(4, 2 + vert_offset, "II. Ideal Battery / Continuous Pilot Signal", **title_style)
ax.text(4, 7 + vert_offset, "III. Ideal Battery / Quantized Pilot Signal", **title_style)
ax.text(4, 12 + vert_offset, "IV. Non-Ideal Battery / Continuous Pilot Signal", **title_style)
ax.text(4, 17 + vert_offset, "V. Non-Ideal Battery / Quantized Pilot Signal", **title_style)
ax.set_xlabel("Transformer Capacity (kW)")
ax.axvline(6.99, 21 / 22, 20 / 22, color="k", linewidth=0.5)
ax.axvline(6.99, 19 / 22, 15 / 22, color="k", linewidth=0.5)
ax.axvline(6.99, 19 / 22, 15 / 22, color="k", linewidth=0.5)
ax.axvline(6.99, 14 / 22, 10 / 22, color="k", linewidth=0.5)
ax.axvline(6.99, 9 / 22, 5 / 22, color="k", linewidth=0.5)
ax.axvline(6.99, 4 / 22, 0 / 22, color="k", linewidth=0.5)

ax.tick_params(axis="y", which="both", length=0)
fig.tight_layout()

# In[ ]:

plt.show()
# fig.savefig('figures/infrastructure_heatmap.pdf', dpi=150)


# In[ ]:


start = "9-1-2018"
end = "10-1-2018"
caps = [70]  # list(range(20, 81, 10)) + [150]
scenario_order = ["II", "III", "IV", "V"]
# scenario_order = ['II', 'V']
algs = ["ASA-QC", "LLF", "EDF", "RR"]
avg_solve_time = dict()
for scenario in scenario_order:
    avg_solve_time[scenario] = np.full((len(algs), len(caps)), np.nan)
    for row, alg in enumerate(algs):
        for col, cap in enumerate(caps):
            config = {"scenario": scenario, "start": start, "end": end, "cap": cap, "alg": alg}
            try:
                stats = pd.DataFrame(get_solve_stats(f"{energy_del_base_dir}/", config))
                stats = stats[stats["active_sessions"] > 25]
                avg_solve_time[scenario][row, col] = stats["solve_time"].mean()
            except:
                pass

# In[ ]:


fig, ax = plt.subplots()
ax.set_title("Average Solve Time")
fig.set_size_inches(5, 6)
labels = [
    "",
    "ASA-QC",
    "LLF",
    "EDF",
    "RR",
    "",
    "ASA-QC",
    "LLF",
    "EDF",
    "RR",
    "",
    "ASA-QC",
    "LLF",
    "EDF",
    "RR",
    "",
    "ASA-QC",
    "LLF",
    "EDF",
    "RR",
]

# labels = ['', 'Optimal',
#           '', 'LLF',#, 'EDF', 'RR',
#           '',  'LLF',# 'EDF', 'RR',
#           '', 'LLF',# 'EDF', 'RR',
#           '', 'LLF',]# 'EDF', 'RR']

stack = []
spacer = np.full((1, len(caps)), np.nan)
for scenario in scenarios:
    stack.append(spacer)
    stack.append(avg_solve_time[scenario])
heatmap = np.vstack(stack)

sns.heatmap(
    heatmap,
    annot=True,
    fmt=".1f",
    linewidth=1.5,
    ax=ax,
    cbar=False,
    annot_kws={"fontsize": 9},
    yticklabels=labels,
    xticklabels=caps,
)
title_style = {"horizontalalignment": "center", "verticalalignment": "center", "fontsize": 10}
vert_offset = 0.6
ax.text(4, 0 + vert_offset, "II. Ideal Battery / Continuous Pilot Signal", **title_style)
ax.text(4, 5 + vert_offset, "III. Ideal Battery / Quantized Pilot Signal", **title_style)
ax.text(4, 10 + vert_offset, "IV. Non-Ideal Battery / Continuous Pilot Signal", **title_style)
ax.text(4, 15 + vert_offset, "V. Non-Ideal Battery / Quantized Pilot Signal", **title_style)
ax.set_xlabel("Transformer Capacity (kW)")

ax.tick_params(axis="y", which="both", length=0)
fig.tight_layout()

# ## Profit maximization with TOU tariffs and demand charge
#

# In[ ]:


profit_max_base_dir = "results/profit_max_results"


# ### Running Experiments
#
# Next, we consider the case where a site host would like to minimize their operating costs. Within this case, we will consider the Southern California Edison TOU EV-4 tariff schedule for separately metered EV charging systems between 20-500~kW. In each case, we assume that the charging system operator has a fixed revenue of \$0.30\/kWh and only delivers energy when their marginal cost is less than this revenue.
#
# In order to maximize profit, we use the objective:
#
# \begin{equation*}
#     U^\mathrm{PM} := u^{EC} + u^{DC} + 10^{-6}u^{QC} + 10^{ -12}u^{ES}
# \end{equation*}
#
# \noindent We denote the ASA algorithm with this objective ASA-PM.
#
# The revenue term $\pi$ in $u^{EC}$ can have several interpretations. In the most straightforward case, $\pi$ is simply the price paid by users. However, $\pi$ can also include subsidies by employers, governments, automakers, or carbon credits through programs like the California Low-Carbon Fuel Standard (LCFS). For example, LCFS credits for EV charging have averaged between $0.13 - $0.16 / kWh in 2018-2019. In these cases, some energy demands might not be met if the marginal price of that energy exceeds $\pi$. This is especially important when demand charge is considered since the marginal cost can be extremely high if it causes a spike above the previous monthly peak. Alternatively, $\pi$ can be set to a very high value (greater than the maximum marginal cost of energy) and act as a non-completion penalty. When this is the case, the algorithm will attempt to minimize costs while meeting all energy demands (when it is feasible to do so).
#
# In $u^{DC}$, $\hat{P}$ and $q'$ are tunable parameters. The demand charge proxy $\hat{P}$ controls the trade-off between energy costs and demand charges in the online problem. In this case, we use the heuristic proposed in (Lee, Pricing, 2020), $\hat{P} = P/(D_p - d)$, where $D_p$ is the number of days in the billing period, and $d$ is the index of the current day. We will consider one version of the algorithm without a peak hint, e.g. $q'=0$, and one where the peak hint is 75\% of the optimal peak calculated using data from the previous month. This percentage is chosen based on maximum historic month-to-month variability in the optimal peak (+11\%/-16\%).
#
# We also include the quick charge objective as a regularizer, which encourages the scheduling algorithm to front-load charging within a TOU period. To ensure that this regularizer does not lead to a large increase in cost, we use a coefficient of $10^{-6}$. This results in an maximum increase in value of \$0.000058 / kWh, which is an three orders of magnitude lower than the minimum cost of energy.
#
# We fix the transformer capacity to 150~kW and consider the previous baselines along with uncontrolled charging, which is the most common type of charging system today.

# In[ ]:


def days_remaining_scale_demand_charge(rates, infrastructure, interface, baseline_peak=0, **kwargs):
    """Demand Charge Proxy which divideds the demand charge over the remaining days in the billing period."""
    day_index = interface.current_time // ((60 / interface.period) * 24)
    days_in_month = 30  # monthrange(year, month)[1]
    day_index = min(day_index, days_in_month - 1)
    scale = 1 / (days_in_month - day_index)
    dc = demand_charge(rates, infrastructure, interface, baseline_peak, **kwargs)
    return scale * dc


ALGS = dict()
ALGS["Unctrl"] = algorithms.UncontrolledCharging()
ALGS["LLF"] = algorithms.SortedSchedulingAlgo(algorithms.least_laxity_first)
ALGS["EDF"] = algorithms.SortedSchedulingAlgo(algorithms.earliest_deadline_first)
ALGS["RR"] = algorithms.RoundRobin(algorithms.first_come_first_served, continuous_inc=1)

profit_max_obj_no_hint = [
    ObjectiveComponent(total_energy, revenue),
    ObjectiveComponent(tou_energy_cost),
    ObjectiveComponent(days_remaining_scale_demand_charge),
    ObjectiveComponent(quick_charge, 1e-6),
    ObjectiveComponent(equal_share, 1e-12),
]
ALGS["ASA-PM"] = AdaptiveSchedulingAlgorithm(profit_max_obj_no_hint, solver="MOSEK", max_recompute=1)

profit_max_obj_w_hint = [
    ObjectiveComponent(total_energy, revenue),
    ObjectiveComponent(tou_energy_cost),
    # This peak estimate is taken from Aug 2018.
    ObjectiveComponent(days_remaining_scale_demand_charge, 1, {"baseline_peak": 66.56 * 0.75}),
    ObjectiveComponent(quick_charge, 1e-6),
    ObjectiveComponent(equal_share, 1e-12),
]
ALGS["ASA-PM-Hint"] = AdaptiveSchedulingAlgorithm(profit_max_obj_w_hint, solver="MOSEK", max_recompute=1)

# In[ ]:


# Online Algorithms
for scenario_id, scenario in scenarios.items():
    for alg in ALGS:
        output_dir = f"{profit_max_base_dir}/{start}:{end}/{tariff_name}/{revenue}/" f"{scenario_id}/{cap}/{alg}/"

        events = get_events(start, end, scenario["ideal_battery"], FORCE_FEASIBLE, MAX_LEN)

        updated_events = update_events_with_true_arrivals(events)
        events = updated_events

        sim = configure_sim(
            cap=cap,
            alg=deepcopy(ALGS[alg]),
            start=start,
            events=events,
            basic_evse=scenario["basic_evse"],
            estimate_max_rate=scenario["estimate_max_rate"],
            uninterrupted_charging=scenario["uninterrupted_charging"],
            quantized=scenario["quantized"],
            tariff_name=tariff_name,
        )
        ex = Experiment(sim)
        ex.run_and_store(output_dir)

# We also consider the optimal profit possible by solving an optimization with perfect foreknowledge of arrivals and objective:
#
# \begin{equation*}
# U^\mathrm{PM\_OFF} := u^{EC} + u^{DC}
# \end{equation*}
#
# with $\hat{P} = P$, $q'=0$.

# In[ ]:


# Offline Optimal
optimal_obj = [
    ObjectiveComponent(total_energy, revenue),
    ObjectiveComponent(tou_energy_cost),
    ObjectiveComponent(demand_charge),
]
optimal_alg = AdaptiveChargingAlgorithmOffline(optimal_obj, solver="MOSEK")
output_dir = f"{profit_max_base_dir}/{start}:{end}/{tariff_name}/{revenue}/I/{cap}/Optimal/"
events = get_events(start, end, True, FORCE_FEASIBLE, MAX_LEN)

updated_events = update_events_with_true_arrivals(events)
events = updated_events

sim = configure_sim(alg=deepcopy(optimal_alg), cap=cap, start=start, events=events, tariff_name=tariff_name)
ex = Experiment(sim)
ex.run_and_store(output_dir)


# ### Analysis
#
# Results of the experiment are shown below, from which we observe:
#
# 1. Profits from both ASA-PM and ASA-PM w/ Hint, are within 3.6\% and 1.9\% of the optimal respectively, and far exceed the profits of all baseline algorithms.
# 1. Uncontrolled, LLF and RR result in \emph{lower} energy costs, but incur *very high* demand charges. These algorithms are not price aware. Instead low energy costs are a result of drivers arriving during off-peak and mid-peak times. In particular, uncontrolled charging, which does not consider an infrastructure limit, leads to \emph{extremely high} demand charges. On the other hand, both ASA-PM algorithms (and the offline optimal) trade-off higher energy costs for much lower peaks resulting in lower overall costs.
# 1. Providing a peak hint to ASA-PM increases revenue by allowing more energy demands to be met. In this case, 97.8\% vs. 95.6\% without peak hints. Accurate hints allow the algorithm to utilize higher capacity earlier in the billing period, increasing throughput without increasing cost. Even with the peak hint, ASA-PM does not meet 100\% of demands even though the offline optimal does. Since ASA-PM does not have knowledge of future arrivals, it must act conservatively in increasing the peak over time. It is, however, important that hints not be too large, as the algorithm can increase the peak as needed, but once a high peak is set, the demand charge cannot be lowered.
# 1. While EVSE quantization and non-ideal batteries each reduce the operator's profit, even in scenario V, ASA-PM w/ Hint still produces 90\% of the optimal profit.
# 1. Interestingly, revenue increases in scenarios with quantization (III and V). It can be hard to reason about exactly why this occurs, though it appears that the post-processing step leads to initial conditions for the next solve of **OPT** to produce a higher revenue, higher cost solution.
# 1. Because we use real tariffs structures, real workloads, and realistic assumptions (scenario V), we can conclude with reasonable certainty that a charging system operator could expect to net approximately \$2,600~/~month using an ACN like system, compared to just \$763~/~month in a conventional, uncontrolled system.

# In[ ]:


def get_profit_max_metrics(results_dir, config):
    path = os.path.join(
        results_dir,
        f"{config['start']}:{config['end']}",
        config["tariff"],
        str(config["revenue"]),
        config["scenario"],
        str(config["cap"]),
        config["alg"],
        "metrics.json",
    )
    if not os.path.exists(path):
        print(path)
        return {}
    with open(path) as f:
        metrics = json.load(f)
    return metrics


# In[ ]:


algs = ["Unctrl", "LLF", "RR", "ASA-PM", "ASA-PM-Hint"]
scenario_order = ["II", "III", "IV", "V"]
# scenario_order = ['II', 'V']
cap = 150
data = []

config = {
    "scenario": "I",
    "start": start,
    "end": end,
    "cap": cap,
    "alg": "Optimal",
    "tariff": tariff_name,
    "revenue": revenue,
}
opt_metrics = get_profit_max_metrics(f"{profit_max_base_dir}/", config)
opt_metrics["scenario"] = "I"
opt_metrics["alg"] = "Optimal"
data.append(opt_metrics)

for scenario in scenario_order:
    for row, alg in enumerate(algs):
        config = {
            "scenario": scenario,
            "start": start,
            "end": end,
            "cap": cap,
            "alg": alg,
            "tariff": tariff_name,
            "revenue": revenue,
        }
        metrics = get_profit_max_metrics(f"{profit_max_base_dir}/", config)
        metrics["scenario"] = scenario
        if alg == "ASA-PM-Hint":
            alg = "ASA-PM w/ Hint"
        metrics["alg"] = alg
        data.append(metrics)

# In[ ]:

"""
df = pd.DataFrame(data)
requested = 20598.87  # total requested kWh
df['revenue'] = df['proportion_delivered'] / 100 * requested * revenue
df['total_cost'] = df['demand_charge'] + df['energy_cost']
df['profit'] = df['revenue'] - df['total_cost']

# In[ ]:


# Show Experiment Results
df

# In[ ]:


cmap = sns.diverging_palette(220, 20, n=10)
colors = [cmap[1], cmap[2], cmap[3], cmap[7], cmap[8]]
opt_color = cmap[9]
bar_width = 0.75

order = ['Unctrl', 'RR', 'LLF', 'ASA-PM', 'ASA-PM w/ Hint']
fig, ax = plt.subplots(4, 1, figsize=(4, 5), sharex=True)
pt = df.pivot(index='scenario', columns='alg')

# Revenue
pt['revenue'][order].loc[scenarios].plot.bar(ax=ax[0], linewidth=1, edgecolor='w', color=colors, width=bar_width,
                                             legend=False)
ax[0].fill_between([-1, 5], 0, pt['revenue']['Optimal']['I'], facecolor='grey', alpha=0.25, label='Optimal')
lgd = ax[0].legend(bbox_to_anchor=(0., 1.1, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0., fontsize=9)
ax[0].set_ylim(5600, 6300)
ax[0].set_ylabel('Revenue ($)')

# Energy Cost
pt['energy_cost'][order].loc[scenarios].plot.bar(ax=ax[1], linewidth=1, edgecolor='w', color=colors, width=bar_width,
                                                 legend=False)
ax[1].fill_between([-1, 5], 0, pt['energy_cost']['Optimal']['I'], facecolor='grey', alpha=0.25)
ax[1].set_ylim(2000, 2550)
ax[1].set_ylabel('Energy\nCost ($)')

# Demand Charge
pt['demand_charge'][order].loc[scenarios].plot.bar(ax=ax[2], linewidth=1, edgecolor='w', color=colors, width=bar_width,
                                                   legend=False)
ax[2].fill_between([-1, 5], 0, pt['demand_charge']['Optimal']['I'], facecolor='grey', alpha=0.25)
ax[2].set_ylim(0, 3500)
ax[2].set_ylabel('Demand\nCharge ($)')

# Profit
pt['profit'][order].loc[scenarios].plot.bar(ax=ax[3], linewidth=1, edgecolor='w', color=colors, width=bar_width,
                                            legend=False)
ax[3].fill_between([-1, 5], 0, pt['profit']['Optimal']['I'], facecolor='grey', alpha=0.25, label='Optimal')
ax[3].set_ylim(0, 3250)
ax[3].set_ylabel('Profit ($)')

plt.xticks(rotation=0)
ax[3].set_xlabel('Scenario')
plt.tight_layout()

# In[ ]:

plt.show()
# fig.savefig('figures/profit_maximization.pdf', dpi=150)


# #### Percentage of Optimal Profit

# In[ ]:


pt['profit'] / pt['profit']['Optimal']['I']

# In[ ]:

"""
