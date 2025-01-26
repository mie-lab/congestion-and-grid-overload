import copy
from datetime import datetime
from typing import Dict

import warnings
import json

# noinspection PyProtectedMember
from pydoc import locate

from .events import *
from .events import UnplugEvent
from .interface import Interface
from .interface import InvalidScheduleError
from acnportal.algorithms import BaseAlgorithm
from .base import BaseSimObj


class Simulator(BaseSimObj):
    """ Central class of the acnsim package.

    The Simulator class is the central place where everything about a particular simulation is stored including the
    network, scheduling algorithm, and events. It is also where timekeeping is done and orchestrates calling the
    scheduling algorithm, sending pilots to the network, and updating the energy delivered to each EV.

    Args:
        network (ChargingNetwork): The charging network which the simulation will use.
        scheduler (BaseAlgorithm): The scheduling algorithm used in the simulation.
            If scheduler = None, Simulator.run() cannot be called.
        events (EventQueue): Queue of events which will occur in the simulation.
        start (datetime): Date and time of the first period of the simulation.
        period (float): Length of each time interval in the simulation in minutes. Default: 1
        signals (Dict[str, ...]):
        store_schedule_history (bool): If True, store the scheduler output each time it is run. Note this can use lots
            of memory for long simulations.
        interface_type (type): The class of interface to register with the scheduler.
    """

    period: float

    def __init__(
        self,
        network,
        scheduler,
        events,
        start,
        period: float = 1,
        signals=None,
        store_schedule_history=False,
        verbose=True,
        interface_type=Interface,
    ):
        self.network = network
        self.scheduler = scheduler
        self.max_recompute = None
        self.event_queue = events
        self.start = start
        self.period = period
        self.signals = signals
        self.verbose = verbose
        self.energy_demand_remaining_at_time_t = {}
        self.EVs_present_in_facility_at_time_t = {}

        self.power_demand_remaining_at_time_t = {}
        # Information storage
        width = 1
        if self.event_queue.get_last_timestamp() is not None:
            width = self.event_queue.get_last_timestamp() + 1
        self.pilot_signals = np.zeros((len(self.network.station_ids), width))
        self.charging_rates = np.zeros((len(self.network.station_ids), width))
        self.peak = 0
        self.ev_history = {}
        self.event_history = []
        if store_schedule_history:
            self.schedule_history = {}
        else:
            self.schedule_history = None

        # Local Variables
        self._iteration = 0
        self._resolve = False
        self._last_schedule_update = None

        # Interface registration is moved here so that copies of this
        # simulator have all attributes.
        if scheduler is not None:
            self.max_recompute = scheduler.max_recompute
            self.scheduler.register_interface(interface_type(self))

    @property
    def iteration(self):
        return self._iteration

    def run(self, save_ev_state=False):
        """
        If scheduler is not None, run the simulation until the event queue is empty.

        The run function is the heart of the simulator. It triggers all actions and keeps the simulator moving forward.
        Its actions are (in order):
            1. Get current events from the event queue and execute them.
            2. If necessary run the scheduling algorithm.
            3. Send pilot signals to the network.
            4. Receive back actual charging rates from the network and store the results.

        Returns:
            None

        Raises:
            TypeError: If called when the scheduler attribute is None.
                The run() method requires a BaseAlgorithm-like
                scheduler to execute.
        """
        if self.scheduler is None:
            raise TypeError("Add a scheduler before attempting to call" " run().")
        while not self.event_queue.empty():
            current_events = self.event_queue.get_current_events(self._iteration)
            for e in current_events:
                self.event_history.append(e)
                self._process_event(e)
            if (
                self._resolve
                or self.max_recompute is not None
                and (
                    self._last_schedule_update is None
                    or self._iteration - self._last_schedule_update
                    >= self.max_recompute
                )
            ):
                new_schedule = self.scheduler.run()
                self._update_schedules(new_schedule)
                if self.schedule_history is not None:
                    self.schedule_history[self._iteration] = new_schedule
                self._last_schedule_update = self._iteration
                self._resolve = False
            if not self.event_queue.empty():
                width_increase = self.event_queue.get_last_timestamp() + 1
            else:
                width_increase = self._iteration + 1
            self.pilot_signals = _increase_width(self.pilot_signals, width_increase)
            self.charging_rates = _increase_width(self.charging_rates, width_increase)
            self.network.update_pilots(self.pilot_signals, self._iteration, self.period)
            self._store_actual_charging_rates()
            self.network.post_charging_update()
            all_evs = self.get_active_evs()
            self.energy_demand_remaining_at_time_t[self._iteration] = []

            import csv

            def write_evs_to_csv(evs, filename):
                with open(filename, 'w', newline='') as csvfile:
                    fieldnames = [
                        'arrival', 'departure', 'session_id', 'station_id',
                        'requested_energy', 'estimated_departure', 'battery_capacity',
                        'battery_current_charge', 'battery_init_charge', 'battery_max_power',
                        'battery_current_charging_power', 'energy_delivered'
                    ]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()

                    for ev in evs:
                        battery = ev._battery
                        writer.writerow({
                            'arrival': ev._arrival,
                            'departure': ev._departure,
                            'session_id': ev._session_id,
                            'station_id': ev._station_id,
                            'requested_energy': ev._requested_energy,
                            'estimated_departure': ev._estimated_departure,
                            'battery_capacity': battery._capacity,
                            'battery_current_charge': battery._current_charge,
                            'battery_init_charge': battery._init_charge,
                            'battery_max_power': battery._max_power,
                            'battery_current_charging_power': battery._current_charging_power,
                            'energy_delivered': ev._energy_delivered
                        })

            if save_ev_state:
                write_evs_to_csv(all_evs, 'evs_tod_'+str(self._iteration)+'.csv')

            count_total_num_evs = 0
            for ind_, ev in enumerate(all_evs):
                self.energy_demand_remaining_at_time_t[self._iteration].append(ev.percent_remaining)
                count_total_num_evs += 1

            self.EVs_present_in_facility_at_time_t[self._iteration] = count_total_num_evs

            self.power_demand_remaining_at_time_t[self._iteration] = []
            for ind_, ev in enumerate(all_evs):
                self.power_demand_remaining_at_time_t[self._iteration].append(\
                    1 - ev._battery.current_charging_power/ev._battery.max_charging_power)
                try:
                    assert ev._battery.current_charging_power/ev._battery.max_charging_power >= 0
                except Exception as e:
                    print (ev._battery.current_charging_power, ev._battery.max_charging_power)
                    raise e
            self._iteration = self._iteration + 1

    def step(self, new_schedule):
        """ Step the simulation until the next schedule recompute is
        required.

        The step function executes a single iteration of the run()
        function. However, the step function updates the simulator with
        an input schedule rather than query the scheduler for a new
        schedule when one is required. Also, step will return a flag if
        the simulation is done.

        Args:
            new_schedule (Dict[str, List[number]]): Dictionary mapping
                station ids to a schedule of pilot signals.

        Returns:
            bool: True if the simulation is complete.
        """
        while (
            not self.event_queue.empty()
            and not self._resolve
            and (
                self.max_recompute is None
                or (self._iteration - self._last_schedule_update < self.max_recompute)
            )
        ):
            self._update_schedules(new_schedule)
            if self.schedule_history is not None:
                self.schedule_history[self._iteration] = new_schedule
            self._last_schedule_update = self._iteration
            self._resolve = False
            if self.event_queue.get_last_timestamp() is not None:
                width_increase = max(
                    self.event_queue.get_last_timestamp() + 1, self._iteration + 1
                )
            else:
                width_increase = self._iteration + 1
            self.pilot_signals = _increase_width(self.pilot_signals, width_increase)
            self.charging_rates = _increase_width(self.charging_rates, width_increase)
            self.network.update_pilots(self.pilot_signals, self._iteration, self.period)
            self._store_actual_charging_rates()
            self.network.post_charging_update()
            self._iteration = self._iteration + 1
            current_events = self.event_queue.get_current_events(self._iteration)
            for e in current_events:
                self.event_history.append(e)
                self._process_event(e)
        return self.event_queue.empty()

    def get_active_evs(self):
        """ Return all EVs which are plugged in and not fully charged at the current time.

        Wrapper for self.network.active_evs. See its documentation for more details.

        Returns:
            List[EV]: List of all EVs which are plugged in but not fully charged at the current time.

        """
        evs = copy.deepcopy(self.network.active_evs)
        return evs

    def _process_event(self, event):
        """ Process an event and take appropriate actions.

        Args:
            event (Event): Event to be processed.

        Returns:
            None
        """
        if event.event_type == "Plugin":
            self._print("Plugin Event...")
            self.network.plugin(event.ev)
            self.ev_history[event.ev.session_id] = event.ev
            self.event_queue.add_event(UnplugEvent(event.ev.departure, event.ev))
            self._resolve = True
            self._last_schedule_update = event.timestamp
        elif event.event_type == "Unplug":
            self._print("Unplug Event...")
            self.network.unplug(event.ev.station_id, event.ev.session_id)
            self._resolve = True
            self._last_schedule_update = event.timestamp
        elif event.event_type == "Recompute":
            self._print("Recompute Event...")
            self._resolve = True

    def _update_schedules(self, new_schedule):
        """ Extend the current self.pilot_signals with the new pilot signal schedule.

        Args:
            new_schedule (Dict[str, List[number]]): Dictionary mapping station ids to a schedule of pilot signals.

        Returns:
            None

        Raises:
            KeyError: Raised when station_id is in the new_schedule but not registered in the Network.
        """
        if len(new_schedule) == 0:
            return

        for station_id in new_schedule:
            if station_id not in self.network.station_ids:
                raise KeyError(
                    "Station {0} in schedule but not found in network.".format(
                        station_id
                    )
                )

        schedule_lengths = set(len(x) for x in new_schedule.values())
        if len(schedule_lengths) > 1:
            raise InvalidScheduleError("All schedules should have the same length.")
        schedule_length = schedule_lengths.pop()

        schedule_matrix = np.array(
            [
                new_schedule[evse_id]
                if evse_id in new_schedule
                else [0] * schedule_length
                for evse_id in self.network.station_ids
            ]
        )
        if not self.network.is_feasible(schedule_matrix):
            aggregate_currents = self.network.constraint_current(schedule_matrix)
            diff_vec = (
                np.abs(aggregate_currents)
                - np.tile(
                    self.network.magnitudes + self.network.violation_tolerance,
                    (schedule_length, 1),
                ).T
            )
            max_idx = np.unravel_index(np.argmax(diff_vec), diff_vec.shape)
            max_diff = diff_vec[max_idx]
            max_timeidx = max_idx[1]
            max_constraint = self.network.constraint_index[max_idx[0]]
            warnings.warn(
                f"Invalid schedule provided at iteration {self._iteration}. "
                f"Max violation is {max_diff} A on {max_constraint} "
                f"at time index {max_timeidx}.",
                UserWarning,
            )
        if self._iteration + schedule_length <= self.pilot_signals.shape[1]:
            self.pilot_signals[
                :, self._iteration : (self._iteration + schedule_length)
            ] = schedule_matrix
        else:
            # We've reached the end of pilot_signals, so double pilot_signal array width
            self.pilot_signals = _increase_width(
                self.pilot_signals,
                max(
                    self.event_queue.get_last_timestamp() + 1,
                    self._iteration + schedule_length,
                ),
            )
            self.pilot_signals[
                :, self._iteration : (self._iteration + schedule_length)
            ] = schedule_matrix

    def _store_actual_charging_rates(self):
        """ Store actual charging rates from the network in the simulator for later analysis."""
        current_rates = self.network.current_charging_rates
        agg = np.sum(current_rates)
        if self.iteration < self.charging_rates.shape[1]:
            self.charging_rates[:, self.iteration] = current_rates.T
        else:
            if not self.event_queue.empty():
                width_increase = self.event_queue.get_last_timestamp() + 1
            else:
                width_increase = self._iteration + 1
            self.charging_rates = _increase_width(self.charging_rates, width_increase)
            self.charging_rates[:, self._iteration] = current_rates.T
        self.peak = max(self.peak, agg)

    def _print(self, s):
        if self.verbose:
            print(s)

    def charging_rates_as_df(self):
        """ Return the charging rates as a pandas DataFrame, with EVSE id as columns
        and iteration as index.

        Returns:
            pandas.DataFrame: A DataFrame containing the charging rates
                of the simulation. Columns are EVSE id, and the index is
                the iteration.
        """
        return pd.DataFrame(
            data=self.charging_rates.T, columns=self.network.station_ids
        )

    def pilot_signals_as_df(self):
        """ Return the pilot signals as a pandas DataFrame

        Returns:
            pandas.DataFrame: A DataFrame containing the pilot signals
                of the simulation. Columns are EVSE id, and the index is
                the iteration.
        """
        return pd.DataFrame(data=self.pilot_signals.T, columns=self.network.station_ids)

    def index_of_evse(self, station_id):
        """ Return the numerical index of the EVSE given by station_id in the (ordered) dictionary
        of EVSEs.
        """
        if station_id not in self.network.station_ids:
            raise KeyError("EVSE {0} not found in network.".format(station_id))
        return self.network.station_ids.index(station_id)

    def _to_dict(
        self, context_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Implements BaseSimObj._to_dict. Certain simulator attributes are
        not serialized completely as they are not ACN-Sim objects
        (signals and scheduler exist in their own modules).

        If the Python version used is less than 3.7, datetimes cannot be
        accurately loaded. As such, a warning is thrown when the start
        attribute is serialized.

        The signals attribute is only serialized if it is natively
        JSON Serializable, otherwise None is stored.

        Only the scheduler's name is serialized.
        """
        attribute_dict = {}

        # noinspection PyProtectedMember
        registry, context_dict = self.network._to_registry(context_dict=context_dict)
        attribute_dict["network"] = registry["id"]

        registry, context_dict = self.event_queue._to_registry(
            context_dict=context_dict
        )
        attribute_dict["event_queue"] = registry["id"]

        attribute_dict["scheduler"] = (
            f"{self.scheduler.__module__}." f"{self.scheduler.__class__.__name__}"
        )

        attribute_dict["start"] = self.start.strftime("%H:%M:%S.%f %d%m%Y")

        try:
            json.dumps(self.signals)
        except TypeError:
            warnings.warn(
                "Not serializing signals as value types"
                "are not natively JSON serializable.",
                UserWarning,
            )
            attribute_dict["signals"] = None
        else:
            attribute_dict["signals"] = self.signals

        # Serialize non-nested attributes.
        nn_attr_lst = [
            "period",
            "max_recompute",
            "verbose",
            "peak",
            "_iteration",
            "_resolve",
            "_last_schedule_update",
            "schedule_history",
            "pilot_signals",
            "charging_rates",
        ]
        for attr in nn_attr_lst:
            attribute_dict[attr] = getattr(self, attr)

        ev_history = {}
        for session_id, ev in self.ev_history.items():
            # noinspection PyProtectedMember
            registry, context_dict = ev._to_registry(context_dict=context_dict)
            ev_history[session_id] = registry["id"]
        attribute_dict["ev_history"] = ev_history

        event_history = []
        for past_event in self.event_history:
            # noinspection PyProtectedMember
            registry, context_dict = past_event._to_registry(context_dict=context_dict)
            event_history.append(registry["id"])
        attribute_dict["event_history"] = event_history

        return attribute_dict, context_dict

    @classmethod
    def _from_dict(
        cls,
        attribute_dict: Dict[str, Any],
        context_dict: Dict[str, Any],
        loaded_dict: Optional[Dict[str, BaseSimObj]] = None,
    ) -> Tuple[BaseSimObj, Dict[str, BaseSimObj]]:
        """
        Implements BaseSimObj._from_dict. Certain simulator attributes
        are not loaded completely as they are not ACN-Sim objects
        (signals and scheduler exist in their own modules).

        If the Python version used is less than 3.7, the start attribute
        is stored in ISO format instead of datetime, and a warning is
        thrown.

        The signals attribute is only loaded if it was natively
        JSON Serializable, in the original object, otherwise None is
        set as the signals attribute. The Simulator's signals can be set
        after the Simulator is loaded.

        The scheduler attribute is only accurate if the scheduler's
        constructor takes no arguments, otherwise BaseAlgorithm is
        stored. The Simulator provides a method to set the scheduler
        after the Simulator is loaded.

        """
        # noinspection PyProtectedMember
        network, loaded_dict = BaseSimObj._build_from_id(
            attribute_dict["network"], context_dict, loaded_dict=loaded_dict
        )

        # noinspection PyProtectedMember
        events, loaded_dict = BaseSimObj._build_from_id(
            attribute_dict["event_queue"], context_dict, loaded_dict=loaded_dict
        )

        scheduler_cls = locate(attribute_dict["scheduler"])
        try:
            scheduler = scheduler_cls()
        except TypeError:
            warnings.warn(
                f"Scheduler {attribute_dict['scheduler']} "
                f"requires constructor inputs. Setting "
                f"scheduler to BaseAlgorithm instead."
            )
            scheduler = BaseAlgorithm()

        start = datetime.strptime(attribute_dict["start"], "%H:%M:%S.%f %d%m%Y")

        out_obj = cls(
            network,
            scheduler,
            events,
            start,
            period=attribute_dict["period"],
            signals=attribute_dict["signals"],
            verbose=attribute_dict["verbose"],
        )
        scheduler.register_interface(Interface(out_obj))

        attr_lst = [
            "max_recompute",
            "peak",
            "_iteration",
            "_resolve",
            "_last_schedule_update",
        ]
        for attr in attr_lst:
            setattr(out_obj, attr, attribute_dict[attr])

        if attribute_dict["schedule_history"] is not None:
            out_obj.schedule_history = {
                int(key): value
                for key, value in attribute_dict["schedule_history"].items()
            }
        else:
            out_obj.schedule_history = None

        out_obj.pilot_signals = np.array(attribute_dict["pilot_signals"])
        out_obj.charging_rates = np.array(attribute_dict["charging_rates"])

        ev_history = {}
        for session_id, ev in attribute_dict["ev_history"].items():
            # noinspection PyProtectedMember
            ev_elt, loaded_dict = BaseSimObj._build_from_id(
                ev, context_dict, loaded_dict=loaded_dict
            )
            ev_history[session_id] = ev_elt
        out_obj.ev_history = ev_history

        event_history = []
        for past_event in attribute_dict["event_history"]:
            # noinspection PyProtectedMember
            loaded_event, loaded_dict = BaseSimObj._build_from_id(
                past_event, context_dict, loaded_dict=loaded_dict
            )
            event_history.append(loaded_event)
        out_obj.event_history = event_history

        return out_obj, loaded_dict

    def update_scheduler(self, new_scheduler):
        """ Updates a Simulator's schedule. """
        self.scheduler = new_scheduler
        self.scheduler.register_interface(Interface(self))
        self.max_recompute = new_scheduler.max_recompute


def _increase_width(a, target_width):
    """ Returns a new 2-D numpy array with target_width number of columns, with the contents
    of a up to the first a.shape[1] columns and 0's thereafter.

    Args:
        a (numpy.Array): 2-D numpy array to be expanded.
        target_width (int): desired number of columns; must be greater than number of columns in a
    Returns:
        numpy.Array
    """
    if target_width <= a.shape[1]:
        return a
    new_matrix = np.zeros((a.shape[0], target_width))
    new_matrix[:, : a.shape[1]] = a
    return new_matrix






############################################################# OPTIMISER REFERENCE #############################################################
from typing import List, Union, Optional
from collections import namedtuple
import numpy as np
import cvxpy as cp
from acnportal.acnsim.interface import Interface, SessionInfo, InfrastructureInfo


class InfeasibilityException(Exception):
    pass


ObjectiveComponent = namedtuple(
    "ObjectiveComponent", ["function", "coefficient", "kwargs"]
)
ObjectiveComponent.__new__.__defaults__ = (1, {})


class AdaptiveChargingOptimization:
    """Base class for all MPC based charging algorithms.

    Args:
        objective (List[ObjectiveComponent]): List of components which make up the optimization objective.
        interface (Interface): Interface providing information used by the algorithm.
        constraint_type (str): String representing which constraint type to use. Options are 'SOC' for Second Order Cone
            or 'LINEAR' for linearized constraints.
        enforce_energy_equality (bool): If True, energy delivered must be equal to energy requested for each EV.
            If False, energy delivered must be less than or equal to request.
        solver (str): Backend solver to use. See CVXPY for available solvers.
    """

    def __init__(
        self,
        objective: List[ObjectiveComponent],
        interface: Interface,
        constraint_type="SOC",
        enforce_energy_equality=False,
        solver="ECOS",
    ):
        self.interface = interface
        self.constraint_type = constraint_type
        self.enforce_energy_equality = enforce_energy_equality
        self.solver = solver
        self.objective_configuration = objective

    @staticmethod
    def charging_rate_bounds(
        rates: cp.Variable, active_sessions: List[SessionInfo], evse_index: List[str]
    ):
        """Get upper and lower bound constraints for each charging rate.

        Args:
            rates (cp.Variable): cvxpy variable representing all charging rates. Shape should be (N, T) where N is the
                total number of EVSEs in the system and T is the length of the optimization horizon.
            active_sessions (List[SessionInfo]): List of SessionInfo objects for all active charging sessions.
            evse_index (List[str]): List of IDs for all EVSEs. Index in evse_index represents the row number of that
                EVSE in rates.

        Returns:
            List[cp.Constraint]: List of lower bound constraint, upper bound constraint.
        """
        lb, ub = np.zeros(rates.shape), np.zeros(rates.shape)
        for session in active_sessions:
            i = evse_index.index(session.station_id)
            lb[
                i,
                session.arrival_offset : session.arrival_offset
                + session.remaining_time,
            ] = session.min_rates
            ub[
                i,
                session.arrival_offset : session.arrival_offset
                + session.remaining_time,
            ] = session.max_rates
        # To ensure feasibility, replace upper bound with lower bound when they conflict
        ub[ub < lb] = lb[ub < lb]
        return {
            "charging_rate_bounds.lb": rates >= lb,
            "charging_rate_bounds.ub": rates <= ub,
        }

    @staticmethod
    def energy_constraints(
        rates: cp.Variable,
        active_sessions: List[SessionInfo],
        infrastructure: InfrastructureInfo,
        period,
        enforce_energy_equality=False,
    ):
        """Get constraints on the energy delivered for each session.

        Args:
            rates (cp.Variable): cvxpy variable representing all charging rates. Shape should be (N, T) where N is the
                total number of EVSEs in the system and T is the length of the optimization horizon.
            active_sessions (List[SessionInfo]): List of SessionInfo objects for all active charging sessions.
            infrastructure (InfrastructureInfo): InfrastructureInfo object describing the electrical infrastructure at
                a site.
            period (int): Length of each discrete time period. (min)
            enforce_energy_equality (bool): If True, energy delivered must be equal to energy requested for each EV.
                If False, energy delivered must be less than or equal to request.

        Returns:
            List[cp.Constraint]: List of energy delivered constraints for each session.
        """
        constraints = {}
        for session in active_sessions:
            i = infrastructure.get_station_index(session.station_id)
            planned_energy = cp.sum(
                rates[
                    i,
                    session.arrival_offset : session.arrival_offset
                    + session.remaining_time,
                ]
            )
            planned_energy *= infrastructure.voltages[i] * period / 1e3 / 60
            constraint_name = f"energy_constraints.{session.session_id}"
            if enforce_energy_equality:
                constraints[constraint_name] = (
                    planned_energy == session.remaining_demand
                )
            else:
                constraints[constraint_name] = (
                    planned_energy <= session.remaining_demand
                )
        return constraints

    @staticmethod
    def infrastructure_constraints(
        rates: cp.Variable, infrastructure: InfrastructureInfo, constraint_type="SOC"
    ):
        """Get constraints enforcing infrastructure limits.

        Args:
            rates (cp.Variable): cvxpy variable representing all charging rates. Shape should be (N, T) where N is the
                total number of EVSEs in the system and T is the length of the optimization horizon.
            infrastructure (InfrastructureInfo): InfrastructureInfo object describing the electrical infrastructure at
                a site.
            constraint_type (str): String representing which constraint type to use. Options are 'SOC' for Second Order
                Cone or 'LINEAR' for linearized constraints.

        Returns:
            List[cp.Constraint]: List of constraints, one for each bottleneck in the electrical infrastructure.
        """
        # If constraint_matrix is empty, no need to add infrastructure
        # constraints.
        if (
            infrastructure.constraint_matrix is None
            or infrastructure.constraint_matrix.shape == (0, 0)
        ):
            return {}
        constraints = {}
        if constraint_type == "SOC":
            if infrastructure.phases is None:
                raise ValueError(
                    "phases is required when using SOC infrastructure constraints."
                )
            phase_in_rad = np.deg2rad(infrastructure.phases)
            for j, v in enumerate(infrastructure.constraint_matrix):
                a = np.stack([v * np.cos(phase_in_rad), v * np.sin(phase_in_rad)])
                constraint_name = (
                    f"infrastructure_constraints." f"{infrastructure.constraint_ids[j]}"
                )
                constraints[constraint_name] = (
                    cp.norm(a @ rates, axis=0) <= infrastructure.constraint_limits[j]
                )
        elif constraint_type == "LINEAR":
            for j, v in enumerate(infrastructure.constraint_matrix):
                constraint_name = (
                    f"infrastructure_constraints.{infrastructure.constraint_ids[j]}"
                )
                constraints[constraint_name] = (
                    np.abs(v) @ rates <= infrastructure.constraint_limits[j]
                )
        else:
            raise ValueError(
                "Invalid infrastructure constraint type: {0}. Valid options are SOC or AFFINE.".format(
                    constraint_type
                )
            )
        return constraints

    @staticmethod
    def peak_constraint(
        rates: cp.Variable, peak_limit: Union[float, List[float], np.ndarray]
    ):
        """Get constraints enforcing infrastructure limits.

        Args:
            rates (cp.Variable): cvxpy variable representing all charging rates. Shape should be (N, T) where N is the
                total number of EVSEs in the system and T is the length of the optimization horizon.
            peak_limit (Union[float, List[float], np.ndarray]): Limit on aggregate peak current. If None, no limit is
                enforced.

        Returns:
            List[cp.Constraint]: List of constraints, one for each bottleneck in the electrical infrastructure.
        """
        if peak_limit is not None:
            return {"peak_constraint": cp.sum(rates, axis=0) <= peak_limit}
        return {}

    def build_objective(
        self, rates: cp.Variable, infrastructure: InfrastructureInfo, **kwargs
    ):
        def _merge_dicts(*args):
            """ Merge two dictionaries where d2 override d1 when there is a conflict. """
            merged = dict()
            for d in args:
                merged.update(d)
            return merged

        obj = cp.Constant(0)
        for component in self.objective_configuration:
            obj += component.coefficient * component.function(
                rates,
                infrastructure,
                self.interface,
                **_merge_dicts(kwargs, component.kwargs),
            )
        return obj

    def build_problem(
        self,
        active_sessions: List[SessionInfo],
        infrastructure: InfrastructureInfo,
        peak_limit: Optional[Union[float, List[float], np.ndarray]] = None,
        prev_peak: float = 0,
    ):
        """Build parts of the optimization problem including variables, constraints, and objective function.

        Args:
            active_sessions (List[SessionInfo]): List of SessionInfo objects for all active charging sessions.
            infrastructure (InfrastructureInfo): InfrastructureInfo object describing the electrical infrastructure at
                a site.
            peak_limit (Union[float, List[float], np.ndarray]): Limit on aggregate peak current. If None, no limit is
                enforced.
            prev_peak (float): Previous peak current draw during the current billing period.

        Returns:
            Dict[str: object]:
                'objective' : cvxpy expression for the objective of the optimization problem
                'constraints': list of all constraints for the optimization problem
                'variables': dict mapping variable name to cvxpy Variable.
        """
        optimization_horizon = max(
            s.arrival_offset + s.remaining_time for s in active_sessions
        )
        num_evses = len(infrastructure.station_ids)
        rates = cp.Variable(shape=(num_evses, optimization_horizon))
        constraints = {}

        # Rate constraints
        constraints.update(
            self.charging_rate_bounds(
                rates, active_sessions, infrastructure.station_ids
            )
        )

        # Energy Delivered Constraints
        constraints.update(
            self.energy_constraints(
                rates,
                active_sessions,
                infrastructure,
                self.interface.period,
                self.enforce_energy_equality,
            )
        )

        # Infrastructure Constraints
        constraints.update(
            self.infrastructure_constraints(rates, infrastructure, self.constraint_type)
        )

        # Peak Limit
        constraints.update(self.peak_constraint(rates, peak_limit))

        # Objective Function
        objective = cp.Maximize(
            self.build_objective(rates, infrastructure, prev_peak=prev_peak)
        )
        return {
            "objective": objective,
            "constraints": constraints,
            "variables": {"rates": rates},
        }

    def solve(
        self,
        active_sessions: List[SessionInfo],
        infrastructure: InfrastructureInfo,
        peak_limit: Union[float, List[float], np.ndarray] = None,
        prev_peak=0,
        verbose: bool = False,
    ):
        """Solve optimization problem to create a schedule of charging rates.

        Args:
            active_sessions (List[SessionInfo]): List of SessionInfo objects for all active charging sessions.
            infrastructure (InfrastructureInfo): InfrastructureInfo object describing the electrical infrastructure at
                a site.
            peak_limit (Union[float, List[float], np.ndarray]): Limit on aggregate peak current. If None, no limit is
                enforced.
            verbose (bool): See cp.Problem.solve()

        Returns:
            np.Array: Numpy array of charging rates of shape (N, T) where N is the number of EVSEs in the network and
                T is the length of the optimization horizon. Rows are ordered according to the order of evse_index in
                infrastructure.
        """
        # Here we take in arguments which describe the problem and build a problem instance.
        if len(active_sessions) == 0:
            return np.zeros((infrastructure.num_stations, 1))
        problem_dict = self.build_problem(
            active_sessions, infrastructure, peak_limit, prev_peak
        )
        prob = cp.Problem(
            problem_dict["objective"], list(problem_dict["constraints"].values())
        )
        prob.solve(solver=self.solver, verbose=verbose)
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise InfeasibilityException(f"Solve failed with status {prob.status}")
        return problem_dict["variables"]["rates"].value


# ---------------------------------------------------------------------------------
#  Objective Functions
#
#
#  All objectives should take rates as their first positional argument.
#  All other arguments should be passed as keyword arguments.
#  All functions should except **kwargs as their last argument to avoid errors
#  when unknown arguments are passed.
#
# ---------------------------------------------------------------------------------


def charging_power(rates, infrastructure, **kwargs):
    """ Returns a matrix with the same shape as rates but with units kW instead of A. """
    voltage_matrix = np.tile(infrastructure.voltages, (rates.shape[1], 1)).T
    return cp.multiply(rates, voltage_matrix) / 1e3


def aggregate_power(rates, infrastructure, **kwargs):
    """ Returns aggregate charging power for each time period. """
    return cp.sum(charging_power(rates, infrastructure=infrastructure), axis=0)


def get_period_energy(rates, infrastructure, period, **kwargs):
    """ Return energy delivered in kWh during each time period and each session. """
    power = charging_power(rates, infrastructure=infrastructure)
    period_in_hours = period / 60
    return power * period_in_hours


def aggregate_period_energy(rates, infrastructure, interface, **kwargs):
    """ Returns the aggregate energy delivered in kWh during each time period. """
    # get charging rates in kWh per period
    energy_per_period = get_period_energy(
        rates, infrastructure=infrastructure, period=interface.period
    )
    return cp.sum(energy_per_period, axis=0)


def quick_charge(rates, infrastructure, interface, **kwargs):
    optimization_horizon = rates.shape[1]
    c = np.array(
        [
            (optimization_horizon - t) / optimization_horizon
            for t in range(optimization_horizon)
        ]
    )
    return c @ cp.sum(rates, axis=0)


def equal_share(rates, infrastructure, interface, **kwargs):
    return -cp.sum_squares(rates)


def tou_energy_cost(rates, infrastructure, interface, **kwargs):
    current_prices = interface.get_prices(rates.shape[1])  # $/kWh
    return -current_prices @ aggregate_period_energy(rates, infrastructure, interface)


def total_energy(rates, infrastructure, interface, **kwargs):
    return cp.sum(get_period_energy(rates, infrastructure, interface.period))


def peak(rates, infrastructure, interface, baseline_peak=0, **kwargs):
    agg_power = aggregate_power(rates, infrastructure)
    max_power = cp.max(agg_power)
    prev_peak = interface.get_prev_peak() * infrastructure.voltages[0] / 1000
    if baseline_peak > 0:
        return cp.maximum(max_power, baseline_peak, prev_peak)
    else:
        return cp.maximum(max_power, prev_peak)


def demand_charge(rates, infrastructure, interface, baseline_peak=0, **kwargs):
    p = peak(rates, infrastructure, interface, baseline_peak, **kwargs)
    dc = interface.get_demand_charge()
    return -dc * p


def load_flattening(rates, infrastructure, interface, external_signal=None, **kwargs):
    if external_signal is None:
        external_signal = np.zeros(rates.shape[1])
    aggregate_rates_kW = aggregate_power(rates, infrastructure)
    total_aggregate = aggregate_rates_kW + external_signal
    return -cp.sum_squares(total_aggregate)


# def smoothing(rates, active_sessions, infrastructure, previous_rates, normp=1, *args, **kwargs):
#     reg = -cp.norm(cp.diff(rates, axis=1), p=normp)
#     prev_mask = np.logical_not(np.isnan(previous_rates))
#     if np.any(prev_mask):
#         reg -= cp.norm(rates[0, prev_mask] - previous_rates[prev_mask], p=normp)
#     return reg
