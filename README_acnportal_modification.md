
- Copy the folder `acnportal` from the official repository at commit 3c76892d78ae7cbdca9017f8e2a4e3114198deba
- Depending on the installation folder: Inside `/acnportal/acnportal/signals/tariffs/tariff_schedules/` or inside `~/opt/anaconda3/envs/congestion-and-grid-overload/lib/python3.8/site-packages/acnportal/signals/tariffs/tariff_schedules/` create cost profile json files as needed. The names of cost profile file can be found inside `compare_without_dollars.py` in variable `tariff_name`
- Depending on the installation folder: Modify the `simulator.py` inside `~/opt/anaconda3/envs/congestion-and-grid-overload/lib/python3.8/site-packages/acnportal/acnsim/simulator.py` with the version inside `optimise_charging_schedule/simulator.py`
- Modify the maximum allowable current for BASIC EVSE to 80 Amps `~/opt/anaconda3/envs/congestion-and-grid-overload/lib/python3.8/site-packages/acnportal/acnsim/models/evse.py`
- Depending on the installation folder: Modify the `__init__.py` inside `/Users/nishant/opt/anaconda3/envs/congestion-and-grid-overload/lib/python3.8/site-packages/acnportal/acnsim/analysis/__init__.py` to replace the function:
- ` for CUTOFF_TIME_FOR_RESULTS_MIXING_value in [-1, 1, 120]:` must have the correct value accoriding to the driver script which has the shell command like this: `sed -i '' "s/CUTOFF_TIME_FOR_RESULTS_MIXING.*/CUTOFF_TIME_FOR_RESULTS_MIXING = 120/" shared_config.py;` simarly the `CUTOFF_TIME_for_no_accident_day_pricing_computation` should be changed in `shared_config.py` 

```python

def energy_cost(sim, tariff=None, cap_remaining=None, cutOfftime=None):
    """ Calculate the total energy cost of the simulation.

    Args:
        sim (Simulator): A Simulator object which has been run.
        tariff (TimeOfUseTariff): Tariff structure to use when calculating energy costs.
        cutOfftime: sets the cost of the remaining day to 0
        cap_remaining: list of ratio of ratio of arrivals for accident day vs. non-accident day
    Returns:
        float: Total energy cost of the simulation ($)

    """
    if tariff is None:
        if "tariff" in sim.signals:
            tariff = sim.signals["tariff"]
        else:
            raise ValueError("No pricing method is specified.")
    agg = aggregate_power(sim, cap_remaining=cap_remaining, cutOfftime=cutOfftime)
    energy_costs = tariff.get_tariffs(sim.start, len(agg), sim.period)

```

```python

def aggregate_power(sim, cap_remaining=None, cutOfftime=None):
    """ Calculate the time series of aggregate power of all EVSEs within a simulation.

    Args:
        sim (Simulator): A Simulator object which has been run.

    Returns:
        np.Array: A numpy ndarray of the aggregate power at each time. [kW]
    """
    agg =  sim.network._voltages.T.dot(sim.charging_rates) / 1000

    if cutOfftime != None and set(cap_remaining) != 1: # so that these extra steps are done only for a single case <no-accident, -1>
        L = sim.ev_history.__len__()
        ev_history_list = list(sim.ev_history.values())
        ev_arrival_list_ordered = []
        C = np.array(sim.charging_rates)
        for i in range(L-1): # This loop is only for testing monotonicity
            ev = ev_history_list[i]
            ev_next = ev_history_list[i+1]
            assert ev.arrival == ev._arrival
            assert ev.arrival <= ev_next.arrival

        for i in range(L):
            ev_arrival_list_ordered.append(ev.arrival)
            if ev.arrival < cutOfftime:
                C[i] *= np.array(cap_remaining[ev.arrival])

        agg = sim.network._voltages.T.dot(C) / 1000
        agg[cutOfftime:] = 0

    return agg
```