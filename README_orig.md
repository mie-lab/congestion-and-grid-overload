# congestion-and-grid-overload

plot flow in colors is incomplete right now.
Pointers to discuss with Yi Wang:
 
- What about connecting the lanes across the segments? 
- When we output outflow, are we actually interested in the outflow? or is it the inflow into the sink cells? Is this just an abuse of terms? 
- Why do we have 6 vaues for Dem(8000) while the input.txt says 7 for demand lines?
- Two types of Δ𝑡 (One for the cell division;  one for the simualtion)
- Why are lanes in floating points? I see lanes numbers like 1.75 etc..




- Must fix the correct name of 

To run:

[//]: # (- Copy the folder `acnportal` from the official repository at commit 3c76892d78ae7cbdca9017f8e2a4e3114198deba)

[//]: # (- Copy the folder `adacharge` from the official repository at commit b7d5fddb25e842333fc2b404d32dd3477ca47297)


Run the following from inside `python_scripts/optimise_charging_schedule`
```bash

# pip install acnportal==0.3.2        
pip install git+https://github.com/zach401/acnportal.git@3c76892d78ae7cbdca9017f8e2a4e3114198deba && 
pip install git+https://github.com/caltech-netlab/adacharge.git@b7d5fddb25e842333fc2b404d32dd3477ca47297


```
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

- prefix = "r3" implies the arrival optimiser files will be called `arrivals_for_optimiser-r2-45mins-ac.....`
- "r2" implies single accident down South
- "r3-set-a" implies accidents at [4485, 4470, 4273]
- "r3-set-b" implies accidents at [4485, 465, 2183]
- "r1" implies single accident at 4485: down south
- "r5" implies all 5 locations immediately after exiting highway.
- 6400 in fortran code is the maximum number of cells that this code can handle.
- Only one type of accident (w.r.t space and duration) can be handled by one shell script
- Voltage is hardcoded to 208 at some places; (if changing, need to check for places where simulator.py or init.py was changed and the plotting codes)
- The whole pipeline is tested on python3.8 
- energy_cost_1 is garbage for no-accident, -1 case. <already taken care of since the C matrix inside aggregate_power inside __init__.py will not be updated. Implies no scaling for such cases>
- sometimes the worst case (i.e. info @ 45 mins) might result in a crash during EV simulation since we might have extra vehiles due to day transfer. Currently, we have just two options when this happesn - 1) Ignore the 45-mins analyses for this case 2) Put all EVs in one EVSE infra

For the `pip install` for `adacharge`, it is safer to use the same commit id as b7d5fddb25e842333fc2b404d32dd3477ca47297 


To-do:   
 * ~~combine multiple cells to plot aggregated arrivals timings~~
 
 
 - ~~Generate documentation 
 Run this at the home folder; 
 rm -rf documentation/* && pdoc --output-dir documentation ./python_scripts/*~~
The timeofdday impact plots need to be run manually one over the other, otherwise we need a script to automate it.

 - Run once
 - Plot Kepler
 - Get cell numbers for OR, Accidents and DE
 - Run again


- What is the point of having accident_file_name = "arrivals_for_optimiser-r3-set-b-45-mins-accident-1-capacity-remaining-start-730am.csv"
no_accident_file_name = "arrivals_for_optimiser-r3-set-b-no-accident.csv" in the shared_config.py?
- Of this is the case, what is the point of passing the filenames through the shell script as python parameters?



Running `compare_without_dollars_organised.py` results in output files being generated inside the python_scripts/ optimise folder
Running the `main_driver.sh` however takes care of this and moves these files to the home folder.
We must rename the folders to have the same number of digits in the tranformer capacity
Otherwise the plots will be inverted if we use plt.plot (plt.scatter doesn't depend on the ordering as expected)

Runnig the pipeline only results in the arrivals for optimiser files
thereafter the main shell script needs to be run.


For this error:
```python
  File "python_scripts/optimise_charging_schedule/compare_without_dollars_organised.py", line 780, in calc_metrics
    _, accident_arrival = total_num_EVS_from_CTM_file_accident_arrivals(fname=shared_config.accident_file_name)
  File "python_scripts/optimise_charging_schedule/compare_without_dollars_organised.py", line 700, in total_num_EVS_from_CTM_file_accident_arrivals
    arrivals = [round(x/frac) for x in arrivals]
TypeError: 'float' object is not iterable
path: above

```
Please check the filename in the `input_output_text_files`
