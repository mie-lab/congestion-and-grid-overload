import matplotlib.pyplot as plt
import numpy as np
import shared_config
max_len = 300  # 240
with open("python_scripts/optimise_charging_schedule/overall_power_side_results_total_power_demand_unmet_at_now.csv") as f:
    df = {}
    scenario_name_set = []
    for row in f:
        listed = row.strip().split(',')
        id = listed[0]
        _,scenario_name, scenario_class, transformer_capacity, algo, run_num = [x.strip().replace("'","").replace("\"","").replace("(","").replace(")", "") for x in listed[:6]]
        unmet_power_values = []
        for x in listed[6:]:
            unmet_power_values.append(float(x))
        pending_length = max_len - len(unmet_power_values)
        unmet_power_values.extend([0] * pending_length)
        if (scenario_name, scenario_class, transformer_capacity, algo) in df:
            df[(scenario_name, scenario_class, transformer_capacity, algo)].append(unmet_power_values)
        else:
            df[(scenario_name, scenario_class, transformer_capacity, algo)] = [unmet_power_values]

        scenario_name_set.append(scenario_name)

    color = {
        "Offline":"tab:green",
        "RR": "tab:blue",
        "Unctrl": "tab:red",
        "ASA":"tab:orange"
    }

    thickness = {
        "Offline": 3,
        "RR": 1,
        "Unctrl": 1,
        "ASA": 2
    }
    scenario_name_set = list(set(scenario_name_set))
    scenario_name_dict = dict(zip(scenario_name_set, ["Scenario " + str(x) for x in range(1, len(scenario_name_set) + 1)]))

    for key in df:
        if "Offline" in str(key):
            color_chosen = color["Offline"]
            thickness_chosen = thickness["Offline"]
        if "RR" in str(key):
            color_chosen = color["RR"]
            thickness_chosen = thickness["RR"]
        if "Unctrl" in str(key):
            color_chosen = color["Unctrl"]
            thickness_chosen = thickness["Unctrl"]
        if "ASA" in str(key):
            color_chosen = color["ASA"]
            thickness_chosen = thickness["ASA"]

        if "no-accident" in str(key):
            plt.plot(np.array(df[key]).sum(axis=0), label=scenario_name_dict[key[0]] + " - " + str(key[3]), linestyle="--", color=color_chosen, alpha=0.4, linewidth = thickness_chosen)
        else:
            plt.plot(range(shared_config.CUTOFF_TIME_FOR_RESULTS_MIXING, max_len), np.array(df[key]).sum(axis=0)[shared_config.CUTOFF_TIME_FOR_RESULTS_MIXING:], label=scenario_name_dict[key[0]] + " - " + str(key[3]), linestyle="-", color=color_chosen, linewidth = thickness_chosen, alpha=0.4)

    plt.xlabel(f"Time Steps 5 minutes (max_len = {max_len})")
    plt.ylabel("Unmet power demand (%)")
    plt.legend(loc="upper left", fontsize=8, ncol=2)
    l = len(str(scenario_name_dict))
    plt.title(str(scenario_name_dict)[:l//2] + "\n" + str(scenario_name_dict)[l//2:])
    plt.ylim(0, 60)
    plt.plot([shared_config.CUTOFF_TIME_FOR_RESULTS_MIXING] * 100, np.arange(0, 30, (30-0)/100),
             linewidth=0.7, alpha=0.5)
    # plt.xlim(90, 120)
    # plt.tight_layout()
    # plt.gca().set_aspect(20.2)
    plt.show()



plt.clf()





with open("python_scripts/optimise_charging_schedule/overall_power_side_results_total_EV_count_at_now.csv") as f:
    df = {}
    scenario_name_set = []
    for row in f:
        listed = row.strip().split(',')
        id = listed[0]
        _,scenario_name, scenario_class, transformer_capacity, algo, run_num = [x.strip().replace("'","").replace("\"","").replace("(","").replace(")", "") for x in listed[:6]]
        unmet_power_values = []
        for x in listed[6:]:
            unmet_power_values.append(float(x))
        pending_length = max_len - len(unmet_power_values)
        unmet_power_values.extend([0] * pending_length)
        if (scenario_name, scenario_class, transformer_capacity, algo) in df:
            df[(scenario_name, scenario_class, transformer_capacity, algo)].append(unmet_power_values)
        else:
            df[(scenario_name, scenario_class, transformer_capacity, algo)] = [unmet_power_values]

        scenario_name_set.append(scenario_name)


    scenario_name_set = list(set(scenario_name_set))
    scenario_name_dict = dict(zip(scenario_name_set, ["Scenario " + str(x) for x in range(1, len(scenario_name_set) + 1)]))

    for key in df:
        if "Offline" in str(key):
            color_chosen = color["Offline"]
        if "RR" in str(key):
            color_chosen = color["RR"]
        if "Unctrl" in str(key):
            color_chosen = color["Unctrl"]
        if "ASA" in str(key):
            color_chosen = color["ASA"]

        if "no-accident" in str(key):
            plt.plot(np.array(df[key]).sum(axis=0), label=scenario_name_dict[key[0]] + " - " + str(key[3]), linestyle="--", color=color_chosen)
        else:
            # plt.plot(range(shared_config.CUTOFF_TIME_FOR_RESULTS_MIXING, max_len), np.array(df[key]).sum(axis=0)[shared_config.CUTOFF_TIME_FOR_RESULTS_MIXING:], label=scenario_name_dict[key[0]] + " - " + str(key[3]), linestyle="-", color=color_chosen)
            plt.plot(range(0, max_len), np.array(df[key]).sum(axis=0)[0:], label=scenario_name_dict[key[0]] + " - " + str(key[3]), linestyle="-", color=color_chosen)

    plt.xlabel(f"Time Steps 5 minutes (max_len = {max_len})")
    plt.ylabel("Total active EVs count ")
    plt.legend(loc="upper left", fontsize=8, ncol=2)
    l = len(str(scenario_name_dict))
    plt.title(str(scenario_name_dict)[:l//2] + "\n" + str(scenario_name_dict)[l//2:])
    plt.ylim(0, 60)
    plt.plot([shared_config.CUTOFF_TIME_FOR_RESULTS_MIXING] * 100, np.arange(0, 30, (30-0)/100),
             linewidth=0.7, alpha=0.5)
    # plt.xlim(105, 112)
    # plt.tight_layout()
    # plt.gca().set_aspect(20.2)
    plt.show()




with open("python_scripts/optimise_charging_schedule/overall_power_side_results_total_energy_demand_unmet_by_now.csv") as f:
    df = {}
    scenario_name_set = []
    for row in f:
        listed = row.strip().split(',')
        id = listed[0]
        _,scenario_name, scenario_class, transformer_capacity, algo, run_num = [x.strip().replace("'","").replace("\"","").replace("(","").replace(")", "") for x in listed[:6]]
        unmet_power_values = []
        for x in listed[6:]:
            unmet_power_values.append(float(x))
        pending_length = max_len - len(unmet_power_values)
        unmet_power_values.extend([0] * pending_length)
        if (scenario_name, scenario_class, transformer_capacity, algo) in df:
            df[(scenario_name, scenario_class, transformer_capacity, algo)].append(unmet_power_values)
        else:
            df[(scenario_name, scenario_class, transformer_capacity, algo)] = [unmet_power_values]

        scenario_name_set.append(scenario_name)


    scenario_name_set = list(set(scenario_name_set))
    scenario_name_dict = dict(zip(scenario_name_set, ["Scenario " + str(x) for x in range(1, len(scenario_name_set) + 1)]))

    for key in df:
        if "Offline" in str(key):
            color_chosen = color["Offline"]
        if "RR" in str(key):
            color_chosen = color["RR"]
        if "Unctrl" in str(key):
            color_chosen = color["Unctrl"]
        if "ASA" in str(key):
            color_chosen = color["ASA"]


        if "no-accident" in str(key):
            plt.plot(np.array(df[key]).sum(axis=0), label=scenario_name_dict[key[0]] + " - " + str(key[3]), linestyle="--", color=color_chosen)
        else:
            # plt.plot(range(shared_config.CUTOFF_TIME_FOR_RESULTS_MIXING, max_len), np.array(df[key]).sum(axis=0)[shared_config.CUTOFF_TIME_FOR_RESULTS_MIXING:], label=scenario_name_dict[key[0]] + " - " + str(key[3]), linestyle="-", color=color_chosen)
            plt.plot(range(0, max_len), np.array(df[key]).sum(axis=0)[0:], label=scenario_name_dict[key[0]] + " - " + str(key[3]), linestyle="-", color=color_chosen)

    plt.xlabel(f"Time Steps 5 minutes (max_len = {max_len})")
    plt.ylabel("Total power demand unmet as of now (%)")
    plt.legend(loc="upper left", fontsize=8, ncol=2)
    l = len(str(scenario_name_dict))
    plt.title(str(scenario_name_dict)[:l//2] + "\n" + str(scenario_name_dict)[l//2:])
    plt.ylim(0, 60)
    plt.plot([shared_config.CUTOFF_TIME_FOR_RESULTS_MIXING] * 100, np.arange(0, 30, (30-0)/100),
             linewidth=0.7, alpha=0.5)
    # plt.xlim(105, 112)
    # plt.tight_layout()
    # plt.gca().set_aspect(20.2)
    plt.show()