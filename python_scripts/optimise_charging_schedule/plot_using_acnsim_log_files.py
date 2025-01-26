# ('45-mins-accident-1-cap', 'II', 100, 'ASA', 1)	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.9676579203109820	0.9353158406219630	0.9029737609329450	1.838289601577240	3.7089212829022900	3.5795529642480000	3.4501846454919200	3.320816326735940	4.159105928295530	4.965053450178290	5.738658892456460	5.512264334676140	5.285869777114500	5.059475219754220	4.833080661931100	5.589833973535990	5.346587285140920	5.103340596837240	5.8600939084441200	5.616847220049120	5.373600531654180	6.130353843259140	8.887107154864130	8.643860466469100	9.400613778073970	10.157367089679300	10.91412040128490	10.670873712890700	13.427627024499400	13.184380336137300	14.941133647791100	15.697886959396000	16.454640271019800	17.211393582624700	16.96814689422960	16.724900205834400	16.48165351743930	16.238406829044100	15.995160140649000	15.751913452253800	15.508666763858700	15.265420075552300	15.022173387209700	14.778926698862100	14.535680010467000	14.29243332207180	14.0491866336767	13.805939945281600	14.56269325689450	15.31944656852070	16.052346799364100	15.78524703030280	16.495105179409000	17.182744579320300	16.87038397962810	19.547859403749500	21.22435902642080	20.900880087693400	22.576901652928800	23.252807906437300	23.92865189131770	23.604496460422300	23.28034164132260	22.95618745593610	22.632033927211900	22.307881086878500	22.983702121574300	23.659523156731100	23.33534419176760	24.946481066920400	24.55761794206940	25.136412737650400	26.65052337393950	28.16425492568680	27.67798647742400	27.191718029157800	27.705449581571200	27.21918113402660	27.732912686173300	27.24664423829480	27.7603757900171	27.27410734176640	26.787838893528900	28.301570445234800	27.81530199694040	27.329033548645900	26.842765100351300	28.356496652059600	27.870228203768200	27.3839597554769	26.89769130745020	26.411422859472600	25.925154411537900	25.438885963614900	24.95261751569630	24.466349067728300	23.980080619704200	23.49381217164960	23.007543723556200	22.521275275420600	22.035006827256600	21.548738379033600	21.062469930822100	20.57620148260440	20.08993303437650	19.603664586137700	19.117396137873200	18.631127689594300	18.144859241308500	17.658590793138000	17.172322345024400	16.68605389695790	16.199785448872200	15.713517000723500	15.227248552434400	14.740980104176700	14.254711655945500	13.768443208120700	13.282174760537100	12.795906312240700	12.309637863958700	11.82336941566720	11.337100967373900	10.85083251907810	10.364564070785100	9.878295622528480	9.392027174238990	8.905758725951630	8.419490278412110	7.933221830144270	7.446953382103630	6.960684933919130	6.474416485680450	0.0

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from slugify import slugify

shape = []
with open ("overall_power_side_results_total_power_demand_unmet_at_now.csv") as f:
    for row in f:
        listed = row.strip().split(",")
        traffic_scenario_id, power_scenario_id, capacity, algo, run_counter = [x.strip().replace('\'','').replace("(","").replace("\"","").replace(")","") for x in listed[:5]]
        capacity = float(capacity); run_counter = int(run_counter)
        values = np.array([float(x) for x in listed[5:]])
        shape.append(len(values.tolist()))

shape = min(shape)
data_dict = {}
data_dict_counter = {}
with open ("overall_power_side_results_total_power_demand_unmet_at_now.csv") as f:
    for row in f:
        listed = row.strip().split(",")
        traffic_scenario_id, power_scenario_id, capacity, algo, run_counter = [x.strip().replace('\'','').replace("(","").replace("\"","").replace(")","") for x in listed[:5]]
        capacity = float(capacity); run_counter = int(run_counter)
        values = np.array([float(x) for x in listed[5:]])

        key = (traffic_scenario_id, power_scenario_id, capacity, algo)
        if key in data_dict:
            data_dict[key] += values[:shape]
            data_dict_counter[key] += 1
        else:
            data_dict[key] = values[:shape]
            data_dict_counter[key] = 1

for key in data_dict:
    plt.plot(data_dict[key]/ data_dict_counter[key], label=key)
plt.xlabel("Time steps (t)")
plt.ylabel("Unmet power demand percentage at time t")
plt.legend(fontsize=6)
plt.tight_layout()
plt.savefig("images_from_acnsim/power_demand_unmet.png", dpi=300)


list_of_scenarios = [
    "no-accident",
    '45-mins-accident-1-capacity-remaining-start-9am',
    "45-mins-accident-5-capacity-remaining-start-9am",
    "45-mins-accident-10-capacity-remaining-start-9am",
    "45-mins-accident-20-capacity-remaining-start-9am",
    "45-mins-accident-40-capacity-remaining-start-9am",
    "45-mins-accident-80-capacity-remaining-start-9am"
]
list_of_scenarios = [slugify(x) for x in list_of_scenarios]

for capacity in [100, 150, 200]:
    for algo in [ "RR", "Unctrl", "Offline"]: # "ASA",
        plt.clf()
        for key in data_dict:
            if key[3] == algo and key[2] == capacity and key[0] in list_of_scenarios:
                plt.plot(data_dict[key]/data_dict_counter[key], label=key)
        plt.xlabel("Time steps (t)")
        plt.ylabel("Unmet power demand percentage at time t")
        plt.legend(fontsize=6)
        plt.tight_layout()
        plt.ylim(0, 27)
        plt.savefig("images_from_acnsim/"+ algo +"_power_demand_unmet" + str(capacity) + ".png", dpi=300)




