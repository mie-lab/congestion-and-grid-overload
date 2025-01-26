accidenttruestarttime=120   # 7:30AM = 90, 800am = 96, 10am = 102, , 10am = 108m,  10am= 114,  10:00am = 120, 1030am=126
accidenttruestarttime15mins=123
accidenttruestarttime30mins=126
accidenttruestarttime45mins=129

sed -i '' "s/TRUE_ACCIDENT_START_TIME.*/TRUE_ACCIDENT_START_TIME = $accidenttruestarttime/" shared_config.py

for accidentcapacity in 1; do # 5 10 20 40 $(seq 1510 20 1590); do
  for capacity in $(seq 400 300 1600); do #  $(seq 10 40 220); do #  $(seq 30 75 1000) ;do # $(seq 25 50 1000);do # 1500 2000 2500 3000 3500 ;do  #850 900 1000 1101 1201 1301 ; do # $(seq 25 10 100) ; do # $(seq 1400 100 3000) #$(seq 1000 100 2000); do
      echo "Processing capacity: $capacity"
      rm -rf python_scripts/optimise_charging_schedule/images_from_acnsim
      mkdir python_scripts/optimise_charging_schedule/images_from_acnsim
      rm -rf python_scripts/optimise_charging_schedule/all_evs_charging_record_*
      rm -rf python_scripts/optimise_charging_schedule/Offline_optimal_failure.*

      sed -i '' "s/CUTOFF_TIME_FOR_RESULTS_MIXING.*/CUTOFF_TIME_FOR_RESULTS_MIXING = -1/" shared_config.py
      sed -i '' "s/capacity_list.*/capacity_list = [$capacity]/" shared_config.py
      pwd
      echo "path: above"
      rm python_scripts/optimise_charging_schedule/events_dict_across_traffic_scenarios.pickle
      rm python_scripts/optimise_charging_schedule/*.png
      rm -rf python_scripts/optimise_charging_schedule/results*
      rm -rf python_scripts/optimise_charging_schedule/overall_power_side_results*.csv
      rm python_scripts/optimise_charging_schedule/events_dict_across_traffic_scenarios.pickle
      python python_scripts/optimise_charging_schedule/compare_without_dollars_organised.py --scenario no-accident --inputfilename arrivals_for_optimiser-r3-set-b-no-accident.csv

      sed -i '' "s/CUTOFF_TIME_FOR_RESULTS_MIXING.*/CUTOFF_TIME_FOR_RESULTS_MIXING = -1/" shared_config.py
      sed -i '' "s/capacity_list.*/capacity_list = [$capacity]/" shared_config.py
      echo "path: above"
      rm python_scripts/optimise_charging_schedule/events_dict_across_traffic_scenarios.pickle
      rm python_scripts/optimise_charging_schedule/*.png
      rm -rf python_scripts/optimise_charging_schedule/results*
      python python_scripts/optimise_charging_schedule/compare_without_dollars_organised.py --scenario 45-mins-accident-${accidentcapacity}-capacity-remaining-start-10am --inputfilename arrivals_for_optimiser-r3-set-b-45-mins-accident-${accidentcapacity}-capacity-remaining-start-10am.csv
      #
      #
      sed -i '' "s/CUTOFF_TIME_FOR_RESULTS_MIXING.*/CUTOFF_TIME_FOR_RESULTS_MIXING = 1/" shared_config.py
      sed -i '' "s/capacity_list.*/capacity_list = [$capacity]/" shared_config.py
      pwd
      echo "path: above"
      rm python_scripts/optimise_charging_schedule/events_dict_across_traffic_scenarios.pickle
      rm python_scripts/optimise_charging_schedule/*.png
      rm -rf python_scripts/optimise_charging_schedule/results*
      rm python_scripts/optimise_charging_schedule/events_dict_across_traffic_scenarios.pickle
      python python_scripts/optimise_charging_schedule/compare_without_dollars_organised.py --scenario no-accident --inputfilename arrivals_for_optimiser-r3-set-b-no-accident.csv

      sed -i '' "s/CUTOFF_TIME_FOR_RESULTS_MIXING.*/CUTOFF_TIME_FOR_RESULTS_MIXING = $accidenttruestarttime/" shared_config.py
      sed -i '' "s/capacity_list.*/capacity_list = [$capacity]/" shared_config.py
      echo "path: above"
      rm python_scripts/optimise_charging_schedule/events_dict_across_traffic_scenarios.pickle
      rm python_scripts/optimise_charging_schedule/*.png
      rm -rf python_scripts/optimise_charging_schedule/results*
      python python_scripts/optimise_charging_schedule/compare_without_dollars_organised.py --scenario 45-mins-accident-${accidentcapacity}-capacity-remaining-start-10am --inputfilename arrivals_for_optimiser-r3-set-b-45-mins-accident-${accidentcapacity}-capacity-remaining-start-10am.csv

      sed -i '' "s/CUTOFF_TIME_FOR_RESULTS_MIXING.*/CUTOFF_TIME_FOR_RESULTS_MIXING = $accidenttruestarttime15mins/" shared_config.py
      sed -i '' "s/capacity_list.*/capacity_list = [$capacity]/" shared_config.py
      echo "path: above"
      rm python_scripts/optimise_charging_schedule/events_dict_across_traffic_scenarios.pickle
      rm python_scripts/optimise_charging_schedule/*.png
      rm -rf python_scripts/optimise_charging_schedule/results*
      python python_scripts/optimise_charging_schedule/compare_without_dollars_organised.py --scenario 45-mins-accident-${accidentcapacity}-capacity-remaining-start-10am --inputfilename arrivals_for_optimiser-r3-set-b-45-mins-accident-${accidentcapacity}-capacity-remaining-start-10am.csv

      sed -i '' "s/CUTOFF_TIME_FOR_RESULTS_MIXING.*/CUTOFF_TIME_FOR_RESULTS_MIXING = $accidenttruestarttime30mins/" shared_config.py
      sed -i '' "s/capacity_list.*/capacity_list = [$capacity]/" shared_config.py
      echo "path: above"
      rm python_scripts/optimise_charging_schedule/events_dict_across_traffic_scenarios.pickle
      rm python_scripts/optimise_charging_schedule/*.png
      rm -rf python_scripts/optimise_charging_schedule/results*
      python python_scripts/optimise_charging_schedule/compare_without_dollars_organised.py --scenario 45-mins-accident-${accidentcapacity}-capacity-remaining-start-10am --inputfilename arrivals_for_optimiser-r3-set-b-45-mins-accident-${accidentcapacity}-capacity-remaining-start-10am.csv

      sed -i '' "s/CUTOFF_TIME_FOR_RESULTS_MIXING.*/CUTOFF_TIME_FOR_RESULTS_MIXING = $accidenttruestarttime45mins/" shared_config.py
      sed -i '' "s/capacity_list.*/capacity_list = [$capacity]/" shared_config.py
      echo "path: above"
      rm python_scripts/optimise_charging_schedule/events_dict_across_traffic_scenarios.pickle
      rm python_scripts/optimise_charging_schedule/*.png
      rm -rf python_scripts/optimise_charging_schedule/results*
      python python_scripts/optimise_charging_schedule/compare_without_dollars_organised.py --scenario 45-mins-accident-${accidentcapacity}-capacity-remaining-start-10am --inputfilename arrivals_for_optimiser-r3-set-b-45-mins-accident-${accidentcapacity}-capacity-remaining-start-10am.csv

      sed -i '' "s/CUTOFF_TIME_FOR_RESULTS_MIXING.*/CUTOFF_TIME_FOR_RESULTS_MIXING = 1/" shared_config.py
      sed -i '' "s/capacity_list.*/capacity_list = [$capacity]/" shared_config.py
      pwd
      echo "path: above"
      rm python_scripts/optimise_charging_schedule/events_dict_across_traffic_scenarios.pickle
      rm python_scripts/optimise_charging_schedule/*.png
      rm -rf python_scripts/optimise_charging_schedule/results*
      rm python_scripts/optimise_charging_schedule/events_dict_across_traffic_scenarios.pickle
      python python_scripts/optimise_charging_schedule/compare_without_dollars_organised.py --scenario 45-mins-accident-${accidentcapacity}-capacity-remaining-start-10am --inputfilename arrivals_for_optimiser-r3-set-b-45-mins-accident-${accidentcapacity}-capacity-remaining-start-10am.csv

      capacityprint=$(printf "%05d" "$capacity")
      mkdir Info_time_duckbufferremoved-30kWh-166kW_${capacityprint}_45-mins-accident-${accidentcapacity}-capacity-remaining-start-10am
      mv python_scripts/optimise_charging_schedule/overall_power_side_results*.csv Info_time_duckbufferremoved-30kWh-166kW_${capacityprint}_45-mins-accident-${accidentcapacity}-capacity-remaining-start-10am/
      mv python_scripts/optimise_charging_schedule/images_from_acnsim Info_time_duckbufferremoved-30kWh-166kW_${capacityprint}_45-mins-accident-${accidentcapacity}-capacity-remaining-start-10am/
  done
done