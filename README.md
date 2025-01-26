
# Congestion and Grid Overload

This repository contains the code accompanying our paper "Quantifying the impacts of non-recurrent congestion on workplace EV charging infrastructures". The codebase includes a Cell Transmission Model (CTM) implemented in Fortran and a Python pipeline that interfaces with the ACM simulator.

## Repository Contents

- **CTM Model**: Implemented in Fortran, dynamically generated from a configuration file to adapt to various traffic scenarios.
- **Python Pipeline**: Integrates with the ACM simulator to analyze the impact of EV charging on power grids. The ACM simulator can be found at [ACM Simulator](https://github.com/zach401/acnportal).
- **Traffic Volume Data**: Utilized from the Traffic Mapping Application provided by the Minnesota Department of Transportation, available [here](https://www.dot.state.mn.us/traffic/data/tma.html). The repository includes sample data for demonstration purposes.
- **Network Data**: Collected from OpenStreetMap (OSM), facilitating the realistic simulation of traffic scenarios.


If you find the code useful or refer to the concepts presented in the paper, please cite our work. The code developed for this study is publicly available on GitHub at [Congestion and Grid Overload](https://github.com/mie-lab/congestion-and-grid-overload).

## Installation and Configuration

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/mie-lab/congestion-and-grid-overload.git
   cd congestion-and-grid-overload
   ```

2. **Set Up the Environment**:
   - For Python dependencies:
     ```bash
     pip install -r requirements.txt
     ```

   - For Fortran, ensure a compatible compiler is available on your system.

3. **Install ACM Portal and AdaCharge**:
   ```bash
   pip install git+https://github.com/zach401/acnportal.git@3c76892d78ae7cbdca9017f8e2a4e3114198deba
   pip install git+https://github.com/caltech-netlab/adacharge.git@b7d5fddb25e842333fc2b404d32dd3477ca47297
   ```
We need to modify one function inside the acnportal to extract more logging information during the simulation. Please follow the instructions in README_acnportal_modification.md to install the acnportal properly.


```

## Usage

- **Configure the Model**:
  Modify the configuration file to suit your specific traffic scenario.
  
- **Run the Simulation**:
  Execute the main script to start the simulation:
  ```bash
  python pipeline.py
  ```


## License

This project is open-sourced under the MIT license. See the [LICENSE](LICENSE) file for more details.
```

This version of the README retains detailed setup and configuration instructions crucial for proper environment setup and modification of dependency packages as required by your simulation setup.
