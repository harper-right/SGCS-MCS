
# Stackelberg Game based quality Control System (SGCS)
This is the code accompanying the paper: "SGCS: A Cost-Effective Quality Control System for Strategic Workers in Mobile Crowd Sensing" by Han Wang, Anfeng Liu, and Neal N. Xiong, under review at IEEE Transactions on Network Science and Engineering.


## Description
This repository contains the implementation of a Stackelberg Game-based Quality Control System (SGCS) designed for Mobile Crowd Sensing (MCS). The system addresses the challenge of ensuring high-quality sensing data by modeling the regulatory efforts of the platform and their impact on workers' behavior.

### Key Features
- **Theoretical Analysis:** Establishes the minimum verification rate required to deter workers from submitting low-quality data by analyzing the game equilibrium.
- **Validation Studies:** Evaluates the robustness of the proposed scheme across various datasets and parameter variations, offering insights into cost-effective data quality control strategies for real-world MCS systems.

### Comparison Algorithms
- **ITD:** P. Wang, Z. Li, B. Guo, S. Long, S. Guo, and J. Cao present a UAV-assisted truth discovery approach with an incentive mechanism design in mobile crowd sensing, published in *IEEE/ACM Transactions on Networking*, vol. 32, no. 2, April 2024, pp. 1738-1752. [Read more](https://ieeexplore.ieee.org/document/10328727).

- **VIR-EA:** M. Huang, Z. Li, A. Liu, X. Zhang, Z. Yang, and M. Yang introduce a proactive trust evaluation system for secure data collection based on sequence extraction, early access in *IEEE Transactions on Dependable and Secure Computing*. [Read more](https://ieeexplore.ieee.org/document/10589359).

- **UITDE:** Z. Chen et al. describe a UAV-assisted intelligent true data evaluation method for ubiquitous IoT systems in intelligent transportation within a smart city, as seen in *IEEE Transactions on Intelligent Transportation Systems*, vol. 25, no. 8, August 2024, pp. 9597-9607. [Read more](https://ieeexplore.ieee.org/document/10475129).


## Dependencies
- Python == 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
### Installation
1. Clone repo
    ```bash
    git clone https://github.com/harper-right/SGCS-MCS.git
    cd SGCS-MCS
    ```
2. Create Virtual Environment
    ```
   conda create -n sgcs python==3.10
   conda activate sgcs
   ```
3. Install dependent packages
    ```
    pip install -r requirements.txt
    cd SGCS
    ```


## Experiments
To plot the distribution of tasks and workers in the experimental scenario
```bash
cd /source_data
python locs_demands_vis.py
```

To verify the theorems in Section V
```
python sec_iv_sg_thm1.py
python sec_iv_sg_thm2.py
```

To verify the discussion on worker-dependent verification in Section V
```
python sec_iv_worker_dep.py
```

To verify the theorems of the SGCS framework on real trajectory datasets
```
python sec_v_thm_p1.py
python sec_v_thm_workers.py
python sec_v_thm_alpha.py
```

To get Verification cost (H) vs. Verification rate, and Utility (U) vs. Verification rate for different baseline data collection algorithms
```
python sec_v_alpha_H_U.py
```

To compare our SGCS with strategies ITD, VIR-EA, and UITDE under different numbers of tasks
```
python sec_v_cmp_apps_n.py
```

To compare our SGCS with strategies ITD, VIR-EA, and UITDE under different numbers of workers
```
python sec_v_cmp_apps_m.py
```

To test the running time of the algorithms
```
python sec_v_time_n.py
```

## Contact
If you have any question, please email `hanwang@csu.edu.cn`.
