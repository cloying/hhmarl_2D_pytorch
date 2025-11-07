HHMARL 2D: A Pure PyTorch Implementation

This repository is a pure PyTorch conversion of the original HHMARL 2D project by IDSIA. The goal of this version is to provide a self-contained implementation that does not require the Ray RLlib framework, making it easier to study, modify, and integrate into other PyTorch-based projects.

Original Repository: https://github.com/IDSIA/hhmarl_2D
Heterogeneous Hierarchical Multi-Agent Reinforcement Learning for Air Combat Maneuvering

This is the implementation of the method proposed in the paper: Hierarchical Multi-Agent Reinforcement Learning for Air Combat Maneuvering.
<p align="center">
<img src="https://raw.githubusercontent.com/IDSIA/hhmarl_2D/main/img/hier_pol.png" width="250">
<img src="https://raw.githubusercontent.com/IDSIA/hhmarl_2D/main/img/fight_pol.png" width="250">
<img src="https://raw.githubusercontent.com/IDSIA/hhmarl_2D/main/img/esc_pol.png" width="250">
</p>
Overview

We use a two-level hierarchy to solve the air combat maneuvering task. Low-level policies are trained to execute basic "fight" or "escape" maneuvers against opponents. These pre-trained policies are then employed as skills by a high-level "commander" policy, which learns to make strategic decisions by selecting which skill to use for each agent.
<p align="center">
<img src="https://raw.githubusercontent.com/IDSIA/hhmarl_2D/main/img/policies.png" width="300">
</p>
Setup and Installation

This project is built with PyTorch and uses Gymnasium for the environment API.
1. Prerequisites

    Python 3.8+

    System libraries for pycairo and cartopy.

        On Debian/Ubuntu: sudo apt-get install -y libcairo2-dev libgeos-dev

        On macOS (Homebrew): brew install cairo geos

2. Create a Virtual Environment (Recommended)
code Bash

    
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

  

3. Install Dependencies

Install all required packages from the requirements.txt file.
code Bash

    
pip install -r requirements.txt

  

The requirements.txt file contains:
code Txt

    
# Core Deep Learning & Reinforcement Learning
torch>=2.0.0
gymnasium==0.26.3
numpy==1.24.3

# Logging and Utilities
tensorboard==2.13.0
tqdm
scipy

# Simulation, Geodesics, and Plotting
pycairo==1.23.0
cartopy>=0.21.0
matplotlib
geographiclib==2.0

  

Training Workflow

The training is a 3-step process. First, train the low-level policies. Second, train the high-level commander. Third, evaluate the final performance.
Step 1: Train Low-Level Policies (train_hetero.py)

This script trains the heterogeneous agents in "fight" or "escape" modes using a curriculum. The low-level policies must be pre-trained and stored before you can train the high-level commander.

Training Procedure:
The training uses a curriculum from level=1 to level=5.

    Train "Fight" Policies (Levels 1-4):
    Start with level=1 and agent_mode="fight". The script will automatically find and restore the previous level's checkpoint, so you can run the commands sequentially.
    code Bash

    
python train_hetero.py --level=1 --agent_mode="fight"
python train_hetero.py --level=2 --agent_mode="fight"
python train_hetero.py --level=3 --agent_mode="fight"
python train_hetero.py --level=4 --agent_mode="fight"

  

Train "Escape" Policy (Level 3):
This trains the basic escape maneuver.
code Bash

    
python train_hetero.py --level=3 --agent_mode="escape"

  

Train Advanced Policies (Level 5):
At this level, the agents learn to fight against opponents that may also try to escape.
code Bash

        
    python train_hetero.py --level=5 --agent_mode="fight"
    python train_hetero.py --level=5 --agent_mode="escape"  # Recommended

      

Upon completion, the trained policy models (.pth files) will be saved in the policies/ directory.
Step 2: Train High-Level Commander (train_hier.py)

This script trains the high-level commander policy. It automatically loads and uses the low-level policies you trained in Step 1.
code Bash

    
python train_hier.py

  

This script will save the final commander model (e.g., Commander_3_vs_3.pth) in the results/ directory.
Step 3: Evaluate the Models (evaluation.py)

Use this script to evaluate the performance of your trained agents.

    To evaluate the full hierarchical model (with commander):
    Set eval_hl=True (this is the default).
    code Bash

    
python evaluation.py --eval_hl=True

  

To evaluate only the low-level policies (without commander):
Set eval_hl=False. You can also specify which pre-trained levels to compare.
code Bash

        
    python evaluation.py --eval_hl=False --eval_level_ag=5 --eval_level_opp=4

      

Evaluation results, including metrics and scenario plots, will be saved in a folder named EVAL_* inside the results/ directory.
Key Concepts
Curriculum Learning

The low-level "fight" policy is trained across 5 levels of increasing difficulty. The opponent behavior for each level is as follows:

    L1: Static opponent.

    L2: Randomly moving opponent.

    L3: Scripted, deterministic opponent.

    L4: Opponent uses the policy trained at L3 (fictitious self-play).

    L5: Opponent uses a mix of L3, L4, and escape policies.

The high-level commander policy is not trained in a curriculum fashion.
Configurations

You can modify training and environment parameters by passing command-line arguments. All available arguments are defined in config.py. Some of the most important ones include:

    --agent_mode: "fight" or "escape" (for low-level training).

    --level: 1 to 5 (for low-level training).

    --num_agents and --num_opps: To change the combat scenario (e.g., 2-vs-2, 3-vs-3).

    --gpu: Set to 1 to enable GPU training.

    --render: Set to True to visualize and save images of the combat scenario during evaluation.

GPU vs CPU

In the original project, experiments showed that training performance was sometimes worse on GPU compared to CPU. The reason is still unknown but might be related to data transfer overhead in this specific environment. In our case, the CPU was an i9-13900H and the GPU was an RTX 3080Ti. You may want to experiment on your own hardware.
Citation

If you use this work in your research, please cite the original paper:
code Bibtex

    
@inproceedings{hhmarl2d,
 title={Hierarchical Multi-Agent Reinforcement Learning for Air Combat Maneuvering},
 author={Selmonaj, Ardian and Szehr, Oleg and Del Rio, Giacomo and Antonucci, Alessandro and Schneider, Adrian and R{\"u}egsegger, Michael},
 booktitle={2023 International Conference on Machine Learning and Applications (ICMLA)},
 pages={1031--1038},
 year={2023},
 organization={IEEE}
}

  

Acknowledgments

A special thanks to the original authors from the Swiss AI Lab IDSIA (Ardian Selmonaj, Oleg Szehr, et al.) for making their excellent research and code publicly available.