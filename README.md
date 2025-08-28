# VessShape Experiments

This repository contains experiments and scripts for vessel shape analysis and related machine learning workflows. It is organized to facilitate training, fine-tuning, and testing of models using vessel shape datasets.

## Repository Structure

- `training/` — Scripts and configurations for training models.
- `fine-tuning/` — Scripts for fine-tuning pre-trained models.
- `testing/` — Scripts for evaluating models.
- `setup.sh` — Shell script to set up the environment and install dependencies.

## Setup Instructions

To set up your environment and install all necessary dependencies, follow these steps:

1. **Clone this repository** (if you haven't already):

   ```bash
   git clone <this-repository-url>
   ```

2. Create a virtual environment (optional but recommended):

    With Python 3.11 or later, you can create a virtual environment to isolate your dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

    If you use conda, you can create an environment with:
    ```bash
    conda create -n <env_name> python=3.11
    conda activate <env_name>
    ```

3. **Run the setup script to install dependencies:**

   ```bash
   bash setup.sh
   ```

   This script will:
   - Clone the [torchtrainer](https://github.com/chcomin/torchtrainer) repository and install it in editable mode.
   - Clone the [vessel-shape-dataset](https://github.com/galvaowesley/vess-shape-dataset) repository and install it in editable mode.

Make sure you have Python and pip installed on your system.


---

Feel free to explore the `training/`, `fine-tuning/`, and `testing/` folders for more details on available experiments and scripts.
