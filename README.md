# Krypton

KryptoCurrency data relay service.

---
## Installation
git and python 3.8+ required. Optionally requires a redis database

- ### Recommended: Install as project

    clone and install requirements
    ```
    git clone https://github.com/BolunHan/Krypton.git
    cd Krypton
    pip install -r requirements.txt
    ```

- ### or Install as package

    clone and pip install with 
    `pip install git+https://github.com/BolunHan/Krypton.git`

- ### or Auto Setup Script

  Ubuntu 18.04+ is required.
  run `wget -O - https://raw.githubusercontent.com/BolunHan/Krypton/main/KryptonSetup.sh | bash`

## Usage

- Optional: set environment variable `KRYPTON_CWD` to a valid location, preferably where the project cloned
  
  e.g. `export KRYPTON_CWD=~/Krypton`

- then init relay with `python Krypton/Relay/Binance.Spot.py`


## Custom Configuration

- copy `config.ini` file to `KRYPTON_CWD` directory
- tweak config file as you wish