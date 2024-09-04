# 🦠 Leishmaniasis Giemsa Macrophages Model (leishmaniasis.giemsa.macrophages _v2.0b1_)
> Part of the LeishmaniappCloudServicesV2 project

Macrophage detection in Leishmaniasis samples with Giemsa stain

## 🧩 ALEF (Adapter Layer Exec Format)
Input path (absolute or relative) must be specified with `--alef-in`, this model does not require `--alef-out` and will be ignored

Input parameter `-i` used instead of `--alef-in`

Identification model is defined in [src/model.py](src/model.py) which exposes the `analyze` function, [src/alef.py](src/alef.py) parses the arguments and translates results into JSON ALEF. format

## 🪛 Installation
1. Create (and/or activate) a python virtual environment (optional)
   ```sh
   # Create a python environment
   python3.8 -m venv .venv
   # Give run permissions and activate
   chmod +x ./.venv/bin/activate
   ./.venv/bin/activate
   ```
2. Update pip
   ```sh
   python -m pip install --upgrade pip
   ```
3. Install requirements
   ```sh
   pip install -r requirements.txt
   ```
4. Install the script
   ```sh
   pip install -e .
   ```

The model will now be available via the `leishmaniasis_giemsa_macrophages_alef` command (check your _$PATH_ to include pip packages)

## Credits
* Analysis model built by _Nicolás Pérez Fonseca (nicolasperezfonseca1@gmail.com)_ in 2023.
* Code cleanup and ALEF adaptation by _Ángel Talero (angelgotalero@outlook.com)_ in 2024.