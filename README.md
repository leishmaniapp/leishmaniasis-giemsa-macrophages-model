# 🦠 Leishmaniasis Giemsa Macrophages Model (leishmaniasis.giemsa.macrophages _v2.0b1_)
> Part of the LeishmaniappCloudServicesV2 project

Macrophage detection in Leishmaniasis samples with Giemsa tinction

## 🧩 ALEF (Adapter Layer Exec Format)
Input path (absolute or relative) must be specified with `--alef-in`, this model does not require `--alef-out` parameter and thus will be ignored.

Input can also be specified using `-i` instead of `--alef-int`

Results will be printed to stdout using ALEF JSON format

Identification model is defined in [src/model.py](src/model.py) which exposes the `analyze` function, [src/alef.py](src/alef.py) parses the arguments and translates results into ALEF. format

## Credits
* Analysis model built by _Nicolás Pérez Fonseca (nicolasperezfonseca1@gmail.com)_ in 2023.
* Code cleanup and ALEF adaptation by _Ángel Talero (angelgotalero@outlook.com)_ in 2024.