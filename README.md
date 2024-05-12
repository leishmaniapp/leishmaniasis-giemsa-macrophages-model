# 🦠 Leishmaniasis Giemsa Macrophages Model (leishmaniasis.giemsa.macrophages)
> Part of the LeishmaniappCloudServicesV2 project

## 🧩 ALEF (Adapter Layer Exec Format)
[Visit this repository for details about ALEF](https://github.com/leishmaniapp/model_wrapper)

Input path (absolute or relative) must be specified with `--alef-in`, this model does not require `--alef-out` parameter and thus will be ignored.

Identification model is defined in [src/model.py](src/model.py) which exposes the `analyze` function, [src/alef.py](src/alef.py) parses the arguments and translates results into ALEF. format

## Credits
Analysis model built by _Nicolás Pérez Fonseca (nicolasperezfonseca1@gmail.com)_ in 2023.