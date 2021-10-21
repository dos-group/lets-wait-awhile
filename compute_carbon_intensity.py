"""Converts the production CSVs to carbon intensity CSVs"""

from typing import Dict

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


# A literature review of numerous total life cycle energy sources CO2 emissions per unit of electricity generated,
# conducted by the Intergovernmental Panel on Climate Change in 2011, found that the CO2 emission value,
# that fell within the 50th percentile of all total life cycle emissions studies were as follows.[6]
# http://www.ipcc-wg3.de/report/IPCC_SRREN_Annex_II.pdf
EMISSIONS_IPCC = dict(
    # Renewable
    biopower=18,
    solar_pv=46,
    solar_csp=22,
    geothermal=45,
    hydro=4,
    ocean=8,
    wind=12,
    # Fossil
    nuclear=16,
    gas=469,
    oil=840,
    coal=1001,
    # Other?
)

# Mapping from ENTSOE to IPCC labels
ENTSOE_MAP = {
    'Biomass': EMISSIONS_IPCC["biopower"],
    'Fossil Brown coal/Lignite': EMISSIONS_IPCC["coal"],
    'Fossil Gas': EMISSIONS_IPCC["gas"],
    'Fossil Hard coal': EMISSIONS_IPCC["coal"],
    'Fossil Oil': EMISSIONS_IPCC["oil"],
    'Geothermal': EMISSIONS_IPCC["geothermal"],
    # 'Hydro Pumped Storage': 43,
    'Hydro Run-of-river and poundage': EMISSIONS_IPCC["hydro"],
    'Hydro Water Reservoir': EMISSIONS_IPCC["hydro"],
    'Nuclear': EMISSIONS_IPCC["nuclear"],
    'Other': 770,  # Avg of fossil
    'Other renewable': 22,  # Avg. of renewables
    'Solar': EMISSIONS_IPCC["solar_pv"],
    'Wind Offshore': EMISSIONS_IPCC["wind"],
    'Wind Onshore': EMISSIONS_IPCC["wind"],
    # Production Mix (https://www.carbonfootprint.com/docs/2020_09_emissions_factors_sources_for_2020_electricity_v14.pdf)
    "AT": 133,
    'BE': 153,
    "CZ": 545,
    "DK": 154,
    'FR': 39,
    "DE": 379,
    'IE': 348,
    "IT": 339,
    "LU": 139,
    'NL': 452,
    "NO": 11,
    "PL": 791,
    "ES": 220,
    "SE": 12,
    "CH": 12,
    "GB": 233,
}

# Mapping from CAISO to IPCC labels
CA_MAP = {  # WNA_map
    "Solar": EMISSIONS_IPCC["solar_pv"],
    "Wind": EMISSIONS_IPCC["wind"],
    "Geothermal": EMISSIONS_IPCC["geothermal"],
    "Biomass": EMISSIONS_IPCC["biopower"],
    "Biogas": EMISSIONS_IPCC["biopower"],
    "Small hydro": EMISSIONS_IPCC["hydro"],
    "Large hydro": EMISSIONS_IPCC["hydro"],
    "Coal": EMISSIONS_IPCC["coal"],
    "Nuclear": EMISSIONS_IPCC["nuclear"],
    "Natural gas": EMISSIONS_IPCC["gas"],
    "Batteries": None,  # TODO
    "Imports": 453,  # https://www.carbonfootprint.com/docs/2020_09_emissions_factors_sources_for_2020_electricity_v14.pdf
    "Other": None,
}


def convert(production_csv: str, mapping: Dict, out_csv: str, interpolation_factor: float = None):
    with open(production_csv, "r") as csvfile:
        result = pd.read_csv(csvfile, index_col=0, parse_dates=True)

    if 'Hydro Pumped Storage' in result.columns:
        result = result.drop('Hydro Pumped Storage', axis=1)
    if 'Batteries' in result.columns:
        result = result.drop('Batteries', axis=1)

    total_energy = result.sum(axis=1)
    total_carbon = (result * pd.Series(mapping)).sum(axis=1)
    carbon_intensity = total_carbon / total_energy

    if interpolation_factor:
        y = carbon_intensity.to_numpy()
        x = range(len(y))
        xnew = np.arange(0, len(y) - 1, interpolation_factor)  # minutely
        interpolate = interp1d(x, y, kind=3)
        new_index = pd.date_range(result.index[0], result.index[-1], freq=pd.DateOffset(minutes=30))
        _write_csv(new_index, interpolate(xnew), out_csv=out_csv)
    else:
        _write_csv(carbon_intensity.index, carbon_intensity.values, out_csv=out_csv)


def _write_csv(x, y, out_csv):
    csv_content = "Time,Carbon Intensity\n"
    for i, value in zip(x, y):
        csv_content += f"{i},{value}\n"
    with open(out_csv, "w") as csvfile:
        csvfile.write(csv_content)


if __name__ == '__main__':
    gb = dict(
        production_csv="data/gb_production.csv",
        mapping=ENTSOE_MAP,
        out_csv="data/gb_ci.csv",
        interpolation_factor=None,
    )
    ger = dict(
        production_csv="data/ger_production.csv",
        mapping=ENTSOE_MAP,
        out_csv="data/ger_ci.csv",
        interpolation_factor=None,
    )
    cal = dict(
        production_csv="data/cal_production.csv",
        mapping=CA_MAP,
        out_csv="data/cal_ci.csv",
        interpolation_factor=None,
    )
    fr = dict(
        production_csv="data/fr_production.csv",
        mapping=ENTSOE_MAP,
        out_csv="data/fr_ci.csv",
        interpolation_factor=1 / 2,  # original data is in hourly intervals so this samples in 30min intervals
    )

    for country in [gb, ger, cal, fr]:
        convert(**country)
