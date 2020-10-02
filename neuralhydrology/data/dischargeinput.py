import logging
import pickle
import sys
from pathlib import Path

from tqdm import tqdm

from neuralhydrology.data.utils import load_basin_file, load_camels_us_forcings, load_camels_us_discharge

LOGGER = logging.getLogger(__name__)


def create_discharge_files(data_dir: Path, basin_file: Path, out_dir: Path, shift: int = 1):

    out_file = out_dir / f"discharge_input_shift{shift}.p"
    if out_file.is_file():
        raise FileExistsError

    basins = load_basin_file(basin_file)

    data = {}

    for basin in tqdm(basins, file=sys.stdout):

        df, area = load_camels_us_forcings(data_dir=data_dir, basin=basin, forcings="daymet")
        df["QObs(mm/d)"] = load_camels_us_discharge(data_dir=data_dir, basin=basin, area=area)

        df[f"QObs(t-{shift})"] = df["QObs(mm/d)"].shift(shift)

        drop_columns = [col for col in df.columns if col != f"QObs(t-{shift})"]

        data[basin] = df.drop(labels=drop_columns, axis=1)

    with out_file.open("wb") as fp:
        pickle.dump(data, fp)

    LOGGER.info(f"Data successfully stored at {out_file}")
