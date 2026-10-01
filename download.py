from pathlib import Path
from mne.datasets.sleep_physionet.age import fetch_data

data_dir = Path("sleep_edf_data").resolve()
data_dir.mkdir(parents=True, exist_ok=True)

files = fetch_data(
    subjects=[0, 1, 2],
    recording=[1],
    path=str(data_dir),
)

for psg, hypnogram in files:
    print("Signals:", psg)
    print("Labels: ", hypnogram)