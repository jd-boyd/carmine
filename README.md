# carmine
Object Tracking with positional alarms

## Dev Install
winget install --id=astral-sh.uv  -e
python 3.12 required (3.13 not yet supported)

uv venv -p python3.12 venv

venv\Scripts\activate

uv pip install -r requirements.txt


## Driver note:
I had to go into system settings, search for login items and extensions, scroll down the the extensions section, then enable both driver extensions and camera extensions for blackmagic.


https://github.com/sergio11/vehicle_detection_tracker

https://github.com/AlessioMichelassi/openPyVision_013
