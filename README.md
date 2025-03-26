# carmine
Object Tracking with positional alarms

## Dev Install

Using uv for package installation is recommended.

On windows:
`winget install --id=astral-sh.uv  -e`

On macOS:
`curl -LsSf https://astral.sh/uv/install.sh | sh`

On linux:
`pip install uv`

python 3.12 required (3.13 not yet supported)

`uv venv -p python3.12 venv`

If you don't have that version of python, the above command will get it for you while making the virtual environment.

`venv\Scripts\activate`
or
`source venv/bin/activate`

At this point, you should download and install bmcapture: https://github.com/jd-boyd/bmcapture

```
git clone https://github.com/jd-boyd/bmcapture
cd bmcapture
uv pip install -e ./
```

`uv pip install -r requirements.txt`

`python ./carmine.py`

Note, startup is always slow, but the first time it also has to download the yolo model, which will take longer.

## Driver note:
I had to go into system settings, search for login items and extensions, scroll down the the extensions section, then enable both driver extensions and camera extensions for blackmagic.


https://github.com/sergio11/vehicle_detection_tracker

https://github.com/AlessioMichelassi/openPyVision_013
