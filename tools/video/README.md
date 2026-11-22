# Walkthrough video

`docs/media/ReefFlow_walkthrough.mp4` (1920 x 1080, 30 fps, about 2 min, captions, no sound) is recorded
from the real app with headless Chrome on the GPU and assembled with FFmpeg.

1. Make a workspace with the demo project and a finished run (about 12 min on an RTX 3070):

       .venv\Scripts\python tools\video\make_demo_run.py D:\tmp\videows

2. Record (needs Chrome, FFmpeg on PATH and `pip install websocket-client`):

       .venv\Scripts\python tools\video\record.py tools\video\walkthrough_steps.json docs\media\ReefFlow_walkthrough.mp4 D:\tmp\videows

`walkthrough_steps.json` is the script: each step runs JavaScript in the page; the helpers `__cap` (caption),
`__title`, `__click`, `__pick` (choose in a list) and `__orbit` (turn the camera) are defined in `record.py`.
Edit the captions there and record again.
