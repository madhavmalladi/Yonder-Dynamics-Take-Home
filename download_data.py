"""Download the Sampled-YD-Object-Detection v2 dataset (YOLOv8 format) from Roboflow.

Reads the API key from the ROBOFLOW_API_KEY environment variable (or a .env file).
"""
import os

from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()
api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    raise SystemExit("ROBOFLOW_API_KEY is not set. Add it to .env (see .env.example).")

rf = Roboflow(api_key=api_key)
project = rf.workspace("malletbottle2").project("sampled-yd-object-detection")
version = project.version(2)
dataset = version.download("yolov8")
