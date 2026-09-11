#!/usr/bin/env python3
"""Invoke the unchanged exact-manifest public model downloader."""
import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'repo/research/refit_round_2026-09-07/deployment'))
from cloud_worker import download_model
p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();download_model(a.manifest,a.out)
