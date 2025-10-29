@echo off
chcp 65001
python validate_ofi_cvd_harvest.py --raw-dir data\ofi_cvd --preview-dir preview\ofi_cvd --lookback-mins 180 --min-events 100 --join-tol-ms 900 --seq-huge-jump 20000

