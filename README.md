# Project Setup & Useful Commands

This repository contains several Python tools and apps used in the lab.  
Below is a quick reference for setting up the environment and running the programs.

---

## 1. Virtual Environment

### 1.1 Create / check / activate environment

Use the provided PowerShell script:

```powershell
. .\setup_env.ps1
```
---

## 2. Running the Applications

All applications are started via Python’s module syntax from the repository root.

```powershell
python -m apps.plot_bot.script
```
```powershell
python -m apps.scan_averaging.script
```
```powershell
python -m apps.single_scan.script
```
```powershell
python -m apps.gui_based_plotting.rungui
```
```powershell
python -m apps.stft_analysis.script
```
---
