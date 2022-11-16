import datetime
import os
import shutil
from pathlib import Path

now = datetime.datetime.now()

to_remove = []
for fname in Path("logs").glob("*/runs/*"):
    if fname.is_dir():
        fdate = datetime.datetime.strptime(fname.name, r"%Y-%m-%d_%H-%M-%S")
        if fdate + datetime.timedelta(days=2) < now:
            to_remove.append(fname)
            print(str(fname) + " ... [too old]")
        elif not (fname / "checkpoints").exists():
            to_remove.append(fname)
            print(str(fname) + " ... [no ckpt]")

if len(to_remove) == 0:
    print("No thing to clean.")

choice = input("Enter y to delete. Or any other key to exit.\n")
if choice == "y":
    for item in to_remove:
        shutil.rmtree(item)
