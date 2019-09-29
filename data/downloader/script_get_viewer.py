import json
import argparse
import time
from main import main



p = argparse.ArgumentParser()
p.add_argument("arg", type=str, help="Streamer ID")
args = p.parse_args()

while True:
    counter = 0
    with open (args.arg+".txt", "a") as f:
        time.sleep(60)
        t, viewer = main("v", args.arg)
        print(counter,t,viewer)
        counter += 1
        f.write(str(t) + " " + str(viewer) + "\n")


