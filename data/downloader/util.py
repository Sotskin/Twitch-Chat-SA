import json
import subprocess
import requests
from datetime import datetime

to_chatdownloader = "../../Twitch-Chat-Downloader/"

def extract_id(json_file, out_name):
    counter = 0
    with open (json_file, "r") as f, open (out_name, "w") as output:
        data = json.load(f)
        for video in data["videos"]:
            output.write(video["_id"] + "\n")
            counter += 1
    print("Finish extracting " + json_file + ". A total of " + str(counter) + " video IDs found.\n")

def download(id_file):
    with open (id_file, "r") as inp:
        for line in inp:
            line = line[1:]
            line = line.rstrip("\n")
            subprocess.call("python3.6 app.py --format irc --output ../final-project/twitch_chat -v "+line, cwd=to_chatdownloader, shell=True)


def get_viewer(channel_id):
    header = {"Client-ID":"juqqpgo8j6zf74fcg7b6dba19hrjo4", "Accept":"application/vnd.twitchtv.v5+json"}
    url = "https://api.twitch.tv/kraken/streams/" + str(channel_id)
    t = str(datetime.now())
    stream_info = requests.get(url, headers = header)
    viewer = stream_info.json()["stream"]["viewers"]
    #print(t+" "+str(viewer))
    return t, viewer

