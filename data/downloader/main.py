import argparse
import util
import subprocess as sub


def main(flag, arg):
    if flag is "e":
        util.extract_id(arg, arg.rstrip(".json")+"_vid.txt")
    if flag is "d":
        util.download(arg)
    if flag is "v":
        time, viewer = util.get_viewer(arg)
        return time, viewer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--extract", help="Extract all video IDs in the given channel info file", action="store_true")
    parser.add_argument("-d", "--download", help="Download all video chat logs according to the given VOD ID file", action="store_true")
    parser.add_argument("-v", "--viewers", help="Get current viewer count of the given streamer (ID)", action="store_true")
    parser.add_argument("true_argument", type=str, help="File name of streamer ID")
    args = parser.parse_args()

    if args.extract:
        main("e", args.true_argument)
    if args.download:
        main("d", args.true_argument)
    if args.viewers:
        main("v", args.true_argument)
