import sys
import subprocess
import os

emote = "../emote/emote.txt"
classified_list = "classified.txt"
directory_list = "directory.txt"
built_direc_list = "built_directory.txt"
to_chat = "../twitch_chat/"
classfied_direc = "classified"
to_classified = "classified_chat/"


def classify(inputfile, outputfile):

    with open (emote, "r") as emo, open (inputfile, "r") as f, open(outputfile, "w") as output:
        e = []
        for line in emo:
            e.append(line.rstrip("\n"))
        for line in f:
            if line is "\n":
                continue
            else:
                e_counter = 0
                e_list = []
                line = line.rstrip("\n")
                l = line.split()

                for i in range(len(l)):
                    token = l[i]
                    if token[len(token)-1] is ">":
                        index = i
                        break

                l = l[i+1:]
                s = ""
                for word in l:
                    if word in e and word not in e_list:
                        e_list.append(word)
                        e_counter += 1
                    elif word not in e_list:
                        s += (word + " ")
                if e_counter == 1 and s != "":
                    output.write(s)
                    output.write(e_list[0])
                    output.write("\n")

def main():
    with open (built_direc_list, "w+") as di, open (classified_list, "w") as done:
        subprocess.call("ls", cwd=to_classified, stdout=di)
        di.seek(0)
        for line in di:
            line = line.rstrip("\n")
            subprocess.call("ls", cwd=to_classified+line, stdout=done)
    with open (directory_list, "w") as folder_list:
        subprocess.call("ls", cwd=to_chat, stdout=folder_list)

    with open (directory_list, "r") as inp, open (classified_list, "r") as exempt:

        exempt_list = []
        for file_name in exempt:
            exempt_list.append((file_name.rstrip("\n")))

        for direc in inp:

            if direc in classified_list:
                continue
            direc = direc.rstrip("\n")
            d = to_chat + direc + "/"
            with open (d+direc+".txt", "w") as file_list:
                subprocess.call("ls", cwd=d, stdout=file_list)

            with open (d+direc+".txt", "r") as in_list:
                for file_name in in_list:
                    file_name = file_name.rstrip("\n")
                    if file_name in exempt_list:
                        print("Skip " + file_name)
                        continue
                    if direc in file_name:
                        continue
                    i = d + file_name
                    o = to_classified + direc + "/" + file_name
                    if not os.path.isdir(to_classified+direc):
                        subprocess.call("mkdir "+direc, cwd=to_classified, shell=True)
                    print(i + " " + o)
                    classify(i, o)

if __name__ == "__main__":
    main()
