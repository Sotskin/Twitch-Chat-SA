import sys
import argparse

emote = "emote.txt"
emote_sentiment = "emote_sentiment.txt"
complete = "complete_emote_sentiment.txt"

def emote_to_id():
    with open ("emote.txt", "r") as i, open ("emoticon.txt", "w") as o:
        counter = 0
        for line in i:
            line = line.rstrip("\n")
            o.write(line + " " + str(counter) + "\n")
            counter += 1

def fill_emote():
    with open (emote_sentiment, "r") as e, open(complete, "w") as o, open (emote, "r") as emo:
        to_sentiment = {}
        for line in e:
            l = line.split()
            sentiment = l[0]
            o.write(sentiment)
            l = l[1:]
            for word in l:
                word = word.rstrip("\n")
                to_sentiment[word] = sentiment
                o.write(" " + word)
            o.write("\n")

        #o.write("spam")
        #with open ("processed_spam_emote.txt", "r") as sp:
        #    for line in sp:
        #        line = line.rstrip("\n")
        #        o.write(" " + line)
        #        to_sentiment[line] = "spam"
        #    o.write("\n")

        o.write("undefined")
        for line in emo:
            line = line.rstrip("\n")
            if line not in to_sentiment:
                o.write(" " + line)
        o.write("\n")



def emote_to_sentiment(f):
    outfile = "sentiment_" + f
    with open (complete, "r") as e, open (f, "r") as i, open(outfile, "w") as o:
        to_sentiment = {}
        for line in e:
            l = line.split()
            sentiment = l[0]
            l = l[1:]
            for word in l:
                word = word.rstrip("\n")
                word = word.rstrip()
                to_sentiment[word] = sentiment

        for line in i:
            line = line.rstrip("\n")
            o.write(line + " " + to_sentiment[line] + "\n") 
    print("Finishing classifying. Input: " + f + ". Output: " + outfile + ".")


def main():
    arg = sys.argv[1:]
    fill_emote()
    #emote_to_sentiment(arg[0])

if __name__ == "__main__":
    main()
