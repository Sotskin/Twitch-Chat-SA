
with open("output.txt", "r") as i, open("sorted.txt", "w") as o:
    l = []
    d = {}

    with open ("complete_emote_sentiment.txt", "r") as e:
        for line in e:
            li = line.split()
            sentiment = li[0]
            li= li[1:]
            for word in li:
                word = word.rstrip("\n")
                word = word.rstrip()
                d[word] = sentiment

    for line in i:
        line = line.rstrip("\n")
        line = line.split()
        tup = (line[0], line[4], d[line[0]])
        l.append(tup)

    sort = sorted(l, key=lambda element:int(element[1]))

    for tup in sort:
        o.write(tup[0]+" "+tup[1]+" "+tup[2]+"\n")
