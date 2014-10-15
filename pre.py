###Cosmo Zhang @Purdue 2014/10
###filename: pre.py
###nlp project
### -*- coding: utf-8 -*- 
import sys


def genetr(filename):
    f = open(filename, 'r')
    rawdata = f.readlines()
    data = []
    subtxt = []
    sublb = ['*', '*']
    removal = "\n" + "\r"
    for line in rawdata:
        if 'URL' not in line.split() and  'USR' not in line.split():
            #print line
            if line.split() != []:
                #print line.split()
                subtxt.append(line.strip(removal).split()[0])
                sublb.append(line.strip(removal).split()[1])
            else:
                sublb.append('stop')
                data.append([subtxt, sublb])
                subtxt = []
                sublb = ['*', '*']
    f.close()
    #print data[100][0], data[1]
    return data

def test():
    filename = sys.argv[1]
    data = genetr(filename)
    print data[101]
    print len(data)

if __name__ == "__main__":
    test()
