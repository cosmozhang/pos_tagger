###Cosmo Zhang @Purdue 2014/10
###filename: strperceptron.py
###nlp project
### -*- coding: utf-8 -*- 

import pre
import random
import math
import time

def para_init(data):
    paradic = {}
    tagls = []
    for eg in data:
        words = eg[0]
        tags = eg[1]
        #print words
        #print tags
        for idx in range(len(tags)-2):
            s1 = tags[idx+1] + ','  + tags[idx+2]
            paradic.setdefault(s1, 0)
            if idx > 1 and idx != len(tags)-1: #return all possible tags as a list
                if tags[idx] not in tagls:
                    tagls.append(tags[idx])
            
        for idx in range(len(words)):
            s2 = words[idx] + ',' + tags[idx+2]
            paradic.setdefault(s2, 0)
    return paradic, tagls
            
def traceback(tb, tags, slen):
    #print slen
    maxvl = float('-Inf')
    i = -1
    #print tb[-1]
    for x in tb[-1]: #get the maximum value in the last column
        #print x
        i += 1
        if x[0] >= maxvl:
            maxvl = x[0]
            pre_tag = x[1]
    cur_tag = 'stop'
    predic_tag = [cur_tag, pre_tag]
    #print tb
    for idx in range(slen-2, 0, -1):
        row = tags.index((pre_tag))
        cur_tag = pre_tag
        #print range(slen-1, 0, -1)
        #print "trace, idx", slen, idx, tb[idx], row
        pre_tag = tb[idx][row][1] #the pre_tag
        predic_tag.append(pre_tag)
    predic_tag.append('*')
    predic_tag.append('*')
    predic_tag.reverse()
    #print 'words lenth', slen, predic_tag, len(tb)
    return predic_tag
        

def scorefunc(idx, cur_tag, pre_tags, words, dptable, paradic):
    #dynamic programming
    #vtdic = {}
    bestvalue = float('-Inf')
    #print 'len', len(pre_pre_tags)
    for pre_tag in pre_tags: #here tag is the tag for idx - 1
        bitags = pre_tag + ',' + cur_tag
        
        #print "idx", idx, pre_pre_tag + ',' + pre_tag + ',' + cur_tag
        if idx != len(words): #last idx is len(words) for no word
            wordtagpair = words[idx] + ',' + cur_tag
            if wordtagpair not in paradic:
                paradic.setdefault(wordtagpair, 0)
        if bitags not in paradic:
            #print bitags
            paradic.setdefault(bitags, 0)

        #calculate the score
        if idx == 0:
            curvalue = paradic[bitags] + paradic[wordtagpair] #for first word
        elif idx == len(words):
            #print "idx", idx
            #print dptable[idx-1][last_cart_tags.index((pre_pre_tag, pre_tag))]
            #print cur_tag
            curvalue = dptable[idx-1][pre_tags.index((pre_tag))][0] + paradic[bitags]
            #print pre_tag
        else:
            '''
            print idx
            print dptable
            print last_cart_tags
            print pre_pre_tag, pre_tag
            print last_cart_tags.index((pre_pre_tag, pre_tag)), idx-1, 0
            '''
            #print last_cart_tags
            #print dptable[1]
            
            curvalue = dptable[idx-1][pre_tags.index((pre_tag))][0] + paradic[bitags] + paradic[wordtagpair] #two scores combined (two features)
        if curvalue >= bestvalue:
            bestvalue = curvalue
            bestpre_tag = pre_tag
        #vtdic.setdefault(pre_pre_tag, curvalue)
    #vtdic_sortkeyls = sorted(vtdic.items(), key = lambda x: x[1], reverse = True)
    #print vtdic_sortkeyls
    #print vtdic[vtdic_sortkeyls[0][0]]
    #print vtdic_sortkeyls[0][0]
    return (bestvalue, bestpre_tag)
            

def viterbi(paras, words, tags):
    dptable = [] #dptable store pre_pre_tag for cur_tag, and value
    for idx in range(len(words)+1): #idx in the word sequence
        #print range(len(words)+1)
        #print '%dth word' % idx
        timeold = time.clock()
        #different tags set
        if idx == 0:
            #print words[idx]
            #print "idx0", idx
            pre_tags, cur_tags= ['*'], tags
        elif idx == len(words): 
            pre_tags, cur_tags =tags, ['stop'] #last tag of the sentence is "stop"
        else:
            pre_tags, cur_tags = tags, tags

        #print cart_tags_tb[0]
        cur_col = [([0, None]) for j in range(len(cur_tags))]
        #print cur_col
        for cur_tag in cur_tags:
            
            cur_col[cur_tags.index(cur_tag)] = scorefunc(idx, cur_tag, pre_tags, words, dptable, paras)
        #print cur_col
        #print 'used time is %f' % (time.clock() - timeold)
        dptable.append(cur_col)
    #print len(dptable)
    predic_tag = traceback(dptable, tags, len(words)+1) #calculate predicted tags by traceback
    #print 'in viterbi', predic_tag
    return predic_tag

def data_partition(data):
    indexls = range(0, len(data))
    random.shuffle(indexls)
    traindata = []
    valdata = []
    testdata = []
    
    trindexls = indexls[0: int(math.floor(len(data)*0.6))] #60% training
    valindexls = indexls[int(math.floor(len(data)*0.6))+1: int(math.floor(len(data)*0.8))] #20 validation
    teindexls = indexls[int(math.floor(len(data)*0.8))+1: len(data)] #20 testing

    for i in trindexls:
        traindata.append(data[i])
    for i in valindexls:
        valdata.append(data[i])
    for i in teindexls:
        testdata.append(data[i])
    return (traindata, valdata, testdata)

def update(predic_tag, words, true_tag, paradic): #update perceptron parameters
    if predic_tag != true_tag:
        #print predic_tag
        #update trigram feature weights
        #print len(predic_tag), len(words), len(true_tag)
        for idx in range(len(words)):
            paradic[true_tag[idx+1] + ',' + true_tag[idx+2]] += 1
            paradic[predic_tag[idx+1] + ',' + predic_tag[idx+2]] -= 1
            paradic[words[idx] + ',' + true_tag[idx+2]] += 1
            paradic[words[idx] + ',' + predic_tag[idx+2]] -= 1
        #paradic[true_tag[-2] + ',' + true_tag[-1]] += 1
        #paradic[predic_tag[-2] + ',' + predic_tag[-1]] -= 1
    return paradic

def test(paradic, data, tags):
    match_sum, num_sum = 0.0, 0.0
    for eg in data:
        predic_tag = viterbi(paradic, eg[0], tags)
        match, num = accuracy(eg[1], predic_tag)
        match_sum += match
        num_sum += num
    return float(match_sum)/num_sum #calculate accuracy

def train(paradic, data, epochs, tags, valdata):
    prevvaliacc = 0.0
    epoch = 1
    while True:
        time_b_v = time.clock()
        print "\nthis is epoch %d" % epoch
        for idx in range(len(data)):
            if idx % 5 == 0: print "\n************%.2f%% is finished************" %(100*idx*1.0/len(data))
            eg = data[idx]
            #print eg
            
            predic_tag = viterbi(paradic, eg[0], tags)
            
            paradic = update(predic_tag, eg[0], eg[1], paradic)
        if epoch > epochs: #early stop creterion
            if (epoch - epochs)%1 == 0:
                valiacc =test(paradic, valdata, tags)
                print "accuracy is %f%%" % (valiacc*100)
                if valiacc < prevvaliacc: break
                prevvaliacc = valiacc
        print "\nused time in this epoch is %f" % (time.clock() - time_b_v) #timing viterbi
        epoch += 1
    return (paradic, epoch, valiacc)

def accuracy(orig, pred):  
    num = len(orig)
    if(num != len(pred)):
	print 'Error!! Num of labels are not equal.'
    	return
    match = 0
    for o_atag, p_atag in zip(orig, pred):
	if(o_atag == p_atag):
	    match += 1
    return (match, num)

def main():
    filename = "./pos_fixed.tsv"
    data = pre.genetr(filename)
    paradic, tags= para_init(data)
    epochs = 5
    traindata, valdata, testdata = data_partition(data)
    #training
    parasup, bestepoch, valiacc = train(paradic, traindata, epochs, tags, valdata)
    #final accuracy
    result_acc = test(parasup, testdata, tags)
    print "~~~~~~~test accuracy is %0.2f%%~~~~~~" % (100*result_acc)

if __name__ == "__main__":
    main()
