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
            s1 = tags[idx] + ',' + tags[idx+1] + ','  + tags[idx+2]
            paradic.setdefault(s1, 0)
            if idx > 1 and idx != len(tags)-1:
                if tags[idx] not in tagls:
                    tagls.append(tags[idx])
            
        for idx in range(len(words)):
            s2 = words[idx] + ',' + tags[idx+2]
            paradic.setdefault(s2, 0)
    return paradic, tagls
            
def traceback(tb, cart, tags, slen):
    maxvl = float('-Inf')
    i = -1
    #print tb[-1]
    for x in tb[-1]: #get the maximum value
        #print x
        i += 1
        if x[0] >= maxvl:
            maxvl = x[0]
            cur_tag = tags[i]
            pre_tag = x[1]
    predic_tag = ['stop', cur_tag, pre_tag]
    for idx in range(slen-1, 1, -1):
        row = cart[idx].index((pre_tag, cur_tag))
        cur_tag = pre_tag
        pre_tag = tb[idx][row][1] #the pre_pre_tag
        predic_tag.append(pre_tag)
    predic_tag.append('*')
    predic_tag.append('*')
    predic_tag.reverse()
    return predic_tag
        

def scorefunc(idx, pre_tag, cur_tag, pre_pre_tags, last_cart_tags, words, dptable, paradic):
    #dynamic programming
    #vtdic = {}
    bestvalue = float('-Inf')
    #print 'len', len(pre_pre_tags)
    for pre_pre_tag in pre_pre_tags: #here tag is the tag for idx - 1
        tritags = pre_pre_tag + ',' + pre_tag + ',' + cur_tag
        
        #print "idx", idx, pre_pre_tag + ',' + pre_tag + ',' + cur_tag
        if idx != len(words): #last idx is len(words) for no word
            wordtagpair = words[idx] + ',' + cur_tag
            if wordtagpair not in paradic:
                paradic.setdefault(wordtagpair, 0)
        if tritags not in paradic:
            paradic.setdefault(tritags, 0)
        #calculate the score
        if idx == 0:
            curvalue = paradic[tritags] + paradic[wordtagpair] #for first word
        elif idx == len(words):
            #print "idx", idx
            #print dptable[idx-1][last_cart_tags.index((pre_pre_tag, pre_tag))]
            
            curvalue = dptable[idx-1][last_cart_tags.index((pre_pre_tag, pre_tag))][0] + paradic[tritags]
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
            
            curvalue = dptable[idx-1][last_cart_tags.index((pre_pre_tag, pre_tag))][0] + paradic[tritags] + paradic[wordtagpair] #two scores combined (two features)
        if curvalue >= bestvalue:
            bestvalue = curvalue
            bestpre_pre_tag = pre_pre_tag
        #vtdic.setdefault(pre_pre_tag, curvalue)
    #vtdic_sortkeyls = sorted(vtdic.items(), key = lambda x: x[1], reverse = True)
    #print vtdic_sortkeyls
    #print vtdic[vtdic_sortkeyls[0][0]]
    #print vtdic_sortkeyls[0][0]
    return (bestvalue, bestpre_pre_tag)
            

def viterbi(paras, words, tags):
    dptable = [] #dptable store pre_pre_tag for cur_tag, and value
    cart_tags_tb = []
    cart_tagsg = [(a, b) for a in tags for b in tags]
    for idx in range(len(words)+1):
        print '%dth word' % idx
        timeold = time.clock()
        #different tags set
        if idx == 0: 
            #print "idx0", idx
            pre_pre_tags, pre_tags= ['*'], ['*']
            cart_tags = [(a, b) for a in pre_tags for b in tags]
        elif idx == 1: 
            pre_pre_tags = ['*']
            cart_tags = [(a, b) for a in tags for b in tags]
        elif idx == len(words): 
            cur_tags = ['stop'] #last tag of the sentence is "stop"
            cart_tags = [(a, b) for a in tags for b in cur_tags]
        else:
            pre_pre_tags, pre_tags, cur_tags = tags, tags, tags
            cart_tags = cart_tagsg

        cart_tags_tb.append(cart_tags)
        #print cart_tags_tb[0]
        cur_col = [([0, None]) for j in range(len(cart_tags))]
        #print cur_col
        print 'cart_tags lenth is %d' % len(cart_tags)
        for element in cart_tags:
            pre_tag = element[0]
            cur_tag = element[1]
            
            cur_col[cart_tags.index(element)] = scorefunc(idx, pre_tag, cur_tag, pre_pre_tags, cart_tags_tb[idx-1], words, dptable, paras)
            #print cur_col
        print 'used time is %f' % (time.clock() - timeold)
        dptable.append(cur_col)
        #print dptable
    predic_tag = traceback(dptable, cart_tags_tb, tags, len(words)) #calculate predicted tags by traceback
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
        for idx in range(len(words)):
            paradic[true_tag[idx] + ',' + true_tag[idx+1] + ',' + true_tag[idx+2]] += 1
            paradic[predic_tag[idx] + ',' + predic_tag[idx+1] + ',' + predic_tag[idx+2]] -= 1
            paradic[words[idx] + ',' + true_tag[idx+2]] += 1
            paradic[words[idx] + ',' + predic_tag[idx+2]] -= 1
        paradic[true_tag[-3] + ',' + true_tag[-2] + ',' + true_tag[-1]] += 1
        paradic[predic_tag[-3] + ',' + predic_tag[-2] + ',' + predic_tag[-1]] -= 1
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
        for idx in range(len(data[:20])):
            if idx % 5 == 0: print "\n************%.2f%% is finished************" %(100*idx*1.0/len(data[:20]))
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
    paras, bestepoch, valiacc = train(paradic, traindata, epochs, tags, valdata)
    #final accuracy
    result_acc = test(testdata, parasup)
    print "~~~~~~~test accuracy is %0.2f%%~~~~~~" % (100*result_acc)

if __name__ == "__main__":
    main()
