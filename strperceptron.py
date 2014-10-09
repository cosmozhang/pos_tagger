###Cosmo Zhang @Purdue 2014/10
###filename: strperceptron.py
###nlp project
### -*- coding: utf-8 -*- 

import pre
import random
import math

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
            
def traceback(tb, cart, tags, last_col, slen):
    maxvl = float('-Inf')
    i = -1
    for x in last_col:
        i += 1
        if x[0] > maxvl:
            maxvl = x[0]
            lasttag = tags[i]
            pre_tag = x[1]
    predic_tag = ['stop', lasttag, pre_tag]
    for idx in range(slen, 1, -1):
        row = cart.index((pre_tag, lasttag))
        lasttag = pre_tag
        pre_tag = [row][idx][1]
        predic_tag.append(pre_tag)
    predic_tag.append('*')
    predic_tag.append('*')
    return predic_tag.reverse()
        

def scorefunc(idx, pre_tag, cur_tag, pre_pre_tags, cart_tags, words, dptable, paradic):
    #dynamic programming
    vtdic = {}
    for pre_pre_tag in pre_pre_tags: #here tag is the tag for idx - 1
        tritags = pre_pre_tag + ',' + pre_tag + ',' + cur_tag
        wordtagpair = words[idx] + ',' + cur_tag
        if idx == len(words)+1:
            curvalue = dptable[cart_tags.index((pre_pre_tag, pre_tag))][idx-1] + paradic[tritags]
        else:
            print tritags, paradic[tritags]
            curvalue = dptable[cart_tags.index((pre_pre_tag, pre_tag))][idx-1] + paradic[tritags] + paradic[wordtagpair] #two scores combined (two features)
            vtdic.setdefault(pre_pre_tag, curvalue)
    vtdic_sortkeyls = sorted(vtdic.items(), key = lambda x: x[1], reverse = True)
    return (vtdic[vtdic_sortkeyls[0]], vtdic_sortkeyls[0])
            

def viterbi(paras, words, tags):
    for idx in range(len(words)+1):
        #different tags set
        if idx == 0: pre_pre_tags, pre_tags, cur_tags= ['*'], ['*'], tags
        elif idx == 1: pre_pre_tags, pre_tags, cur_tags = ['*'], tags, tags
        elif idx == len(words)+1: #last tag of the sentence is "stop"
            pre_pre_tags, pre_tags, cur_tags = tags, tags, ['stop']
            cart_tags = [(a, b) for a in pre_tags for b in cur_tags]
            last_col = [([0, None]) for j in range(len(cart_tags))]
            for element in cart_tags:
                pre_tag = element[0]
                cur_tag = element[1]
                last_col[cart_tags.index(element)] = scorefunc(idx, pre_tag, cur_tag, pre_pre_tags, cart_tags, words, dptable, paras)
        else: 
            pre_pre_tags, pre_tags, cur_tags = tags, tags, tags
            cart_tags = [(a, b) for a in pre_tags for b in cur_tags]
            dptable = [([([0, None]) for j in range(len(words))]) for k in  range(len(cart_tags))] #dptable store pre_pre_tag for cur_tag and value
            for element in cart_tags:
                pre_tag = element[0]
                cur_tag = element[1]
                dpinx = idx + 2
                dptable[cart_tags.index(element)][dpidx] = scorefunc(idx, pre_tag, cur_tag, pre_pre_tags, cart_tags, words, dptable, paras)
    #calculate predicted tags by traceback
    predic_tag = traceback(dptable, cart_tags, tags, last_col, len(words))
    return predic_tag

def data_partition(data):
    indexls = range(0, len(data))
    random.shuffle(indexls)
    traindata = []
    valdata = []
    testdata = []
    
    trindexls = indexls[0: int(math.floor(len(data)*0.6))]
    valindexls = indexls[int(math.floor(len(data)*0.6))+1: int(math.floor(len(data)*0.8))]
    teindexls = indexls[int(math.floor(len(data)*0.8))+1: len(data)]

    for i in trindexls:
        traindata.append(data[i])

    for i in valindexls:
        valdata.append(data[i])

    for i in teindexls:
        testdata.append(data[i])

    return (traindata, valdata, testdata)

def update(predic_tag, words, true_tag, paradic):
    if predic_tag != true_tag:
        #update trigram feature weights
        for idx in range(len(words)):
            paradic[true_tag[idx] + ',' + ture_tag[idx+1] + ',' + ture_tag[idx+2]] += 1
            paradic[predic_tag[idx] + ',' + predic_tag[idx+1] + ',' + predic_tag[idx+2]] -= 1
            paradic[words[idx] + ',' + ture_tag[idx+2]] += 1
            paradic[words[idx] + ',' + predic_tag[idx+2]] -= 1
        paradic[true_tag[-3] + ',' + ture_tag[-2] + ',' + ture_tag[-1]] += 1
        paradic[predic_tag[-3] + ',' + predic_tag[-2] + ',' + predic_tag[-1]] -= 1
    return paradic


def train(paradic, data, epochs, tags, valdata):
    prevvaliacc = 0.0
    while True:
        for eg in data:
            predic_tag = viterbi(paradic, eg[0], tags)
            paradic = update(predic_tag, eg[0], eg[1], paradic)
        if epoch > epochs: #early stop creterion
            if (epoch - epochs)%5 == 0:
                valiacc = validate(paradic, valdata)
                if valiacc < prevvaliacc: break
                prevvaliacc = valiacc
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

def test(paradic, data):
    match_sum, num_sum = 0.0, 0.0
    for eg in data:
        predic_tag = viterbi(paradic, eg[0], tags)
        match, num = accuracy(eg[1], predic_tag)
        match_sum += match
        num_sum += num
    return float(match_sum)/num_sum #calculate accuracy

def main():
    filename = "./pos_fixed.tsv"
    data = pre.genetr(filename)
    paradic, tags= para_init(data)
    epochs = 100
    traindata, valdata, testdata = data_partition(data)
    #training
    paras, bestepoch, valiacc = train(paradic, traindata, epochs, tags, valdata)
    #final accuracy
    result_acc = test(testdata, parasup)

if __name__ == "__main__":
    main()
