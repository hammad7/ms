
import json
import re
from multiprocessing import Pool
from copy import copy, deepcopy
from gensim.models.doc2vec import TaggedDocument,Doc2Vec
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm
from GPyOpt.methods import BayesianOptimization
import time,logging,random,collections,os
import numpy as np
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
from pprint import pprint
import pickle
_corpus = 3


def print_stats(dataset):
    print("Distinct Q",len(set([i["question"] for i in dataset])))
    # 58876 questions
    print("Q with ans",len([1 for i in dataset if len(i["positive_ctxs"])>0]))
    # 58880
    print("Total P",sum([len(i["positive_ctxs"]) for i in dataset if len(i["positive_ctxs"])>0]))
    # 498816 total refs to wiki i.e each question has ~8 refs
    print("max P",max([len(i["positive_ctxs"]) for i in dataset if len(i["positive_ctxs"])>0]))
    # upto 101 refs
    print("Distinct P",sum([len(set([j['passage_id'] for j in i["positive_ctxs"]])) for i in dataset if len(i["positive_ctxs"])>0]))
    # 464609 certain questions have repeated passages as answer contexts
    # k=109
    # for i in nq[110:]:
    #   k=k+1
    #   if(len(set([j['passage_id'] for j in i["positive_ctxs"]]))!=len(i["positive_ctxs"])):
    #       print(i)
    #       break
    # k
    # Eg: 
    dataset[112]["positive_ctxs"][0]
    # 'score': 1000, 'title_score': 1
    dataset[112]["positive_ctxs"][4]
    # 'score': 12.370535, 'title_score': 0






def cleanDoc(doc):
    text = doc.lower().replace(",", " , ").replace("(", " ( ").replace(")", " ) ").replace("[", " [ ").replace("]", " ] ").replace("&amp;", "and")
    text = text.replace("&quot;", ' " ').replace("&lt;", " < ").replace("&gt;", " > ").replace("%age","%").replace("%le","percentile").replace("%."," % . ").replace("%ile","percentile").replace("%tile","percentile")
    text = re.sub(r"[\?]+","?",text)
    text = re.sub(r"[\!]+"," ! ",text) 
    text = re.sub(r"\?"," ? ",text)
    text = re.sub(r"[\.]+",".",text)
    text = re.sub(r"\. "," . ",text) #nn
    # text = re.sub(r"[0-9]+\.[0-9]+"," decimal ",text)
    # text = re.sub(r"[0-9]+"," number ",text)
    text = re.sub(r' ([!"#$%&\'()*+,\-\/.:;<=>?@[\\\]^_`{|}~])([A-Za-z]{1,})', r' \1 \2 ',text) # space.abc
    text = re.sub(r'([A-Za-z]{1,})([!"#$%&\'()*+,\-\/.:;<=>?@[\\\]^_`{|}~]) ', r' \1 \2 ',text) # abc.space
    text = re.sub(r'([A-Za-z]{1,})([!"#$%&\'()*+,\-\/.:;<=>?@[\\\]^_`{|}~])$', r' \1 \2 ',text) # abc.eos
    text = re.sub(r"-","",text)
    text = " ".join(text.split())
    return text


with open('datasets/nqdev_tagged_testq.pickle', 'rb') as handle:
    nqdev_tagged_testq = pickle.load(handle)

import json
f = open('datasets/nq_ref_train.json', encoding='utf-8')
nq_refs = json.load(f)
f.close()


def taggedDoc(nq_pids_batch):
    tagged_passage_list = []
    for row in tqdm(nq_pids_batch):
        try:
            doc = wiki.loc[int(row)-1]
            # print(doc)
            docStr = doc["title"] + " " + doc["text"]  ## append title
            docStr = cleanDoc(docStr)
            single_tagd_doc = TaggedDocument(words = docStr.split(), tags = ["p_"+str(row)])
            tagged_passage_list.append(single_tagd_doc)
        except Exception as e:
            print(doc)
    return tagged_passage_list




with open('datasets/train3x.pickle', 'rb') as handle:
    global_merged_list = pickle.load(handle)

with open('datasets/nqdev_passages.pickle', 'rb') as handle:
    nqdev_passages = pickle.load(handle)

global_merged_list = global_merged_list + nqdev_passages ## index



def get_training_q(dataset):
    tagged_sentence_list = [] 
    for row in tqdm(dataset): 
        question = cleanDoc(row["q"])
        question = row["q"]
        tokens = question.split()
        single_tagd_doc = TaggedDocument(words = tokens, tags = ["p_"+j for j in row["pid"]])   
        tagged_sentence_list.append(single_tagd_doc)
    return tagged_sentence_list

tagged_sentence_list = get_training_q(nq_refs)








class EpochLogger(CallbackAny2Vec):
    "Callback to log information about training"
    def __init__(self,testIds,tagged_sentence_list,params):
        self.epoch = 1
        self.modelSeed = 2
        self.testIds = testIds 
        self.tagged_sentence_list = tagged_sentence_list
        self.params = params
        # test set
        # self.testData = testData
    def on_epoch_begin(self, model):
        self.time1 = time.time() 
        print("Epoch #{} start".format(self.epoch))
    def on_epoch_end(self, model):
        self.time2 = time.time() 
        print("Epoch #{} end".format(self.epoch)+"  time(s):"+str(int(self.time2-self.time1)))
        ### get copy not in testmodel since takes memory and this is debugging feature
        results = testModel(getCopy(model),self.testIds,self.tagged_sentence_list,self.modelSeed,False,self.params)
        # results = testModel(getCopy(model),self.testData["testIds"],self.testData["tagged_sentence_list"],self.modelSeed)
        # del modelcopy #=getCopy(model)
        print (results)
        self.epoch += 1

    

def blackboxAgent(params):  ## DEBUGGER METHOD
    print(params,"PARAMS: -----------")
    # print(self.OnlyTestData["tagged_sentence_list"][0:10])
    # print(self.OnlyTestData["tagged_sentence_list"][-10:])
    # print(len(self.trainTestData["tagged_sentence_list"]))
    min_alpha = params[0][0]
    alpha = params[0][1]
    min_count = params[0][2]
    wind = params[0][3]
    siz_ = int(params[0][4])
    dm_ = params[0][5]
    # min_alpha = 0.0005
    # alpha = 0.025967
    # min_count = 2
    # wind = 3
    # siz_ = 300
    # dm_ = 0
    # min_alpha = 0.005763
    # alpha = 0.01297
    # min_count = 2
    # wind = 3
    # siz_ = 500
    # dm_ = 0
    # dm_m = params[0][6]
    # tagged_sentence_list = self.trainTestData["tagged_sentence_list"]
    # print(len(tagged_sentence_list),tagged_sentence_list[0].tags,tagged_sentence_list[0].words)
    # assert docIds are int
    # for row in tagged_sentence_list:
    #     try:
    #         (int(row.tags[0].strip("'")))
    #     except Exception as e:
    #         print("==== string doc ids: ",row.tags[0])
    e_parametrs = {
        "min_alpha":min_alpha,
        "alpha":alpha,
        "min_count" : min_count,
        "wind" : wind,
        "siz_" : siz_,
        "dm_" : dm_,
        "steps":100
    }
    epochlogger = EpochLogger(random.sample(range(len(tagged_sentence_list)),100),tagged_sentence_list,e_parametrs)
    # epochSaver = epochLoggers.EpochSaver(self.modelHisDir)
    model = Doc2Vec(window=wind, vector_size = siz_ ,alpha=alpha,min_alpha=min_alpha, epochs=1, 
                    min_count = min_count, workers=8, # atleast 8 workers
                    dm=dm_, dbow_words=1,dm_mean=1,callbacks=[])
    model.build_vocab(global_merged_list)
    start = time.time()
    _ep = 10
    model.train(global_merged_list,total_examples = model.corpus_count,epochs=_ep, report_delay=2,callbacks=[] )
    end = time.time()
    # if self.debuglevel:
    print("Model Params:",model.train_count,model.alpha,model.min_alpha,model.min_alpha_yet_reached)
    print ("Training time = " + str(round((end - start)/1000,2)) + " seconds")
    modelSeed = 2
    # insModelParam = {
    #     "alpha" : alpha,
    #     "min_alpha" : min_alpha,
    #     "steps" : 100
    # }
    saveModel(model,e_parametrs)
    results = testModel(model,random.sample(range(len(tagged_sentence_list)),2000),tagged_sentence_list,123,False,e_parametrs)
    print(results,"RESULTS: *************",_ep,_corpus)############### in file, with params
    with open('logs.txt', 'at') as out:
        pprint(params, stream=out)
        pprint(results, stream=out)
    resultsValid = testModel(model,random.sample(range(len(nqdev_tagged_testq)),2000),nqdev_tagged_testq,123,False,e_parametrs)
    print(resultsValid,"RESULTS Valid: *************",_ep,_corpus)############### in file, with params
    with open('logs.txt', 'at') as out:
        pprint(resultsValid, stream=out)
    # with open("logs.txt", "a") as file:
    #     file.write(params)
    #     file.write(results)
    # return -float(results[1]) ## top 1 retrieval,   mean rank
    return -float(results[0]) ## top 20 retrieval,   not mean rank


def trainOpt(trainTestData,OnlyTestData=None):
    # self.trainTestData = trainTestData
    # self.OnlyTestData = OnlyTestData
    domain = [
        # {'name': 'var_1', 'type': 'continuous', 'domain': (0.0001,0.01)},
        # {'name': 'var_2', 'type': 'continuous', 'domain': (0.001,0.01)},
        {'name': 'var_1', 'type': 'discrete', 'domain': (0.001,)},
        {'name': 'var_2', 'type': 'discrete', 'domain': (0.014196,)},
        {'name': 'var_3', 'type': 'discrete', 'domain': (1,)},
        {'name': 'var_4', 'type': 'discrete', 'domain': (5,)},
        {'name': 'var_5', 'type': 'discrete', 'domain': (800,)},
        {'name': 'var_6', 'type': 'discrete', 'domain': (0,)},
        # {'name': 'var_7', 'type': 'discrete', 'domain': (0,1)},
        ]
    constraint = [
        {'name': 'constr_1', 'constraint': '-x[:,1]+x[:,0]'},  # min_alpha < alpha
        # {'name': 'constr_1', 'constraint': '-0.9*x[:,1]+x[:,0]'},  # min_alpha < alpha
        ]
    myBopt = BayesianOptimization(f=blackboxAgent, domain=domain,  constraints = constraint ,
        acquisition_type='EI', num_cores=8, verbosity=True, verbosity_model=False, maximize=True)
    myBopt.run_optimization(max_iter=100, max_time=36000, eps=1e-04, verbosity=True,report_file="gpyopt100_aacor.log",
        evaluations_file="eval100_aacor.log")
    myBopt.plot_convergence()
    print((myBopt.fx_opt, myBopt.x_opt))
    myBopt.plot_acquisition(filename="plotted")
    return myBopt



def saveModel(model,insModelParam,sep="_"):
    try:
        print("Saving model to disk")
        fileName = "models/dpr"+sep+str(insModelParam)
        model.save(fileName);
        # del model
        return fileName
    except Exception as e:
        print("Error saving")
        return ""

def testModel(model,testIds,tagged_list,modelSeed,returnNonTop1=False,modelParameters=None):
    ranks = []
    print("Docs: ",len(tagged_list)," modelSeed: ",modelSeed)
    print("TestSet: ",len(testIds))
    print(modelParameters)
    alpha_ = modelParameters["alpha"]
    min_alpha_ = modelParameters["min_alpha"]
    steps_ = modelParameters["steps"]
    model.docvecs.vectors_docs_norm = None
    model.docvecs.init_sims()
    nonTop1 = []
    for doc_id in tqdm(testIds):
        if modelSeed:
            model.random.seed(modelSeed)
        # print(tagged_list[doc_id].words)
        inferred_vector = model.infer_vector(tagged_list[doc_id].words,alpha=alpha_,min_alpha=min_alpha_,epochs=steps_) ## global tagged_list
        p_ids = tagged_list[doc_id].tags
        passages = model.docvecs.most_similar([inferred_vector], topn=1000+1)## because top 1000
        array_of_p_ids = []
        for p_id, sim in passages:
            try:
                array_of_p_ids.append(p_id)
            except Exception as e:
                print("####### string doc ids: ",p_id)
        q_rank=[]
        match = list(set(p_ids) & set(array_of_p_ids))
        if len(match)>0:
            #consider avg rank
            for p_id in p_ids:
                if p_id in array_of_p_ids:
                    q_rank.append(array_of_p_ids.index(p_id)+1)
                else:
                    q_rank.append(1001)
            # if rank!=1:
            #     if rank >= 1001:
            #         print(rank,ques_id,tagged_sentence_list[doc_id].words,array_of_p_ids[0:15]) ### DEBUGGER
            #     nonTop1.append(ques_id)
        else:
            q_rank.append(1001)
            nonTop1.append(doc_id)
        ranks.append(np.mean(q_rank))
    meanRank = str(np.mean([row for row in ranks if row>0]))
    # print(nonTop1) #DEBUGGER
    hits=collections.Counter(ranks)
    print(hits)
    sum1=0
    sum5=0
    sum10=0
    sum20=0
    sum50=0
    sum100=0
    sum500=0
    sum1000=0
    for key,cnt in dict(hits).items():
        if key!=-1:
            if key==1:
                sum1+=(cnt)
            if key<=5:
                sum5+=(cnt)
            if key<=10:
                sum10+=(cnt)
            if key<=20:
                sum20+=(cnt)
            if key<=50:
                sum50+=(cnt)
            if key<=100:
                sum100+=(cnt)
            if key<=500:
                sum500+=(cnt)
            if key<=1000:
                sum1000+=(cnt)
    results = [meanRank,float(sum1)/len(ranks),float(sum5)/len(ranks),float(sum10)/len(ranks),float(sum20)/len(ranks),float(sum50)/len(ranks),float(sum100)/len(ranks),float(sum500)/len(ranks),float(sum1000)/len(ranks)]
    if returnNonTop1:
        results.append(nonTop1)
    return results




class Copyable:
    __slots__ = 'a', '__dict__'
    def __init__(self, a):
        self.a = a
    def __copy__(self):
        return type(self)(self.a)
    def __deepcopy__(self, memo): # memo is a dict of id's to copies
        id_self = id(self)        # memoization avoids unnecesary recursion
        _copy = memo.get(id_self)
        if _copy is None:
            _copy = type(self)(
                deepcopy(self.a, memo))
            memo[id_self] = _copy 
        return _copy

def getCopy(model):
    c1 = Copyable(model)
    c2 = deepcopy(c1)
    return c2.a


def loadModel(insModelParam,sep="_"):
    try:
        fileName = "models/dpr"+sep+str(insModelParam)
        # fname = get_tmpfile(fileName)
        model = Doc2Vec.load(fileName);
        return model
    except Exception as e:
        print("Error loading model")
        return ""



best = trainOpt(None)
