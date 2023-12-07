import os
import json
from rouge_score import rouge_scorer
import numpy as np


class ExplanationSimilarity:
    def __init__(self, fimeDataPath, datasets):
        self.fimeDataPath = fimeDataPath
        self.datasets = datasets
        self.similarityScore = {}
        self.expTypes = ["explanation", "llava-raw-explain", "llava-p2-explain","llava-p3-explain"]
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    def loadJson(self, dataset):
        jsonPath = os.path.join(self.fimeDataPath, dataset, dataset+".json")
        with open(jsonPath, "r", encoding="utf-8") as fd:
            return json.load(fd)

    def printSimScore(self):
        # with open("./simscore.json", "w", encoding="utf-8") as fd:
        #     json.dump(self.similarityScore,fd,indent=2)
        for dataset in self.similarityScore:
            print("Rouge1", dataset)
            print(f'{"    ".join(self.expTypes)}')
            for i in range(len(self.expTypes)):
                print("        ".join([str(v) for v in self.similarityScore[dataset]["Rouge1"][i]]))
        # print("\n\nRougeL")
        # print(f'Dataset    {"    ".join(self.expTypes)}')
        for dataset in self.similarityScore:
            print("RougeL", dataset)
            print(f'{"    ".join(self.expTypes)}')
            for i in range(len(self.expTypes)):
                print("        ".join([str(v) for v in self.similarityScore[dataset]["RougeL"][i]]))
    def computeSimilarity(self):
        for dataset in self.datasets:
            self.similarityScore[dataset] = {}
            data = self.loadJson(dataset)
            # scoresR1 = [[0.0]*len(self.expTypes)]*len(self.expTypes)
            
            # scoresRL = [[0.0]*len(self.expTypes)]*len(self.expTypes)
            scoresR1 = np.zeros((len(self.expTypes), len(self.expTypes)))
            scoresRL = np.zeros((len(self.expTypes), len(self.expTypes)))
            # print(scoresR1)
            dataLen = len(data.keys())
            for imgData in data:                
                for i in range(len(self.expTypes)):
                    for j in range(i+1, len(self.expTypes)):
                        exptype1 = self.expTypes[i]
                        exptype2 = self.expTypes[j]
                        if exptype1 not in data[imgData] or exptype2 not in data[imgData]:
                            continue
                        if len(data[imgData][exptype1]) == 0 or len(data[imgData][exptype2]) == 0:
                            continue
                        # print(exptype1, data[imgData][exptype1])
                        # print(exptype2, data[imgData][exptype2])
                        scores = self.scorer.score(data[imgData][exptype1], data[imgData][exptype2])
                        # print(scores)
                        scoresR1[i][j] += scores['rouge1'][2]
                        scoresRL[i][j] += scores['rougeL'][2]
                        # print(scoresR1)
                        # print(scoresRL)
                        # exit(0)
            for i in range(len(self.expTypes)):
                for j in range(len(self.expTypes)):
                    if scoresR1[i][j] != 0.0:
                        scoresR1[i][j] /= dataLen
                    if scoresRL[i][j] != 0.0:
                        scoresRL[i][j] /= dataLen
            self.similarityScore[dataset]['Rouge1'] = scoresR1
            self.similarityScore[dataset]['RougeL'] = scoresRL
        # print(self.similarityScore)
        self.printSimScore()



if __name__ == "__main__":
    fimeDataPath = os.path.join('..', '..', 'Data', 'FimeData')
    #name of dataste in fimeDataPath
    datasets = ['MEMEX-MCC', 'HVV-COVID19', 'HVV-USPOLITICS']
    simObj = ExplanationSimilarity(fimeDataPath, datasets)
    simObj.computeSimilarity()