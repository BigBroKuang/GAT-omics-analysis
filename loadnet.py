from config import Config
import csv
from collections import defaultdict
import OmicsProcess
import networkx as nx

class LoadNetwork():
    def __init__(self, config, genes):
        self.config = config
        self.load_net(genes)
        
    def load_net(self, genes):
        G = nx.Graph()
        self.genedict = defaultdict(list)
        gene2index = {ge:idx for idx,ge in enumerate(genes)}
        
        with open(self.config.network_path,'r', encoding = 'utf8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row[0] in genes and row[1] in genes:
                    G.add_edge(gene2index[row[0]], gene2index[row[1]])
        for ge in range(len(genes)):
            if ge not in G.nodes():
                self.genedict[ge] = []
            else:
                self.genedict[ge] = list(G.neighbors(ge))
        #print(self.genedict)
        
    def sub_net(self, batch):
        neis ={}
        for b in batch:
            neis[b] = self.genedict[b]
        
        return neis
        
if __name__=="__main__":
    config = Config()
    omics = OmicsProcess.LoadOmicsData(config)
    train_iter, test_iter = omics.load_omics_label()
    data = LoadNetwork(config, omics.genes)