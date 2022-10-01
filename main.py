"""
Practice on LSA: (SVD)
https://blog.csdn.net/zhzhji440/article/details/47193731?spm=1001.2014.3001.5501
"""

from cmath import isnan
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
# from string import maketrans

titles = [
            "The Neatest Little Guide to Stock Market Investing", #T1
            "Investing For Dummies, 4th Edition", #T2
            "The Little Book of Common Sense Investing The Only Way to Guarantee Your Fair Share of Stock Market Returns",
            "The Little Book of Value Investing", "Value Investing From Graham to Buffett and Beyond",
            "Rich Dads Guide to Investing: What the Rich Invest in, That the Poor and the Middle Class Do Not!",
            "Investing in Real Estate, 5th Edition",
            "Stock Investing For Dummies",
            "Rich Dads Advisors: The ABC's of Real Estate Investing The Secrets of Finding Hidden Profits Most Investors Miss",
            "Harry Potter and his adventure"
        ]
stopwords = ['and','edition','for','in','little','of','the','to']
ignorechars = ",:'!"

class LSA(object):
    def __init__(self, stopwords, ignorechars):
        self.stopwords = stopwords
        self.ignorechars_trans_table = "".maketrans(ignorechars, "    ")
        self.word_dict = {}
        self.doc_cout = 0
    
    def parse(self, doc):
        words = doc.split()

        for word in words:
            # translate()
            word = word.lower().translate(self.ignorechars_trans_table)
            
            if word in self.stopwords:
                continue
            elif word in self.word_dict:
                self.word_dict[word].append(self.doc_cout)
            else:
                self.word_dict[word] = [self.doc_cout]
        
        self.doc_cout += 1
        return

    def build(self):
        self.keys = [ key for key in self.word_dict.keys() if len(self.word_dict[key]) > 1 ] # 只取出現一次以上的
        self.keys.sort()
        self.count_matrix = np.zeros([ len(self.keys), self.doc_cout ])
        self.TFIDF_matrix = np.zeros([ len(self.keys), self.doc_cout ])

        for key_idx, key in enumerate(self.keys):
            for doc_num in self.word_dict[key]:
                self.count_matrix[key_idx, doc_num] += 1
        return
    
    def print_count_matrix(self):
        print("keys>1:", self.keys)
        print("# of documents:", self.doc_cout)
        print("count_matrix:\n", self.count_matrix)
        return
    
    def print_TFIDF_matrix(self):
        print(self.TFIDF_matrix)
        return
    
    # Optional
    def TFIDF(self):
        # 不會用到的變數不需要進入 class (self.)
        words_per_doc = np.sum(self.count_matrix, axis=0)
        docs_per_word = np.sum(np.asarray(self.count_matrix>0, "int"), axis=1) # 過濾所有大於 0 的成為 1
        rows, cols = self.count_matrix.shape

        for i in range(rows):
            for j in range(cols):
                val = (self.count_matrix[i][j] / words_per_doc[j]) * np.log(float(cols) / docs_per_word[i])
                self.TFIDF_matrix[i][j] = val
                if np.isnan(val):
                    self.TFIDF_matrix[i][j] = 0.0
        return

    def do_svd(self, dim=None, TFIDF=False):
        self.U, self.S, self.Vt = svd(self.count_matrix)
        if TFIDF == True:
            self.U, self.S, self.Vt = svd(self.TFIDF_matrix)

        # 取前三維（沒特別意義，方便視覺化而已）
        # print(self.Vt) # WHy (10,)?
        print(self.U[:, :3]) # 單字在語意空間中的座標
        print(self.S[:3]) # 奇異值
        print(self.Vt[:, :3]) # 文章在語意空間中的座標
        
        if dim != None:
            self.U = self.U[:, :dim]
            self.S = self.S[:dim]
            self.Vt = self.Vt[:, :dim]
        return
    
    def draw_fig(self):
        plt.rcParams["figure.autolayout"] = True

        x, y = self.U[:, 1], self.U[:, 2]
        plt.plot(x, y, 'r*')
        for idx, xy in enumerate(zip(x, y)):
            # print(self.keys)
            # plt.annotate('(%.2f, %.2f)' % xy, xy=xy)
            plt.annotate(self.keys[idx], xy=xy)
        
        x, y = self.Vt[:, 1], self.Vt[:, 2]
        plt.plot(x, y, 'b*')
        for idx, xy in enumerate(zip(x, y)):
            # plt.annotate("({:.2f}, {:.2f})".format(xy[0], xy[1]), xy=xy)
            plt.annotate("T{}".format(idx+1), xy=xy)

        plt.axis([-1,1,-1,1])
        plt.show()
        return


lsa = LSA(stopwords=stopwords, ignorechars=ignorechars)

for title in titles:
    lsa.parse(title)
lsa.build()
# lsa.print_count_matrix()

lsa.TFIDF()
# lsa.print_TFIDF_matrix()

lsa.do_svd(dim=3, TFIDF=False)

# Draw figure
lsa.draw_fig() # T10 在 (0,0)