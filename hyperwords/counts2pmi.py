from docopt import docopt
from scipy.sparse import dok_matrix, csr_matrix
import numpy as np
import pandas as pd
import scipy as sp
from representations.matrix_serializer import save_matrix, save_vocabulary, load_count_vocabulary
import time

def main():
    args = docopt("""
    Usage:
        counts2pmi.py [options] <counts> <output_path>
    
    Options:
        --cds NUM    Context distribution smoothing [default: 1.0]
    """)
    
    counts_path = args['<counts>']
    vectors_path = args['<output_path>']
    cds = float(args['--cds'])
    
    
    o = open(counts_path + '-new',"w")
    for line in open(counts_path):
        o.write(line.strip()+"\n")
    o.close()
    
    
    counts_path_new = counts_path + '-new'
    
    
    counts, iw, ic = read_counts_matrxi_fast(counts_path, counts_path_new)

    pmi = calc_pmi(counts, cds)

    save_matrix(vectors_path, pmi)
    save_vocabulary(vectors_path + '.words.vocab', iw)
    save_vocabulary(vectors_path + '.contexts.vocab', ic)
    
    savePmiNonzeroTerm_fast(counts,vectors_path + '.cooccurrence')
    

    remain_index = pmi.data > 1 

    
    pmi.data = np.log(pmi.data)
    savePmiNonzeroTerm_fast(pmi,vectors_path + '.PMI')
    
    counts.data = counts.data * remain_index
    counts.eliminate_zeros()
    savePmiNonzeroTerm_fast(counts,vectors_path + '.PPMIcooccurrence')

    

    pmi.data[pmi.data < 0] = 0
    pmi.eliminate_zeros()
    
    savePmiNonzeroTerm_fast(pmi,vectors_path + '.PPMI')


def read_counts_matrxi_fast(counts_path, counts_path_new):
    """
    Reads the counts into a sparse matrix (CSR) from the count-word-context textual format.
    """
    df = pd.read_csv(counts_path_new, sep=" ", names = ["num", "word", "context"],converters =  {"num": np.float32, "word": str, "context": str}, header=None)

    words = load_count_vocabulary(counts_path + '.words.vocab')#this is a dict, contains (word: how many this word appears) pair
    contexts = load_count_vocabulary(counts_path + '.contexts.vocab')

    words = list(words.keys())#is a list contains all the words
    contexts = list(contexts.keys())#is a list contains all the words

    iw = sorted(words)#this is a sorted words list
    ic = sorted(contexts)#this is a sorted context word list 


    wi=pd.Series(index=iw, data=sp.arange(len(iw)))#this should be a dictionary, word: index
    ci=pd.Series(index=ic, data=sp.arange(len(ic)))#this should be a dictionary, context word: index
    
    return csr_matrix((df.num, (wi[df.word], ci[df.context])), [len(iw),len(ic)], dtype=np.float32), list(iw), list(ic)



def read_counts_matrix(counts_path, counts_path_new):
    """
    Reads the counts into a sparse matrix (CSR) from the count-word-context textual format.
    """
    words = load_count_vocabulary(counts_path + '.words.vocab')
    contexts = load_count_vocabulary(counts_path + '.contexts.vocab')
    words = list(words.keys())
    contexts = list(contexts.keys())
    iw = sorted(words)
    ic = sorted(contexts)
    wi = dict([(w, i) for i, w in enumerate(iw)])
    ci = dict([(c, i) for i, c in enumerate(ic)])
    
    counts = csr_matrix((len(wi), len(ci)), dtype=np.float32)
    tmp_counts = dok_matrix((len(wi), len(ci)), dtype=np.float32)
    update_threshold = 100000
    i = 0
    with open(counts_path_new) as f:
        for line in f:
            count, word, context = line.strip().split()
            if word in wi and context in ci:
                tmp_counts[wi[word], ci[context]] = int(count)
            i += 1
            if i == update_threshold:
                counts = counts + tmp_counts.tocsr()
                tmp_counts = dok_matrix((len(wi), len(ci)), dtype=np.float32)
                i = 0
    counts = counts + tmp_counts.tocsr()
    
    return counts, iw, ic


def calc_pmi(counts, cds):
    """
    Calculates e^PMI; PMI without the log().
    """
    sum_w = np.array(counts.sum(axis=1))[:, 0]#this is a column number, all words' frequency
    sum_c = np.array(counts.sum(axis=0))[0, :]#this is a row rumber, all context words' frequency
    if cds != 1:
        sum_c = sum_c ** cds#for each context frequency, calculate its frequency's square
    sum_total = sum_c.sum()
    sum_w = np.reciprocal(sum_w)
    sum_c = np.reciprocal(sum_c)
    
    pmi = csr_matrix(counts, dtype=np.float32)
    
    pmi = multiply_by_rows(pmi, sum_w)#each row multiple same times
    pmi = multiply_by_columns(pmi, sum_c)#each column multiple same times
    pmi = pmi * sum_total
    
    return pmi


def multiply_by_rows(matrix, row_coefs):
    normalizer = dok_matrix((len(row_coefs), len(row_coefs)))
    normalizer.setdiag(row_coefs)
    return normalizer.tocsr().dot(matrix)


def multiply_by_columns(matrix, col_coefs):
    normalizer = dok_matrix((len(col_coefs), len(col_coefs)))
    normalizer.setdiag(col_coefs)
    return matrix.dot(normalizer.tocsr())

def savePmiNonzeroTerm_fast(csrmatrix,name):
    time1 = time.time()
    with open(name,'w') as f:
        for i in range(len(csrmatrix.indptr)-1):
            columnIndices=[]
            dataInLine=[]
            columnIndices=csrmatrix.indices[csrmatrix.indptr[i]:csrmatrix.indptr[i+1]] 
            dataInLine=csrmatrix.data[csrmatrix.indptr[i]:csrmatrix.indptr[i+1]]
            for j in range(len(columnIndices)):
                f.write("%d %d %.6f\n"% (i+1, columnIndices[j]+1, dataInLine[j]))         

def savePmiNonzeroTerm(csrmatrix,name):
    dokmatrix = csrmatrix.todok()
    coordinate = dokmatrix.keys()
    count = dokmatrix.values()
    
    with open(name, 'w') as f:
        for i in range(len(coordinate)):
            f.write("%d %d %.6f\n"% ((coordinate[i][0]+1), (coordinate[i][1]+1), count[i]))

        
    
def normalize(matrix):
    m2 = matrix.copy()
    m2.data **= 2
    norm = np.reciprocal(np.sqrt(np.array(m2.sum(axis=1))[:, 0]))
    normalizer = dok_matrix((len(norm), len(norm)))
    normalizer.setdiag(norm)
    return normalizer.tocsr().dot(matrix)


if __name__ == '__main__':
    main()
