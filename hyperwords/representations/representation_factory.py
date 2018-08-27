from embedding import SVDEmbedding, EnsembleEmbedding, ConcatenateEmbedding, PMFEmbedding, Embedding
from explicit import PositiveExplicit


def create_representation(args):
    rep_type = args['<representation>']
    path = args['<representation_path>']
    
    neg = int(args['--neg'])
    w_c = args['--w+c']
    eig = float(args['--eig'])
    concatenate = args['--concatenate']
    contexts = args['--contexts']
    
    if rep_type == 'PPMI':
        if w_c:
            raise Exception('w+c is not implemented for PPMI.')
        else:
            return PositiveExplicit(path, True, neg)
        
    elif rep_type == 'SVD':
        if w_c:
            return EnsembleEmbedding(SVDEmbedding(path, False, eig, False), SVDEmbedding(path, False, eig, True), True)
        else:
            return SVDEmbedding(path, True, eig)
        
    elif rep_type == 'PMF':
        return PMFEmbedding(path, True)
        
    else:
        if w_c:
            return EnsembleEmbedding(Embedding(path + '.words', False), Embedding(path + '.contexts', False), True)
        elif concatenate:
            return ConcatenateEmbedding(Embedding(path + '.words', False), Embedding(path + '.contexts', False), True)
        elif contexts:
            return Embedding(path + '.contexts', True)        
        else:
            return Embedding(path + '.words', True)
