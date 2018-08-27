import os
import numpy as np
from docopt import docopt

def main():
    args = docopt("""
    Usage: 
        restore_matrix_to_npy_file.py <output_file_path>
    """)

    matrix = np.loadtxt(args['<output_file_path>']+'.words')
    np.save(args['<output_file_path>'] +'.words', matrix)
    
    matrix = np.loadtxt(args['<output_file_path>']+'.contexts')
    np.save(args['<output_file_path>'] +'.contexts', matrix)

if __name__ == '__main__':
    main()