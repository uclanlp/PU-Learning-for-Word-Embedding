import os
from docopt import docopt

def main():
    args = docopt("""
    Usage: 
        extract_model_file_name.py <output_file_name>
    """)

    a = os.listdir('./')
    b = [ai for ai in a if (ai.find('words') != -1 and ai.find('final') != -1)]
    c = [ai for ai in a if (ai.find('words') != -1 and ai.find('final') == -1 and ai.find('swp') == -1)]
    name_list = []
    for i in range(len(b)):
        name_list.append(b[i][:-len(".words")])
    for i in range(len(c)):
        name_list.append(c[i][:-len(".words")])

    with open('./model_name', 'w') as f:
        for ci in name_list:
            f.write(ci+'\n')

if __name__ == '__main__':
    main()