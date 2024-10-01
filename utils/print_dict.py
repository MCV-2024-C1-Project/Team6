"""
Usage:
  print_dict.py <dictName> [--iPosition=<ip>]
  print_dict.py -h | --help
Options:
  --iposition=<ip>      Index of a single element. Only this will be printed [default: None]
"""

from docopt import docopt
import pickle


if __name__ == "__main__":

    # read args
    args = docopt(__doc__)
    dict_name = args['<dictName>']
    ip        = args['--iPosition']

    with open(dict_name, 'rb') as fd:
        dict = pickle.load(fd)


    if ip == None:
        print (dict)
        print ('Dictionary contains {} elements'.format(len(dict)))
    else:
        ip = int(ip)
        print (dict[ip])
        
