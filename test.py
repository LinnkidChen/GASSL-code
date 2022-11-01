# """
# author: Chen Tong
# email: linnkid.chen@gmail.com
# """

import os
from time import sleep

# datasets = ["MUTAG", "PTC-MR", "IMDB-BINARY", "IMDB-MULTI", "COLLAB", "NCI1"]
datasets = ["PTC_MR", "IMDB-BINARY", "IMDB-MULTI", "COLLAB", "NCI1"]
OPT1s = ["gin", "gcn"]
OPT2s = ["312", "888"]
OPT3s = ["H", "X"]


for OPT1 in OPT1s:
    for OPT2 in OPT2s:
        for OPT3 in OPT3s:
            for dataset in datasets:
                command = "python /root/GASSL_Supplementary_Material/main.py --dataset "+dataset+" --gnn " + OPT1 + \
                    " --m 3 --seed "+OPT2+" --lr 1e-4 --pp "+OPT3 + \
                    " --device 0 > results/"+dataset+'_'+OPT1+'_'+OPT2+'_'+OPT3 + \
                    '.log 2> results/'+dataset+'_'+OPT1+'_'+OPT2+'_'+OPT3+'_err.log'
                print(command, 'BEGIN')
                os.system(command)
                sleep(5)
                print(command, 'DONE')

# python test.py > test.log 2>&1
