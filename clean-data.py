f = open("./mylog1.log", "r")
datasets = ['MUTAG', 'PTC_MR', 'IMDB-BINARY',
            'IMDB-MULTI',
            'COLLAB',
            'NCI1']

GNNs = ['gin', 'gcn']

best_results = {}

for lineID, line in enumerate(f):
    # print(lineID, line)
    # break
    if lineID % 2 == 0:  # model detail
        datas = line.split('-')
        key = tuple([datas[0], datas[1]])
    else:
        # accuracies and std
        datas = line.replace(":", " ")
        datas = datas.split()
        value = [datas[1], datas[3]]
        if(key not in best_results.keys()):
            best_results[key] = value
        elif(best_results[key][0] < value[0]
             or (best_results[key][0] == value[0] and best_results[key][1] > value[1])):
            best_results[key] = value

with open("cleand-data.txt", "w+") as out:
    for key in best_results.keys():
        out.write(
            f"{key[0]} , {key[1]} , accuracy= {best_results[key][0]} ± {best_results[key][1]}\n")
        # print(key, value)
