import requests

lfw_pair = 'pairs/lfw.txt'
calfw_pair = 'pairs/calfw.txt'
cplfw_pair = 'pairs/cplfw.txt'

lfw_dataset = 'dataset/lfw/images/'
calfw_dataset = 'dataset/calfw/aligned images/'
cplfw_dataset = 'dataset/cplfw/aligned images/'

pair_path = [lfw_pair, calfw_pair, cplfw_pair]
dataset_path = [lfw_dataset,calfw_dataset, cplfw_dataset]

accuracy = 0
threshold = 30

FMR = 0
FNMR = 0

for index, pair in (pair_path):
    with open(pair, 'r') as myfile:
        for line in myfile:
            columns = line.split()
            dist = requests.post("http://localhost:8000/face-recognition",
                                    files={"image1" : open(f"{dataset_path[index]}/{columns[0]}", "rb"), 
                                    "image2" : open(f"{dataset_path[index]}/{columns[0]}", "rb")})
            
            if dist.text < f"{threshold}":
                if columns[2] == "True":
                    accuracy += 1
                else:
                    FNMR += 1    
                
            else:
                if columns[2] == "False":
                    accuracy += 1
                else: 
                    FMR +=1 

            print(dist)    

    # false_match_rate = (FMR / 2000) * 100
    # false_not_match_rate = (FNMR / 2000) * 100

    # print("FMR : ", false_match_rate )
    # print("FNMR : ", false_not_match_rate)

acc = (accuracy / 6000) * 100
print("Accuracy : ", acc)

