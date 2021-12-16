import json
import torch
from torch.utils import data
import torchvision.transforms as transforms
from PIL import Image

class DataLoader(data.Dataset):
    def __init__(self, path):
    	self.dataset = TrainData(path)

    def __len__(self):
    	return len(self.dataset)

    def __getitem__(self, index):
    	path = './data/training_data/' + self.dataset[index]['img_name']
    	img = Image.open(path).convert('RGB')
    	
    	transform = transforms.Compose([
			transforms.Resize((64, 64)),
 			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    	img = transform(img)

    	condition = self.dataset[index]['condition']
    	condition = torch.FloatTensor(condition)

    	return img, condition


def TrainData(path):
    training_data = []
    with open(path, 'r') as f:
        data = json.load(f)
        for key, value in data.items():
            training_data.append({'img_name': key, 'condition': OneHot(value)})
    return training_data

def TestData(path):
    testing_data = []
    with open(path, 'r') as f:
        condition = json.load(f)
        for i in condition:
            testing_data.append(OneHot(i)) 
    return testing_data

def OneHot(obj_list):
	with open('./data_property/objects.json', 'r') as reader:
		mapping_list = json.loads(reader.read())

	object_OneHot = [0]*24
	for obj in obj_list:
		object_OneHot[mapping_list[obj]] = 1
	return object_OneHot
