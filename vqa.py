import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import object_classification as obj_cls
import random

#Reasonnet
class NNet(nn.Module):
	def __init__(self):
		super(NNet, self).__init__()
		# encoder layers
		self.fc1 = nn.Linear(84, 40)
		self.fc2 = nn.Linear(40, 60)
		# Bilinear layer
		self.bil = nn.Bilinear(60,60,60)
		# answer classification network
		self.fc3 = nn.Linear(60, 15)      
		self.fc4 = nn.Linear(15, 10)

	def forward(self, x):
		x = F.tanh(self.fc1(x))
		x = F.tanh(self.fc2(x))
		x=  self.bil(x,que_rep)
		x = F.tanh(self.fc3(x))
		x = F.tanh(self.fc4(x))
		return x

def get_layer(x,word_idx):
	x[word_idx] = 1.0
	return x


def tokenize_corpus(corpus):
	tokens = [x.split() for x in corpus]
	return tokens

def questions(predicted):
	print ("Which object is this?")
	print(classes[predicted])
	predicted = int(predicted)
	x = random.randint(0,9)
	if x == 0:
		print("Is this plane?")
		if(predicted==0):
			print("YES")
		else:
			print("NO")
	if x == 1:
		print("Is this car?")
		if(predicted==1):
			print("YES")
		else:
			print("NO")
	if x == 2:
		print("Is this bird?")
		if(predicted==2):
			print("YES")
		else:
			print("NO")
	if x == 3:
		print("Is this cat?")
		if(predicted==3):
			print("YES")
		else:
			print("NO")
	if x == 4:
		print("Is this deer?")
		if(predicted==4):
			print("YES")
		else:
			print("NO")
	if x == 5:
		print("Is this dog?")
		if(predicted==5):
			print("YES")
		else:
			print("NO")
	if x == 6:
		print("Is this frog?")
		if(predicted==6):
			print("YES")
		else:
			print("NO")
	if x == 7:
		print("Is this horse?")
		if(predicted==7):
			print("YES")
		else:
			print("NO")
	if x == 8:
		print("Is this ship?")
		if(predicted==8):
			print("YES")
		else:
			print("NO")
	if x == 9:
		print("Is this truck?")
		if(predicted==9):
			print("YES")
		else:
			print("NO")
	



#neural net for object classification
if __name__=='__main__':

	net = obj_cls.Net()


	transform = transforms.Compose(
		[transforms.ToTensor(),
		 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	#load the data
	trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
												download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
												  shuffle=True, num_workers=2)
	testset = torchvision.datasets.CIFAR10(root='./data', train=False,
										   download=True, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=1,
											 shuffle=False, num_workers=2)

	#load pretrained weights
	net.load_state_dict(torch.load('object_pretrained_weights'))

	#load questions
	with open('questions.txt') as f:
			corpus = f.read().splitlines()
	tokenized_corpus = tokenize_corpus(corpus)
	for tokens in tokenized_corpus:
		ques_words = [word for word in tokens if word.isalnum()]
	vocabulary = []
	for sentence in tokenized_corpus:
		for token in sentence:
			if token not in vocabulary:
				vocabulary.append(token)

	word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
	idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

	vocabulary_size = len(vocabulary)



	wordtovec_weights=torch.load('w1')
	test_corpus = tokenize_corpus(corpus)
	lr = 0.001
	epoch_count = 2
	nnet = NNet()
	criterion = nn.CrossEntropyLoss()
	que_rep = 0

	#train the model

	for epoch in range(1,epoch_count):
	    running_loss = 0.0
	    optimizer = optim.SGD(net.parameters(), lr= lr,weight_decay = lr/epoch,momentum=0.9, nesterov = False)
	    for data in trainloader:
	        images, labels = data
	        labels = Variable(labels)
	        output = net(Variable(images))
	        for tokens in test_corpus:
	            ques_words = [word for word in tokens if word.isalnum()]
	            x = torch.zeros(vocabulary_size).float()
				
	            # retrieving index
	            for word in ques_words:
	                index=word2idx.get(word)        
	                inp = Variable(get_layer(x,index)).float()

				
	            #question representation
	            que_rep = torch.matmul(wordtovec_weights,inp) 

	            optimizer.zero_grad()
	            module_output = nnet(output)
				
	            loss = criterion(module_output, labels)

	            loss.backward()
	            optimizer.step()

	            # print statistics
	            running_loss += loss.data[0]
	            if epoch % 2000 == 1999:    # print every 2000 mini-batches
	                print('[%d, %5d] loss: %.3f' %
	                      (epoch + 1, i + 1, running_loss / 2000))
	                running_loss = 0.0

	print('Finished Training')            
	classes = ('plane', 'car', 'bird', 'cat',
			   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	correct = 0
	total = 0
	count = 1
	# i = 0
	for data in testloader:

		images, labels = data
		outputs = nnet(Variable(images))
		#outputs = net(images)
		_, predicted = torch.max(outputs.data, 1)
		
		c = classes[int(labels)]+'.png'
		torchvision.utils.save_image(images,c)
		print("Actual Image: %s " %(classes[int(labels)]))
		predicted = int(predicted)
		questions(predicted)
		# if i < 4:
		# 	print("\n")
		# i = i+1
		# if i >= 5:
		# 	break

		