import torch
from torchvision.utils import save_image, make_grid
import time
import math
import numpy as np

from model import *
from dataloader import *
from evaluator import evaluation_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.0002
epochs = 1000
batch_size = 64
latent_size = 40
w_loss = 0.5

def train(model_g, model_d, optimizer_g, optimizer_d, criterion_d, criterion_c, loader):
	model_g.train()
	model_d.train()
	d_d_loss, d_c_loss, g_d_loss, g_c_loss = 0, 0, 0 ,0
	for idx, (real_img, real_c) in enumerate(loader):
		data_size = real_img.size(0)

		model_d_d_loss, model_d_c_loss = TrainDiscriminator(real_img, real_c, data_size)
		d_loss, c_loss = TrainGenerator(data_size)

		d_d_loss += model_d_d_loss.item()
		d_c_loss += model_d_c_loss.item()
		g_d_loss += d_loss.item()
		g_c_loss += c_loss.item()

	d_d_loss /= len(loader)
	d_c_loss /= len(loader)
	g_d_loss /= len(loader)
	g_c_loss /= len(loader)

	print('Loss: {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(d_d_loss, d_c_loss, g_d_loss, g_c_loss))

def TrainDiscriminator(real_img, real_c, data_size):
    model_d.zero_grad()

    # real
    real_img, real_c = real_img.to(device), real_c.to(device)
    real_label = torch.ones(data_size, requires_grad=False).to(device)

    # fake
    latent = torch.randn(data_size, latent_size, 1, 1).to(device)
    fake_c = torch.FloatTensor(generate_condition(data_size)).to(device)
    fake_img = model_g(latent, fake_c)
    fake_label = torch.zeros(data_size, requires_grad=False).to(device)

    real_score, real_class = model_d(real_img, real_c)
    fake_score, fake_class = model_d(fake_img, fake_c)

    real_d_loss = criterion_d(real_score, real_label)
    fake_d_loss = criterion_d(fake_score, fake_label)

    real_c_loss = criterion_c(real_class, real_c)
    fake_c_loss = criterion_c(fake_class, fake_c)

    model_d_d_loss = (real_d_loss + fake_d_loss) / 2
    model_d_c_loss = (real_c_loss + fake_c_loss) / 2

    model_d_loss = model_d_d_loss + w_loss * model_d_c_loss
    model_d_loss.backward()
    optimizer_d.step()

    return model_d_d_loss, model_d_c_loss

def TrainGenerator(data_size):
    model_g.zero_grad()

    # fake
    z = torch.randn(data_size, latent_size, 1, 1).to(device)
    fake_c = torch.FloatTensor(generate_condition(data_size)).to(device)
    fake_img = model_g(z, fake_c)
    fake_score, fake_class = model_d(fake_img, fake_c)
    real_label = torch.ones(data_size, requires_grad=False).to(device)

    d_loss = criterion_d(fake_score, real_label)
    c_loss = criterion_c(fake_class, fake_c)

    model_g_loss = d_loss + w_loss * c_loss
    model_g_loss.backward()
    optimizer_g.step()

    return d_loss, c_loss

def evaluate(test_dataset, eval_model, model_g, epoch):
	model_g.eval()

	latent = torch.randn(32, latent_size, 1, 1).to(device)
	condition = torch.FloatTensor(test_dataset).to(device)
	with torch.no_grad():
		out = model_g(latent, condition)

	score = eval_model.eval(out, condition)
	img = denormalize(out)
	img = make_grid(img)
	save_image(img, './data/result/img_{}.png'.format(epoch))
	
	print('Score: {}'.format(score))
	return score

def weight_init(m):
    if isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

def generate_condition(size):
	cond_list = np.zeros((size, 24))
	for i in range(size):
		rand_class = np.random.randint(0, 24, np.random.randint(1, 4))
		for j in rand_class:
			cond_list[i][j] = 1.0
	return cond_list

def denormalize(img): # (-1, 1) -> (0, 1)
	mean = torch.Tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).expand(img.size(0), 3, 64, 64).cuda()
	std = torch.Tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).expand(img.size(0), 3, 64, 64).cuda()
	return ((img * std) + mean)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

if __name__ == '__main__':

    train_dataset = DataLoader('./data_property/train.json')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_dataset = TestData('./data_property/test.json')

    eval_model = evaluation_model()

    model_g = Generator().to(device)
    model_g.apply(weight_init)

    model_d = Discriminator().to(device)
    model_d.apply(weight_init)

    criterion_d = torch.nn.BCELoss()
    criterion_c = torch.nn.BCELoss()
    optimizer_g = torch.optim.Adam(model_g.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(model_d.parameters(), lr=lr, betas=(0.5, 0.999))

    start = time.time()
    best_score = 0
    for epoch in range(epochs):
        print('Epoch {}:'.format(epoch))
        train(model_g, model_d, optimizer_g, optimizer_d, criterion_d, criterion_c, train_loader)
        score = evaluate(test_dataset, eval_model, model_g, epoch)
        print('Time: {}\n'.format(timeSince(start, (epoch+1) / epochs)))

        if score >= best_score:
            print('best score: ', score)
            best_score = score
            torch.save({
                'score': score,
                'model_g': model_g.state_dict(),
                'model_d': model_d.state_dict(),
            }, './data/result/model.tar')
        if score >= 0.7:
            torch.save({
                'score': score,
                'model_g': model_g.state_dict(),
                'model_d': model_d.state_dict(),
            }, './data/result/model_{:3f}.tar'.format(score))
    
    print('final best acc: ', best_score)