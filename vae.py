import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
The encoder or the recognition model is the density function q_φ(z|x) which is trying to estimate the posterior function
p_θ(z|x).

The generative model, aims to compute p(x,z) given that we specify the decoder model p_θ(x|z) and a prior p_θ(z)
ß
Once we have an estimate of p(x,z) using a DLVM and we have defined q_φ(z|x), we can use Bayes law to get an estimate of 
p_θ(x)=p_θ(x,z)/q_φ(z|x)

In order to tune the parameters we find the log likelihood of p_θ(x). We can show that maximising the log likelihood is equivalent to maximising the ELBO function:
E_p(eps)[log p_θ(x,z)-q_φ(z|x)]. 

To use SGD we find the derivatives by using the reparamaterisation trick. Saying that z=g(φ,x,eps) and that eps is drawn randomly from a prob distribution.
Doing this allows us to draw zs and find the log likelihood MC estimate. We minimise this over SGD.




We are using factorised Gaussian Posteriors model, where we have
that q_φ(z|x)=N(z|mu,diag(sigma^2))

repar we get
epsilon ~ N(o,I)
(mu,log sigma)=NN_φ(x)
z=mu+sigma x epsilon
'''


class Encoder(nn.Module): 
    ''' the recognition model, the estimate q_φ(z|x) of the posterior function p_θ(z|x) and the reparamaterisation trick'''
    def __init__(self, input_dim, z_dim, hidden_dims,dropout):
        super(Encoder,self).__init__()
        
        self.hidden1 = nn.Linear(input_dim, hidden_dims[0])
        self.hidden2=nn.Linear(hidden_dims[0],hidden_dims[1]) 
        self.nn_mu = nn.Linear(hidden_dims[1], z_dim)
        self.nn_log_sigma = nn.Linear(hidden_dims[1], z_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.tanh=nn.Tanh()
        self.relu=nn.ReLU()
        self.softplus=nn.Softplus()

    def reparameterization(self,mean,sigma): #this adds the probabalistic element to the function
        epsilon=torch.randn_like(sigma) 
        z=mean+sigma*epsilon 
        return z 

    def forward(self,x):
        hidden_layer1=(self.relu(self.hidden1(x)))
        hidden_layer2=(self.relu(self.hidden2(hidden_layer1)))

        z_mu = self.nn_mu(hidden_layer2)
        z_log_sigma = self.softplus(self.nn_log_sigma(hidden_layer2))

        z=self.reparameterization(z_mu,torch.exp(z_log_sigma)**2)

        return z, z_mu, z_log_sigma

'''
The genaritive distribution will use the decoder p_θ(x|z) and the prior draws taken from the reparamaterised step, to compute the joint distribution

'''

class Decoder(nn.Module): #likelihood function p(x given z)
    def __init__(self,z_dim, hidden_dims, output_dim,dropout):
        super(Decoder,self).__init__()
        self.hidden1 = nn.Linear(z_dim, hidden_dims[0])
        self.hidden2=nn.Linear(hidden_dims[0],hidden_dims[1]) #changed from linear
        self.nn_out = nn.Linear(hidden_dims[1], output_dim)

        self.dropout = nn.Dropout(p=dropout)


        self.tanh=nn.Tanh()
        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()
    def forward(self,z):
        layer1=(self.relu(self.hidden1(z)))
        layer2=(self.relu(self.hidden2(layer1))) # z --> nn fully connected --> softplus activation --> hidden
        x_recon=self.sigmoid(self.nn_out(layer2))  #hidden --> nn fully connected --> sigmoid --> reconstructed spectrum
        return x_recon



class VAE(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(VAE,self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder


    def forward(self,x):
        z,mean,log_var=self.Encoder(x)
        x_recon=self.Decoder(z)
        return x_recon, mean, log_var, z

'''

Now we calcualte the loss function

'''


BCE_loss = nn.BCELoss()

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')  
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp()) # is this right, my dimensionalty could be off
    kldweight=1.0
    return reproduction_loss + KLD*kldweight, KLD

def model_train(vae_spec,batch_size,optimizer,model,loss_function,epochs):
    z_an=[]
    kld=[] #this is done for analysis of posterior collapse
    for epoch in range(epochs):
        
        overall_loss = 0
        for batch_idx, x in enumerate(vae_spec):

            x = x.view(batch_size, len(x[0]))
            #x = x.to(DEVICE)

            optimizer.zero_grad()

            x_hat, mean, log_var, z = model(x)
            loss, KLD = loss_function(x, x_hat, mean, log_var)
            
            overall_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        z_an.append(z)   
        kld.append(KLD) 
                
        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
        print("Overall Loss: ", overall_loss)
        print("KLD Loss: ", KLD)
    print("Finish!!")
    return z_an,kld



