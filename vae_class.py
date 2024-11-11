import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data import DataLoader

class VAE(nn.Module):

    def __init__(self, input_dim,
                  output_dim ,
                  hidden_dim=128, 
                  latent_dim=10,
                  likelihood= 'NB', 
                  var_eps = 1e-4,
                  dropout_rate = 0.01,
                  device=device):
        super(VAE, self).__init__()
        self.var_eps = var_eps

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_rate),
            nn.ReLU(),
            )
        
        # latent mean and variance 
        self.z_mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.z_logvar_layer = nn.Linear(hidden_dim, latent_dim)
        
        #l_encoder
        self.l_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
            )
        
        # latent mean and variance 
        self.l_mean_layer = nn.Linear(hidden_dim, 1)
        self.l_logvar_layer = nn.Linear(hidden_dim, 1)
        
        # decoder
        self.intermediate_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Dropout(p=dropout_rate),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=dropout_rate),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
            )
        
        self.scale_decoder = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
            )
        
        self.dispersion_decoder = nn.Linear(hidden_dim, output_dim)
        self.dropout_decoder = nn.Linear(hidden_dim, output_dim)
        
    # def encode(self, x):
    #     x = self.encoder(x)
    #     z_mean, z_logvar = self.z_mean_layer(x), self.z_logvar_layer(x)
        
    #     x = self.encoder(x)
    #     l_mean, l_logvar = self.l_mean_layer(x), self.l_logvar_layer(x)
    #     return z_mean, z_logvar, l_mean, l_logvar

    def reparameterisation_z(self, z_mean, z_logvar):
        z_std = torch.exp(0.5*z_logvar) + torch.tensor(self.var_eps).sqrt()
        # print(z_std.shape)
        # print(torch.tensor(self.var_eps))
        epsilon = torch.randn_like(z_std).to(device)      
        z = z_mean + z_std * epsilon
        return z
    
    def reparameterisation_l(self, l_mean, l_logvar):
        l_std = torch.exp(0.5*l_logvar) + torch.tensor(self.var_eps).sqrt()
        epsilon = torch.randn_like(l_std).to(device)      
        logl = l_mean + l_std * epsilon
        return logl

    def decode(self, z, library, dispersion ): #library may be wrong
        px = self.intermediate_decoder(z)   #fw
        px_scale = self.scale_decoder(px)   # w_ng
        px_rate = torch.exp(library) * px_scale #w_ng * l
        
        px_dispersion = self.dispersion_decoder(px)
        px_dropout = self.dropout_decoder(px) #fh
    
        return px_scale, px_rate, px_dispersion, px_dropout

    def forward(self, x):
        enc = self.encoder(x)
        z_mean  = self.z_mean_layer(enc)
        z_logvar = self.z_logvar_layer(enc)
        z = self.reparameterisation_z(z_mean, z_logvar)

        l_enc = self.l_encoder(x)
        l_mean, l_logvar = self.l_mean_layer(l_enc), self.l_logvar_layer(l_enc)
        logl = self.reparameterisation_l(l_mean, l_logvar)

        px_scale, px_rate, px_dispersion, px_dropout = self.decode(z, logl, dispersion=1)
        return  {'px_scale':px_scale, 
                 'px_rate':px_rate, 
                 'px_dispersion':px_dispersion, 
                 'px_dropout':px_dropout, 
                 'z_mean':z_mean, 
                 'z_logvar':z_logvar, 
                 'l_mean': l_mean, 
                 'l_logvar': l_logvar
                }       

def loss_fn(xi, decode_output, gene_likelihood = False, kl_weight=0.1,max_rate = 1e8):
    z_mean, z_logvar = decode_output['z_mean'], decode_output['z_logvar']
    z_KLD = -0.5* torch.sum(1 + z_logvar - z_mean.pow(2)-torch.exp(z_logvar))

    l_mean, l_logvar = decode_output['l_mean'], decode_output['l_logvar']
    l_KLD = -0.5* torch.sum(1 + l_logvar - l_mean.pow(2)-torch.exp(l_logvar))
    
    recon_loss  = -1*torch.distributions.Poisson(rate  = torch.clamp(decode_output['px_rate'],max = max_rate)).log_prob(xi).sum(dim = -1)
    return torch.mean(recon_loss + kl_weight*z_KLD + l_KLD)

def training_step(epochs, model, train_loader, optimizer, device = device):

    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        for batch_idx, (data,) in enumerate(train_loader):
            
            data = data.to(device)
            
            optimizer.zero_grad()  # Zero the gradients
            
            output = model(data)  # Forward pass
            loss = loss_fn(data, output)
            
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch+1} Loss: {loss.item():.4f}")
    pass
# def training_step(self, batch, batch_idx):
            
#     x,_ = batch
#     #data = data.to(device)
#     output = self.forward(x)
#     loss = loss_fn(x, output)
    
#     self.log('train_loss',loss.detach(),on_step = True,on_epoch = True,prog_bar = True)

#     return loss
# def validation_step(self, batch, batch_idx):
#     x,_ = batch
#     output = self.forward(x)
#     loss = loss_fn(x, output)

#     self.log('val_loss',loss.detach(),on_step = True,on_epoch = True)
#     return loss

# def test_step(self, batch, batch_idx):
#         return self.validation_step(batch, batch_idx)

# def train_loader(self):

# def val_loader(self):

# def test_loader(self):
