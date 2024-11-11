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
                  device=device,
                  num_classes = 10, 
                  beta=1.0):
        super(VAE, self).__init__()
        self.var_eps = var_eps
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.beta = beta

        # encoder of x : parameters of q(z|x,y)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        #parameters of q(y|x)
        self.q_y_logits = nn.Linear(hidden_dim, num_classes)
        
        # mean and variance in q(z|x,y) 
        self.mu_z = nn.Linear(hidden_dim + num_classes, latent_dim)
        self.log_var_z = nn.Linear(hidden_dim + num_classes, latent_dim)
        
        # decoder from z to x
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  
        )
     
    def encode(self, x):
        h = self.encoder(x)
        logits_y = self.q_y_logits(h)
        q_y = Categorical(logits=logits_y)  # q(y|x)
        return h, q_y
        
    def decode(self, z):
        return self.decoder(z)


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    
    def forward(self, x):
        # Encode x pour obtenir h et q(y|x)
        h, q_y = self.encode(x)
        
        # sampling y from q(y|x)
        y = q_y.sample() 
        y_one_hot = F.one_hot(y, self.num_classes).float()
        h_y = torch.cat([h, y_one_hot], dim=-1)

        # Parameters from q(z|x, y)
        mu_z = self.mu_z(h_y)
        log_var_z = self.log_var_z(h_y)
        
        # sampling z from q(z|x, y) with reparametrization
        z = self.reparameterize(mu_z, log_var_z)
     
        x_recon = self.decode(z)
        elbo, recon_loss, kl_y, kl_z = self.elbo(x, x_recon, q_y, y, mu_z, log_var_z)
        
        return x_recon, elbo, recon_loss, kl_y, kl_z

    def elbo(self, x, x_recon, q_y, y, mu_z, log_var_z):
       
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction="sum")
        
        # KL divergence  y 
        prior_y = Categorical(logits=torch.ones_like(q_y.logits))
        kl_y = kl_divergence(q_y, prior_y).sum()

        # KL divergence z
        prior_z = Normal(torch.zeros_like(mu_z), torch.ones_like(mu_z))
        q_z = Normal(mu_z, torch.exp(0.5 * log_var_z))
        kl_z = kl_divergence(q_z, prior_z).sum()

        elbo = recon_loss + self.beta * (kl_y + kl_z)
        
        return elbo, recon_loss, kl_y, kl_z
    
    def train(model, data_loader, optimizer, epochs=10):
        model.train()
        for epoch in range(epochs):
            total_elbo = 0
            for x, _ in data_loader:
                x = x.view(-1, input_dim)  # Mise en forme des données d'entrée
                optimizer.zero_grad()
                _, elbo, recon_loss, kl_y, kl_z = model(x)
                elbo.backward()
                optimizer.step()
                total_elbo += elbo.item()
            print(f"Epoch {epoch + 1}, ELBO: {-total_elbo:.2f}")
    
