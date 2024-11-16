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

        # Get h from x : the reprensentation in the hidden dim of x
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            
            
        )
        
        #from  h to Pi_phi(x) : parameters of q(y|x)
        self.y_logits = nn.Linear(hidden_dim, num_classes)
        
        # mean and variance in q(z|x,y) : from h and y to z
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
        logits_y = self.y_logits(h)
        y = F.softmax(logits_y) 
        
        return h, y
        
    def decode(self, z):
        return self.decoder(z)


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    
    def forward(self, x):
        
        h, y = self.encode(x)
          
        # variable (h,y) equivalent to (x,y)
        h_y = torch.cat([h, y], dim=-1)

        
        mu_z = self.mu_z(h_y)
        log_var_z = self.log_var_z(h_y)
        
        
        z = self.reparameterize(mu_z, log_var_z)
     
        x_recon = self.decode(z)
        loss_elbo, recon_loss, kl_y, kl_z = self.elbo(x, x_recon, y, mu_z, log_var_z)
        
        return x_recon, elbo, recon_loss, kl_y, kl_z

    def elbo(self, x, x_recon, y, mu_z, log_var_z):
       
        
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction="sum")
        
        # KL divergence pour y : q(y|x) vs p(y) 
        prior_y = torch.ones_like(y) / self.num_classes  # Prior uniforme
        kl_y = torch.sum(y * torch.log(y / prior_y), dim=-1).sum()

        # KL divergence pour z : q(z|x, y) vs p(z|y)
        prior_z_y = Normal(torch.zeros_like(mu_z), torch.ones_like(mu_z))  
        z = Normal(mu_z, torch.exp(0.5 * log_var_z))  
        kl_z = kl_divergence(z, prior_z_y).sum()

        # ELBO = Reconstruction + KL divergences
        elbo = - recon_loss + kl_y + kl_z
        loss_elbo = -elbo #max elbo = min loss
        return loss_elbo, recon_loss, kl_y, kl_z
    
    def train(model, data_loader, optimizer, epochs=10):
        model.train()
        for epoch in range(epochs):
            total_elbo = 0
            for x, _ in data_loader:
                x = x.view(-1, input_dim)  # Mise en forme des données d'entrée
                optimizer.zero_grad()
                _, loss_elbo, recon_loss, kl_y, kl_z = model(x)
                elbo.backward()
                optimizer.step()
                total_elbo += elbo.item()
            print(f"Epoch {epoch + 1}, ELBO: {-total_elbo:.2f}")
    
