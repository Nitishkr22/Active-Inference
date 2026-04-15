import torch
import torch.nn as nn


# class Encoder(nn.Module):
#     def __init__(self, latent_dim: int = 32):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),  # 64 -> 32. # stride is shift of the kernel, padding is how many pixels to pad on each side of the input
#             nn.ReLU(),
#             nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1), # 32 -> 16
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 16 -> 8
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 8 -> 4
#             nn.ReLU(),
#         )
#         self.fc = nn.Linear(128 * 4 * 4, latent_dim)

#     def forward(self, x):
#         h = self.net(x)
#         h = h.view(h.size(0), -1) # flatten the feature maps into a vector of size [batch_size, 128*4*4]
#         z = self.fc(h) # project the flattened features into the latent space of size [batch_size, latent_dim]
#         return z

## change encoder for configurable input channels (e.g. for obs_stack with 2 channels) ##
class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 32, in_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=4, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),            # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),            # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),           # 8 -> 4
            nn.ReLU(),
        )
        self.fc = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.net(x)
        h = h.view(h.size(0), -1)
        z = self.fc(h)
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4) # project the latent vector back to the feature map space of size [batch_size, 128*4*4]
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 4 -> 8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),   # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),    # 32 -> 64
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(z.size(0), 128, 4, 4)
        x_recon = self.net(h)
        return x_recon


# class AutoEncoder(nn.Module):
#     def __init__(self, latent_dim: int = 32):
#         super().__init__()
#         self.encoder = Encoder(latent_dim=latent_dim)
#         self.decoder = Decoder(latent_dim=latent_dim)

#     def forward(self, x):
#         z = self.encoder(x)
#         x_recon = self.decoder(z)
#         return x_recon, z
    
## change autoencoder for configurable input channels (e.g. for obs_stack with 2 channels) ##
class AutoEncoder(nn.Module):
    def __init__(self, latent_dim: int = 32, in_channels: int = 1):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim, in_channels=in_channels)
        self.decoder = Decoder(latent_dim=latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

## dynamics model ##
class DynamicsModel(nn.Module):
    def __init__(self, latent_dim: int = 32, num_actions: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.num_actions = num_actions

    def forward(self, z, actions):
        # actions: [B] integer action ids
        action_onehot = torch.nn.functional.one_hot(actions, num_classes=self.num_actions).float()
        x = torch.cat([z, action_onehot], dim=-1)
        z_next_pred = self.net(x)
        return z_next_pred
    
    ## joint trainig ##

# class WorldModel(nn.Module):
#     def __init__(self, latent_dim: int = 32, num_actions: int = 4):
#         super().__init__()
#         self.encoder = Encoder(latent_dim=latent_dim)
#         self.decoder = Decoder(latent_dim=latent_dim)
#         self.dynamics = DynamicsModel(latent_dim=latent_dim, num_actions=num_actions)

#     def forward(self, obs, actions, next_obs=None):
#         z = self.encoder(obs)
#         obs_recon = self.decoder(z)

#         z_next_pred = self.dynamics(z, actions)
#         next_obs_pred = self.decoder(z_next_pred)

#         out = {
#             "z": z,
#             "obs_recon": obs_recon,
#             "z_next_pred": z_next_pred,
#             "next_obs_pred": next_obs_pred,
#         }

#         if next_obs is not None:
#             z_next_true = self.encoder(next_obs)
#             out["z_next_true"] = z_next_true

#         return out

# update the world model for configurable input channels (e.g. for obs_stack with 2 channels) ##
class WorldModel(nn.Module):
    def __init__(self, latent_dim: int = 32, num_actions: int = 4, in_channels: int = 1):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim, in_channels=in_channels)
        self.decoder = Decoder(latent_dim=latent_dim)
        self.dynamics = DynamicsModel(latent_dim=latent_dim, num_actions=num_actions)

    def forward(self, obs, actions, next_obs=None):
        z = self.encoder(obs)
        obs_recon = self.decoder(z)

        z_next_pred = self.dynamics(z, actions)
        next_obs_pred = self.decoder(z_next_pred)

        out = {
            "z": z,
            "obs_recon": obs_recon,
            "z_next_pred": z_next_pred,
            "next_obs_pred": next_obs_pred,
        }

        if next_obs is not None:
            z_next_true = self.encoder(next_obs)
            out["z_next_true"] = z_next_true

        return out