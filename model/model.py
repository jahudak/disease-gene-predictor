import torch
from torch_geometric.nn import HeteroConv, SAGEConv, VGAE
from torch_geometric.nn.models.autoencoder import InnerProductDecoder


class HeteroVGAE(torch.nn.Module):
    def __init__(self, in_channels_disease, in_channels_gene, out_channels):
        super(HeteroVGAE, self).__init__()

        self.encoder = HeteroConv(
            {
                ("disease", "to", "gene"): SAGEConv(
                    [in_channels_disease, in_channels_gene],
                    out_channels,
                    # FXI: It doesnt like when the data is indexed from 0
                ),
                ("gene", "rev_to", "disease"): SAGEConv(
                    [in_channels_gene, in_channels_disease], out_channels
                ),
            },
            aggr="mean",
        )

        self.fc_mu_disease = torch.nn.Linear(out_channels, out_channels)
        self.fc_logvar_disease = torch.nn.Linear(out_channels, out_channels)
        self.fc_mu_gene = torch.nn.Linear(out_channels, out_channels)
        self.fc_logvar_gene = torch.nn.Linear(out_channels, out_channels)

        self.vgae = VGAE(self.encoder, decoder=InnerProductDecoder())

    def encode(self, x_dict, edge_index_dict):
        h_dict = self.encoder(
            x_dict, edge_index_dict
        )  # Todo: currently only return with a gene tensor (dpi, dsi)
        mu = {
            # no disease
            # "disease": self.fc_mu_disease(h_dict["disease"]),
            "gene": self.fc_mu_gene(h_dict["gene"]),
        }
        logvar = {
            # No desease
            # "disease": self.fc_logvar_disease(h_dict["disease"]),
            "gene": self.fc_logvar_gene(h_dict["gene"]),
        }
        return mu, logvar

    def decode(self, z, edge_index_dict):
        return self.vgae.decode_all(z, edge_index_dict)

    def forward(self, x_dict, edge_index_dict):
        # Embed to Latent space
        mu, logvar = self.encode(x_dict, edge_index_dict)
        # ToDo ne kelljen tenzorrá alakítani
        mu_tensor = torch.tensor(mu["gene"])
        logvar_tensor = torch.tensor(logvar["gene"])
        z = self.vgae.reparametrize(mu_tensor, logvar_tensor)
        # Latent reprezentation
        return z

    def train(self, x_dict, edge_index_dict):
        # ToDo
        return

    def evaluate(self, x_dict, edge_index_dict):
        # ToDo
        return
