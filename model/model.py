import torch
from torch_geometric.nn import HeteroConv, SAGEConv, VGAE


class EncoderClass(torch.nn.Module):
    def __init__(self, in_channels_disease, in_channels_gene, out_channels):
        super(EncoderClass, self).__init__()
        hidden_channels = 64
        self.conv1 = HeteroConv(
            {
                ("disease", "to", "gene"): SAGEConv(
                    [in_channels_disease, in_channels_gene],
                    hidden_channels,
                    # FYI: It doesnt like when the data is indexed from 0
                ),
                ("gene", "rev_to", "disease"): SAGEConv(
                    [in_channels_gene, in_channels_disease], hidden_channels
                ),
            },
            aggr="sum",
        )
        self.conv2 = HeteroConv(
            {
                ("disease", "to", "gene"): SAGEConv(
                    [hidden_channels, hidden_channels], out_channels
                ),
                ("gene", "rev_to", "disease"): SAGEConv(
                    [hidden_channels, hidden_channels], out_channels
                ),
            },
            aggr="sum",
        )

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict


class InnerProductDecoder:
    def forward_all(self, z_disease, z_gene):
        return torch.matmul(z_disease, z_gene.t())  # valoszinuseg


class HeteroVGAE(torch.nn.Module):
    def __init__(self, in_channels_disease, in_channels_gene, out_channels):
        super(HeteroVGAE, self).__init__()
        hidden = 32
        self.encoder = EncoderClass(in_channels_disease, in_channels_gene, hidden)

        self.fc_mu_disease = torch.nn.Linear(hidden, out_channels)
        self.fc_logvar_disease = torch.nn.Linear(hidden, out_channels)
        self.fc_mu_gene = torch.nn.Linear(hidden, out_channels)
        self.fc_logvar_gene = torch.nn.Linear(hidden, out_channels)

        self.vgae = VGAE(self.encoder, decoder=InnerProductDecoder())

    def encode(self, x_dict, edge_index_dict):
        h_dict = self.encoder(
            x_dict, edge_index_dict
        )  # TODO: currently only return with a gene tensor (dpi, dsi)

        mu = {
            # no disease
            "disease": self.fc_mu_disease(h_dict["disease"]),
            "gene": self.fc_mu_gene(h_dict["gene"]),
        }
        logvar = {
            # no desease
            "disease": self.fc_logvar_disease(h_dict["disease"]),
            "gene": self.fc_logvar_gene(h_dict["gene"]),
        }
        return mu, logvar

    def decode(self, z_disease, z_gene):
        return self.vgae.decoder.forward_all(z_disease, z_gene)

    def forward(self, x_dict, edge_index_dict):
        # Embed to Latent space
        mu, logvar = self.encode(x_dict, edge_index_dict)

        gene_z = self.vgae.reparametrize(mu["gene"], logvar["gene"])
        disease_z = self.vgae.reparametrize(mu["disease"], logvar["disease"])
        # Latent representation
        return self.decode(disease_z, gene_z)
