import torch
from torch_geometric.nn import HeteroConv, SAGEConv, VGAE


# Encoder module that performs graph convolution on heterogeneous graph data.
class Encoder(torch.nn.Module):
    def __init__(self, in_channels_disease, in_channels_gene, hidden_channels, out_channels):
        super(Encoder, self).__init__()
                
        # Define the first and second layer of the encoder
        # The encoder consists of two layers of SAGEConv
        self.conv1 = HeteroConv(
            {
                ("disease", "to", "gene"): SAGEConv(
                    [in_channels_disease, in_channels_gene],
                    hidden_channels,
                ),
                # The reverse edge direction
                ("gene", "rev_to", "disease"): SAGEConv(
                    [in_channels_gene, in_channels_disease], 
                    hidden_channels
                ),
            },
            aggr="sum",
        )
        
        self.conv2 = HeteroConv(
            {
                ("disease", "to", "gene"): SAGEConv(
                    [hidden_channels, hidden_channels], 
                    out_channels
                ),
                ("gene", "rev_to", "disease"): SAGEConv(
                    [hidden_channels, hidden_channels], 
                    out_channels
                ),
            },
            aggr="sum",
        )

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict


# The decoder module is a simple dot product between the latent embeddings of genes and diseases
class Decoder:
    def forward_all(self, z_disease, z_gene):
        return torch.matmul(z_disease, z_gene.t())


# The HeteroVGAE model combines the encoder, decoder, and VGAE modules to create a full model
class HeteroVGAE(torch.nn.Module):
    def __init__(self, in_channels_disease, in_channels_gene, encoder_hidden_channels, encoder_out_channels, out_channels):
        super(HeteroVGAE, self).__init__()
                
        self.encoder = Encoder(in_channels_disease, in_channels_gene, encoder_hidden_channels, encoder_out_channels)
        self.decoder = Decoder()

        # Define the fully connected layers for the mean and log variance of the latent embeddings
        self.fc_mu_disease = torch.nn.Linear(encoder_out_channels, out_channels)
        self.fc_logvar_disease = torch.nn.Linear(encoder_out_channels, out_channels)
        self.fc_mu_gene = torch.nn.Linear(encoder_out_channels, out_channels)
        self.fc_logvar_gene = torch.nn.Linear(encoder_out_channels, out_channels)

        self.vgae = VGAE(self.encoder, self.decoder)

    def encode(self, x_dict, edge_index_dict):
        encoded_features = self.encoder(
            x_dict, edge_index_dict
        )

        mu = {
            "disease": self.fc_mu_disease(encoded_features["disease"]),
            "gene": self.fc_mu_gene(encoded_features["gene"]),
        }
        logvar = {
            "disease": self.fc_logvar_disease(encoded_features["disease"]),
            "gene": self.fc_logvar_gene(encoded_features["gene"]),
        }
        
        return mu, logvar

    def decode(self, z_disease, z_gene):
        # Decode the latent embeddings to produce the output
        return self.vgae.decoder.forward_all(z_disease, z_gene)

    def forward(self, x_dict, edge_index_dict):
        # Encode input features into the latent space
        mu, logvar = self.encode(x_dict, edge_index_dict)

        # Reparameterize to obtain latent embeddings for genes and diseases
        gene_z = self.vgae.reparametrize(mu["gene"], logvar["gene"])
        disease_z = self.vgae.reparametrize(mu["disease"], logvar["disease"])
        
        # Decode the latent embeddings to produce the output
        return self.decode(disease_z, gene_z)
