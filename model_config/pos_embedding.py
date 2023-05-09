import torch

class PosEmbeder(torch.nn.Module):
    def __init__(self, sim_dim):
        super(PosEmbeder, self).__init__()
        self.sim_dim = sim_dim
        self.embedding_x1 = torch.nn.Embedding(num_embeddings=1000,
                                               embedding_dim=sim_dim)
        self.embedding_y1 = torch.nn.Embedding(num_embeddings=1000,
                                               embedding_dim=sim_dim)
        self.embedding_x2 = torch.nn.Embedding(num_embeddings=1000,
                                               embedding_dim=sim_dim)
        self.embedding_y2 = torch.nn.Embedding(num_embeddings=1000,
                                               embedding_dim=sim_dim)
        self.normalize = torch.nn.LayerNorm(normalized_shape=sim_dim)

    def forward(self, data):
        bbox = data['bbox']
        result = []
        for index_1 in range(bbox.shape[0]):
            result_sub = []
            for index_2 in range(bbox.shape[1]):
                x1 = self.embedding_x1(bbox[index_1][index_2][0])
                y1 = self.embedding_y1(bbox[index_1][index_2][1])
                x2 = self.embedding_x2(bbox[index_1][index_2][2])
                y2 = self.embedding_y2(bbox[index_1][index_2][3])
                result_sub.append(torch.stack([x1, y1, x2, y2]))
            result.append(torch.stack(result_sub))

        return self.normalize(torch.sum(torch.stack(result), dim=2))
