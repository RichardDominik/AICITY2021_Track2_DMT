import torch

class LFTDModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.lftd = LFTD(input_dim, output_dim)

    def forward(self, x):
        return self.lftd(x)

    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i.replace('module.','')].copy_(param_dict[i])

class LFTD(torch.nn.Module):
    """
    Learning Features in Temporal Domain
    Layer for aggregation of features in temporal domain
    Accepts inputs of shape batch_size x time_samples x feature_dimensionality
    Produces outputs of shape batch_size x output_dim
    See paper Learning Feature Aggregation in Temporal Domain for Re-Identification for more details
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.compression_kernel = torch.nn.Parameter(torch.zeros(input_dim, output_dim))
        torch.nn.init.xavier_uniform_(self.compression_kernel)

        self.compression_bias = torch.nn.Parameter(torch.zeros(output_dim))

        self.weights_gen_kernel = torch.nn.Parameter(torch.zeros(2 * output_dim, output_dim))
        torch.nn.init.xavier_uniform_(self.weights_gen_kernel)

    def _matmul_in_time(self, x, kernel):
        input_size = x.size()
        x = torch.reshape(x, [-1, input_size[-1]])
        x = torch.matmul(x, kernel)
        x = torch.reshape(x, [-1, input_size[1], x.size(dim=-1)])
        return x

    def forward(self, x):
        y = torch.tanh(self._matmul_in_time(x, self.compression_kernel) + self.compression_bias)
        avg = torch.mean(y, 1, keepdim=True).repeat([1, x.size(dim=1), 1])
        y_avg = torch.cat([y, avg], dim=2)
        weights = self._matmul_in_time(y_avg, self.weights_gen_kernel)
        weights = torch.softmax(weights, dim=1)
        features = torch.sum(y * weights, dim=1)
        features = torch.nn.functional.normalize(features, p=2.0, dim=1)

        return features