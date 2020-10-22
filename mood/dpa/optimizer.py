from torch import nn
from torch.optim import Adam


class Optimizer(nn.Module):
    def __init__(self, enc_params, dec_params, image_rec_loss, adam_kwargs):
        super(Optimizer, self).__init__()

        self.adam_kwargs = adam_kwargs
        self.image_rec_loss = image_rec_loss
        self.opt = None

        self.set_new_params(enc_params, dec_params)

    def set_new_params(self, enc_params, dec_params, image_rec_loss=None):

        def preprocess(params):
            params = [p for p in params if p.requires_grad]
            return params

        params = preprocess(enc_params) + preprocess(dec_params)
        self.opt = Adam(params, **self.adam_kwargs)

        if image_rec_loss is not None:
            self.image_rec_loss = image_rec_loss

    def compute_loss(self, real_x, rec_x, update_parameters=True):
        loss = self.image_rec_loss(real_x, rec_x)

        if update_parameters:
            if loss.item() > 1e9:
                raise ValueError("Too large value of loss function (>10^9)!")

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        loss_info = {"image_rec_loss": loss.item()}
        return loss_info
