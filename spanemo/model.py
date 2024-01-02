from transformers import AutoModel
import torch.nn.functional as F
import torch.nn as nn
import torch
import transformers 


class SpanEmo(nn.Module):
    def __init__(self, output_dropout=0.1, backbone = "bert-base-uncased", joint_loss='joint', alpha=0.2, bce_weight=None):
        """ casting multi-label emotion classification as span-extraction
        :param output_dropout: The dropout probability for output layer
        :param lang: encoder language
        :param joint_loss: which loss to use cel|corr|cel+corr
        :param alpha: control contribution of each loss function in case of joint training
        """
        super(SpanEmo, self).__init__()
        self.bert = AutoModel.from_pretrained(backbone)
        self.joint_loss = joint_loss
        self.alpha = alpha
        self.bce_weight = torch.FloatTensor(bce_weight) if bce_weight is not None else None
        
        self.ffn = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size),
            nn.Tanh(),
            nn.Dropout(p=output_dropout),
            nn.Linear(self.bert.config.hidden_size, 1)
        )

    def forward(self, batch, device):
        """
        :param batch: tuple of (input_ids, labels, length, label_indices)
        :param device: device to run calculations on
        :return: loss, num_rows, y_pred, targets
        """
        #prepare inputs and targets
        inputs, targets, lengths, label_idxs = batch
        num_rows = lengths.size(0)
        inputs = {k: inputs[k].to(device) for k in inputs}
        label_idxs, targets = label_idxs[0].long().to(device), targets.float().to(device)

        if self.bce_weight is not None: self.bce_weight = self.bce_weight.to(device)


        #Bert encoder
        last_hidden_state = self.bert(**inputs).last_hidden_state

        # FFN---> 2 linear layers---> linear layer + tanh---> linear layer
        # select span of labels to compare them with ground truth ones
        logits = self.ffn(last_hidden_state).squeeze(-1).index_select(dim=1, index=label_idxs)

        #Loss Function
        if self.joint_loss == 'joint':
            cel = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.bce_weight).cuda()
            cl = self.corr_loss(logits, targets)
            loss = ((1 - self.alpha) * cel) + (self.alpha * cl)
        elif self.joint_loss == 'cross-entropy':
            loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.bce_weight).cuda()
        elif self.joint_loss == 'corr_loss':
            loss = self.corr_loss(logits, targets)

        y_pred = self.compute_pred(logits)
        return loss, num_rows, y_pred, logits, targets.cpu().numpy()
    
    def predict(self, batch, device, targets=False):
        if targets:
            inputs, targets, lengths, label_idxs = batch
        else:
            inputs, lengths, label_idxs = batch

        num_rows = lengths.size(0)
        inputs = {k: inputs[k].to(device) for k in inputs}
        label_idxs = label_idxs[0].long().to(device)

        #Bert encoder
        last_hidden_state = self.bert(**inputs).last_hidden_state

        # FFN---> 2 linear layers---> linear layer + tanh---> linear layer
        # select span of labels to compare them with ground truth ones
        logits = self.ffn(last_hidden_state).squeeze(-1).index_select(dim=1, index=label_idxs)

        y_pred = self.compute_pred(logits)
        return num_rows, y_pred, logits



    @staticmethod
    def corr_loss(y_hat, y_true, reduction='mean'):
        """
        :param y_hat: model predictions, shape(batch, classes)
        :param y_true: target labels (batch, classes)
        :param reduction: whether to avg or sum loss
        :return: loss
        """
        loss = torch.zeros(y_true.size(0)).cuda()
        for idx, (y, y_h) in enumerate(zip(y_true, y_hat.sigmoid())):
            y_z, y_o = (y == 0).nonzero(), y.nonzero()
            if y_o.nelement() != 0:
                output = torch.exp(torch.sub(y_h[y_z], y_h[y_o][:, None]).squeeze(-1)).sum()
                num_comparisons = y_z.size(0) * y_o.size(0)
                loss[idx] = output.div(num_comparisons)
        return loss.mean() if reduction == 'mean' else loss.sum()
        
    @staticmethod
    def compute_pred(logits, threshold=0.5):
        """
        :param logits: model predictions
        :param threshold: threshold value
        :return:
        """
        y_pred = torch.sigmoid(logits) > threshold
        return y_pred.float().cpu().numpy()
