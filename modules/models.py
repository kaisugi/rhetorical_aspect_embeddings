import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from transformers.file_utils import ModelOutput, WEIGHTS_NAME

from dataclasses import dataclass
from typing import Optional

from modules.distances import (
    LpDistance,
    CosineSimilarity
)
from modules.reducers import (
    AvgNonZeroReducer,
    MeanReducer
)
from modules.utils import (
    check_shapes, 
    small_val,
    to_device, 
    to_dtype,
    torch_arange_from_size,
)
from modules.loss_utils import (
    assert_distance_type, 
    convert_to_triplets,
    convert_to_pairs,
    logsumexp,
    mean_pooling,
    neg_inf
)


@dataclass
class SequenceClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    embeddings: torch.FloatTensor = None



class BertCls(torch.nn.Module):
    def __init__(self, model_name_or_path):
        super(BertCls, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.bert = AutoModel.from_pretrained(model_name_or_path)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs.last_hidden_state[:, 0, :] # obtain [CLS] embeddings

        return SequenceClassifierOutput(
            embeddings=pooled_output
        )


class BertClsWithPooling(torch.nn.Module):
    def __init__(self, model_name_or_path):
        super(BertClsWithPooling, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.bert = AutoModel.from_pretrained(model_name_or_path)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs.pooler_output # obtain embeddings after pooling layer

        return SequenceClassifierOutput(
            embeddings=pooled_output
        )


# used for SentenceBERT
class BertMeanPooling(torch.nn.Module):
    def __init__(self, model_name_or_path):
        super(BertMeanPooling, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.bert = AutoModel.from_pretrained(model_name_or_path)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = mean_pooling(outputs, attention_mask) # obtain average embeddings

        return SequenceClassifierOutput(
            embeddings=pooled_output
        )


class BertClsSoftmaxCrossEntropyLoss(torch.nn.Module):
    def __init__(self, model_name_or_path, num_labels):
        super(BertClsSoftmaxCrossEntropyLoss, self).__init__()
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.bert = AutoModel.from_pretrained(model_name_or_path)

        self.dropout = torch.nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(self.config.hidden_size, self.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs.last_hidden_state[:, 0, :] # obtain [CLS] embeddings

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output) # convert [CLS] embeddings into N logits

        loss_fct = torch.nn.CrossEntropyLoss()
        # logits: torch.Size([batch_size, num of classes])
        # labels: torch.Size([batch_size])
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1)) # softmax + cross entropy

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            embeddings=pooled_output
        )


class BertClsArcFaceLoss(torch.nn.Module):
    def __init__(self, model_name_or_path, num_labels, margin, scale):
        super(BertClsArcFaceLoss, self).__init__()
        
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.bert = AutoModel.from_pretrained(model_name_or_path)

        self.dropout = torch.nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(self.config.hidden_size, self.num_labels)

        self.margin = margin
        self.scale = scale
        self.distance = self.get_distance()
        assert_distance_type(self, CosineSimilarity)

    def get_distance(self):
        return CosineSimilarity()

    def get_cosine(self, embeddings):
        return self.distance(embeddings, self.classifier.weight)

    def get_angles(self, cosine_of_target_classes):
        angles = torch.acos(torch.clamp(cosine_of_target_classes, -1, 1))
        return angles

    def get_target_mask(self, embeddings, labels):
        batch_size = labels.size(0)
        mask = torch.zeros(
            batch_size,
            self.num_labels,
            dtype=embeddings.dtype,
            device=embeddings.device,
        )
        mask[torch.arange(batch_size), labels] = 1
        return mask

    # ArcFace specific
    def modify_cosine_of_target_classes(self, cosine_of_target_classes):
        angles = self.get_angles(cosine_of_target_classes)
        return torch.cos(angles + self.margin)

    # ArcFace specific
    def scale_logits(self, logits, *_):
        return logits * self.scale

    # see also https://arxiv.org/pdf/1801.07698.pdf (Algorithm 1)
    def compute_loss(self, embeddings, labels):
        mask = self.get_target_mask(embeddings, labels)
        cosine = self.get_cosine(embeddings)
        cosine_of_target_classes = cosine[mask == 1]
        modified_cosine_of_target_classes = self.modify_cosine_of_target_classes(
            cosine_of_target_classes
        )
        diff = (modified_cosine_of_target_classes - cosine_of_target_classes).unsqueeze(1)
        logits = cosine + (mask * diff)
        logits = self.scale_logits(logits, embeddings)

        loss_fct = torch.nn.CrossEntropyLoss()
        # logits: torch.Size([batch_size, num of classes])
        # labels: torch.Size([batch_size])
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1)) # softmax + cross entropy

        return loss

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs.last_hidden_state[:, 0, :] # obtain [CLS] embeddings
        pooled_output = self.dropout(pooled_output)
        pooled_output = F.normalize(pooled_output, p=2, dim=1) # normalize embedding feature

        loss = self.compute_loss(pooled_output, labels)
        return SequenceClassifierOutput(
            loss=loss,
            embeddings=pooled_output
        )


# distance_type = LpDistance(p=2)
class TripletMarginLoss(torch.nn.Module):
    """
    Args:
        margin: The desired difference between the anchor-positive distance and the
                anchor-negative distance.
        swap: Use the positive-negative distance instead of anchor-negative distance,
              if it violates the margin more.
        smooth_loss: Use the log-exp version of the triplet loss
    """
    def __init__(
        self,
        margin=0.05,
        swap=False,
        smooth_loss=False,
        triplets_per_anchor="all"
    ):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.swap = swap
        self.smooth_loss = smooth_loss
        self.triplets_per_anchor = triplets_per_anchor

        self.distance = self.get_distance()
        self.reducer = self.get_reducer()

    def compute_loss(self, embeddings, labels, indices_tuple):
        indices_tuple = convert_to_triplets(
            indices_tuple, labels, t_per_anchor=self.triplets_per_anchor
        )
        anchor_idx, positive_idx, negative_idx = indices_tuple

        mat = self.distance(embeddings)
        ap_dists = mat[anchor_idx, positive_idx]
        an_dists = mat[anchor_idx, negative_idx]
        if self.swap:
            pn_dists = mat[positive_idx, negative_idx]
            an_dists = self.distance.smallest_dist(an_dists, pn_dists)

        current_margins = self.distance.margin(ap_dists, an_dists)  # d_p - d_n
        violation = current_margins + self.margin  # d_p - d_n + m
        if self.smooth_loss:
            loss = torch.nn.functional.softplus(violation)
        else:
            loss = torch.nn.functional.relu(violation)  # max(0, d_p - d_n + m)

        return {
            "loss": {
                "losses": loss,
                "indices": indices_tuple,
                "reduction_type": "triplet",
            }
        }

    def get_default_reducer(self):
        return AvgNonZeroReducer()

    def get_reducer(self):
        return self.get_default_reducer()

    def get_default_distance(self):
        return LpDistance(p=2)

    def get_distance(self):
        return self.get_default_distance()

    def forward(self, embeddings, labels, indices_tuple=None):
        check_shapes(embeddings, labels)
        labels = to_device(labels, embeddings)
        loss_dict = self.compute_loss(embeddings, labels, indices_tuple)
        return self.reducer(loss_dict, embeddings, labels)


# distance_type = CosineSimilarity
class NTXentLoss(torch.nn.Module):
    def __init__(
        self, 
        temperature=0.07
    ):
        super(NTXentLoss, self).__init__()

        self.temperature = temperature

        self.distance = self.get_distance()
        self.reducer = self.get_reducer()
        assert_distance_type(self, CosineSimilarity)

    def compute_loss(self, embeddings, labels, indices_tuple):
        indices_tuple = convert_to_pairs(indices_tuple, labels)
        mat = self.distance(embeddings)
        return self.pair_based_loss(mat, labels, indices_tuple)

    def _compute_loss(self, pos_pairs, neg_pairs, indices_tuple):
        a1, p, a2, _ = indices_tuple

        if len(a1) > 0 and len(a2) > 0:
            dtype = neg_pairs.dtype
            # if dealing with actual distances, use negative distances
            if not self.distance.is_inverted:
                pos_pairs = -pos_pairs
                neg_pairs = -neg_pairs

            pos_pairs = pos_pairs.unsqueeze(1) / self.temperature
            neg_pairs = neg_pairs / self.temperature
            n_per_p = to_dtype(a2.unsqueeze(0) == a1.unsqueeze(1), dtype=dtype)
            neg_pairs = neg_pairs * n_per_p
            neg_pairs[n_per_p == 0] = neg_inf(dtype)

            max_val = torch.max(
                pos_pairs, torch.max(neg_pairs, dim=1, keepdim=True)[0]
            ).detach()
            numerator = torch.exp(pos_pairs - max_val).squeeze(1)
            denominator = torch.sum(torch.exp(neg_pairs - max_val), dim=1) + numerator
            log_exp = torch.log((numerator / denominator) + small_val(dtype))
            return {
                "loss": {
                    "losses": -log_exp,
                    "indices": (a1, p),
                    "reduction_type": "pos_pair",
                }
            }
        return self.zero_losses()

    def zero_loss(self):
        return {"losses": 0, "indices": None, "reduction_type": "already_reduced"}

    def zero_losses(self):
        return {loss_name: self.zero_loss() for loss_name in self.sub_loss_names()}

    def sub_loss_names(self):
        return ["loss"]

    def pair_based_loss(self, mat, labels, indices_tuple):
        a1, p, a2, n = indices_tuple
        pos_pair, neg_pair = [], []
        if len(a1) > 0:
            pos_pair = mat[a1, p]
        if len(a2) > 0:
            neg_pair = mat[a2, n]
        return self._compute_loss(pos_pair, neg_pair, indices_tuple)

    def get_default_reducer(self):
        return MeanReducer()

    def get_reducer(self):
        return self.get_default_reducer()

    def get_default_distance(self):
        return CosineSimilarity()

    def get_distance(self):
        return self.get_default_distance()

    def forward(self, embeddings, labels, indices_tuple=None):
        check_shapes(embeddings, labels)
        labels = to_device(labels, embeddings)
        loss_dict = self.compute_loss(embeddings, labels, indices_tuple)
        return self.reducer(loss_dict, embeddings, labels)


# distance_type = CosineSimilarity
class MultiSimilarityLoss(torch.nn.Module):
    def __init__(
        self, 
        alpha=2, 
        beta=50, 
        base=0.5
    ):
        super(MultiSimilarityLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.base = base

        self.distance = self.get_distance()
        self.reducer = self.get_reducer()
        assert_distance_type(self, CosineSimilarity)

    def compute_loss(self, embeddings, labels, indices_tuple):
        indices_tuple = convert_to_pairs(indices_tuple, labels)
        mat = self.distance(embeddings)
        return self.mat_based_loss(mat, labels, indices_tuple)

    def _compute_loss(self, mat, pos_mask, neg_mask):
        pos_exp = self.distance.margin(mat, self.base) # lambda - S_ij
        neg_exp = self.distance.margin(self.base, mat) # S_ij - lambda
        pos_loss = (1.0 / self.alpha) * logsumexp(
            self.alpha * pos_exp, keep_mask=pos_mask.bool(), add_one=True # alpha (lambda - S_ij)
        )
        neg_loss = (1.0 / self.beta) * logsumexp(
            self.beta * neg_exp, keep_mask=neg_mask.bool(), add_one=True # beta (lambda - S_ij)
        )
        return {
            "loss": {
                "losses": pos_loss + neg_loss,
                "indices": torch_arange_from_size(mat),
                "reduction_type": "element",
            }
        }

    def mat_based_loss(self, mat, labels, indices_tuple):
        a1, p, a2, n = indices_tuple
        pos_mask, neg_mask = torch.zeros_like(mat), torch.zeros_like(mat)
        pos_mask[a1, p] = 1
        neg_mask[a2, n] = 1
        return self._compute_loss(mat, pos_mask, neg_mask)

    def zero_loss(self):
        return {"losses": 0, "indices": None, "reduction_type": "already_reduced"}

    def zero_losses(self):
        return {loss_name: self.zero_loss() for loss_name in self.sub_loss_names()}

    def sub_loss_names(self):
        return ["loss"]

    def pair_based_loss(self, mat, labels, indices_tuple):
        a1, p, a2, n = indices_tuple
        pos_pair, neg_pair = [], []
        if len(a1) > 0:
            pos_pair = mat[a1, p]
        if len(a2) > 0:
            neg_pair = mat[a2, n]
        return self._compute_loss(pos_pair, neg_pair, indices_tuple)

    def get_default_reducer(self):
        return MeanReducer()

    def get_reducer(self):
        return self.get_default_reducer()

    def get_default_distance(self):
        return CosineSimilarity()

    def get_distance(self):
        return self.get_default_distance()

    def forward(self, embeddings, labels, indices_tuple=None):
        check_shapes(embeddings, labels)
        labels = to_device(labels, embeddings)
        loss_dict = self.compute_loss(embeddings, labels, indices_tuple)
        return self.reducer(loss_dict, embeddings, labels)