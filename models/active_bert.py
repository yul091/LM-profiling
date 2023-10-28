
from typing import Dict, Union, Any, List, Tuple, Optional, Callable

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import (
    BertForSequenceClassification,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    add_code_sample_docstrings,
)

from transformers.models.bert.modeling_bert import (
    BERT_START_DOCSTRING,
    BERT_INPUTS_DOCSTRING,
    _CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
    _CONFIG_FOR_DOC,
    _SEQ_CLASS_EXPECTED_OUTPUT,
    _SEQ_CLASS_EXPECTED_LOSS,
)
from transformers.modeling_outputs import (
    SequenceClassifierOutput, 
)
from .active_base import ActiveSelectionBase



@add_start_docstrings(
    """
    Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    BERT_START_DOCSTRING,
)
class ActiveSelectionBertForSequenceClassification(BertForSequenceClassification, ActiveSelectionBase):
    def __init__(self, config):
        super().__init__(config)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        compute_loss: bool = True,
        idx: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if compute_loss:
            loss = self._get_loss(logits, labels)
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
    def _get_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
        batch_loss: bool = False,
        irreducible_loss: Optional[torch.Tensor] = None,
    ):
        if self.config.problem_type is None:
            if self.num_labels == 1:
                self.config.problem_type = "regression"
            elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                self.config.problem_type = "single_label_classification"
            else:
                self.config.problem_type = "multi_label_classification"

        if self.config.problem_type == "regression":
            loss_fct = MSELoss() if not batch_loss else MSELoss(reduction='none')
            if self.num_labels == 1:
                if indices is not None:
                    loss = loss_fct(logits.squeeze()[indices], labels.squeeze()[indices])
                else:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                if indices is not None:
                    loss = loss_fct(logits[indices], labels[indices])
                else:
                    loss = loss_fct(logits, labels)
            if batch_loss and irreducible_loss is not None:
                loss = loss - irreducible_loss

        elif self.config.problem_type == "single_label_classification":
            loss_fct = CrossEntropyLoss() if not batch_loss else CrossEntropyLoss(reduction='none')
            if indices is not None:
                # print("Using active selection for CE loss computation") 
                loss = loss_fct(logits[indices].view(-1, self.num_labels), labels[indices].view(-1))
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            if batch_loss and irreducible_loss is not None:
                loss = loss - irreducible_loss
  
        elif self.config.problem_type == "multi_label_classification":
            loss_fct = BCEWithLogitsLoss() if not batch_loss else BCEWithLogitsLoss(reduction='none')
            if indices is not None:
                loss = loss_fct(logits[indices], labels[indices])
            else:
                loss = loss_fct(logits, labels)
            if batch_loss and irreducible_loss is not None:
                loss = loss - irreducible_loss
        
        return loss