
import inspect
import logging
from typing import Dict, Union, Any, List, Tuple, Optional, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from packaging import version
from transformers import Trainer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DataCollator
from transformers.utils import (
    is_sagemaker_mp_enabled,
    is_apex_available,
    is_peft_available,
    is_datasets_available,
)
from transformers.trainer_utils import (
    EvalPrediction,
    seed_worker,
)
from transformers.modeling_utils import unwrap_model, PreTrainedModel
from transformers.trainer_callback import TrainerCallback
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

if is_datasets_available():
    import datasets

if is_apex_available():
    from apex import amp

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION
    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")
    from transformers.trainer_pt_utils import smp_forward_backward
else:
    IS_SAGEMAKER_MP_POST_1_10 = False
    
if is_peft_available():
    from peft import PeftModel

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class ActiveSelectionTrainer(Trainer):
    UNCERTAINTY_STRATEGIES = set(['entropy', 'vanilla', 'perplexity', 'dropout'])
    
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        minibatch: Optional[int] = None,
        strategy: Optional[str] = None,
        record_mode: bool = False,
        irreducible_loss: Optional[Dict[int, torch.Tensor]] = None,
        ue_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)
        self.minibatch = -1 if minibatch is None else minibatch
        if self.minibatch > 0:
            logger.info(f'Only using {self.minibatch} samples of each batch')
        self.strategy = 'all' if strategy is None else strategy
        
        self.record_mode = record_mode
        if irreducible_loss is None:
            self._irreducible_loss = {}
        else:
            loss_dict = irreducible_loss
            sorted_items = sorted(loss_dict.items(), key=lambda x: x[0])
            self._irreducible_loss = torch.zeros(
                max(loss_dict.keys()) + 1, 
                dtype=torch.float32, 
                device=next(iter(loss_dict.values())).device,
            )
            for (old_idx, loss_value) in sorted_items:
                self._irreducible_loss[old_idx] = loss_value
                
        self.ue_config = ue_config if ue_config is not None else {}
        self._num_samples = self.ue_config.get('stochastic_samples', 10)
        self._input_len_rate = self.ue_config.get('input_len_rate', 1.0)
        
            
    def _selection_and_forward(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Tuple[torch.Tensor, SequenceClassifierOutput]:
        if self.strategy == 'random':
            outputs = model(**inputs, compute_loss=False)
            # indices = torch.randperm(inputs['input_ids'].shape[0])[:self.minibatch]
            indices = torch.arange(self.minibatch)
            
        elif self.strategy == 'entropy':
            outputs = model(**inputs, compute_loss=False)
            probs = torch.softmax(outputs.logits, dim=-1)  # (B, C)
            entropy = - torch.sum(probs * torch.log(probs), dim=-1)  # (B,)
            indices = torch.argsort(entropy, descending=True)[:self.minibatch]  # (B',)
            
        elif self.strategy == 'vanilla':
            outputs = model(**inputs, compute_loss=False)
            probs = torch.softmax(outputs.logits, dim=-1)  # (B, C)
            indices = torch.argmax(probs, dim=-1)[:self.minibatch]  # (B',)
            
        elif self.strategy == 'all':
            outputs = model(**inputs, compute_loss=False)
            indices = None
            
        elif self.strategy == 'perplexity':
            outputs = model(**inputs, compute_loss=False)
            batch_losses = model._get_loss(outputs.logits, inputs['labels'], batch_loss=True)  # (B,)
            perplexity = torch.exp(batch_losses)
            indices = torch.argsort(perplexity, descending=True)[:self.minibatch]  # (B',)
            
        elif self.strategy == 'dropout':
            model.train()
            logits_sum = None
            logits_samples = []
            for _ in range(self._num_samples):
                outputs = model(**inputs, compute_loss=False)
                if logits_sum is None:
                    logits_sum = outputs.logits # (B, C)
                else:
                    logits_sum += outputs.logits
                logits_samples.append(outputs.logits.detach())
            outputs.logits = logits_sum / self._num_samples # (B, C)
            # Compute probability variance (PV)
            pv_scores = torch.var(torch.stack(logits_samples, dim=0), dim=0).mean(dim=-1)  # (B,)
            indices = torch.argsort(pv_scores, descending=True)[:self.minibatch]  # (B',)
            
        elif self.strategy == 'IL':
            outputs = model(**inputs, compute_loss=False)
            irreducible_loss = self._irreducible_loss[inputs['idx']]
            batch_losses = model._get_loss(outputs.logits, inputs['labels'], batch_loss=True, irreducible_loss=irreducible_loss)
            indices = torch.argsort(batch_losses, descending=True)[:self.minibatch]  # (B',)
        else:
            raise NotImplementedError(f'Unknown strategy: {self.strategy}')
        
        # minibatch_inputs = {key: val[indices] for key, val in inputs.items() if isinstance(val, torch.Tensor)}
        return indices, outputs
        
    def _compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        
        ###############################################################################################
        indices, outputs = self._selection_and_forward(model, inputs)
        if 'loss' not in outputs:
            if self.record_mode:
                batch_losses = model._get_loss(outputs.logits, inputs['labels'], indices, batch_loss=True)
                outputs['loss'] = batch_losses.mean()
                for id, single_loss in zip(inputs['idx'], batch_losses):
                    self._irreducible_loss[id.item()] = single_loss
            else:
                outputs['loss'] = model._get_loss(outputs.logits, inputs['labels'], indices)
        ###############################################################################################
        
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if is_peft_available() and isinstance(model, PeftModel):
                model_name = unwrap_model(model.base_model)._get_name()
            else:
                model_name = unwrap_model(model)._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True, indices=indices)
            else:
                loss = self.label_smoother(outputs, labels, indices=indices)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
        
        
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        # {labels: B, idx: B, input_ids: B X T, token_type_ids: B X T, attention_mask: B X T}
        # print("inputs (before cut): ", {k: v.shape if isinstance(v, torch.Tensor) else v for k, v in inputs.items()})
        if self._input_len_rate != 1 and self.strategy in self.UNCERTAINTY_STRATEGIES:
            cutted_inputs = {k: v[:, :int(v.shape[1] * self._input_len_rate)] if len(v.shape) == 2 else v for k, v in inputs.items()}
        else:
            cutted_inputs = inputs
        
        inputs = self._prepare_inputs(cutted_inputs)
        # print("inputs (after cut): ", {k: v.shape if isinstance(v, torch.Tensor) else v for k, v in inputs.items()})

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self._compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps
    
    def _set_signature_train_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
            # Add the idx of the example
            self._signature_columns += ["idx"]
            
    def _remove_unused_train_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_train_columns_if_needed()
        signature_columns = self._signature_columns

        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"
            logger.info(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
                f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
                " you can safely ignore this message."
            )

        columns = [k for k in signature_columns if k in dataset.column_names]
        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)
    
    
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        # print("train_dataset (before remove columns): ", train_dataset)
        
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_train_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        # print("train_dataset (after remove columns): ", train_dataset)
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
    
    