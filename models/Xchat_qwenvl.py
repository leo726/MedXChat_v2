import os
import torch
from typing import List
import lightning.pytorch as pl
import einops
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchmetrics.text import BLEUScore
import numpy as np
import transformers
# from deepspeed.ops.adam import DeepSpeedCPUAdam
from torch.optim.lr_scheduler import StepLR
from peft import get_peft_model, LoraConfig
import pdb
from transformers.generation import GenerationConfig
from PIL import Image
from dataclasses import dataclass, field
import sys
sys.path.append('/mnt/sdc/yangling/MedXchat/')
from configs.config import parser
from typing import Dict
from transformers.trainer_pt_utils import LabelSmoother
import time
import datetime
a=datetime.datetime.now()

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

# @dataclass
# class LoraArguments:
#     lora_r: int = 64
#     lora_alpha: int = 16
#     lora_dropout: float = 0.05
#     lora_target_modules: List[str] = field(
#         default_factory=lambda: ["c_attn", "attn.c_proj", "w1", "w2"] ##["in_proj","out_proj","c_fc"]
#     )
#     lora_weight_path: str = ""
#     lora_bias: str = "none"
#     q_lora: bool = False

class XChatModel(pl.LightningModule):
    """
    XChatModel.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
        print("Loadding model")
        if args.accelerator == 'cpu':
            self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
        else:
            self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True, bf16 = True)
            # self.model = MplugOwlForConditionalGeneration.from_pretrained(args.llm_model, torch_dtype=torch.float32)
        self.model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
        if args.llm_use_lora:
            # for param in self.model.parameters():
            #     # freeze base model's layers
            #     param.requires_grad = False
            # for name, param in self.model.named_parameters():
            #     if 'language_model' in name:
            #         param.requires_grad = False
            #     else:
            #         param.requires_grad = True

            # peft_config = LoraConfig(
            #     target_modules=r'.*language_model.*\.(q_proj|v_proj)|.*abstractor.*\.(query|value)', 
            #     inference_mode=args.lora_inference, 
            #     r=args.llm_r, 
            #     lora_alpha=args.llm_alpha, 
            #     lora_dropout=args.lora_dropout
            # )
            # peft_config = LoraConfig(
            #     target_modules=r'.*QWenBlock.*\.（c_attn)', 
            #     inference_mode=args.lora_inference, 
            #     r=args.llm_r, 
            #     lora_alpha=args.llm_alpha, 
            #     lora_dropout=args.lora_dropout
            # )
            
            # parser = transformers.HfArgumentParser(LoraArguments)
            # (lora_args) = parser.parse_args_into_dataclasses()
            
            lora_config = LoraConfig(
                r=64,
                lora_alpha=16,
                target_modules=["c_attn", "attn.c_proj", "w1", "w2"], ##["in_proj","out_proj","c_fc"]
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                modules_to_save=["wte", "lm_head"]  # This argument serves for adding new tokens.
        )
            self.model = get_peft_model(self.model, lora_config)
            # for name, param in self.model.named_parameters():
            #     if 'language_model' not in name:
            #         param.requires_grad = True
            self.model.print_trainable_parameters()
        else:
            for name, param in self.model.named_parameters():
                if 'language_model' in name or 'vision_model' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        self.input_embeddings = self.model.get_input_embeddings()

        self.val_step_outputs = []
        self.bleu_scorers = [BLEUScore(n_gram=i) for i in [1, 2, 3, 4]]
        self.val_score = 100.0
        self.max_len = 768
        
        # query = self.tokenizer.from_list_format([
        #     {'image': '/mnt/sda/yangling/MedXchat/models/Qwen-VL/assets/apple.jpeg'},
        #     {'text': 'Generate the caption in English with grounding:'},
        # ])

        # query = self.tokenizer.from_list_format([
        #     {'text': 'Generate the story about a cat for me:'},
        # ])
        # inputs = self.tokenizer(query, return_tensors='pt')
        # pred = self.model.generate(**inputs)
        # response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)

        if args.delta_file is not None:
            # state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))['model']
            state_dict = torch.load(args.delta_file, map_location='cpu')['model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load checkpoint from {args.delta_file}')

# tokenizer.pad_token_id = tokenizer.eod_id
# to_return = dict(
#         input_ids=input_ids,
#         labels=targets,
#         attention_mask=input_ids.ne(tokenizer.pad_token_id),
#     )
# output = self.model(**to_return)
# output.keys()
# output['loss']

    def preprocess(
        qwen_model,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        max_len: int,
        system_message: str = "You are a helpful assistant."
    ) -> Dict:
        roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

        im_start = tokenizer.im_start_id
        im_end = tokenizer.im_end_id
        nl_tokens = tokenizer('\n').input_ids
        _system = tokenizer('system').input_ids + nl_tokens
        _user = tokenizer('user').input_ids + nl_tokens
        _assistant = tokenizer('assistant').input_ids + nl_tokens

        # Apply prompt templates
        input_ids, targets = [], []
        #print(sources)

        for i, source in enumerate(sources):

            count = source[0]["value"].count("\n")
            if roles[source[0]["from"]] != roles["user"]:
                source = source[1:]
             
            input_id, target = [], []
            system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens

            input_id += system
            target += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens

            assert len(input_id) == len(target)

            for j, sentence in enumerate(source):

                role = roles[sentence["from"]]
                _input_id = tokenizer(role).input_ids + nl_tokens + \
                    tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens

                if role == '<|im_start|>user':
                    _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + nl_tokens

                elif role == '<|im_start|>assistant':
                    _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                        _input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens

                else:
                    raise NotImplementedError
 
                input_id += _input_id
                target += _target 

            assert len(input_id) == len(target)
            input_id += [tokenizer.eod_id] * (max_len - len(input_id))

            target += [IGNORE_TOKEN_ID] * (max_len - len(target))
            input_ids.append(input_id[:max_len])
            targets.append(target[:max_len])
  
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.long)

        return dict(
            input_ids=input_ids,
            labels=targets,
            attention_mask=input_ids.ne(tokenizer.eod_id),
        )


    def training_step(self, batch, batch_idx):
        ret = self.preprocess(batch["conversations"], self.tokenizer, self.max_len)
        ret['input_ids'] = ret['input_ids'].to('cuda')
        ret['labels'] = ret['labels'].to('cuda')
        ret['attention_mask'] = ret['attention_mask'].to('cuda')
        output = self.model(**ret)
        loss = output['loss']

        to_log = {}
        # loss.requires_grad= True
        to_log['loss'] = loss
        # self.log('train_loss', loss, prog_bar=True)
        self.log_dict(to_log, prog_bar=True)
        return loss

    def save_checkpoint(self, eval_res):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad
        }
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if k not in param_grad_dic.keys():
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step":global_step
        }
        os.makedirs(os.path.join(self.hparams.savedmodel_path, 'checkpoints'), exist_ok=True)
        save_to = os.path.join(
            self.hparams.savedmodel_path, 'checkpoints',
            "checkpoint_epoch{}_step{}_val_loss{:3f}.pth".format(current_epoch, global_step, eval_res),
        )
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)

    def validation_step(self, batch, batch_idx):
        ret = self.preprocess(batch["conversations"], self.tokenizer, self.max_len)
        ret['input_ids'] = ret['input_ids'].to('cuda')
        ret['labels'] = ret['labels'].to('cuda')
        ret['attention_mask'] = ret['attention_mask'].to('cuda')
        output = self.model(**ret)
        loss = output['loss']

        to_log = {}
        # loss.requires_grad= True  
        to_log['val_loss'] = loss
        # self.log_dict(to_log)
        self.val_step_outputs.append({"val_loss": loss})
        return to_log

    def decode(self, output_token):
        output_text = self.tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.replace('<unk>', '').replace('<s>', '').replace('</s>', '').strip()
        return output_text

    def on_validation_epoch_end(self):
        # last_emb, video_path, val_loss = [], [], []
        val_loss = []
        for i in self.val_step_outputs:
            val_loss.append(i['val_loss'].item())
        val_loss = np.mean(val_loss)
        if self.trainer.local_rank == 0:
            self.save_checkpoint(val_loss)

    def configure_optimizers(self):
        # if 'deepspeed' in self.hparams.strategy:
        #     optimizer = DeepSpeedCPUAdam(self.parameters(), lr=self.hparams.learning_rate)
        #     scheduler = StepLR(optimizer, step_size=1, gamma=0.85)

        #     return {"optimizer": optimizer, "lr_scheduler": scheduler}
        # else:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.85)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    # def get_progress_bar_dict(self):
    #     # don't show the version number
    #     items = super().get_progress_bar_dict()
    #     items.pop("v_num", None)
    #     return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()

if __name__== "__main__" :
    args = parser.parse_args()
    xchat = XChatModel(args)
    print(xchat)