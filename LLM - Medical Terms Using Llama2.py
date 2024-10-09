# pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7
# pip install huggingface_hub


import torch
import trl
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline)
from peft import LoraConfig


print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))




llama_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path='aboonaji/llama2finetune-v2',
                                                   quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                                                                                          bnb_4bit_compute_dtype=getattr(torch, "float16"),
                                                                                          bnb_4bit_quant_type="nf4"))


llama_model.config.use_cache = False
llama_model.config.pretraining_tp = 1


llama_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='aboonaji/llama2finetune-v2',
                                                trust_remote_code=True)

llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = 'right'

training_arguments = TrainingArguments(output_dir=r'C:\Users\green\Meu Drive\Documentos\Engenharia da Computação\I.A. - Aprendizado', max_steps=100)

llama_sft_trainer = trl.SFTTrainer(model=llama_model, args=training_arguments, train_dataset=load_dataset('aboonaji/wiki_medical_terms_llam2_format', split='train'),
                                   tokenizer=llama_tokenizer, dataset_text_field='text', peft_config= LoraConfig(task_type='CAUSAL_LM', r = 64, lora_alpha= 16, lora_dropout= 0.1))


llama_sft_trainer.train()

user_prompt = input()
text_generation_pipeline = pipeline(task="text-generation", model=llama_model, tokenizer=llama_tokenizer, max_len = 300)
model_answer = text_generation_pipeline(f"<s>[INST] {user_prompt} [/INST]")
print(model_answer[0]['generated_text'])
