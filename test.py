from transformers import AutoConfig, LlamaForCausalLM, LlamaTokenizer,AutoModelForSeq2SeqLM, AutoTokenizer,AutoModelForCausalLM,pipeline

import argparse
import pandas as pd
from tqdm import tqdm
import datetime

parser=argparse.ArgumentParser()
parser.add_argument("--model",type=str,default='vicuna-13b')
parser.add_argument("--task",type=str)
parser.add_argument("--prompt_type",type=str,choices=['basic_prompt','instructive_prompt','CoT_prompt'])
args=parser.parse_args()

prompt_file=pd.read_csv("./prompt/"+args.task+"_task_prompt.csv")


def generate(model_name,prompt_list,max_new_tokens=512,do_sample=False,num_beams=1,diversity_penalty=0.0,temperature=1.0,top_k=50,\
             top_p=1,repetition_penalty=1):
    result_list=[]
    if model_name=='vicuna-13b':
        tokenizer = LlamaTokenizer.from_pretrained("/lustre/S/zhaoyunpu/vicuna-13b/")
        model     = LlamaForCausalLM.from_pretrained("/lustre/S/zhaoyunpu/vicuna-13b/",device_map='auto')
        generator = pipeline(model=model,tokenizer=tokenizer,device_map='auto',framework='pt',task='text-generation',\
                           max_new_tokens=max_new_tokens,do_sample=do_sample,num_beams=num_beams,diversity_penalty=diversity_penalty,\
                                temperature=temperature,top_k=top_k,top_p=top_p,repetition_penalty=repetition_penalty,min_new_tokens=10)

    elif model_name=='vicuna-33b':
        tokenizer = LlamaTokenizer.from_pretrained("/lustre/S/zhaoyunpu/vicuna-33b/")
        model     = LlamaForCausalLM.from_pretrained("/lustre/S/zhaoyunpu/vicuna-33b/",device_map='auto')
        generator = pipeline(model=model,tokenizer=tokenizer,device_map='auto',framework='pt',task='text-generation',\
                           max_new_tokens=max_new_tokens,do_sample=do_sample,num_beams=num_beams,diversity_penalty=diversity_penalty,\
                                temperature=temperature,top_k=top_k,top_p=top_p,repetition_penalty=repetition_penalty,min_new_tokens=10)
        
    elif model_name=='flan-t5-xxl':
        tokenizer = AutoTokenizer.from_pretrained("/lustre/S/zhaoyunpu/flan-t5-xxl/")
        model     = AutoModelForSeq2SeqLM.from_pretrained("/lustre/S/zhaoyunpu/flan-t5-xxl/",device_map='auto')
        
    elif model_name=="flan-UL2":
        tokenizer = AutoTokenizer.from_pretrained("/lustre/S/zhaoyunpu/flan-UL2/")
        model     = AutoModelForSeq2SeqLM.from_pretrained("/lustre/S/zhaoyunpu/flan-UL2/",device_map='auto')
        
    elif model_name=='mpt-30b-chat':
        config    = AutoConfig.from_pretrained("/lustre/S/zhaoyunpu/mpt-30b-chat/",trust_remote_code=True)
        model     = AutoModelForCausalLM.from_pretrained("/lustre/S/zhaoyunpu/mpt-30b-chat/",config=config,device_map='auto',trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("/lustre/S/zhaoyunpu/mpt-30b-chat/")
    
    elif model_name=='llama-2-13b':
        config    = AutoConfig.from_pretrained("/lustre/S/liwenyi/llm/Llama-2-13b-chat-hf/",trust_remote_code=True)
        model     = AutoModelForCausalLM.from_pretrained("/lustre/S/liwenyi/llm/Llama-2-13b-chat-hf/",config=config,device_map='auto',trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("/lustre/S/liwenyi/llm/Llama-2-13b-chat-hf/")
        generator = pipeline(model=model,tokenizer=tokenizer,device_map='auto',framework='pt',task='text-generation',\
                           max_new_tokens=max_new_tokens,do_sample=do_sample,num_beams=num_beams,diversity_penalty=diversity_penalty,\
                                temperature=temperature,top_k=top_k,top_p=top_p,repetition_penalty=repetition_penalty,min_new_tokens=10)

    elif model_name=='llama-2-70b':
        config    = AutoConfig.from_pretrained("/lustre/S/liwenyi/llm/Llama-2-70b-chat-hf/",trust_remote_code=True)
        model     = AutoModelForCausalLM.from_pretrained("/lustre/S/liwenyi/llm/Llama-2-70b-chat-hf/",config=config,device_map='auto',trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("/lustre/S/liwenyi/llm/Llama-2-70b-chat-hf/")
        generator = pipeline(model=model,tokenizer=tokenizer,device_map='auto',framework='pt',task='text-generation',\
                           max_new_tokens=max_new_tokens,do_sample=do_sample,num_beams=num_beams,diversity_penalty=diversity_penalty,\
                                temperature=temperature,top_k=top_k,top_p=top_p,repetition_penalty=repetition_penalty,min_new_tokens=10)


    else:
        raise Exception("Invalid model name.")

    for prompt in tqdm(prompt_list):
        result = generator(prompt,return_full_text=False)[0]['generated_text']
        result_list.append(result)
        print(result_list[-1])
    return result_list
        
'''
    for prompt in tqdm(prompt_list):
        inputs=tokenizer(prompt,return_tensors='pt').to('cuda')
        generate_ids=model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=do_sample,num_beams=num_beams,diversity_penalty=diversity_penalty,\
                                temperature=temperature,top_k=top_k,top_p=top_p,repetition_penalty=repetition_penalty,min_new_tokens=10)
        result_list.append(tokenizer.batch_decode(generate_ids,skip_special_tokens=True,clean_up_tokenization_spaces=False))
    return result_list
'''
output=generate(model_name=args.model,prompt_list=prompt_file[args.prompt_type].tolist())
output=pd.Series(output)
output.to_csv("./result/"+args.model+'_'+args.task+'_'+args.prompt_type+'_'+datetime.datetime.now().strftime("%Y%m%d")+".csv",index=False)







