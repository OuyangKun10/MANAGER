import pickle as pkl 

import os
import shutil
import json
from tqdm.notebook import tqdm
import pandas as pd

from datasets import load_dataset
import datasets
import datasets
from transformers import AutoTokenizer, AutoConfig



def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    return {"context": context, "target": target}


# The preprocess function tokenizes the prompt and target, combines them into input IDs,
# and then trims or pads the sequence to the maximum sequence length.
def preprocess(tokenizer, config, example, max_seq_length):
    prompt = example["context"]
    target = example["target"]
    prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
    target_ids = tokenizer.encode(
        target,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False)
    input_ids = prompt_ids + target_ids + [config.eos_token_id]
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}

# The read_jsonl function reads each line from the JSONL file, preprocesses it using the preprocess function,
# and then yields each preprocessed example.
def read_jsonl(path, max_seq_length, skip_overlength=False):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(
        model_name, trust_remote_code=True, device_map='auto')
    with open(path, "r") as f:
        for line in tqdm(f.readlines()):
            example = json.loads(line)
            feature = preprocess(tokenizer, config, example, max_seq_length)
            # print(example)
            # print(feature)
            # exit()
            # if skip_overlength and len(feature["input_ids"]) > max_seq_length:
            #     continue
            feature["input_ids"] = feature["input_ids"][:max_seq_length]
            yield feature

# dic = {
#     0:"negative",
#     1:'positive',
#     2:'neutral',
# }


# tfns = load_dataset('zeroshot/twitter-financial-news-sentiment')
# tfns = tfns['train']
# print(tfns)
# print(type(tfns))
# # exit()
# tfns = tfns.to_pandas()

# tfns['label'] = tfns['label'].apply(lambda x:dic[x])
# tfns['instruction'] = 'What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}.'
# tfns.columns = ['input', 'output', 'instruction']
# tfns = datasets.Dataset.from_pandas(tfns)
# print(tfns)

# tmp_dataset = datasets.concatenate_datasets([tfns]*2)
# train_dataset = tmp_dataset
# print(tmp_dataset.num_rows)

# all_dataset = train_dataset.shuffle(seed = 42)
# all_dataset.shape

# data_list = []
# for item in all_dataset.to_pandas().itertuples():
#     tmp = {}
#     tmp["instruction"] = item.instruction
#     tmp["input"] = item.input
#     tmp["output"] = item.output
#     data_list.append(tmp)
# # with open("../data/dataset_new.jsonl", 'w') as f:
# # for example in tqdm(data_list, desc="formatting.."):
# for example in data_list:
#     print(example)
#     print(type(example))
#     formax_ex=format_example(example)
#     print(formax_ex)
#     print(type(formax_ex))
#     exit()
#         # f.write(json.dumps(format_example(example)) + '\n')
predict_day='3'
pkl_list=['scripts_boc','scripts_boe','scripts_bonz','scripts_bosa','scripts_ecb','scripts_frb']
scripts_boc=['S&P/TSX 60','S&P/TSX Composite Index','XAUCAD','CADUSD','10-year Canadian Bond','3-month Canadian Bond']
scripts_boe=['FTSE 100','FTSE-All Share','Gold','GBP Currency','10 year Britain Bond','3 Month British Bond']
scripts_bonz=['NZSE50FG Index','NZSE Index','XAUNZD','NZDUSD','10-year Bond GTNZD10Y Govt','NZB 3MAY Index 3 month yield']
scripts_bosa=['FTSE South Africa','South Africa Top 40','XAUZAR','ZARUSD','10-year Bond','OEZAR004 Index']
scripts_ecb=['SXXP Index','N100 Index','XAUEUR Curncy ','EURUSD Curncy','GTEUR10Y Govt','GTEUR3M Govt']
scripts_frb=['SPX Index','NASDAQ','GOLD USD','US Dollar Index','GT10 Govt','GB3 Govt']
scripts_list=[scripts_boc,scripts_boe,scripts_bonz,scripts_bosa,scripts_ecb,scripts_frb]
dic = {
    0:"decrease",
    1:'increase',
}
index_scripts=0
error_list=[]
data_list = []
# jsonl_path = '../data/'+predict_day+'_'+'all/'+'data_text.jsonl'#./data/1_scripts_boc/XX.jsonl
# save_path = '../data/'+predict_day+'_'+'all/'

for pkl_path in pkl_list:
  print("Processing:",pkl_path)
  for col_index in range(6):
   
    file = open('./pkl_data/'+predict_day+'/'+pkl_path+'.pkl','rb')  # 以二进制读模式（rb）打开pkl文件
    data = pkl.load(file)
    scripts_col=scripts_list[index_scripts]#预测object的名称
    object=scripts_col[col_index]
    # print()
    print(object)
    jsonl_path = '../data/'+predict_day+'_'+pkl_path+'/'+object.replace('/',' ')+'data_text.jsonl'#./data/1_scripts_boc/XX.jsonl
    save_path = '../data/'+predict_day+'_'+pkl_path+'/'+object.replace('/',' ')

    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)

    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    directory = "../data"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # data = data.to_pandas()
    new_data={}
    text=data['text']
    label=data['label']
    for idx in range(len(text)):
        # print(text[idx])
        # print(label[idx][0])
        new_data[text[idx]]=label[idx][col_index]
        # exit()
    # print(new_data)
    # print(len(text))
    # print(len(label))
    # exit()
    # print(data.items())
    # print(data.keys())
    # exit()
    data=pd.DataFrame(list(new_data.items()),
                    columns=['text', 'label'])
    # for i in range(len(data)):
    #     row = data.iloc[i].values.tolist()
    #     print(row)
    #     exit()
    # print("label:",data['label'])
    # exit()
    # print(data['label'])
    # exit()
    data['label'] = data['label'].apply(lambda x:dic[x])
    data['instruction']= 'What is the price movement of the '+object+' '+predict_day+' days later according to the given transcript? Please choose an answer from {decrease/increase}.'
    # print(data)
    # exit()
    data.columns = ['input', 'output', 'instruction']
    data = datasets.Dataset.from_pandas(data)

    tmp_dataset = datasets.concatenate_datasets([data]*2)
    train_dataset = tmp_dataset
    print(tmp_dataset.num_rows)

    all_dataset = train_dataset.shuffle(seed = 42)
    all_dataset.shape

  
    max_len=0
    for item in all_dataset.to_pandas().itertuples():
        tmp = {}
        tmp["instruction"] = item.instruction
        tmp["input"] = item.input
        tmp["output"] = item.output
        max_len=max(max_len,len(item.instruction.split(' '))+len(item.input.split(' ')))
        data_list.append(tmp)

    # save to a jsonl file
    if(len(data_list)==0):
       error_list.append(predict_day+'_'+pkl_path+'_'+object)
       continue
    
  index_scripts+=1
with open(jsonl_path, 'w') as f:
     for example in tqdm(data_list, desc="formatting.."):
        f.write(json.dumps(format_example(example)) + '\n')
model_name = "/home/shared/LargeLanguageModels/others/chatglm2-6b"
# jsonl_path = "../data/dataset_new.jsonl"  # updated path
# save_path = '../data/dataset_new'  # updated path
# print(max_len)
max_seq_length = max_len
# exit()
skip_overlength = False
# The script then creates a Hugging Face Dataset object from the generator and saves it to disk.

dataset = datasets.Dataset.from_generator(
    lambda: read_jsonl(jsonl_path, max_seq_length, skip_overlength)
    )
dataset.save_to_disk(save_path)
  
# with open("../data/dataset_new.jsonl", 'w') as f:
# for example in tqdm(data_list, desc="formatting.."):
# for example in data_list:
    # print(example)
    # print(type(example))
    # formax_ex=format_example(example)
    # print(formax_ex)
    # print(type(formax_ex))


#tokenization



print(error_list)