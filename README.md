# Official Repo for Multi Resolution Analysis (MRA) for Approximate Self-Attention

## Dataset Preparation
All codes for dataset preparation are located at `src/datasets_scripts`

### Finetuning
The codes simply download HuggingFace's Dataset or convert json files downloaded from official source to HuggingFace's Dataset
Code for downstream datasets is very short and should be self-explanatory.

### Pretraining
Use the following command to download wikipedia english dataset from HuggingFace
```
python3 src/datasets_scripts/pretrain/wiki_en_dataset.py --output <output_path>
```

Then use the following command to segment or pack articles in wikipedia english dataset to examples of `<sequence_length>` length and stores examples to multiple jsonl files
```
python3 src/datasets_scripts/pretrain/example_packing.py \
  --output_folder <jsonl_files_output_folder> \
  --data_file <wiki_en_output_path>/wikipedia.20220301.en/train/ \
  --example_pack_length <sequence_length> \
  --mp
```
mp option will use all available cpu cores to preprocess the dataset.

Finally, use the following command to combine all jsonl files
```
python3 src/datasets_scripts/pretrain/jsonl_to_arrow.py \
  --jsonl_folder <jsonl_files_output_folder> \
  --output_file <arrow_file_output_path>
```
You can move some jsonl files from `<jsonl_files_output_folder>` to a different folder and use it as validation set.

## Training using PyTorch Lightning
Both pretraining and finetuning are launched by `main.py`. All configuration including training pipeline, model, dataset, data collator, and optimizer are specified in a config file, such as `cfgs/roberta/base-512/postnorm.py`
```
python3 main.py --config cfgs/roberta/base-512/postnorm.py
python3 main.py --config cfgs/roberta/base-512/mra2.py
python3 main.py --config cfgs/roberta/base-4k/mra2.py
python3 main.py --config cfgs/roberta/base-512/wikihop/postnorm.py
python3 main.py --config cfgs/roberta/base-512/wikihop/mra2.py
python3 main.py --config cfgs/roberta/base-4k/wikihop/mra2.py
```
