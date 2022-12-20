from src.args import Args

seed = 111111
output_dir = "../outputs/"
save_top_k = 50

deepspeed_config = "deepspeed/stage1.json"

model = Args()
model.pl_module = "src.roberta.tasks.mlm.MLMModelModule"
model.encoder_type = "src.roberta.models.efficient_attention.EfficientPostnormRobertaModel"
model.load_pretrain = {
    "model_type":"src.roberta.tasks.mlm.RobertaForMLM",
    "checkpoint":"/home/zhanpeng/ssd/outputs/cfgs/roberta/base-512/postnorm.py/hf_ckpts/epoch=00000-step=00010000"
}
model.initializer_range = 0.02
model.vocab_size = 50265
model.type_vocab_size = 1
model.pad_token_id = 1
model.num_hidden_layers = 12
model.num_attention_heads = 12
model.layer_norm_eps = 1e-05
model.intermediate_size = 3072
model.hidden_size = 768
model.hidden_dropout_prob = 0.1
model.attention_probs_dropout_prob = 0.1
model.max_position_embeddings = 4096

model.attention_type = "src.roberta.models.postnorm_efficient_attentions.mra2.Attention"
model.attention_block_per_row = 16
model.attention_initial_prior_first_n_blocks = 1
model.attention_initial_prior_diagonal_n_blocks = 5
model.attention_approx_mode = "full"

trainer = Args()
trainer.strategy = "ddp"
trainer.gpus = -1
trainer.precision = 16
trainer.val_check_interval = 10000
trainer.limit_val_batches = 100
trainer.max_steps = 300000

data = Args()
data.pl_module = "src.roberta.data.data_module_pretrain.PretrainDataModule"
data.num_workers = 8
data.training_dataset_path = "/home/zhanpeng/ssd/nlp_data/train.arrow"
data.validation_dataset_path = "/home/zhanpeng/ssd/nlp_data/val.arrow"
data.tokenizer = "roberta-base"
data.collator = "src.roberta.data.pretrain.mlm.MLMCollator"
data.collator_args = Args()
data.collator_args.num_masked_tokens = 77 * 8
data.collator_args.max_sequence_length = 4096

optimizer = Args()
optimizer.optimizer = "adam"
optimizer.adam_beta1 = 0.9
optimizer.adam_beta2 = 0.98
optimizer.adam_epsilon = 1e-6
optimizer.batch_size = 1
optimizer.weight_decay = 0.01
optimizer.base_learning_rate = 3e-5
optimizer.min_lr_ratio = 0.001
optimizer.lr_scheduler_type = "linear"
optimizer.warmup_steps = 24000
