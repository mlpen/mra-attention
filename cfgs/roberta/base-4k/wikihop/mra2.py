from src.args import Args, Options

seed = 1234
output_dir = "../outputs/"
save_top_k = 0

model = Args()
model.pl_module = "src.roberta.tasks.multiple_choice.MultipleChoiceModelModule"
model.encoder_type = "src.roberta.models.efficient_attention.EfficientPostnormRobertaModel"
model.load_pretrain = {
    "model_type":"src.roberta.tasks.mlm.RobertaForMLM",
    "checkpoint":"<path>"
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
model.attention_block_per_row = 24
model.attention_initial_prior_first_n_blocks = 2
model.attention_initial_prior_diagonal_n_blocks = 5
model.attention_approx_mode = "full"

trainer = Args()
trainer.strategy = "ddp"
trainer.gradient_clip_val = 1.0
trainer.max_epochs = 10
trainer.gpus = -1
trainer.precision = 16

data = Args()
data.pl_module = "src.roberta.data.data_module_downstream.DownstreamDataModule"
data.num_workers = 8
data.training_dataset_path = "/home/zhanpeng/ssd/nlp_data/wikihop/train"
data.validation_dataset_path = "/home/zhanpeng/ssd/nlp_data/wikihop/dev"
data.tokenizer = "roberta-base"
data.collator = "src.roberta.data.downstream.wikihop.WikiHopCollator"
data.collator_args = Args()
data.collator_args.max_sequence_length = model.max_position_embeddings
data.collator_args.encode_type = "original"
data.collator_args.max_num_candidates = 128
data.collator_args.shuffle_candidates = True
data.collator_args.shuffle_supports = True
data.collator_args.question_first = True

optimizer = Args()
optimizer.optimizer = "adam"
optimizer.batch_size = 1
optimizer.weight_decay = 0.01
optimizer.base_learning_rate = 5e-5
optimizer.min_lr_ratio = 0.001
optimizer.lr_scheduler_type = "linear"
optimizer.warmup_steps = 1000
