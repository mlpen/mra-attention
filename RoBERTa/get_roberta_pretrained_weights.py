from transformers import RobertaForMaskedLM
from collections import OrderedDict
import pickle

def convert_weight_keys(weights):
    mappings = {
        "roberta.embeddings.word_embeddings.weight":"model.embeddings.word_embeddings.weight",
        "roberta.embeddings.position_embeddings.weight":"model.embeddings.position_embeddings.weight",
        "roberta.embeddings.token_type_embeddings.weight":"model.embeddings.token_type_embeddings.weight",
        "roberta.embeddings.LayerNorm.weight":"model.embeddings.norm.weight",
        "roberta.embeddings.LayerNorm.bias":"model.embeddings.norm.bias",
    }
    for idx in range(12):
        mappings[f"roberta.encoder.layer.{idx}.attention.self.query.weight"] = f"model.backbone.backbone.encoders.{idx}.mha.W_q.weight"
        mappings[f"roberta.encoder.layer.{idx}.attention.self.query.bias"] = f"model.backbone.backbone.encoders.{idx}.mha.W_q.bias"
        mappings[f"roberta.encoder.layer.{idx}.attention.self.key.weight"] = f"model.backbone.backbone.encoders.{idx}.mha.W_k.weight"
        mappings[f"roberta.encoder.layer.{idx}.attention.self.key.bias"] = f"model.backbone.backbone.encoders.{idx}.mha.W_k.bias"
        mappings[f"roberta.encoder.layer.{idx}.attention.self.value.weight"] = f"model.backbone.backbone.encoders.{idx}.mha.W_v.weight"
        mappings[f"roberta.encoder.layer.{idx}.attention.self.value.bias"] = f"model.backbone.backbone.encoders.{idx}.mha.W_v.bias"
        mappings[f"roberta.encoder.layer.{idx}.attention.output.dense.weight"] = f"model.backbone.backbone.encoders.{idx}.mha.ff.weight"
        mappings[f"roberta.encoder.layer.{idx}.attention.output.dense.bias"] = f"model.backbone.backbone.encoders.{idx}.mha.ff.bias"
        mappings[f"roberta.encoder.layer.{idx}.attention.output.LayerNorm.weight"] = f"model.backbone.backbone.encoders.{idx}.norm1.weight"
        mappings[f"roberta.encoder.layer.{idx}.attention.output.LayerNorm.bias"] = f"model.backbone.backbone.encoders.{idx}.norm1.bias"
        mappings[f"roberta.encoder.layer.{idx}.intermediate.dense.weight"] = f"model.backbone.backbone.encoders.{idx}.ff.0.weight"
        mappings[f"roberta.encoder.layer.{idx}.intermediate.dense.bias"] = f"model.backbone.backbone.encoders.{idx}.ff.0.bias"
        mappings[f"roberta.encoder.layer.{idx}.output.dense.weight"] = f"model.backbone.backbone.encoders.{idx}.ff.2.weight"
        mappings[f"roberta.encoder.layer.{idx}.output.dense.bias"] = f"model.backbone.backbone.encoders.{idx}.ff.2.bias"
        mappings[f"roberta.encoder.layer.{idx}.output.LayerNorm.weight"] = f"model.backbone.backbone.encoders.{idx}.norm2.weight"
        mappings[f"roberta.encoder.layer.{idx}.output.LayerNorm.bias"] = f"model.backbone.backbone.encoders.{idx}.norm2.bias"

    mappings[f"lm_head.dense.weight"] = "mlm.dense.weight"
    mappings[f"lm_head.dense.bias"] = "mlm.dense.bias"
    mappings[f"lm_head.layer_norm.weight"] = "mlm.norm.weight"
    mappings[f"lm_head.layer_norm.bias"] = "mlm.norm.bias"
    mappings[f"lm_head.decoder.weight"] = "mlm.mlm_class.weight"
    mappings[f"lm_head.decoder.bias"] = "mlm.mlm_class.bias"

    target_weights = OrderedDict()
    for key in weights:
        if key in mappings:
            target_weights[mappings[key]] = weights[key]
        else:
            print(f"Missing key: {key}")

    return target_weights

model = RobertaForMaskedLM.from_pretrained('roberta-base')
weights = convert_weight_keys(model.state_dict())
with open("roberta-base-pretrained.pickle", "wb") as f:
    pickle.dump(weights, f)
