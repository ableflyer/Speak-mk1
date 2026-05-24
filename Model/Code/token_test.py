from multimodal_training import get_tokenizer, MultimodalTrainConfig
cfg = MultimodalTrainConfig()
tokenizer = get_tokenizer(cfg)
print(tokenizer.convert_tokens_to_ids("<|endturn|>"))   # should be non-zero
print(tokenizer.convert_tokens_to_ids("<|ctc_blank|>")) # should be non-zero
print(len(tokenizer))                                    # should be 50291