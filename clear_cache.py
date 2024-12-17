from transformers import AutoModel, AutoTokenizer
AutoModel.from_pretrained.cache_clear()
AutoTokenizer.from_pretrained.cache_clear()
