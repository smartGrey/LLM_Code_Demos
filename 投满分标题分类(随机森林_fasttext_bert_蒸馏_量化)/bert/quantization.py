import torch
from bert.bert_model_and_config import Config, Model
from bert.train_eval_test_predict import final_test
from torchao import quantize_
from torchao.quantization.quant_api import int8_weight_only


config = Config(torch.device('cpu'))
model = Model(config).to(config.device)
model.load_state_dict(torch.load(config.save_model_path, map_location='cpu'))


# 先对原模型进行测试
final_test(model, config)

# 动态量化
quantize_(model, int8_weight_only(group_size=32)) # 按组量化
print('量化完成.')


# 测试量化模型的性能
final_test(model, config)


# 保存量化后的模型
torch.save(model.state_dict(), config.save_quantify_model_path)