import fasttext


# 简单使用 fasttext
def do_easy():
	model = fasttext.train_supervised(input='./data/train_for_fast_text.txt', epoch=10, wordNgrams=2)
	model.words.__len__() # 词的总数：118456
	model.labels
	# ic| model.labels: ['__label__4',
	#                    '__label__1',
	#                    '__label__7',
	#                    '__label__5',
	#                    '__label__9',
	#                    '__label__8',
	#                    '__label__2',
	#                    '__label__6',
	#                    '__label__0',
	#                    '__label__3']


	result = model.test('./data/test_for_fast_text.txt')
	# (测试样本数,精确率,召回率)
	# (10000, 0.9107, 0.9107)
# do_easy()


# 自动调参
def auto_tune():
	# 会在验证集上使用随机搜索的方法寻找最优的超参数
	# 调节的超参数包含这些内容:
	# lr                         学习率 default 0.1
	# dim                        词向量维度 default 100
	# ws                         上下文窗口大小 default 5， cbow
	# epoch                      epochs 数量 default 5
	# minCount                   最低词频 default 5
	# wordNgrams                 n-gram设置 default 1
	# loss                       损失函数 {hs,softmax} default softmax
	# minn                       最小字符长度 default 0
	# maxn                       最大字符长度 default 0
	model = fasttext.train_supervised(
		input='./data/train_for_fast_text.txt',
		autotuneValidationFile='./data/test_for_fast_text.txt',
		autotuneDuration=120, # 训练时长 60s
		verbose=3 # 显示训练过程
	)

	# 这里会用得到的最优训练参数重新进行训练，然后再在测试集进行测试
	result = model.test('./data/test_for_fast_text.txt')
	# (10000, 0.9179, 0.9179)

	print('最优参数：', {k:getattr(model, k) for k in ['epoch', 'lr', 'dim', 'minCount', 'wordNgrams', 'ws', 'minCountLabel']})
	# 最优参数： {'epoch': 1, 'lr': 0.9485433223965603, 'dim': 452, 'minCount': 1, 'wordNgrams': 3, 'ws': 5, 'minCountLabel': 0}

	model.save_model('./models/fast_text_auto_tune.bin') # 保存模型
# auto_tune()