from typing import Union


class ClassMapper(object):
	def __init__(self, config):
		self.main_class_to_sub_class_dict = config.main_class_to_sub_class_dict
		# {
		# 	'电脑': ['电脑'],
		# 	'水果': ['水果'],
		# 	'平板': ['平板'],
		# 	'衣服': ['衣服'],
		# 	'酒店': ['酒店'],
		# 	'洗浴': ['洗浴'],
		# 	'书籍': ['书籍'],
		# 	'蒙牛': ['蒙牛'],
		# 	'手机': ['手机'],
		# 	'电器': ['电器'],
		# }

		self.tokenizer = config.tokenizer
		self.words_to_ids = lambda words: tuple(self.tokenizer.convert_tokens_to_ids(list(words)))
		self.ids_to_words = lambda ids: tuple(self.tokenizer.convert_ids_to_tokens(ids))
		# 因为 id 要作为 dict 的 key，所以要转换成 tuple

		self.main_class_ids__to__sub_class_ids_list__dict = {
			self.words_to_ids(main_class_words): [self.words_to_ids(sub_class_words) for sub_class_words in sub_class_words_list]
			for main_class_words, sub_class_words_list in self.main_class_to_sub_class_dict.items()
		}
		# {
		# 	(4510, 5554): [(4510, 5554)],
		# 	(3717, 3362): [(3717, 3362)],
		# 	(2398, 3352): [(2398, 3352)],
		# 	(6132, 3302): [(6132, 3302)],
		# 	(6983, 2421): [(6983, 2421)],
		# 	(3819, 3861): [(3819, 3861)],
		# 	(741, 5093): [(741, 5093)],
		# 	(5885, 4281): [(5885, 4281)],
		# 	(2797, 3322): [(2797, 3322)],
		# 	(4510, 1690): [(4510, 1690)]
		# }

		self.sub_class_ids__to__main_class_ids__dict = {
			sub_class_ids: main_class_ids
			for main_class_ids, sub_class_ids_list in self.main_class_ids__to__sub_class_ids_list__dict.items()
			for sub_class_ids in sub_class_ids_list
		}
		# {
		# 	(4510, 5554): (4510, 5554),
		# 	(3717, 3362): (3717, 3362),
		# 	(2398, 3352): (2398, 3352),
		# 	(6132, 3302): (6132, 3302),
		# 	(6983, 2421): (6983, 2421),
		# 	(3819, 3861): (3819, 3861),
		# 	(741, 5093): (741, 5093),
		# 	(5885, 4281): (5885, 4281),
		# 	(2797, 3322): (2797, 3322),
		# 	(4510, 1690): (4510, 1690)
		# }

	# 返回 LCS 的长度
	# 没有则返回 0
	@staticmethod
	def _get_lcs_len(a: Union[str, list, tuple], b: Union[str, list, tuple]) -> int:
		a, b = list(a), list(b)
		a_len, b_len = len(a), len(b)

		# 存储动态规划的结果
		# 每个位置记录的是到达当前位置的 LCS 的长度
		results_matrix = [[0] * (b_len + 1) for _ in range(a_len + 1)]
		# size: [a_len+1, b_len+1]
		# +1 是因为加入了序列为空串''时的情况
		# 初始化为 0, 表示序列为空时 LCS 一定为0

		# 从两个串有第一个字符开始
		for i in range(1, a_len + 1):
			for j in range(1, b_len + 1):
				if a[i - 1] == b[j - 1]:
					# 如果一个位置的字符相等, 则 LCS 的长度可以比左上角加 1，表示 LCS 长度变长了
					results_matrix[i][j] = results_matrix[i - 1][j - 1] + 1
				else:
					# 如果一个位置的字符不相等，则 LCS 的长度只能取左、上两个位置的 LCS 最大值
					# 代表必须舍弃(跳过)两方其中一个的字符
					results_matrix[i][j] = max(
						results_matrix[i][j - 1],
						results_matrix[i - 1][j],
					)

		return results_matrix[a_len][b_len]

	def _hard_match_main_class_ids_by_inaccurate_sub_class_ids(self, inaccurate_sub_class_ids):
		main_class_ids = None # 最相似的 main_class_ids
		max_lcs_len = -1 # 最长匹配长度
		# 如果没找到，则会找一个 max_lcs_len 为 0 的 main_class_ids 作为匹配结果

		# 遍历所有的 sub_class_ids
		for sub_class_ids, main_class_ids in self.sub_class_ids__to__main_class_ids__dict.items():
			lcs_len = self._get_lcs_len(sub_class_ids, inaccurate_sub_class_ids) # 计算 LCS 的长度
			if lcs_len > max_lcs_len: # 如果有更匹配的，则更新
				main_class_ids = self.sub_class_ids__to__main_class_ids__dict[sub_class_ids]
				max_lcs_len = lcs_len
		return main_class_ids # 返回最相似的 main_class_ids

	def _find_main_class_ids_by_inaccurate_sub_class_ids(self, inaccurate_sub_class_ids):
		# 先进行精确匹配，看是否有匹配的 sub_class_ids
		# 如果有，则直接返回对应的 main_class_ids
		if inaccurate_sub_class_ids in self.sub_class_ids__to__main_class_ids__dict:
			return self.sub_class_ids__to__main_class_ids__dict[inaccurate_sub_class_ids]
		# 如果没有，则进行硬匹配，返回最相似的 main_class_ids
		return self._hard_match_main_class_ids_by_inaccurate_sub_class_ids(inaccurate_sub_class_ids)

	# 用于评估时判断对 main_class 的预测是否准确
	# inaccurate_sub_class_ids: [(4510, 5554), (3717, 3362), ...]
	# 返回：[(4510, 5554), (3717, 3362), ...]
	def batch_find_main_class_ids_by_inaccurate_sub_class_ids(self, inaccurate_sub_class_ids_list):
		return [
			# inaccurate_sub_class_ids(tensor) -> tuple -> main_class_ids - (4510, 5554)
			self._find_main_class_ids_by_inaccurate_sub_class_ids(tuple(inaccurate_sub_class_ids.tolist()))
			for inaccurate_sub_class_ids in inaccurate_sub_class_ids_list
		]

	# 用于训练时计算损失
	# main_class_ids_list - [(4510, 5554), (3717, 3362), ...]
	def batch_find_sub_classes_ids_list_by_main_class_ids(self, main_class_ids_list):
		return [
			# main_class_ids(tensor) -> tuple -> sub_class_ids_list - [(4510, 5554), ...]
			self.main_class_ids__to__sub_class_ids_list__dict[tuple(main_class_ids.tolist())]
			for main_class_ids in main_class_ids_list
		]

# from config import Config
# config = Config('hard')
# classMapper = ClassMapper(config)
# print(classMapper.batch_find_main_class_ids_by_inaccurate_sub_class_ids([(4510, 1620)]))