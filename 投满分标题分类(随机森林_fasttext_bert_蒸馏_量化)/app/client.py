import requests
import time


start_time = time.time()

res = requests.post(
	url="http://0.0.0.0:5001/classify/",
	data={"content": "公共英语(PETS)写作中常见的逻辑词汇汇总"}
)

print(f'分类结果: {res.content} 耗时: {(time.time() - start_time)*1000:.2f} ms')
# 分类结果: b'{"Result": "education", "Status": "success"}' 耗时: 2.83 ms
