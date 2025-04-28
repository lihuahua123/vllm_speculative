curl --noproxy "*" -X POST "http://localhost:8000/v1/chat/completions" \
	-H "Content-Type: application/json" \
	--data '{
		"model": "/data/model/deepseek-aiDeepSeek-R1-Distill-Qwen-7B",
		"messages": [
			{
				"role": "user",
				"content": "下列选项中，找出与众不同的一个：1.铝 2.锡 3.钢 4.铁 5.铜。"
			}
		]
	}'

curl --noproxy "*" -X POST "http://localhost:8000/speculative_action" \
	-H "Content-Type: application/json" \
	--data '{
		"action": 0
	}'