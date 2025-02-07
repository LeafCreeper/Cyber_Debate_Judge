from zhipuai import ZhipuAI
from concurrent.futures import ThreadPoolExecutor, as_completed

FILEPATH = "<辩论赛的录音转文字>.txt"   # 填入文本文件路径
API_KEY = "<你的apikey>"    # 智谱清言API密钥

# 使用飞书妙记把录音文件转换成带说话人标记的文本
# 飞书妙记转换成的文本文件格式里，用\n\n来分隔多个发言
# 本程序调用zhipuai，将文本转换成json格式

i = 0

def clean_model_response(response):
    start = response.find('{')
    end = response.rfind('}')
    if start != -1 and end != -1:
        return response[start:end+1]
    return response

def toJson(content, api_key):
    client = ZhipuAI(api_key=api_key)  # 请填写您自己的APIKey
    systemprompt = (
        "你负责把录音转文字的文本转换成json格式。你应当对文本进行适当的清洗，要求如下：\n"
        "1. 去掉因录音转文字产生的无意义的语气词;\n"
        "2. 去掉重复的语句，处理识别错误的同音字，修改错误的识别，让文本更加连贯自然;\n"
        "3. 如果同一个说话人的发言被分成了多段，你应当将其合并成一段。\n"
        "4. 除此之外，你不应该改动文本的原义，更不应擅自删除文本。但是，在不改动原义的前提下，你有权修改文本让文本更可读\n"
        "5. 你的输出应当只包含json格式，没有任何其他的字符。\n"
        "6. 严格按照文本中已标注的说话人名字来标注speaker，不要随意更改说话人的名字，哪怕通过内容可以判断其身份也不可以修改说话人名字\n"
        "下面是json格式的示例：\n"
        '[\n'
        '{"speaker": "说话人 1", "text": "这是A说的话"},\n'
        '{"speaker": "说话人 3", "text": "这是B说的话"},\n'
        "...\n"
        ']'
    )
    
    response = client.chat.completions.create(
        model="GLM-4-Air-0111",  # 请填写您要调用的模型名称
        messages=[
            {"role": "system", "content": systemprompt},
            {"role": "user", "content": f"下面，请你清洗这段文本，并且按照要求转换成json格式：{content}"},    
        ],
    )
    output = clean_model_response(response.choices[0].message.content)
    global i
    i += 1
    print(f"已处理chunks：{i}\n")
    return output

def parse_text(filepath, chunk_length):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    lines = content.split('\n\n')
    
    chunks = []
    for i in range(0, len(lines), chunk_length):
        chunk = '\n'.join(lines[i:i + chunk_length])
        chunks.append(chunk)
        
    print(f"Total lines: {len(lines)}")
    print(f"Total chunks: {len(chunks)}")
    
    return chunks

def merge_jsons(jsons):
    merged = '[' + ','.join(jsons) + ']'
    return merged

def process_chunk(index, chunk, api_key, jsons):
    jsons[index] = toJson(chunk, api_key)

def main(filepath, api_key, chunk_length, max_threads):
    chunks = parse_text(filepath, chunk_length)
    jsons = [None] * len(chunks)
    
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(process_chunk, i, chunk, api_key, jsons) for i, chunk in enumerate(chunks)]
        for future in as_completed(futures):
            future.result()  # 等待所有线程完成
    
    merged_json = merge_jsons(jsons)
    return merged_json

def output_to_jsonfile(json, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(json)

if __name__ == '__main__':
    filepath = FILEPATH
    api_key = API_KEY
    chunk_length = 5
    max_threads = 50  # 设置最大线程数量
    merged_json = main(filepath, api_key, chunk_length, max_threads)
    output_path = "toinput"+f"{filepath[:filepath.rfind('.')] if '.' in filepath else filepath}.json"
    output_to_jsonfile(merged_json, output_path)