import json
import scipy.spatial
from zhipuai import ZhipuAI

FILEPATH = "<生成的论点图json文件>.json"
API_KEY = "<你的智谱API密钥>" 

def embedding(text_list,api_key=API_KEY):
    client = ZhipuAI(api_key=api_key)
    embeddings = []
    batch_size = 64
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        response = client.embeddings.create(
            model="embedding-3",
            input=batch,
            dimensions=2048
        )
        embeddings.extend([item.embedding for item in response.data])
    return embeddings

def cosine_similarity(vec1, vec2):
    return 1 - scipy.spatial.distance.cosine(vec1, vec2)

def get_id_number(node_id):
    """提取节点ID中的数字部分并转为整数"""
    return int(node_id.split('_')[-1])

def main(filepath=FILEPATH, api_key=API_KEY, output_path="cleaned_" + FILEPATH):
    with open(filepath, "r", encoding="utf-8") as f:
        argument_graph = json.load(f)

    text_list = [node["text"] for node in argument_graph]
    vec_list = embedding(text_list,api_key)

    similarity_pairs = []

    # 生成相似对时按ID序号排序
    for i in range(len(text_list)):
        for j in range(i + 1, len(text_list)):
            # 比较ID序号，确保smaller_idx对应更小的ID
            i_id = get_id_number(argument_graph[i]["id"])
            j_id = get_id_number(argument_graph[j]["id"])
            if i_id < j_id:
                smaller_idx, larger_idx = i, j
            else:
                smaller_idx, larger_idx = j, i
            
            similarity = cosine_similarity(vec_list[i], vec_list[j])
            similarity_pairs.append((similarity, smaller_idx, larger_idx))

    similarity_pairs.sort(reverse=True, key=lambda x: x[0])

    # # 输出最相似的前十对
    # print("最相似的前十对节点：")
    # for i in range(min(10, len(similarity_pairs))):
    #     similarity, idx1, idx2 = similarity_pairs[i]
    #     print(f"余弦相似度 ({text_list[idx1]}, {text_list[idx2]}): {similarity}")

    threshold = 0.85
    to_delete = set()
    redirect_map = {}

    # 处理相似对，建立重定向映射
    for similarity, smaller_idx, larger_idx in similarity_pairs:
        if similarity <= threshold:
            break
        if larger_idx in to_delete:
            continue
        
        # 获取保留节点的最终ID（处理链式重定向）
        retained_id = argument_graph[smaller_idx]["id"]
        while retained_id in redirect_map:
            retained_id = redirect_map[retained_id]
        
        # 记录映射关系
        deleted_id = argument_graph[larger_idx]["id"]
        redirect_map[deleted_id] = retained_id
        to_delete.add(larger_idx)

    # 构建更新后的图结构
    deleted_ids = {argument_graph[idx]["id"] for idx in to_delete}
    updated_graph = [node for node in argument_graph if node["id"] not in deleted_ids]

    # 更新所有节点的target_id
    for node in updated_graph:
        target_id = node.get("target_id")
        if target_id in redirect_map:
            node["target_id"] = redirect_map[target_id]

    # 保存结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(updated_graph, f, ensure_ascii=False, indent=4)

    print("处理完成，已删除重复节点并更新引用关系。")
    
if __name__ == "__main__":
    main()