import json
import matplotlib.pyplot as plt
from zhipuai import ZhipuAI
from 绘制论点拓扑图 import main as visualize_graph
from 论点查重 import main as check_similarity

# 配置区
MODEL_READ1 = 'GLM-4-Air-0111'  # 长文本模型
MODEL_READ2 = 'GLM-Zero-Preview'  # 短文本模型
MODEL_EVALUATION = 'GLM-4-Plus' # 评委点评模型
TOPIC = "<辩题完整表述>"  # 辩题
API_KEY = "<你的apikey>"  # 请填入你的 API Key
FILEPATH = "<你的辩论赛json>" # 请填入已处理成json格式的辩论赛文本，用“录音转文字toJson.py”处理
ARGUMENT_ROUNDS = [1,3]  # 立论环节的轮数
WINDOW_LENGTH = 3 # 窗口长度

# 设置字体为 SimHei
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 用于截取文本片段，避免图中节点标签过长
def text_snippet(text, length=30):
    return text if len(text) <= length else text[:length] + "..."

# 添加清洗大模型返回的回复的函数
def clean_model_response(response):
    start = response.find('[')
    end = response.rfind(']')
    if start != -1 and end != -1:
        return response[start:end+1]
    else:
        start = response.find('{')
        end = response.rfind('}')
        if start != -1 and end != -1:
            return '[' + response[start:end+1] + ']'
    return response

############################################
# 数据结构定义：每个节点代表一次发言更新
############################################
class UtteranceNode:
    def __init__(self, node_id, speaker, text, node_type, base_importance=0.0, target_id=None, delta=0.0, round_number=None):
        """
        node_type: 
          - "new_argument": 新增论点（需要 base_importance），
          - "support": 支持（需要 target_id 与正 delta），
          - "attack": 反驳（需要 target_id 与负 delta）。
        """
        self.node_id = node_id
        self.speaker = speaker
        self.text = text
        self.node_type = node_type
        self.base_importance = base_importance
        self.target_id = target_id
        self.delta = delta
        self.round_number = round_number  # 用于记录该节点所在的辩论轮数

    def to_dict(self):
        return {
            "id": self.node_id,
            "speaker": self.speaker,
            "text": self.text,
            "node_type": self.node_type,
            "base_importance": self.base_importance,
            "target_id": self.target_id,
            "delta": self.delta,
            "round_number": self.round_number
        }

############################################
# 辩论论点拓扑图：管理所有发言节点
############################################
class DebateGraph:
    def __init__(self):
        # 所有节点存放在 nodes 字典中，键为 node_id
        self.nodes = {}
        
    def add_node(self, node: UtteranceNode):
        self.nodes[node.node_id] = node

    def remove_node(self, node_id):
        if node_id in self.nodes:
            del self.nodes[node_id]

    def remove_duplicate_nodes(self):
        # 修改：遍历 self.nodes，而不是 self.graph.nodes
        seen_texts = {}
        nodes_to_remove = []
        for node_id, node in self.nodes.items():
            text = node.text
            if text in seen_texts:
                nodes_to_remove.append(node_id)
            else:
                seen_texts[text] = node_id

        for node_id in nodes_to_remove:
            self.remove_node(node_id)

    def to_json(self):
        nodes_list = [node.to_dict() for node in self.nodes.values()]
        return json.dumps(nodes_list, ensure_ascii=False, indent=2)
    
    ##### 请移步“绘制论点拓扑图.py”程序处理.json文件
    # def visualize_graph(self, filename=f"{TOPIC}.png"):
        
############################################
# LLM 客户端：调用 ChatGLM 接口
############################################
class LLMClient:
    def __init__(self, api_key):
        self.client = ZhipuAI(api_key=api_key)

    def extract_information(self, round_text, graph_snapshot):
        
        prompt_system = ("你是一位专业的辩论分析专家，熟悉辩论评委模型的原理和论点拓扑图的数据结构。当前的论点拓扑图以 JSON 格式表示，是一个数组，每个元素是一个节点，包含字段：\n"
            "  id: 唯一标识符\n"
            "  speaker: 发言持方，可为'Pro' 或 'Con'\n"
            "  text: 发言内容\n"
            "  node_type: 发言类型，可为 'new_argument'（新增论点）、'support'（支持）、'attack'（反驳）\n"
            "  base_importance: 如果是新论点，该值为初始重要性（浮点数）；否则为 0\n"
            "  target_id: 如果是支持或反驳，该字段表示目标论点的 id，否则为 null\n"
            "  delta: 如果是支持或反驳，该字段表示对目标论点的重要性增减值（支持为正，反驳为负）\n"
            "  round_number: 发言所在的辩论轮数\n"
            "\n"
            "  你可以根据当前的论点拓扑图情况，了解当前的辩论战局。如果拓扑图为空，意味着比赛刚刚开始，是立论环节\n\n"
            
            "你需要根据当前的辩论环节发言，分析其中的论点、支持和反驳，并按照要求更新论点拓扑图。"
            "请为每个更新生成一条指令。每条指令必须是一个 JSON 对象，包含如下字段：\n"
            "  - speaker: 发言者\n"
            "  - action: 'new_argument' 或 'support' 或 'attack'\n"
            "  - 如果 action 为 'new_argument'，请提供 'text' 和 'importance'（重要性取值 [0,1.5]）\n"
            "  - 如果 action 为 'support' 或 'attack'，请提供 'target_id'、'text' 和 'delta'（支持为正，反驳为负，对应取值[0,0.5]或者[-0.5,0]）\n"
            "\n"
            "下面是importance和delta的赋值标准：\n"
            "  1. 新增论点：根据论点的清晰度和新颖性，初始重要性取值范围为[0,1.5]，0为“很弱”，1.5为“很强”。\n"
            "  2. 支持：根据论据支持力度，delta取值范围为[0,0.5]\n"
            "  3. 反驳：根据反驳力度，delta取值范围为[-0.5,0]\n"
            "你可以在取值范围内按照你对论点的分析进行自由取值。你的评分应当大胆而非中庸，不必排斥给出很低的分数。\n\n"
            
            "下面是你的注意事项："
            "  1. 请注意，一个环节中很可能不止提供一个论点/支持/反驳，需要更新的拓扑图节点可能有多个，因此，你可能需要生成多个更新。\n"
            "  2. 请为每个更新生成一条指令，确保输出严格符合 JSON 格式，不要附加其他任何内容。\n"
            "  3. 如果没有任何更新，请输出 []\n\n"
            "  4. 如果某项内容可以被识别为support或者attack，则尽量不要识别为new_argument。一般情况下，一场比赛的一个持方可以被识别为新论点的，不超过5个\n"
            "  5. 忽略“主席”的发言\n"
            "  6. 在质询和对辩（双方都会发言的环节），你需要精准地从双方的问题中提炼出合适的攻防逻辑，概括后作为support或者attack的text写入，而不是直接复述原问题，特别是不要保留口语化的表达\n"
            "  7. 对于已出现过的论点，如果选手重复发言，千万千万不要再次添加论点！！无论如何，你都不应该添加与已有节点的text相似甚至相同的节点！！！！\n"
            "  8. 你需要对发言内容进行精简且恰当的**概括**然后再写入text字段，而不是直接复读原文\n"
            "  9. 论据的类型可能包括：**事实、数据、逻辑推理、权威引用、案例分析、历史事件、比较分析、价值观、常识判断**等，你需要把论据的内容概括作为support或attack的text写入\n"
            " 10. 如果论点拓扑图为空，意味着比赛刚刚开始，是立论环节。在立论环节，你需要识别一辩立论中的论据（即support），程序会帮你处理未知的target_id\n\n"
            
            f"这一场比赛的辩题为：{TOPIC}，你可以根据论点对辩题的论证力度来判断论点的重要性。"
        )
            
        prompt_user = (
            "请你分析整个辩论环节的发言，提取其中的论点、支持和反驳，并按照要求更新论点拓扑图。注意任何情况下都不应该添加重复或者相似的论点。"
            
            "现有一整个辩论环节的发言如下，每一行格式为 'Speaker: 发言内容'：\n"
            f"{round_text}\n\n"
            
            "当前的论点拓扑图如下：\n"
            f"{graph_snapshot}\n"
        )
        
        if(len(round_text)+len(graph_snapshot)+ 800 > 16000):
            model = MODEL_READ1
        else:
            model = MODEL_READ2
        # 如果文本长度过长，使用更大的模型
        
        response = self.client.chat.completions.create(
            model = model,
            messages=[
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user}
            ],
            temperature=0.5,
            max_tokens=4025
        )
        
        llm_output = response.choices[0].message.content
        
        try:
            data = json.loads(clean_model_response(llm_output))
        except json.JSONDecodeError:
            raise ValueError("LLM输出无法解析为JSON: " + llm_output)
        return data

    def generate_commentary(self, details):
        prompt = (
            "你是一位资深辩论评委，请根据以下辩论比赛数据生成评委点评，要求清晰地梳理场上的攻防过程，让观众信服这场比赛的结果。数据如下：\n"
            "论点拓扑图以 JSON 格式表示，是一个数组，每个元素是一个节点，包含字段：\n"
            "  id: 唯一标识符\n"
            "  speaker: 发言持方，可为'Pro' 或 'Con'\n"
            "  text: 发言内容\n"
            "  node_type: 发言类型，可为 'new_argument'（新增论点）、'support'（支持）、'attack'（反驳）\n"
            "  base_importance: 如果是新论点，该值为初始重要性（浮点数）；否则为 0\n"
            "  target_id: 如果是支持或反驳，该字段表示目标论点的 id，否则为 null\n"
            "  delta: 如果是支持或反驳，该字段表示对目标论点的重要性增减值（支持为正，反驳为负）\n"
            "  round_number: 发言所在的辩论轮数\n"
            "\n"
            "你可以根据当前的论点拓扑图情况，了解当前的辩论战局。\n\n"
            f"比赛辩题：{TOPIC}\n"
            f"Pro 总论点得分: {details.get('Pro_score')}\n"
            f"Con 总论点得分: {details.get('Con_score')}\n"
            f"比赛结果: {details.get('result')}\n"
            f"论点拓扑图：{details.get('graph_snapshot')}\n"
            "请输出中文点评，2000字左右。你需要像一个人类评委那样点评，也就是最好不要表现出“我是从论点拓扑图里得到的比赛信息”的姿态。而是不露痕迹地梳理攻防，判断哪些论点立住了，哪些论点被挑战了。"
        )
        response = self.client.chat.completions.create(
            model=MODEL_EVALUATION,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=4025
        )
        commentary = response.choices[0].message.content
        return commentary

############################################
# 辩论评委模型：整合调用与图更新
############################################
class DebateJudgeModel:
    def __init__(self, api_key, argument_rounds):
        self.graph = DebateGraph()
        self.node_counter = 0
        self.llm_client = LLMClient(api_key)
        self.argument_rounds = argument_rounds  # 保存立论环节的轮数
        global WINDOW_LENGTH
        self.window_length = WINDOW_LENGTH  # 保存窗口长度

    def process_round(self, transcript, round_number):
        # 整合本轮发言文本
        round_text_lines = [f"{text['speaker']}: {text['text']}" for text in transcript]
        round_text = "\n".join(round_text_lines)
        
        # 构造 graph_snapshot，根据轮次判断
        if round_number in self.argument_rounds:
            # 立论轮：传入空的论点拓扑图
            graph_snapshot = "[]"
        else:
            if round_number <= (2 + self.window_length):
                # 如果当前轮次较小，则传入完整的图
                graph_snapshot = self.graph.to_json()
            else:
                # 传入立论轮和最近 window_length 轮的节点
                relevant_nodes = []
                for node in self.graph.nodes.values():
                    # 注意：要求每个节点在创建时记录了所属的 round_number
                    if (node.round_number in self.argument_rounds) or (node.round_number is not None and node.round_number >= round_number - self.window_length):
                        relevant_nodes.append(node.to_dict())
                graph_snapshot = json.dumps(relevant_nodes, ensure_ascii=False, indent=2)
        
        try:
            updates = self.llm_client.extract_information(round_text, graph_snapshot)
        except Exception as e:
            print(f"Round {round_number}: LLM调用出错：{e}")
            return

        if not updates:
            print(f"Round {round_number}: 无更新指令。")
            return

        # 在 process_round 方法中，替换原来的更新处理循环：
        last_new_argument_id = None  # 记录最新加入的 new_argument 节点的 node_id

        for update in updates:
            action = update.get("action", "none")
            speaker = update.get("speaker", "Unknown")
            if action == "new_argument":
                self.node_counter += 1
                node_id = f"node_{self.node_counter}"
                text = update.get("text", "")
                try:
                    importance = float(update.get("importance", 1.0))
                except Exception:
                    importance = 1.0
                new_node = UtteranceNode(node_id, speaker, text, "new_argument", base_importance=importance, round_number = round_number)
                self.graph.add_node(new_node)
                last_new_argument_id = node_id  # 更新记录
                print(f"Round {round_number}: 添加新论点 {node_id}（{speaker}）：{text}，初始重要性：{importance}")
            elif action in ("support", "attack"):
                # 直接尝试获取 target_id
                target_id = update.get("target_id")
                # 如果当前为第一轮或者 target_id 无效，则使用上一个 new_argument 的 node_id
                if round_number == 1 or not target_id or target_id not in self.graph.nodes:
                    if last_new_argument_id is None:
                        print(f"Round {round_number}: {speaker} 的更新 {update} 无法找到对应的 new_argument 节点，跳过。")
                        continue
                    target_id = last_new_argument_id
                self.node_counter += 1
                node_id = f"node_{self.node_counter}"
                text = update.get("text", "")
                try:
                    delta = float(update.get("delta", 0.0))
                except Exception:
                    delta = 0.0
                if action == "attack" and delta > 0:
                    delta = -delta
                new_node = UtteranceNode(node_id, speaker, text, action, target_id=target_id, delta=delta, round_number=round_number)
                self.graph.add_node(new_node)
                print(f"Round {round_number}: {speaker} 的更新 {node_id} 被识别为 {action}，针对 {target_id}：{text}，影响值：{delta}")
            else:
                print(f"Round {round_number}: {speaker} 的更新指令未识别：{update}")

        # 移除重复的节点（用文本匹配去重）
        self.graph.remove_duplicate_nodes()
        
    def evaluate_debate(self):
        """
        计算各队得分：对每个新增论点，其得分为该论点的 base_importance 加上所有对其的支持/反驳 delta（仅计正贡献）。
        """
        team_scores = {"Pro": 0.0, "Con": 0.0}
        for node in self.graph.nodes.values():
            if node.node_type == "new_argument":
                aggregated = node.base_importance
                # 累加所有支持/反驳对该论点的贡献
                for other in self.graph.nodes.values():
                    if other.target_id == node.node_id:
                        aggregated += other.delta
                # 只计入正向贡献
                if node.speaker not in team_scores:
                    print(f"错误：发现未知发言者 '{node.speaker}'")
                    continue
                team_scores[node.speaker] += aggregated if aggregated > 0 else 0
        Pro_score = team_scores.get("Pro", 0.0)
        Con_score = team_scores.get("Con", 0.0)
        result = "Tie"
        if Pro_score > Con_score:
            result = "Pro"
        elif Con_score > Pro_score:
            result = "Con"
        print("\n【评估阶段】")
        print(f"Pro 总得分：{Pro_score:.2f}")
        print(f"Con 总得分：{Con_score:.2f}")
        print(f"比赛结果：{result}")
        return Pro_score, Con_score, result

    def generate_judgement_commentary(self, Pro_score, Con_score, result, graph_snapshot):
        details = {
            "Pro_score": Pro_score,
            "Con_score": Con_score,
            "result": result,
            "graph_snapshot": graph_snapshot
        }
        try:
            commentary = self.llm_client.generate_commentary(details)
        except Exception as e:
            print(f"LLM生成点评出错：{e}")
            commentary = "辩论十分激烈，双方都展现了出色的实力。"
        return commentary

############################################
# 主流程
############################################
def main():
    filepath = FILEPATH
    with open(filepath, 'r', encoding='utf-8') as f:
        transcripts = json.load(f)
        
    api_key = API_KEY
    judge_model = DebateJudgeModel(api_key, ARGUMENT_ROUNDS)
    round_number = 1
    for transcript in transcripts:
        print(f"\n==== 开始第 {round_number} 轮辩论 ====")
        judge_model.process_round(transcript, round_number)
        round_number += 1

    print("\n==== 辩论结束 ====\n")

    with open(f"{TOPIC}.json", 'w', encoding='utf-8') as f:
        f.write(judge_model.graph.to_json())
    print(f"\n论点拓扑图json已保存为 {TOPIC}.json")

    # 使用词向量技术对论点拓扑图进行清洗查重
    check_similarity(f"{TOPIC}.json", api_key, f"{TOPIC}.json")
    
    # 读取清洗后的论点拓扑图
    with open(f"{TOPIC}.json", 'r', encoding='utf-8') as f:
        cleaned_graph = json.load(f)
        
    # 把清洗后的论点拓扑图传入评分函数
    judge_model.graph = DebateGraph()
    for node_dict in cleaned_graph:
        node = UtteranceNode(
            node_id=node_dict["id"],
            speaker=node_dict["speaker"],
            text=node_dict["text"],
            node_type=node_dict["node_type"],
            base_importance=node_dict.get("base_importance", 0.0),
            target_id=node_dict.get("target_id"),
            delta=node_dict.get("delta", 0.0),
            round_number=node_dict.get("round_number")
        )
        judge_model.graph.add_node(node)
        
    Pro_score, Con_score, result = judge_model.evaluate_debate()
    
    print("\n==== 最终结果 ====")
    print(f"Pro 得分：{Pro_score:.2f}")
    print(f"Con 得分：{Con_score:.2f}")
    print(f"比赛结果：{result}")
    print("\n评委点评：")
    commentary = judge_model.generate_judgement_commentary(Pro_score, Con_score, result, json.dumps(cleaned_graph, ensure_ascii=False, indent=2))
    print(commentary)
    
    # 输出论点拓扑图图片
    visualize_graph(f"{TOPIC}.json",TOPIC)


if __name__ == "__main__":
    main()
