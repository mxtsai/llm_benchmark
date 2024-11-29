import time
import argparse

from llm import Local_LLM

english_questions = [
    "Analyze the concept of justice in Plato's 'Republic,' comparing it to modern interpretations. Please provide a lengthy answer.",
    "Explain in detail the process of photosynthesis, including both the light-dependent and light-independent reactions. Please give a lengthy answer.",
    "Discuss the themes of identity and alienationåå in Franz Kafka's 'The Metamorphosis.' Provide a detailed and lengthy response.",
    "Examine the causes and consequences of the French Revolution. Please provide a comprehensive and lengthy answer.",
    "Analyze the effects of globalization on emerging economies, providing examples. Please give a lengthy response.",
    "Discuss the ethical implications of artificial intelligence in modern society. Provide a detailed and lengthy answer.",
    "Explore the evolution of Impressionism in art, citing key artists and works. Please provide a lengthy response.",
    "Explain Jean Piaget's stages of cognitive development, providing examples for each stage. Please give a lengthy answer.",
    "Discuss the impacts of deforestation on biodiversity and climate change. Provide a detailed and lengthy response.",
    "Analyze the role of social media in shaping public opinion. Please provide a lengthy answer.",
    "Examine the concept of democracy in ancient Athens compared to modern democratic systems. Please give a lengthy answer.",
    "Discuss the benefits and challenges of universal healthcare systems. Provide a detailed and lengthy response.",
    "Analyze the impact of technology on modern education. Please provide a lengthy answer.",
    "Discuss the cultural significance of rituals in indigenous societies. Please give a lengthy response.",
    "Examine the role of religion in shaping moral values across different societies. Please provide a detailed and lengthy answer.",
    "Discuss the importance of the separation of powers in a democratic government. Please provide a lengthy answer.",
    "Analyze the ethical considerations in genetic engineering. Please give a detailed and lengthy response.",
    "Explore the influence of African music on contemporary genres. Please provide a lengthy answer.",
    "Explain the life cycle of a star, from birth to death. Please give a detailed and lengthy answer.",
    "Discuss the impact of leadership styles on organizational performance. Provide a lengthy answer."
]

traditional_chinese_questions = [
    "分析柏拉圖《理想國》中正義的概念，並與現代詮釋進行比較。請提供詳盡的回答。",
    "詳細解釋光合作用的過程，包括光反應和暗反應。請給出長篇回答。",
    "討論卡夫卡《變形記》中的身份和疏離主題。請提供詳細且長篇的回應。",
    "考察法國大革命的原因和後果。請提供全面且詳盡的回答。",
    "分析全球化對新興經濟體的影響，並舉例說明。請提供長篇回應。",
    "討論人工智能在現代社會的倫理影響。請給出詳細且長篇的回答。",
    "探索印象派藝術的演變，列舉主要藝術家和作品。請提供長篇回應。",
    "解釋皮亞傑提出的認知發展階段，為每個階段提供例子。請給出詳盡的回答。",
    "討論森林砍伐對生物多樣性和氣候變化的影響。請提供長篇回答。",
    "分析社交媒體在塑造公眾意見中的角色。請提供詳細且長篇的回應。",
    "考察古雅典的民主概念與現代民主制度的比較。請提供長篇回答。",
    "討論全民醫療保健系統的優點和挑戰。請給出長篇回答。",
    "分析科技對現代教育的影響。請提供長篇回應。",
    "討論儀式在原住民社會中的文化意義。請提供詳細且長篇的回答。",
    "考察宗教在塑造不同社會道德價值觀中的角色。請給出長篇回答。",
    "討論權力分立在民主政府中的重要性。請提供長篇回答。",
    "分析基因工程的倫理考量。請提供詳細且長篇的回應。",
    "探索非洲音樂對當代音樂流派的影響。請給出長篇回答。",
    "解釋恆星的生命週期，從誕生到死亡。請提供詳細且長篇的回答。",
    "討論領導風格對組織績效的影響。請提供長篇回應。"
]


def get_tokens_per_second(model, questions, batch_size=1):

    batch_start = time.time()
    responses = model.batch_inference(questions, batch_size=batch_size)
    batch_end = time.time()

    # batch stats
    token_stats = model.get_token_count()
    tokens_per_second = token_stats["completion_tokens"] / (batch_end - batch_start)

    # per requrest stats
    tok_per_sec = []
    for response in responses:
        tok_per_sec.append(response["completion_tokens"] / response["latency"])
    
    avg_per_req_tok_per_sec = sum(tok_per_sec) / len(tok_per_sec)

    return {"batch": tokens_per_second, "per_request": avg_per_req_tok_per_sec}, responses

def parse_args():

    parser = argparse.ArgumentParser(description="Benchmark LLM model")
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=str, default="9000")
    parser.add_argument("--batch_size", type=int, default=1)

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    model = Local_LLM(args.ip, args.port)

    questions = english_questions + traditional_chinese_questions

    tokens_per_second, responses = get_tokens_per_second(model, questions, args.batch_size)

    print(f"Batch Size {args.batch_size} Tok/Sec for {len(questions)} inputs: {tokens_per_second['batch']:.4f} tok/sec")
    print(f"Per Request Token/Sec: {tokens_per_second['per_request']:.4f} tok/sec")
