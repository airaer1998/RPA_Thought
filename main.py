import json
from typing import List, Dict
import argparse
from models.llm import LLMHandler
from data import TAExtractor
from prompts.prompt_base import API_KEY
from utils.logger import setup_logger

logger = setup_logger()

def save_results(results: List[Dict], output_file: str):
    """保存结果到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description='T-A数据抽取工具')
    parser.add_argument('input_file', help='输入文本文件路径')
    parser.add_argument('output_file', help='输出JSON文件路径')
    parser.add_argument('--type', type=int, choices=[1, 2, 3], default=1,
                      help='T-A数据类型 (1: 近距离, 2: 远距离, 3: 多想法)')
    args = parser.parse_args()
    
    try:
        # 读取输入文件
        with open(args.input_file, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # 初始化LLM和抽取器
        llm_handler = LLMHandler(API_KEY)
        extractor = TAExtractor(llm_handler)
        
        # 根据类型进行抽取
        if args.type == 1:
            results = extractor.extract_type1(text)
        elif args.type == 2:
            results = extractor.extract_type2(text)
        else:
            results = extractor.extract_type3(text)
            
        # 保存结果
        save_results(results, args.output_file)
        logger.info(f"处理完成，共抽取出 {len(results)} 个T-A对")
        
    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    main() 