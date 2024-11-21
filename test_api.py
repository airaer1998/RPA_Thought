import json
from models.llm import LLMHandler
from data import TAExtractor
from prompts.prompt_base import API_KEY
from utils.logger import setup_logger

logger = setup_logger("test_api")

def test_llm_connection():
    """测试LLM API连接"""
    try:
        llm = LLMHandler(API_KEY)
        messages = [
            {"role": "user", "content": "Hello, this is a test message."}
        ]
        response = llm.get_completion(messages)
        logger.info("LLM API连接测试成功")
        logger.info(f"测试响应: {response}")
        return True
    except Exception as e:
        logger.error(f"LLM API连接测试失败: {str(e)}")
        return False

def test_character_extraction():
    """测试角色提取功能"""
    test_text = """
    罗恩的嘴巴蠕动着，却发不出声音，就像一条出水的金鱼。
    这时赫敏猛地转身，气呼呆地登上女生宿舍楼梯，回去睡觉了。
    罗恩转过头来望着哈利。"你看看，"他结结巴巴地说。
    """
    try:
        llm = LLMHandler(API_KEY)
        extractor = TAExtractor(llm)
        characters = extractor._extract_characters(test_text)
        logger.info("角色提取测试成功")
        logger.info(f"提取到的角色: {characters}")
        return True
    except Exception as e:
        logger.error(f"角色提取测试失败: {str(e)}")
        return False

def test_ta_extraction_type1():
    """测试第一类T-A数据提取"""
    test_text = """
    罗恩的嘴巴蠕动着，却发不出声音，就像一条出水的金鱼。这时赫敏猛地转身，气呼呆地登上女生宿舍楼梯，回去睡觉了。
    罗恩转过头来望着哈利。"你看看，"他结结巴巴地说，看样子完全被惊呆了，"你看看——这叫什么事儿——完全没有抓住问题的实质——"
    哈利没有吭声。他很珍惜现在他和罗恩又说话了，因此他谨慎地保持沉默，没有说出自己的观点——实际上，他认为跟罗恩比起来，赫敏才更准确地抓住了问题的实质。
    """
    try:
        llm = LLMHandler(API_KEY)
        extractor = TAExtractor(llm)
        results = extractor.extract_type1(test_text)
        logger.info("第一类T-A数据提取测试成功")
        logger.info(f"提取结果: {json.dumps(results, ensure_ascii=False, indent=2)}")
        return True
    except Exception as e:
        logger.error(f"第一类T-A数据提取测试失败: {str(e)}")
        return False

def test_text_splitting():
    """测试文本分块功能"""
    from utils.utils import split_text_into_chunks
    test_text = "这是第一句话。这是第二句话。这是第三句话。这是第四句话。这是第五句话。"
    try:
        chunks = split_text_into_chunks(test_text, chunk_size=10, overlap=2)
        logger.info("文本分块测试成功")
        logger.info(f"分块结果: {chunks}")
        return True
    except Exception as e:
        logger.error(f"文本分块测试失败: {str(e)}")
        return False

def run_all_tests():
    """运行所有测试"""
    logger.info("开始运行测试...")
    
    tests = [
        ("LLM API连接测试", test_llm_connection),
        ("文本分块测试", test_text_splitting),
        ("角色提取测试", test_character_extraction),
        ("T-A数据提取测试", test_ta_extraction_type1)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n开始{test_name}")
        success = test_func()
        results.append((test_name, success))
        
    logger.info("\n测试结果汇总:")
    for test_name, success in results:
        status = "通过" if success else "失败"
        logger.info(f"{test_name}: {status}")

if __name__ == "__main__":
    run_all_tests() 