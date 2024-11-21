import tiktoken
from typing import List

def split_text_into_chunks(
    text: str,
    chunk_size: int = 5000,
    overlap: int = 200
) -> List[str]:
    """
    将文本分割成指定大小的chunk
    
    Args:
        text: 输入文本
        chunk_size: chunk大小（token数）
        overlap: 重叠部分大小（token数）
    
    Returns:
        List[str]: chunk列表
    """
    # 使用tiktoken计算token
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        # 确定chunk结束位置
        end = start + chunk_size
        if end > len(tokens):
            end = len(tokens)
            
        # 获取当前chunk的tokens
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        
        # 调整到合适的分隔符位置
        if end < len(tokens):
            # 寻找最后一个句号、问号、感叹号或换行符
            separators = ['.', '。', '!', '！', '?', '？', '\n']
            last_sep_pos = -1
            for sep in separators:
                pos = chunk_text.rfind(sep)
                if pos > last_sep_pos:
                    last_sep_pos = pos
            
            if last_sep_pos != -1:
                chunk_text = chunk_text[:last_sep_pos + 1]
        
        chunks.append(chunk_text)
        
        # 更新起始位置，考虑重叠
        if end >= len(tokens):
            break
            
        start = end - overlap
        
    return chunks 