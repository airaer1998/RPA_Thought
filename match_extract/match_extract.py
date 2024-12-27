import os
from openai import OpenAI
import openai
import json
import re
from nltk.tokenize import sent_tokenize
import tiktoken
from pydantic import BaseModel
from rapidfuzz import fuzz, process
import logging
from logging.handlers import RotatingFileHandler


client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
directory_path='extracted'
file_list=os.listdir(directory_path)
def setup_logging(log_file='match_extract.log'):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)


    file_handler = RotatingFileHandler(
        log_file, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)


    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
def chat_completion(prompt,text):
    completion=client.beta.chat.completions.parse(
        messages=[
            {"role":"user","content":prompt},
            {"role":"user","content":text}
        ],
        model="gpt-4o",
        response_format=pair
    )
    return completion.choices[0].message.content
def check_completion(prompt):
    completion=client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role":"user","content":prompt}
        ]

    )
    return completion.choices[0].message.content
def pre_process(file_name):
    location="extracted\\"+file_name
    book_name = file_name
    start = book_name.index("extracted_data_") + len("extracted_data_")
    end = book_name.index(".json")
    book_name = book_name[start:end].strip()
    with open(location,"r",encoding='utf-8') as file:
        data=json.load(file)
    return data,book_name
class content(BaseModel):
    content:str
    type:str
class unit(BaseModel):
    character:str
    reason:str
    thought:content
    action:content
    first_sentence:str
    last_sentence:str
class pair(BaseModel):
    ta_pairs:list[unit]



#nltk.download('punkt')
#nltk.download('punkt_tab')



def split_text_into_chunks_by_sentence(text, max_tokens=2000, model="gpt-4o"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = encoding.encode(sentence)
        sentence_length = len(sentence_tokens)
        if current_tokens + sentence_length > max_tokens:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
            current_tokens = sentence_length
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
            current_tokens += sentence_length
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def split_text_into_chunks(text, max_tokens=1500, model="gpt-4o"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    total_tokens = len(tokens)

    chunks = []

    for i in range(0, total_tokens, max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)

    return chunks

def extract_match_response(file_name):
    data,book_name=pre_process(file_name)

    prompt= f"""
First, I will give you a piece of original text from the book <<{book_name}>>.Based on the text,complete the following tasks:
1.Extract the thoughts (Thought) and actions (Action) of the character  from the text, along with an analysis of how the two are related.

===Output Format===
{{
    "ta_pairs": [
        {{
            "character": "Character name",
            "reason": "The reason for the pairing of actions and thoughts"
            "thought": {{
                "content": "The content of the thought",
                "type": "The type of the thought: reasoning-type or feeling-type"
            }},
            "action": {{
                "content": "The content of the action",
                "type": "The type of the action: behavior of speech"
            }},
            "first_sentence": "The first sentence in the original text contains this set of matches.use the text"
            "last_sentence":"The last sentence in the original text contains this set of matches.use the text"
        }}
    ]
}}

===Requirements===
1.Only extract pairs of thoughts and actions that appear within a distance of no more than five sentences.
2.Only extract pairs of thoughts and actions that have a clear causal relationship, where the character's thoughts must be a significant cause of the character's actions
3.Thoughts are categorized into "reasoning-type" and "feeling-type."
4.Actions are categorized into "behavior" and "speech."
5.Only return high-quality pairs; if no suitable pairs are found, return an empty list.
6.Thoughts cannot be the character’s spoken words or actual actions.Thought cannot be a character's lines or dialogue, nor can it contain a character's lines or dialogue.
7.Actions are specific behaviors performed by the character.
8.Both Thoughts and Actions must be from the original text.
9.Both Thoughts and Actions must come from the same character.
10.Thoughts are the character's internal ideas, not expressed outwardly, while Actions are behaviors that the character exhibits.
11.The first_sentence and last_sentence must be the unaltered original text.
12.Thought and action must appear consecutively in the original text without being split or pieced together.
13.Please ensure that your output content is in the same language as the provided text.

===Examples===
Text:Ron's mouth moved soundlessly, like a goldfish out of water. At that moment, Hermione spun around abruptly, stomped up the girls' dormitory stairs in a huff, and went back to bed. Ron turned to look at Harry."Look at that," he stammered, appearing utterly stunned. "Just look at that—completely missing the point—"Harry said nothing. He valued the fact that he and Ron were talking again, so he carefully held his tongue and refrained from expressing his own opinion—in fact, he thought that compared to Ron, Hermione had a much clearer grasp of the real issue.
According to the text,we can extract the following match:

{{
    "ta_pairs": [
        {{
            "character": "Harry",
            "reason": "Although Harry feels Hermione is right and she has more accurately grasped the essence of the problem, he values the fact that he and Ron are speaking to each other again, so he cautiously remains silent and does not voice his opinion.",
            "thought": {{
                "content": "He valued the fact that he and Ron were talking again, so he carefully held his tongue and refrained from expressing his own opinion—in fact, he thought that compared to Ron, Hermione had a much clearer grasp of the real issue.",
                "type": "Reasoning-type"
            }},
            "action": {{
                "content": "Harry said nothing",
                "type": "Behavior"
            }},
            "first_sentence": "Ron's mouth moved soundlessly, like a goldfish out of water."
            "last_sentence":Hermione had a much clearer grasp of the real issue."
        }}
    ]
}}

"""
    dict_list=[]
    plot_chunk=[]
    for plot in data["plots"]:
        text = re.sub(r'\s+', ' ', plot["text"])
        plot_chunk+= split_text_into_chunks_by_sentence(text)

    for chunk in plot_chunk:
        try:
            response=chat_completion(prompt,chunk)
            res_dict=json.loads(response)
            dict_list.append(res_dict)
        except openai.ContentFilterFinishReasonError as e:
            logger.warning(f"内容过滤错误: {e}. 跳过此chunk并继续执行。")
            dict_list.append([])
        except openai.error.OpenAIError as e:
            logger.error(f"OpenAI API 错误: {e}")
            dict_list.append([])
        except Exception as e:
            logger.error(f"未知错误: {e}")
            dict_list.append([])
    with open('match.json', 'w', encoding='utf-8') as f2:
        json.dump(dict_list, f2, ensure_ascii=False, indent=2)

def extract_substring_regex(text, f, l):

    pattern = re.escape(f) + r'(.*?)' + re.escape(l)
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    return ""
def check_thought_is_speech_response(match_file):
        with open(match_file,'r',encoding='utf-8') as file:
            match_data=json.load(file)
        new_match=[]
        for matchs in match_data:
            match_list=[]
            if matchs==[]:
                continue
            for match in matchs:

                prompt=f"""
First,I will provide a piece of article ,the text is as follows:
{match["raw_text"]}
Please, based on the above article fragment, determine whether the following content belongs to or includes character dialogue or speeches. If so, please return True; otherwise, return False.
sentences:{match["thought"]["content"]}
===Examples===
The text:Ron's mouth moved soundlessly, like a goldfish out of water. At that moment, Hermione spun around abruptly, stomped up the girls' dormitory stairs in a huff, and went back to bed. Ron turned to look at Harry."Look at that," he stammered, appearing utterly stunned. "Just look at that—completely missing the point—"Harry said nothing. He valued the fact that he and Ron were talking again, so he carefully held his tongue and refrained from expressing his own opinion—in fact, he thought that compared to Ron, Hermione had a much clearer grasp of the real issue.
sentences:Just look at that—completely missing the point
Obviously, based on the text, these sentences are character dialogues,so the output should be True.
"""
                try:
                    if check_completion(prompt)=="False":
                        continue
                    match_list.append(match)
                except openai.ContentFilterFinishReasonError as e:
                    logger.warning(f"内容过滤错误: {e}. 跳过此match并继续执行。")
                except openai.error.OpenAIError as e:
                    logger.error(f"OpenAI API 错误: {e}")
                except Exception as e:
                    logger.error(f"未知错误: {e}")
            new_match.append(match_list)
        with open('match_thought_check.json', 'w', encoding='utf-8') as f2:
            json.dump(new_match, f2, ensure_ascii=False, indent=2)



def find_best_match_positions_rapidfuzz(M, target, window_size=20):
    best_match = process.extractOne(
        target,
        [M[i:i + window_size] for i in range(len(M) - window_size + 1)],
        scorer=fuzz.ratio
    )
    if best_match:
        matched_str, similarity, idx = best_match
        return idx, idx + window_size, similarity / 100
    return 0, window_size, 0


def extract_substring_similarity_rapidfuzz(M, f, l, window_size=20, similarity_threshold=0.5):
    f_start, f_end, f_similarity = find_best_match_positions_rapidfuzz(M, f, window_size)
    if f_similarity < similarity_threshold:
        print(f"未找到与起始字符串 '{f}' 相似度高于 {similarity_threshold} 的匹配。")
        return ""

    l_search_start = f_end
    if l_search_start >= len(M):
        print("起始字符串之后没有足够的内容来查找结束字符串。")
        return ""


    M_after_f = M[l_search_start:]
    l_start_rel, l_end_rel, l_similarity = find_best_match_positions_rapidfuzz(M_after_f, l, window_size)
    if l_similarity < similarity_threshold:
        print(f"未找到与结束字符串 '{l}' 相似度高于 {similarity_threshold} 的匹配。")
        return ""

    l_start = l_search_start + l_start_rel
    l_end = l_start + window_size

    return M[f_start:l_end]
def generate_raw_text(match_file_name,book_file_name):
    data,book_name=pre_process(book_file_name)
    plot_chunk = []
    for plot in data["plots"]:
        text = re.sub(r'\s+', ' ', plot["text"])
        plot_chunk += split_text_into_chunks_by_sentence(text)
    with open(match_file_name,'r',encoding='utf-8') as f:
        match_data=json.load(f)
    new_match=[]
    for plot,matchs in zip(plot_chunk,match_data):
        match_list=[]
        if matchs==[]:
            continue
        for match in matchs["ta_pairs"]:
            raw_text=extract_substring_similarity_rapidfuzz(M=plot,f=match["first_sentence"],l=match["last_sentence"],window_size=max(len(match["first_sentence"]),len(match["last_sentence"])))
            if raw_text=="":
                continue
            else:
                match["raw_text"] = raw_text

                match_list.append(match)
        new_match.append(match_list)
    with open('match_with_raw_text.json', 'w', encoding='utf-8') as f2:
        json.dump(new_match, f2, ensure_ascii=False, indent=2)



def check_match_have_reason(match_file,book_name):
    with open(match_file,'r',encoding='utf-8') as f:
        match_data=json.load(f)
    new_match=[]
    for matchs in match_data:
        match_list=[]
        if matchs==[]:
            continue
        for match in matchs:
            prompt=f"""
First, I will provide a piece of text, as follows:
{match["raw_text"]}
Based on the provided passage, determine whether the character's thought directly drive his action, that is, whether there is a clear causal relationship between the character's thought and action.If so return True,else return False.
==Question==
the thought of the character {match["character"]}:{match["thought"]["content"]}
the action of the character {match["character"]}:{match["action"]["content"]}
"""
            try:
                if check_completion(prompt) == "False":
                    continue
                match["valid"]="False"
                match_list.append(match)
            except openai.ContentFilterFinishReasonError as e:
                logger.warning(f"内容过滤错误: {e}. 跳过此match并继续执行。")
            except openai.error.OpenAIError as e:
                logger.error(f"OpenAI API 错误: {e}")
            except Exception as e:
                logger.error(f"未知错误: {e}")
        new_match.append(match_list)
    with open(f"match_data_from_claude/{book_name}.json",'w',encoding='utf-8') as f:
        json.dump(new_match, f, ensure_ascii=False, indent=2)


if __name__=='__main__':
    #extract_match_response("extracted_data_权力的游戏.json")
    #check_thought_is_speech_response("match_test.json")
    #generate_raw_text("test.json", "extracted_data_权力的游戏.json")
    #check_match_have_reason("match_thought_check.json")
    logger = setup_logging()
    datas=[]
    count=1
    """
    with open("books_data_nonfiction_full.jsonl", 'r', encoding='utf-8') as file:
        for line in file:
            try:
                json_obj = json.loads(line.strip())
                datas.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"解析JSON时出错: {e}")
                continue
    for data in datas:
        if count<49:
            count=count+1
            continue
        extract_match_response(data)
        generate_raw_text("match.json",data)
        check_thought_is_speech_response("match_with_raw_text.json")
        check_match_have_reason("match_thought_check.json",book_name=data["title"])
        print(f"第{count}本书已抽取完")
        count=count+1
    """

    for book_name in file_list:
        if count<66:
            count+=1
            continue
        _,name=pre_process(book_name)
        extract_match_response(book_name)
        generate_raw_text("match.json", book_name)
        check_thought_is_speech_response("match_with_raw_text.json")
        check_match_have_reason("match_thought_check.json", book_name=name)
        print(f"第{count}已抽取完")
        count+=1

