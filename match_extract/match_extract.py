import os
from openai import OpenAI
import json
from pydantic import BaseModel
client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
directory_path='extracted'
file_list=os.listdir(directory_path)
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
def pre_process(file_name):
    location="extracted\\"+file_name
    with open(location,"r",encoding='utf-8') as file:
        data=json.load(file)
    return data
class content(BaseModel):
    content:str
    type:str
class unit(BaseModel):
    character:str
    reason:str
    thought:content
    action:content
    raw_text:str
class pair(BaseModel):
    ta_pairs:list[unit]


def get_response(file_name):
    data=pre_process(file_name)
    plot_len=len(data["plots"])
    book_name = file_name
    start = book_name.index("extracted_data_") + len("extracted_data_")
    end = book_name.index(".json")
    book_name = book_name[start:end].strip()
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
            "raw_text": "The original text that contains the pair"
        }}
    ]
}}

===Requirements===
1.Only extract pairs of thoughts and actions that appear within a distance of no more than five sentences.
2.Only extract pairs of thoughts and actions that have a clear causal relationship, where the character's thoughts must be a significant cause of the character's actions
3.Thoughts are categorized into "reasoning-type" and "feeling-type."
4.Actions are categorized into "behavior" and "speech."
5.Only return high-quality pairs; if no suitable pairs are found, return an empty list.
6.Thoughts cannot be the character’s spoken words or actual actions.
7.Actions are specific behaviors performed by the character.
8.Both Thoughts and Actions must be from the original text.
9.Both Thoughts and Actions must come from the same character.
10.Thoughts are the character's internal ideas, not expressed outwardly, while Actions are behaviors that the character exhibits.
11.The raw text must be the unaltered original text, including complete  actions and thoughts,as well as the original text fragments between the thought and action.=. The raw text can be expanded with additional context to provide necessary background information for better understanding.If the thought and action are far apart in the original text, unnecessary parts in between can be omitted using "......".

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
            "raw_text": "Ron's mouth moved soundlessly, like a goldfish out of water. At that moment, Hermione spun around abruptly, stomped up the girls' dormitory stairs in a huff, and went back to bed. Ron turned to look at Harry."Look at that," he stammered, appearing utterly stunned. "Just look at that—completely missing the point—"Harry said nothing. He valued the fact that he and Ron were talking again, so he carefully held his tongue and refrained from expressing his own opinion—in fact, he thought that compared to Ron, Hermione had a much clearer grasp of the real issue."
        }}
    ]
}}

"""
    dict_list=[]
    for plot in data['plots']:
        response=chat_completion(prompt, plot['text'])
        res_dict=json.loads(response)
        dict_list.append(res_dict)
    f2 = open('test.json', 'w')
    json.dump(dict_list,f2,indent=2)

if __name__=='__main__':
    for book_name in file_list:
        get_response(book_name)
        break