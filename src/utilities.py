import time
import random
import openai
import func_timeout
import requests
import numpy as np
import os 
import langchain 

import logging
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

from typing import Union, Any
from math import isclose

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)



import google.generativeai as genai
# logger = logging.getLogger(__name__)



from langchain.chains import LLMChain

# Langchain
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


openai.api_type =  "azure"
openai.api_base = os.environ['OPENAI_API_BASE']
openai.api_version = os.environ['OPENAI_API_VERSION']
openai.api_key = os.environ['OPENAI_API_KEY']
openai.deployment_name = os.environ["OPENAI_DEPLOYMENT_NAME"]
openai.model_name = os.environ['MODEL_NAME']

# Set up Google API Key
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


# Initiate a connection to the LLM from Azure OpenAI Service via LangChain.
llm = AzureChatOpenAI(
    openai_api_key=openai.api_key,
    deployment_name=openai.deployment_name ,
    openai_api_version = openai.api_version,
    openai_api_base = openai.api_base ,
    model_name = openai.model_name, 
    temperature=0.5
)


def safe_execute(code_string: str, keys=None):
    import codecs
    from io import StringIO
    from contextlib import redirect_stdout
    import matplotlib
    matplotlib.use('Agg')
    import func_timeout

    new_code_string = codecs.decode(code_string, 'unicode_escape')

    # 🔥 remove markdown fences if present
    new_code_string = new_code_string.strip()
    if new_code_string.startswith("```python"):
        new_code_string = new_code_string[len("```python"):].strip()
    if new_code_string.startswith("```"):
        new_code_string = new_code_string[len("```"):].strip()
    if new_code_string.endswith("```"):
        new_code_string = new_code_string[:-3].strip()

    output = None
    error_message = None

    def _run_code():
        buffer = StringIO()
        with redirect_stdout(buffer):
            exec(new_code_string, globals())
        return buffer.getvalue()

    try:
        output = func_timeout.func_timeout(10, _run_code)
    except func_timeout.FunctionTimedOut:
        error_message = "TLE"
    except Exception as e:
        error_message = str(e)

    return output, error_message


def get_llama_response(tokenizer,pipeline,prompt,temperature=0.5):
    
    system = "Follow the format of the examples given below"

    # Prompt in CodeLlama format
    #prompt = f"<s>[INST] <<SYS>>\\n{system}\\n<</SYS>>\\n\\n{prompt}[/INST]"
    
    #prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n{prompt} [/INST]"

    prompt = f"<s>[INST] {prompt.strip()} [/INST]"
    
 
    sequences = pipeline(
    prompt,
    do_sample=True,
    top_k=10,
    temperature=temperature,
    top_p=0.5,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=300,
    return_full_text=False
    )
    
    response = ""

    for seq in sequences:
        response = response + (seq['generated_text'])
    
    # Remove the part after the first word "Question"
    index = response.find("Question")

    if index !=-1:
        response = response[:index]

    return response   

def get_llama_13bresponse(tokenizer,pipeline,prompt,temperature=0.5):
    system = "Follow the format of the examples given below"

    # Prompt in CodeLlama format
    #prompt = f"<s>[INST] <<SYS>>\\n{system}\\n<</SYS>>\\n\\n{prompt}[/INST]"
    
    #prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n{prompt} [/INST]"

    #prompt = f"<s>[INST] {prompt.strip()} [/INST]"
    
 
    sequences = pipeline(
    prompt,
    do_sample=True,
    top_k=10,
    temperature=temperature,
    top_p=0.5,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=300,
    return_full_text=False
    )
    
    response = ""

    for seq in sequences:
        response = response + (seq['generated_text'])
    
    # Remove the part after the first word "Question"
    index = response.find("Question")

    if index !=-1:
        response = response[:index]

    return response   


def get_textdavinci002_response(prompt, temperature, max_tokens, n=1, patience=1, sleep_time=2):
   
    while patience > 0:
        patience -= 1
        try:
            response = openai.Completion.create(engine=os.environ['OPENAI_TEXTDAVC002_DEPLOYMENT_NAME'],
                                                prompt=prompt,
                                                api_key=openai.api_key,
                                                temperature=temperature,
                                                max_tokens=500,
                                                top_p=0.5,
                                                best_of=1,
                                                stop=["Question"],
                                                frequency_penalty=0,
                                                presence_penalty=0)
            prediction = response["choices"][0]["text"].strip()
            if prediction != "" and prediction != None:
                time.sleep(sleep_time)
                return prediction
        except Exception as e:
            logging.info(f"{e}")
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    return ""
    
def get_textdavinci003_response(prompt, temperature, max_tokens, n=1, patience=1, sleep_time=2):
   
    logging.info(f"-------Text davinci 003 response------")
    while patience > 0:
        patience -= 1
        try:
            response = openai.Completion.create(engine=os.environ['OPENAI_TEXTDAVC003_DEPLOYMENT_NAME'],
                                                prompt=prompt,
                                                api_key=openai.api_key,
                                                temperature=temperature,
                                                max_tokens=500,
                                                top_p=0.5,
                                                best_of=1,
                                                stop=["Question"],
                                                frequency_penalty=0,
                                                presence_penalty=0)
            
            prediction = response["choices"][0]["text"].strip()
            if prediction != "" and prediction != None:
                logging.info(f"-----Sleep starts---")
                time.sleep(sleep_time)
                logging.info(f"-----Sleep ends---")
                return prediction
        except Exception as e:
            logging.info(f"{e}")
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    return ""



def get_gpt3_response(prompt, api_key, engine="text-davinci-002", temperature=0, max_tokens=256, top_p=1, n=1, patience=100, sleep_time=0):
    while patience > 0:
        patience -= 1
        try:
            response = openai.Completion.create(engine=engine,
                                                prompt=prompt,
                                                api_key=api_key,
                                                temperature=temperature,
                                                max_tokens=max_tokens,
                                                top_p=top_p,
                                                n=n,
                                                stop=['\n\n'],
                                                frequency_penalty=0,
                                                presence_penalty=0)
            prediction = response["choices"][0]["text"].strip()
            if prediction != "" and prediction != None:
                return prediction
        except Exception as e:
            logging.info(f"{e}")
            if sleep_time > 0:
                time.sleep(sleep_time)
    return ""


def get_chat_response_code(context, temperature=0.5, max_tokens=256, system_mess=None,stop=None,n=1, patience=10, sleep_time=5):
    
    from langchain.schema.messages import HumanMessage, SystemMessage
    import time
    logging.info(f"----Response starts-----")
    
    if system_mess is not None:
        try:
            if stop!=None:
                response = llm([SystemMessage(content=system_mess),HumanMessage(content=context)],max_tokens=max_tokens,temperature=temperature,stop=stop)
            else:
                response = llm([SystemMessage(content=system_mess),HumanMessage(content=context)],max_tokens=max_tokens,temperature=temperature)
        except:
            response = ""
    else:
        try:
            if stop!=None:
                response = llm([HumanMessage(content=context)],max_tokens=max_tokens,temperature=temperature,stop=stop)
            else:
                response = llm([HumanMessage(content=context)],max_tokens=max_tokens,temperature=temperature)
        except:
            response = ""
    logging.info(f"----Response ends-----")
    
    try:
      logging.info(f"---------Sleep starts----------")
      time.sleep(sleep_time)
      logging.info(f"---------Sleep ends----------")
      return response.content
    except:
      logging.info(f"---------Sleep starts----------")
      time.sleep(sleep_time)
      logging.info(f"---------Sleep ends----------")
      return ""


def get_gemini_response(full_prompt):
    
    flag=0
    while(flag==0):
        gemini_model = genai.GenerativeModel('gemini-pro')
        response = gemini_model.generate_content(full_prompt)
        
        try :
           return response.text
        except:
           try:
            return response.candidates[0].content.parts[0].text 
           except:
            continue 
    
   

def get_chat_response(messages, temperature=0, max_tokens=256, n=1, patience=1, sleep_time=2):
    
    import time
    
    # Context
    context = (messages[0]["content"])
    logging.info(f"CHATGPT CALLED")
    logging.info(f"----Response starts-----")

    try:
        response = llm([HumanMessage(content=context)],max_tokens=max_tokens,temperature=temperature)

    except:
        response = ""
    
    logging.info(f"----Response ends-----")

    try:
      logging.info(f"---------Sleep starts----------")
      time.sleep(sleep_time)
      logging.info(f"---------Sleep ends----------")
      #logging.info(f"{response.content}")
      return response.content
    
    except:
      logging.info(f"---------Sleep starts----------")
      time.sleep(sleep_time)
      logging.info(f"---------Sleep ends----------")
      return ""        
    
    '''
    try:
            
                response = openai.ChatCompletion.create(
                    engine=engine,
                    #model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=api_key,
                )
                gen = response['choices'][0]['message']['content']
                return gen

    except:
          logging.info(f"Does not work")
          return ""
    '''
       

# Helper functions 
def floatify_ans(ans):
    if ans is None:
        return None
    elif type(ans) == dict:
        ans = list(ans.values())[0]
    elif type(ans) == bool:
        ans = ans
    elif type(ans) in [list, tuple]:
        if not ans:
            return None
        else:
            try:
                ans = float(ans[0])
            except Exception:
                ans = str(ans[0])
    else:
        try:
            ans = float(ans)
        except Exception:
            ans = str(ans)
    return ans


def score_string_similarity(str1, str2):
    if str1 == str2:
        return 2.0
    elif " " in str1 or " " in str2:
        str1_split = str1.split(" ")
        str2_split = str2.split(" ")
        overlap = list(set(str1_split) & set(str2_split))
        return len(overlap) / max(len(str1_split), len(str2_split))
    else:
        return 0.0
        


def _validate_server(address):
    if not address:
        raise ValueError('Must provide a valid server for search')
    if address.startswith('http://') or address.startswith('https://'):
        return address
    PROTOCOL = 'http://'
    logging.info(f'No protocol provided, using "{PROTOCOL}"')
    return f'{PROTOCOL}{address}'

def call_bing_search(endpoint, bing_api_key, query, count):
    
    headers = {'Ocp-Apim-Subscription-Key': bing_api_key}
    params = {"q": query, "textDecorations": True,
      "textFormat": "HTML", "count": count}
    
    try:
        response = requests.get(endpoint, headers=headers, params=params)
        logging.info(f"BING CALLED")
        #response.raise_for_status()
        resp_status = response.status_code
        
        if resp_status == 200:
            result = response.json()
            return result 
    except:
        pass
        
    return None

def parse_bing_result(result):
    responses = []
    try:
        value = result["webPages"]["value"]
    except:
        return responses
    
    try:
        for i in range(len(value)):
            snippet = value[i]['snippet'] if 'snippet' in value[i] else ""
            snippet = snippet.replace("<b>", "").replace("</b>", "").strip()
            if snippet != "":
                responses.append(snippet)
        return responses

    except:
        return []    


        
def get_webpage_content():
       
        from bs4 import BeautifulSoup
        import requests

        url = "https://en.wikipedia.org/wiki/Ryan_Gosling"  # replace with your URL
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Get all the text in the webpage
        text = soup.get_text()
        #logging.info(f"{text}")
        # Split the text by line
        lines = text.splitlines()
        logging.info(f"{lines[1000:]}")

        # Get the first few lines
        #first_few_lines = lines[:20]  # replace 5 with the number of lines you want

        #logging.info(f"{'\n'.join(first_few_lines)}")
    
        
