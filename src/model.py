import os
import sys
import json
import openai
import re
import ast
from tqdm import tqdm 
import random
import csv 
import argparse
import pprint
import time
from huggingface_hub import login
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
import transformers
import torch
from huggingface_hub import snapshot_download

import logging
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv(".env")

try:
    import wolframalpha
except Exception:
    wolframalpha = None
from core.answer_parsing import (
    extract_final_answer_option_letter,
    extract_numeric_answer,
    extract_option_letter,
    extract_tagged_answer,
)
from core.pipeline_shared import (
    extract_example_metadata,
    normalize_module_sequence,
    build_wolfram_next_step_prompt,
    build_knowledge_retrieval_prompt,
    detect_wolfram_loop_reason,
    format_wolfram_trace,
    normalize_wolfram_query_signature,
    parse_wolfram_next_step_response,
    parse_wolfram_query_response,
    remove_wrapping_backticks,
    resolve_wolfram_answer,
)
from core.wolfram_utils import extract_wolfram_plaintext_answer, query_wolfram_alpha
from presentation.asy_rendering import strip_asy_blocks_for_model_input
from core.python_pipeline import (
    execution_failure_reason,
    extract_missing_dependency,
    install_missing_dependency,
    repair_program_until_runnable,
    sanitize_generated_python,
)
from core.app_support import solution_prompt_family


# Set up huggingface token 
huggingface_token = os.getenv('HUGGINGFACE_TOKEN')


# Helper functions 
def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0] 
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    # linebreaks  
    string = string.replace("\n", "")
    
    # remove inverse spaces
    string = string.replace("\\!", "")
    

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    
    
    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$","")
    
    # remove " 
    string = string.replace('"',"")
    
    # Extract the numbers 

    # remove units (on the right)
    string = _remove_right_units(string)
    
    # remove percentage
    string = string.replace("\\%", "")
    # string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def is_equiv(str1, str2, verbose=True):
    if str1 is None and str2 is None:
        logging.warning("Both None")
        return True, str1, str2
    if str1 is None or str2 is None:
        return False,str1,str2
    else:
      try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)

        return ss1 == ss2,ss1,ss2
      except:
        return str1 == str2,str1,str2
      

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def last_boxed_only(sample):
    """
    Given a (q,a) sample, filter the answers so that they only contain 
    the last \boxed{...} or \fbox{...} element
    """
    q, a = sample
    a = last_boxed_only_string(a)
    if a == None:
        return None
    return (q, a)

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval

def only_until_first_boxed_from_tokens(string, tokens):
    idx = string.find("\\boxed")
    if idx < 0:
        idx = string.find("\\fbox")
        if idx < 0:
            return None
    
    cum_length = 0
    for i, t in enumerate(tokens):
        cum_length += len(t)
        if cum_length >= idx:
            break
    
    return tokens[:i]



def clean_numbers(sample):
    if not sample:
        return None
    new_sample = list()
    for s in sample:
        new_sample.append(_clean_numbers(s))

    return tuple(new_sample)

def _clean_numbers(string):
    """
    Clean Numbers in the given string

    >>> _clean_numbers(None, "Hello 123")
    'Hello 123'
    >>> _clean_numbers(None, "Hello 1234")
    'Hello 1,234'
    >>> _clean_numbers(None, "Hello 1234324asdasd")
    'Hello 1,234,324asdasd'
    """
    num_prev_digits = 0
    new_string = ""
    for i, c in enumerate(string):
        # isdigit() doesnt work here because of weird unicode chars.
        if c in {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0'}:
            num_prev_digits += 1
        else:
            if num_prev_digits > 3:
                # Some fixing
                string_number = new_string[-num_prev_digits:]
                new_string = new_string[:-num_prev_digits] + "{0:,}".format(int(string_number))
            num_prev_digits = 0
        new_string += c

    if num_prev_digits > 3:
        # Some fixing
        string_number = new_string[-num_prev_digits:]
        new_string = new_string[:-num_prev_digits] + "{0:,}".format(int(string_number))

    return new_string



def append_csv(save_file_path, data):
        
        # Check if the file exists
        file_exists = os.path.isfile(save_file_path)

        # Open the CSV file in append mode
        with open(save_file_path, 'a', newline='') as file:
            
            writer = csv.writer(file)

            # Write the header row if the file is newly created
            if not file_exists:
                writer.writerow(['Model_name','Temperature','Question','Gold_Solution','Gold_Answer','COT_Output','COT_final_answer'])  # Replace with your column names

            # Write the data to the CSV file
            writer.writerow(data)


def save_output(question, gold_answer,gold_final_answer, COT_output, COT_final_answer, method, args, save_file_dir="math_outputs_cot_variants"):
    file_name = "math_cot_" +"_"+ args.model_name +"_" +str(args.temperature) + method + ".csv" 
    save_file_path = os.path.join(save_file_dir,file_name)
   
    data_to_append = [args.model_name,args.temperature,question,gold_answer,gold_final_answer,COT_output,COT_final_answer]
    append_csv(save_file_path, data_to_append)


def extract_last_number(output):
    # Find all numbers in the text
    numbers = re.findall(r'\d+\.?\d*', output)
    
    # Return the last number
    return float(numbers[-1]) if numbers else None



def get_answer(output, string="The answer is "):
    del string
    return extract_tagged_answer(output)



def _read_text_file_with_fallbacks(file_path):
    last_error = None
    for encoding in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            with open(file_path, "r", encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError as exc:
            last_error = exc

    raise last_error


def read_jsonl_file(file_path):
    data = []
    raw_text = _read_text_file_with_fallbacks(file_path)

    for line in raw_text.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        data.append(record)

    return data


def load_json_file(file_path):
    return json.loads(_read_text_file_with_fallbacks(file_path))

def extract_boxed_value(text):
    boxed_value = re.search(r'\\boxed{(.*?)}', text)
    if boxed_value:
      return (boxed_value.group(1))

    else:
       return None 
    

def extract_model_answer(output):
    answer = extract_tagged_answer(output)
    if answer:
        return answer

    try:
        return extract_boxed_value(output)
    except Exception:
        return None

def extract_vals(string):
  extracted_value = extract_numeric_answer(string)
  if extracted_value is None:
    return None

  extracted_value = extracted_value.replace(",", "")
  try:
    return float(extracted_value)
  except Exception:
    return extracted_value


# add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilities import *
from custom_prompts import (
    prompt_bing_answer_extractor,
    prompt_bing_query,
    prompt_kr,
    prompt_kr_pg,
    prompt_kr_pg_sg,
    prompt_pg,
    prompt_pot,
    prompt_walpha_context_withthought,
    prompt_walpha_kr_sg,
)
from custom_prompts.core_prompts import (
    prompt_codefixer,
    prompt_for_cot,
    prompt_kr_sg,
    prompt_policy,
)
from core.gsm_prompt_templates import (
    build_gsm_knowledge_demo_prompt,
    build_gsm_program_demo_prompt,
    build_gsm_solution_demo_prompt,
    build_gsm_wolfram_demo_prompt,
    gsm_story_leak_tokens,
    looks_like_gsm_story_leak,
)
from core.question_isolation import (
    build_math_knowledge_demo_prompt,
    build_option_knowledge_demo_prompt,
    build_option_program_demo_prompt,
    build_option_solution_demo_prompt,
    build_option_wolfram_demo_prompt,
    cross_problem_leak_details,
    knowledge_leak_details,
    looks_like_knowledge_leak,
    looks_like_cross_problem_leak,
    program_uses_hidden_geometry_coordinates,
    question_explicitly_uses_coordinates,
)
from core.option_reasoning import (
    program_uses_unsupported_closest_estimate_rounding,
    question_requests_closest_option,
    select_option_from_values,
)


class solver:
    def _is_geometry_problem(self):
        topic = str((self.get_metadata() or {}).get("topic") or "").lower()
        return "geometry" in topic

    def _geometry_forbids_coordinate_methods(self, question_text=None):
        if not self._is_geometry_problem():
            return False
        return not question_explicitly_uses_coordinates(question_text or self.get_question_text())

    def _call_python_generation_backend(self, full_prompt, backend_name=None):
        selected_backend = self.python_model if backend_name is None else backend_name
        messages = [
            {"role": "user", "content": full_prompt},
        ]

        if selected_backend in (None, "", "no"):
            return get_chat_response(
                messages=messages,
                temperature=self.pg_temperature,
                max_tokens=self.pg_max_tokens
            )

        if selected_backend == 'gemini':
            return get_gemini_response(full_prompt)

        if selected_backend == 'code_davinci002':
            return get_codex_response(
                prompt=full_prompt,
                temperature=self.pg_temperature
            )

        if selected_backend in [
            'code_llama7b_python',
            'code_llama13b_python',
            'code_llama34b',
            'code_llama34b_pythonV1'
        ]:
            return get_codellama_response(
                self.python_tokenizer,
                self.python_pipeline,
                prompt=full_prompt,
                temperature=self.pg_temperature
            )

        if selected_backend == 'wizardcoder_34B':
            return get_wizard_coder_response(
                self.python_tokenizer,
                self.model_code,
                prompt=full_prompt,
                temperature=self.pg_temperature
            )

        return ""

    def _python_generation_is_usable(self, candidate):
        if candidate in (None, ""):
            return False
        if not str(candidate).strip():
            return False
        cleaned = sanitize_generated_python(candidate)
        if not cleaned.strip():
            return False
        try:
            ast.parse(cleaned)
        except SyntaxError:
            return False
        code_markers = ("\n", "print(", "=", "import ", "from ", "for ", "while ", "if ", "def ")
        return any(marker in cleaned for marker in code_markers)

    def _build_empty_python_retry_prompt(self, full_prompt):
        return (
            full_prompt
            + "\nIMPORTANT: Your previous response did not contain usable executable Python code.\n"
            + "Return ONLY runnable Python code.\n"
            + "The first line must be exactly: from sympy import *\n"
            + "Do not include markdown, explanations, bullet points, or prose.\n"
            + "The final line of the program must print the derived answer.\n"
            + "Code:\n"
        )

    def _build_last_resort_python_program(self):
        fallback_answer = "0"
        options = self._current_options() or []
        if options:
            first_option_key = options[0].get("key")
            if first_option_key not in (None, ""):
                fallback_answer = repr(str(first_option_key))
        return (
            "from sympy import *\n"
            "# Last-resort fallback so the Python lane still emits a parseable answer.\n"
            f"print({fallback_answer})"
        )

    def _generate_python_program_with_trace(self, full_prompt):
        attempts = []

        def try_backend(prompt_text, backend_name, reason):
            backend_label = backend_name if backend_name not in (None, "", "no") else "default-chat"
            try:
                raw_output = self._call_python_generation_backend(prompt_text, backend_name=backend_name)
                attempts.append(
                    {
                        "backend": backend_label,
                        "reason": reason,
                        "raw_output": raw_output,
                        "usable": self._python_generation_is_usable(raw_output),
                    }
                )
                return raw_output
            except Exception as exc:
                attempts.append(
                    {
                        "backend": backend_label,
                        "reason": reason,
                        "raw_output": None,
                        "usable": False,
                        "error": str(exc),
                    }
                )
                return None

        raw_program = try_backend(full_prompt, self.python_model, "initial")
        if self._python_generation_is_usable(raw_program):
            return raw_program, attempts

        retry_prompt = self._build_empty_python_retry_prompt(full_prompt)
        retry_program = try_backend(retry_prompt, self.python_model, "retry_after_blank_or_noncode")
        if self._python_generation_is_usable(retry_program):
            return retry_program, attempts

        if self.python_model not in (None, "", "no"):
            fallback_program = try_backend(retry_prompt, "no", "fallback_default_chat_backend")
            if self._python_generation_is_usable(fallback_program):
                return fallback_program, attempts

        last_resort_program = self._build_last_resort_python_program()
        attempts.append(
            {
                "backend": "last_resort_stub",
                "reason": "all_python_generation_attempts_failed",
                "raw_output": last_resort_program,
                "usable": True,
            }
        )
        warning = (
            "program_generator: all model generation attempts returned blank or non-code output; "
            "used a last-resort stub so the Python lane still emits an answer."
        )
        warnings = self.cache.setdefault("module_warnings", [])
        if warning not in warnings:
            warnings.append(warning)
        return last_resort_program, attempts

    def _generate_python_program(self, full_prompt):
        raw_program, attempts = self._generate_python_program_with_trace(full_prompt)
        self.cache["program_generator:attempts"] = attempts
        self.cache["program_generator:raw_output"] = raw_program
        return raw_program

    def _generate_solution_text(self, full_prompt, temperature):
        messages = [
            {"role": "user", "content": full_prompt},
        ]

        if self.sg_model == 'text_davinci_003':
            return get_textdavinci003_response(full_prompt, temperature=temperature, max_tokens=self.sg_max_tokens)

        if self.sg_model == 'gemini':
            return get_gemini_response(full_prompt)

        return get_chat_response(messages=messages, temperature=temperature, max_tokens=self.sg_max_tokens)

    def _generate_knowledge_text(self, full_prompt):
        messages = [
            {"role": "user", "content": full_prompt},
        ]

        if self.knowledge_model == 'text_davinci_002':
            return get_textdavinci002_response(prompt=full_prompt, temperature=self.kr_temperature)

        if self.knowledge_model == 'text_davinci_003':
            return get_textdavinci003_response(prompt=full_prompt, temperature=self.kr_temperature)

        if self.knowledge_model == 'gemini':
            return get_gemini_response(full_prompt)

        if self.knowledge_model == "llama2_13b":
            return get_llama_response(
                self.knowledge_tokenizer,
                self.knowledge_pipeline,
                prompt=full_prompt,
                temperature=self.kr_temperature,
            )

        if self.knowledge_model == "llama2_7b":
            return get_llama_13bresponse(
                self.knowledge_tokenizer,
                self.knowledge_pipeline,
                prompt=full_prompt,
                temperature=self.kr_temperature,
            )

        return get_chat_response(messages=messages, temperature=self.kr_temperature, max_tokens=self.kr_max_tokens)

    def _option_letters_for_dataset(self):
        if self.dataset == "MMLU":
            return "A-D"
        return "A-E"

    def _current_options(self):
        example = self.cache.get("example") or {}
        raw_options = example.get("options")
        if raw_options:
            return raw_options

        if self.dataset == "MMLU":
            options = []
            for key in ("A", "B", "C", "D"):
                label = example.get(f"Option {key}")
                if label not in (None, ""):
                    options.append({"key": key, "label": str(label)})
            return options or None

        return None

    def _allowed_option_letters(self):
        return self._option_letters_for_dataset().replace("-", "")

    def _resolve_option_crosscheck(self, *text_values, question_text=None):
        options = self._current_options()
        if not options:
            return None
        return select_option_from_values(
            text_values,
            options,
            question_text=question_text or self.get_question_text(),
        )

    def _format_resolved_option_answer(self, resolved_option):
        if not resolved_option:
            return None
        return f"{resolved_option['key']}. {resolved_option['label']}"

    def _format_option_crosscheck_completion(self, resolved_option):
        if not resolved_option:
            return None

        if resolved_option.get("match_type") == "closest-numeric":
            candidate = resolved_option.get("candidate")
            if candidate not in (None, ""):
                return (
                    f"The computed quantity is {candidate}. Comparing every option numerically, "
                    f"option {resolved_option['key']} ({resolved_option['label']}) has the smallest absolute difference."
                )
        return f"The computed result aligns with option {resolved_option['key']} ({resolved_option['label']})."

    def _maybe_reverse_check_option_program(self, full_prompt, question_text, program):
        options = self._current_options()
        if not options or not program or not question_requests_closest_option(question_text):
            return program

        suspicious_rounding = program_uses_unsupported_closest_estimate_rounding(question_text, program)
        if not suspicious_rounding:
            return program

        verification_prompt = (
            full_prompt
            + "\nIMPORTANT: This is a closest, nearest, approximate, or estimate multiple-choice question.\n"
            + "Rewrite the Python code so it follows these rules exactly:\n"
            + "- Compute the exact target quantity asked in the question before comparing options.\n"
            + "- Evaluate every option expression numerically and choose the option with the smallest absolute difference.\n"
            + "- Do not round intermediate quantities unless the problem text explicitly instructs that rounding.\n"
            + "- Do not invent a per-term rounding rule from the options. Compare against the actual option expressions instead.\n"
            + "- Print the computed target quantity and each option's numeric value before printing the final answer letter.\n"
            + "- Output executable Python code only.\n"
        )
        if suspicious_rounding:
            verification_prompt += (
                "Your previous draft rounded values too early. Replace that shortcut with an exact computation plus direct option comparison.\n"
            )
        verification_prompt += f"\nPrevious draft:\n{program}\n\nCode:\n"

        regenerated = self._generate_python_program(verification_prompt)
        if regenerated:
            regenerated = sanitize_generated_python(regenerated)
            if program_uses_unsupported_closest_estimate_rounding(question_text, regenerated):
                retry_prompt = (
                    verification_prompt
                    + "Your previous rewrite still rounded or estimated intermediate quantities without an explicit rounding instruction in the question.\n"
                    + "Remove that shortcut and compare against the exact numeric values of the options instead.\n"
                )
                retried = self._generate_python_program(retry_prompt)
                if retried:
                    regenerated = sanitize_generated_python(retried)
            return regenerated
        return program

    def _maybe_reverse_check_geometry_program(self, full_prompt, question_text, program):
        if not program or not self._geometry_forbids_coordinate_methods(question_text):
            return program

        if not program_uses_hidden_geometry_coordinates(question_text, program):
            return program

        verification_prompt = (
            full_prompt
            + "\nIMPORTANT: This is a geometry problem without explicit coordinates in the problem statement.\n"
            + "Rewrite the Python code so it follows these rules exactly:\n"
            + "- Do not assign coordinates to points.\n"
            + "- Do not use Point(...), coordinate tuples, or analytic coordinate geometry.\n"
            + "- Do not extract coordinates from the diagram or drawing code.\n"
            + "- Use only stated geometric relationships, lengths, angles, ratios, congruence, similarity, area formulas, or algebra directly implied by the problem.\n"
            + "- Output executable Python code only.\n"
            + "Your previous draft introduced coordinates for a non-coordinate geometry problem. Replace that shortcut with geometry grounded in the stated facts.\n"
            + f"\nPrevious draft:\n{program}\n\nCode:\n"
        )

        regenerated = self._generate_python_program(verification_prompt)
        if regenerated:
            regenerated = sanitize_generated_python(regenerated)
            if program_uses_hidden_geometry_coordinates(question_text, regenerated):
                retry_prompt = (
                    verification_prompt
                    + "Your previous rewrite still used coordinates. Remove all coordinate assignments and coordinate-geometry objects.\n"
                )
                retried = self._generate_python_program(retry_prompt)
                if retried:
                    regenerated = sanitize_generated_python(retried)
            return regenerated

        return program

    def _maybe_annotate_option_program_output(self, question_text, output):
        if output in (None, "") or not question_requests_closest_option(question_text):
            return output

        resolved_option = self._resolve_option_crosscheck(output, question_text=question_text)
        if not resolved_option or resolved_option.get("match_type") != "closest-numeric":
            return output

        explicit_option = (
            extract_final_answer_option_letter(output, allowed=self._allowed_option_letters())
            or extract_option_letter(output, allowed=self._allowed_option_letters())
        )
        resolved_line = f"Resolved option by numeric comparison: {resolved_option['key']}. {resolved_option['label']}"

        lines = [line.strip() for line in str(output).splitlines() if line.strip()]
        if resolved_line in lines:
            return output

        if explicit_option != resolved_option.get("key") or explicit_option is None:
            warning = (
                "program_executor: numeric option cross-check overrode or supplemented a closest-estimate option "
                f"with {resolved_option['key']} ({resolved_option['label']})."
            )
            warnings = self.cache.setdefault("module_warnings", [])
            if warning not in warnings:
                warnings.append(warning)

            rendered = str(output).rstrip()
            if rendered:
                rendered += "\n"
            rendered += resolved_line
            return rendered

        return output

    def _maybe_regenerate_cross_problem_program(self, full_prompt, question_text, program):
        if not looks_like_cross_problem_leak(question_text, program, mode="program"):
            return program

        details = cross_problem_leak_details(question_text, program)
        leaked_terms = details["foreign_tokens"][:4] + details["foreign_numbers"][:4]
        leak_summary = ", ".join(leaked_terms) or "unrelated details"
        retry_prompt = (
            full_prompt
            + "\nIMPORTANT: Your previous draft appears copied from a different problem.\n"
            + f"Unrelated leaked details: {leak_summary}\n"
            + "Regenerate from scratch for ONLY the current question, grounded in its own numbers and wording.\n"
            + "Code:\n"
        )
        regenerated = self._generate_python_program(retry_prompt)
        if regenerated:
            return sanitize_generated_python(regenerated)
        return program

    def _maybe_regenerate_cross_problem_solution(self, full_prompt, question_text, solution, temperature):
        if not looks_like_cross_problem_leak(question_text, solution, mode="solution"):
            return solution

        details = cross_problem_leak_details(question_text, solution)
        leaked_terms = details["foreign_tokens"][:4] + details["foreign_numbers"][:4]
        leak_summary = ", ".join(leaked_terms) or "unrelated details"
        retry_prompt = (
            full_prompt
            + "\nIMPORTANT: The previous draft appears grounded in a different problem.\n"
            + f"Unrelated leaked details: {leak_summary}\n"
            + "Rewrite the solution for ONLY the current question and ignore any stray unrelated context.\n"
            + "Solution: "
        )
        regenerated = self._generate_solution_text(retry_prompt, temperature)
        if regenerated:
            return regenerated
        return solution

    def _maybe_regenerate_cross_problem_knowledge(self, full_prompt, question_text, knowledge):
        if not knowledge or not looks_like_knowledge_leak(question_text, knowledge):
            return knowledge

        details = knowledge_leak_details(question_text, knowledge)
        leaked_terms = details["foreign_tokens"][:4] + details["foreign_numbers"][:4]
        leak_summary = ", ".join(leaked_terms) or "unrelated details"
        retry_prompt = (
            full_prompt
            + "\nIMPORTANT: Your previous draft described other problems or copied prompt examples instead of the current question.\n"
            + f"Unrelated leaked details: {leak_summary}\n"
            + "Rewrite the background knowledge for ONLY the current question.\n"
            + "- Do not enumerate multiple problems.\n"
            + "- Do not echo any earlier examples.\n"
            + "- Keep it to 3 to 6 short bullet points.\n"
            + "Knowledge Retrieval:\n"
        )
        regenerated = self._generate_knowledge_text(retry_prompt)
        if regenerated and not looks_like_knowledge_leak(question_text, regenerated):
            warning = "knowledge_retrieval: initial knowledge output leaked unrelated prompt examples and was regenerated."
            warnings = self.cache.setdefault("module_warnings", [])
            if warning not in warnings:
                warnings.append(warning)
            return regenerated

        warning = "knowledge_retrieval: dropped leaked knowledge output that referenced unrelated prompt examples."
        warnings = self.cache.setdefault("module_warnings", [])
        if warning not in warnings:
            warnings.append(warning)
        return ""

    def _maybe_reverse_check_option_solution(self, question_text, response, solution, temperature):
        options = self._current_options()
        if not options:
            return solution

        resolved_option = self._resolve_option_crosscheck(
            self.cache.get("wolfram_alpha_search:output"),
            self.cache.get("program_executor:output"),
            solution,
            question_text=question_text,
        )
        extracted_option = (
            extract_final_answer_option_letter(solution, allowed=self._allowed_option_letters())
            or extract_option_letter(solution, allowed=self._allowed_option_letters())
        )
        needs_reverse_check = bool(
            question_requests_closest_option(question_text)
            or (resolved_option and extracted_option != resolved_option.get("key"))
            or (resolved_option and not extracted_option)
        )
        if not needs_reverse_check:
            return solution

        verification_prompt = (
            "You are performing a mandatory reverse-check on a multiple-choice math solution.\n"
            "Rules:\n"
            "- Recompute the quantity asked using only the current question and context.\n"
            "- Compare against every option before choosing the answer.\n"
            "- For closest, nearest, approximate, or estimate questions, evaluate each option numerically and choose the smallest absolute difference.\n"
            "- If the options are expressions, evaluate the expressions before choosing.\n"
            "- Do not assume a rounding rule unless the options clearly justify it.\n"
            "- Keep the solution concise and grounded in the current question.\n"
            f"- Valid answer letters: {self._option_letters_for_dataset()}.\n"
        )
        if resolved_option:
            verification_prompt += (
                f"- A numeric cross-check from the current tool outputs points to option {resolved_option['key']}"
                f" ({resolved_option['label']}). Make sure your comparison agrees with the options.\n"
            )
        verification_prompt += (
            "\n"
            f"Question: {question_text}\n\n"
        )
        if response:
            verification_prompt += f"Context:\n{response}\n\n"
        if solution:
            verification_prompt += f"Previous draft:\n{solution}\n\n"
        verification_prompt += (
            "Rewrite the concise solution.\n"
            'End with exactly one final line in the form "Final Answer: X".\n'
        )

        corrected = self._generate_solution_text(verification_prompt, min(max(temperature, 0.2), 0.5))
        if corrected:
            corrected = self._maybe_regenerate_cross_problem_solution(
                verification_prompt,
                question_text,
                corrected,
                min(max(temperature, 0.2), 0.5),
            )
            corrected_option = (
                extract_final_answer_option_letter(corrected, allowed=self._allowed_option_letters())
                or extract_option_letter(corrected, allowed=self._allowed_option_letters())
            )
            if resolved_option is None or corrected_option == resolved_option.get("key"):
                return corrected

        if resolved_option:
            warning = (
                "Option reverse-check adjusted the final answer after comparing the computed value "
                "against the provided choices."
            )
            warnings = self.cache.setdefault("module_warnings", [])
            if warning not in warnings:
                warnings.append(warning)
            return (solution or "").rstrip() + f"\nFinal Answer: {resolved_option['key']}"

        return corrected or solution

    def _required_env(self, name, feature):
        value = os.getenv(name)
        if value is None or str(value).strip() == "":
            raise RuntimeError(f"Missing required environment variable '{name}' for {feature}")
        return value

    def _optional_env(self, name):
        value = os.getenv(name)
        if value is None or str(value).strip() == "":
            return None
        return value

    def _disable_model_backend(self, attr_name, requested_model, env_names):
        missing = [name for name in env_names if self._optional_env(name) is None]
        if missing:
            logging.warning(
                "Disabling %s backend '%s' because environment variables are missing: %s",
                attr_name,
                requested_model,
                ", ".join(missing),
            )
            setattr(self, attr_name, "no")
            return False
        return True

    def _skip_optional_module(self, module_name, reason):
        message = f"Skipped: {reason}"
        self.cache.setdefault("module_warnings", []).append(f"{module_name}: {reason}")
        self.cache[f"{module_name}:input"] = None
        self.cache[f"{module_name}:output"] = message
        self.cache[f"{module_name}:error"] = reason
        return None, message

    def _extract_wolfram_answer_from_result(self, result):
        return extract_wolfram_plaintext_answer(result)

    def __init__(self, args):
        # arguments
        
        # Set the attributes
        for key, value in vars(args).items():
            setattr(self, key, value)

        self.requested_dataset = self.dataset


        
        # external arguments
        #self.current_index = 0
        self.api_key = openai.api_key
        self.examples = self.load_data()
        self.modules = []
        self.cache = {}
        self.dependency_install_attempts = set()

        cloud_backend_requirements = [
            ("knowledge_model", "text_davinci_002", ["OPENAI_TEXTDAVC002_DEPLOYMENT_NAME"]),
            ("knowledge_model", "text_davinci_003", ["OPENAI_TEXTDAVC003_DEPLOYMENT_NAME"]),
            ("bing_model", "text_davinci_002", ["OPENAI_TEXTDAVC002_DEPLOYMENT_NAME"]),
            ("bing_model", "text_davinci_003", ["OPENAI_TEXTDAVC003_DEPLOYMENT_NAME"]),
            ("sg_model", "text_davinci_002", ["OPENAI_TEXTDAVC002_DEPLOYMENT_NAME"]),
            ("sg_model", "text_davinci_003", ["OPENAI_TEXTDAVC003_DEPLOYMENT_NAME"]),
            ("wolfram_model", "text_davinci_002", ["OPENAI_TEXTDAVC002_DEPLOYMENT_NAME"]),
            ("wolfram_model", "text_davinci_003", ["OPENAI_TEXTDAVC003_DEPLOYMENT_NAME"]),
            ("knowledge_model", "gemini", ["GOOGLE_API_KEY"]),
            ("bing_model", "gemini", ["GOOGLE_API_KEY"]),
            ("sg_model", "gemini", ["GOOGLE_API_KEY"]),
            ("wolfram_model", "gemini", ["GOOGLE_API_KEY"]),
            ("python_model", "gemini", ["GOOGLE_API_KEY"]),
        ]
        for attr_name, backend_name, env_names in cloud_backend_requirements:
            if getattr(self, attr_name) == backend_name:
                self._disable_model_backend(attr_name, backend_name, env_names)

        if self.knowledge_model == 'llama2_13b' and self._disable_model_backend("knowledge_model", "llama2_13b", ['HUGGINGFACE_TOKEN', 'LLAMA2_13B_CACHE_DIR', 'LLAMA2_13B_LOCAL_DIR']):
            # Huggingface login
            login(token=huggingface_token,new_session=False)  

            repo = "meta-llama/Llama-2-13b-hf"

            # Set cache dir and local dir 
            cache_dir = os.environ['LLAMA2_13B_CACHE_DIR']
            local_dir = os.environ['LLAMA2_13B_LOCAL_DIR']

            snapshot_download(repo_id=repo,cache_dir=cache_dir,local_dir=local_dir,local_dir_use_symlinks=True)

            # path to model
            logging.info("=====Running Llama2-13B-hf========")

            # Check if CUDA (GPU) is available 
            if torch.cuda.is_available():
                    # Get the number of available GPUs
                    num_gpus = torch.cuda.device_count()
                    logging.info(f"Number of available GPUs: {num_gpus}")
                    # Iterate through available GPUs and print information about each
                    for i in range(num_gpus):
                        gpu = torch.cuda.get_device_name(i)
                        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9  # in GB
                        logging.info(f"GPU,Total Memory: {gpu},{gpu_memory}")
            else:
                    logging.info("No GPU available. Using CPU.")

            
            # Define the tokenizer of python generator module
            self.knowledge_tokenizer = AutoTokenizer.from_pretrained(local_dir)

            
            # Define the model pipeline of the python generator module
            self.knowledge_pipeline = transformers.pipeline(
            "text-generation",
            model=local_dir,
            torch_dtype=torch.float32,
            device_map="auto",
        
            )
            
        if self.knowledge_model == 'llama2_7b' and self._disable_model_backend("knowledge_model", "llama2_7b", ['HUGGINGFACE_TOKEN', 'LLAMA2_7B_CACHE_DIR', 'LLAMA2_7B_LOCAL_DIR']):
            
            # Huggingface login
            login(token=huggingface_token,new_session=False)  

            repo = "meta-llama/Llama-2-7b-hf"

            cache_dir = os.environ['LLAMA2_7B_CACHE_DIR']
            local_dir = os.environ['LLAMA2_7B_LOCAL_DIR']

            snapshot_download(repo_id=repo,cache_dir=cache_dir,local_dir=local_dir,local_dir_use_symlinks=True)

            # Path to model
            logging.info("=====Running Llama2-7B-hf========")

            # Check if CUDA (GPU) is available 
            if torch.cuda.is_available():
                    # Get the number of available GPUs
                    num_gpus = torch.cuda.device_count()
                    logging.info(f"Number of available GPUs: {num_gpus}")
                    # Iterate through available GPUs and print information about each
                    for i in range(num_gpus):
                        gpu = torch.cuda.get_device_name(i)
                        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9  # in GB
                        logging.info(f"GPU,Total Memory: {gpu},{gpu_memory}")
            else:
                    logging.info("No GPU available. Using CPU.")

            # Define the tokenizer of python generator module
            self.knowledge_tokenizer = AutoTokenizer.from_pretrained(local_dir)

            
            # Define the model pipeline of the python generator module
            self.knowledge_pipeline = transformers.pipeline(
            "text-generation",
            model=local_dir,
            torch_dtype=torch.float32,
            device_map="auto",
        
            )
            
            

        if self.python_model == 'code_llama7b_python' and self._disable_model_backend("python_model", "code_llama7b_python", ['HUGGINGFACE_TOKEN', 'CODELLAMA_7B_PYTHON_CACHE_DIR', 'CODELLAMA_7B_PYTHON_LOCAL_DIR']):

            # Huggingface login
            login(token=huggingface_token,new_session=False)  

            repo = "codellama/CodeLlama-7b-Python-hf"
            cache_dir = os.environ['CODELLAMA_7B_PYTHON_CACHE_DIR']
            local_dir = os.environ['CODELLAMA_7B_PYTHON_LOCAL_DIR']
            snapshot_download(repo_id=repo,cache_dir=cache_dir,local_dir=local_dir,local_dir_use_symlinks=True)

            # Path to model
            logging.info("=====Running CodeLLama-7B-Python========")

            # Check if CUDA (GPU) is available 
            if torch.cuda.is_available():
                    # Get the number of available GPUs
                    num_gpus = torch.cuda.device_count()
                    logging.info(f"Number of available GPUs: {num_gpus}")
                    # Iterate through available GPUs and print information about each
                    for i in range(num_gpus):
                        gpu = torch.cuda.get_device_name(i)
                        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9  # in GB
                        logging.info(f"GPU,Total Memory: {gpu},{gpu_memory}")
            else:
                    logging.info("No GPU available. Using CPU.")

           
            # Define the tokenizer of python generator module
            self.python_tokenizer = AutoTokenizer.from_pretrained(local_dir)

            
            # Define the model pipeline of the python generator module
            self.python_pipeline = transformers.pipeline(
            "text-generation",
            model=local_dir,
            torch_dtype=torch.float32,
            device_map="auto",
        
            )
            
        if self.python_model == 'code_llama13b_python' and self._disable_model_backend("python_model", "code_llama13b_python", ['HUGGINGFACE_TOKEN', 'CODELLAMA_13B_PYTHON_CACHE_DIR', 'CODELLAMA_13B_PYTHON_LOCAL_DIR']):

            # Huggingface login
            login(token=huggingface_token,new_session=False)  
            repo = "codellama/CodeLlama-13b-Python-hf"

            cache_dir = os.environ['CODELLAMA_13B_PYTHON_CACHE_DIR']
            local_dir = os.environ['CODELLAMA_13B_PYTHON_LOCAL_DIR']
      

            snapshot_download(repo_id=repo,cache_dir=cache_dir,local_dir=local_dir,local_dir_use_symlinks=True)

            # Path to model
            logging.info("=====Running CodeLLama-13B-python========")

            # Check if CUDA (GPU) is available 
            if torch.cuda.is_available():
                    # Get the number of available GPUs
                    num_gpus = torch.cuda.device_count()
                    logging.info(f"Number of available GPUs: {num_gpus}")
                    # Iterate through available GPUs and print information about each
                    for i in range(num_gpus):
                        gpu = torch.cuda.get_device_name(i)
                        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9  # in GB
                        logging.info(f"GPU,Total Memory: {gpu},{gpu_memory}")
            else:
                    logging.info("No GPU available. Using CPU.")

          
            # Define the tokenizer of python generator module
            self.python_tokenizer = AutoTokenizer.from_pretrained(local_dir)

            
            # Define the model pipeline of the python generator module
            self.python_pipeline = transformers.pipeline(
            "text-generation",
            model=local_dir,
            torch_dtype=torch.float32,
            device_map="auto",
        
            )
           

        if self.python_model == 'code_llama34b' and self._disable_model_backend("python_model", "code_llama34b", ['HUGGINGFACE_TOKEN', 'CODELLAMA_34B_CACHE_DIR', 'CODELLAMA_34B_LOCAL_DIR']):

            # Huggingface login
            login(token=huggingface_token,new_session=False)  

            repo = "Phind/Phind-CodeLlama-34B-v2"
            cache_dir = os.environ['CODELLAMA_34B_CACHE_DIR']
            local_dir = os.environ['CODELLAMA_34B_LOCAL_DIR']
          
            snapshot_download(repo_id=repo,cache_dir=cache_dir,local_dir=local_dir,local_dir_use_symlinks=True)

            # Path to model
            logging.info("=====Running CodeLLama-34B-V2========")

            # Check if CUDA (GPU) is available 
            if torch.cuda.is_available():
                    # Get the number of available GPUs
                    num_gpus = torch.cuda.device_count()
                    logging.info(f"Number of available GPUs: {num_gpus}")
                    # Iterate through available GPUs and print information about each
                    for i in range(num_gpus):
                        gpu = torch.cuda.get_device_name(i)
                        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9  # in GB
                        logging.info(f"GPU,Total Memory: {gpu},{gpu_memory}")
            else:
                    logging.info("No GPU available. Using CPU.")

            
            # Define the tokenizer of python generator module
            self.python_tokenizer = AutoTokenizer.from_pretrained(local_dir)

            
            # Define the model pipeline of the python generator module
            self.python_pipeline = transformers.pipeline(
            "text-generation",
            model=local_dir,
            torch_dtype=torch.float32,
            device_map="auto",
        
            )


        if self.python_model =='code_llama34b_pythonV1' and self._disable_model_backend("python_model", "code_llama34b_pythonV1", ['HUGGINGFACE_TOKEN', 'CODELLAMA_34B_PYTHONV1_CACHE_DIR', 'CODELLAMA_34B_PYTHONV1_LOCAL_DIR']) :

            # Huggingface login
            login(token=huggingface_token,new_session=False)  

            repo = "Phind/Phind-CodeLlama-34B-Python-v1"

            cache_dir = os.environ['CODELLAMA_34B_PYTHONV1_CACHE_DIR']
            local_dir = os.environ['CODELLAMA_34B_PYTHONV1_LOCAL_DIR']
          

            snapshot_download(repo_id=repo,cache_dir=cache_dir,local_dir=local_dir,local_dir_use_symlinks=True)

            # Path to model
            logging.info("=====Running CodeLLama-34B-Python-V1========")

            # Check if CUDA (GPU) is available 
            if torch.cuda.is_available():
                    # Get the number of available GPUs
                    num_gpus = torch.cuda.device_count()
                    logging.info(f"Number of available GPUs: {num_gpus}")
                    # Iterate through available GPUs and print information about each
                    for i in range(num_gpus):
                        gpu = torch.cuda.get_device_name(i)
                        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9  # in GB
                        logging.info(f"GPU,Total Memory: {gpu},{gpu_memory}")
            else:
                    logging.info("No GPU available. Using CPU.")

            # Define the tokenizer of python generator module
            self.python_tokenizer = AutoTokenizer.from_pretrained(local_dir)

            
            # Define the model pipeline of the python generator module
            self.python_pipeline = transformers.pipeline(
            "text-generation",
            model=local_dir,
            torch_dtype=torch.float32,
            device_map="auto",
        
            )

        
        if self.python_model == 'wizardcoder_34B' and self._disable_model_backend("python_model", "wizardcoder_34B", ['HUGGINGFACE_TOKEN', 'WIZARDCODER_34B_PYTHON_CACHE_DIR', 'WIZARDCODER_34B_PYTHON_LOCAL_DIR']):
            
            # Huggingface login
            login(token=huggingface_token,new_session=False)  
            repo = "WizardLM/WizardCoder-Python-34B-V1.0"

            
            cache_dir = os.environ['WIZARDCODER_34B_PYTHON_CACHE_DIR']
            local_dir = os.environ['WIZARDCODER_34B_PYTHON_LOCAL_DIR']
          
        

            snapshot_download(repo_id=repo,cache_dir=cache_dir,local_dir=local_dir,local_dir_use_symlinks=True)

            # Path to model
            logging.info("=====Running Wizard-Coder-34B========")

            # Check if CUDA (GPU) is available 
            if torch.cuda.is_available():
                    # Get the number of available GPUs
                    num_gpus = torch.cuda.device_count()
                    logging.info(f"Number of available GPUs: {num_gpus}")
                    # Iterate through available GPUs and print information about each
                    for i in range(num_gpus):
                        gpu = torch.cuda.get_device_name(i)
                        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9  # in GB
                        logging.info(f"GPU,Total Memory: {gpu},{gpu_memory}")
            else:
                    logging.info("No GPU available. Using CPU.")

            
            # Define the tokenizer of python generator module
            self.python_tokenizer = AutoTokenizer.from_pretrained(local_dir)

            # Define the model 
            self.model_code = AutoModelForCausalLM.from_pretrained(local_dir)
            

    def _load_examples_for_dataset(self, dataset_name):
        logging.info(f"Dataset: {dataset_name}")

        if dataset_name == "AQUA":
            examples = read_jsonl_file(self._required_env('TEST_AQUA_DATA_FILE_PATH', "AQUA dataset loading"))
        elif dataset_name == "MMLU":
            examples = read_jsonl_file(self._required_env('TEST_MMLU_DATA_FILE_PATH', "MMLU dataset loading"))
        elif dataset_name == "GSM":
            examples = read_jsonl_file(self._required_env('TEST_GSM8K_DATA_FILE_PATH', "GSM dataset loading"))
        elif dataset_name == "MATH":
            if self.data_file == 'yes':
                examples = read_jsonl_file(self._required_env('MATH_DATA_FILE_PATH', "MATH dataset loading"))
            else:
                examples = read_jsonl_file(self._required_env('SHUFFLED_MATH_DATA_FILE_PATH', "MATH dataset loading"))
        else:
            raise NotImplementedError(f"Unsupported dataset: {dataset_name}")

        if examples:
            logging.info(f"First Sample: {examples[0]}")
        logging.info(f"{type(examples)}")
        return examples


    def _tag_examples_with_dataset(self, examples, dataset_name):
        tagged_examples = []
        for example in examples:
            tagged_example = dict(example)
            tagged_example["dataset"] = dataset_name
            tagged_examples.append(tagged_example)
        return tagged_examples


    def _mix_all_datasets(self, dataset_names):
        datasets = {}
        for dataset_name in dataset_names:
            dataset_examples = self._tag_examples_with_dataset(
                self._load_examples_for_dataset(dataset_name),
                dataset_name,
            )
            random.shuffle(dataset_examples)
            datasets[dataset_name] = dataset_examples

        strategy = getattr(self, "mixed_dataset_strategy", "balanced")
        if strategy == "proportional":
            examples = []
            for dataset_name in dataset_names:
                examples.extend(datasets[dataset_name])
            random.shuffle(examples)
            return examples

        target_total = getattr(self, "test_number", None)
        if target_total is not None:
            selected = []
            quotas = {name: 0 for name in dataset_names}
            active = [name for name in dataset_names if datasets[name]]
            remaining_target = target_total

            # Allocate near-even quotas across datasets, but randomize which
            # datasets receive the extra slots when the target is not divisible.
            while remaining_target > 0 and active:
                round_order = list(active)
                random.shuffle(round_order)
                assigned_this_round = False
                for dataset_name in round_order:
                    if remaining_target == 0:
                        break
                    if quotas[dataset_name] >= len(datasets[dataset_name]):
                        continue
                    quotas[dataset_name] += 1
                    remaining_target -= 1
                    assigned_this_round = True
                if not assigned_this_round:
                    break
                active = [
                    dataset_name
                    for dataset_name in active
                    if quotas[dataset_name] < len(datasets[dataset_name])
                ]

            for dataset_name in dataset_names:
                if quotas[dataset_name]:
                    selected.extend(datasets[dataset_name][:quotas[dataset_name]])
            random.shuffle(selected)
            return selected

        # For full mixed runs, keep the order random while removing dataset-size
        # bias by choosing the next dataset uniformly from the remaining non-empty
        # datasets until every example is consumed.
        examples = []
        remaining = {name: list(dataset_examples) for name, dataset_examples in datasets.items()}
        available = [name for name in dataset_names if remaining[name]]
        while available:
            dataset_name = random.choice(available)
            examples.append(remaining[dataset_name].pop())
            if not remaining[dataset_name]:
                available.remove(dataset_name)
        return examples


    def load_data(self):
        requested_dataset = self.requested_dataset

        if requested_dataset == "ALL":
            examples = self._mix_all_datasets(["MATH", "GSM", "AQUA", "MMLU"])
        else:
            examples = self._tag_examples_with_dataset(
                self._load_examples_for_dataset(requested_dataset),
                requested_dataset,
            )
            random.shuffle(examples)

        self.len_examples = len(examples)
        # limit the number of test examples
        if self.test_number is None:
            self.test_number = len(examples)
        else:
            if self.test_number < len(examples):
                examples = examples[:self.test_number]
            elif self.test_number > len(examples):
                raise Exception("test_number cannot be greater than number of samples!!")
                
        return examples


    def get_question_text(self):
        
        if "question_text" in self.cache:
            question_text = strip_asy_blocks_for_model_input(self.cache["question_text"])
            self.cache["question_text"] = question_text
            return question_text 
        
        # question text
        question = self.cache["example"]["problem"]
        question_text = strip_asy_blocks_for_model_input(question)
        question_text = f"{question_text}\n\n"
        self.cache["question_text"] = question_text
        return question_text

    def get_metadata(self):
        
        if "metadata" in self.cache:
            return self.cache["metadata"] 
        metadata = extract_example_metadata(self.cache.get("example"))
        self.cache["metadata"] = metadata
        return metadata

    def build_prompt_for_policy(self):
        # get the example
        question_text = self.get_question_text()
        
        # build the prompt
        demo_prompt = prompt_policy.prompt.strip() # demo prompt
        
        #test_prompt = f"Question: {question_text}\n\nMetadata: {metadata}\n\nModules: " # test prompt

        metadata = self.get_metadata()
        typ = str(metadata.get("topic") or "Unknown")
        lvl = str(metadata.get("level") or "")

        test_prompt = f"Question: {question_text}\n\nMathematics Problem Type:{typ}\n"
        if lvl:
            test_prompt += f"Level of Problem:{lvl}\n"
        test_prompt += "Thought:"
        full_prompt = demo_prompt + "\n\n" + test_prompt  # full prompt

        return test_prompt, full_prompt

    def _recover_missing_program_for_execution(self):
        recovery_notes = []

        cached_program = sanitize_generated_python(self.cache.get("program"))
        if cached_program:
            self.cache["program"] = cached_program
            return cached_program, None

        generated_program = sanitize_generated_python(self.cache.get("program_generator:output"))
        if generated_program:
            self.cache["program"] = generated_program
            recovery_notes.append("Recovered Python program from cached generator output.")
            return generated_program, " ".join(recovery_notes)

        refine_round_keys = sorted(
            (
                key for key in self.cache
                if str(key).startswith("refine_round")
            ),
            key=lambda key: int(re.search(r"(\d+)$", str(key)).group(1)) if re.search(r"(\d+)$", str(key)) else -1,
        )
        for round_key in reversed(refine_round_keys):
            round_data = self.cache.get(round_key) or {}
            recovered_round_program = sanitize_generated_python(round_data.get("code"))
            if recovered_round_program:
                self.cache["program"] = recovered_round_program
                recovery_notes.append(f"Recovered Python program from {round_key}.")
                return recovered_round_program, " ".join(recovery_notes)

        generator_name = "python_generator_refine_executor" if "python_generator_refine_executor" in (self.modules or []) else "program_generator"
        generator_fn = getattr(self, generator_name, None)
        if callable(generator_fn):
            try:
                _, regenerated_program = generator_fn()
                regenerated_program = sanitize_generated_python(regenerated_program)
                if regenerated_program:
                    self.cache["program"] = regenerated_program
                    recovery_notes.append(f"Regenerated Python program during execution via {generator_name}.")
                    return regenerated_program, " ".join(recovery_notes)
                recovery_notes.append(f"{generator_name} returned an empty program during execution recovery.")
            except Exception as exc:
                recovery_notes.append(f"{generator_name} failed during execution recovery: {exc}")

        return "", " ".join(recovery_notes).strip() or "Unable to recover a Python program for execution."

    def _retry_python_program_for_repo_echo(self, full_prompt, question_text, program):
        suspicious = "import os" in program and "class solver" in program
        if not suspicious:
            return program

        retry_prompt = (
            full_prompt
            + "\nIMPORTANT: Your previous response copied repository code instead of solving the current question.\n"
            + "Do not reproduce the solver, helper functions, imports like os/sys for project code, or any class definitions.\n"
            + "Return ONLY a short standalone Python program for the current math problem.\n"
            + f"Current question:\n{question_text}\n\n"
            + "Code:\n"
        )
        regenerated = self._generate_python_program(retry_prompt)
        regenerated = sanitize_generated_python(regenerated)
        if regenerated and not ("import os" in regenerated and "class solver" in regenerated):
            warning = "program_generator: repo-echo draft was regenerated into standalone code."
            warnings = self.cache.setdefault("module_warnings", [])
            if warning not in warnings:
                warnings.append(warning)
            return regenerated

        warning = "program_generator: repo-echo draft could not be repaired; using last-resort stub."
        warnings = self.cache.setdefault("module_warnings", [])
        if warning not in warnings:
            warnings.append(warning)
        return self._build_last_resort_python_program()

    def predict_modules(self):
        # get the module input
        test_prompt, full_prompt = self.build_prompt_for_policy()
       
        messages=[
            {"role": "user", "content": full_prompt},
        ]
        
        # execute the module
     
        modules = get_chat_response(messages=messages, temperature = self.policy_temperature, max_tokens=self.policy_max_tokens)
        logging.info(f"Response by planner: {modules}")
            
        # Get part starting with '['
        try:
            index = modules.index('[')
            modules = modules[index:]

        except:
            modules = '''['solution_generator']'''

        modules = self.update_modules(modules)

        logging.info(f"Modules selected by planner: {modules}")

        # update the cache
        self.cache["modules:input"] = test_prompt
        self.cache["modules:output"] = modules
        

        return modules

    def update_modules(self, _modules):
        
        # default modules
        default_end_modules = ["solution_generator"]
        valid_modules = {
            "knowledge_retrieval",
            "bing_search",
            "wolfram_alpha_search",
            "program_generator",
            "python_generator_refine_executor",
            "program_executor",
            "solution_generator",
            "answer_generator",
        }
       
        try:
            
            logging.info(f"Modules before eval {_modules}")
            modules = ast.literal_eval(str(_modules).strip())
            if not isinstance(modules, list):
                raise ValueError("Planner did not return a list")
            modules = normalize_module_sequence(modules)
            if any(module not in valid_modules for module in modules):
                raise ValueError("Planner returned unknown modules")
            
            assert modules[-1:] == default_end_modules
               
        except Exception:

            modules = default_end_modules

        return modules
    

    def call_answer_cleaner(self, q, res):
        extracted_answer = self._extract_wolfram_answer_from_result(res)
        if extracted_answer not in (None, ""):
            return extracted_answer

        return resolve_wolfram_answer(
            q,
            res,
            chat_callable=lambda prompt, max_tokens: self._call_wolfram_llm(prompt, temperature=0.5, max_tokens=max_tokens),
            max_tokens=5000,
            wolfram_model=self.wolfram_model,
            text_davinci003_callable=lambda prompt, temperature, max_tokens: get_textdavinci003_response(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            gemini_callable=get_gemini_response,
        )

    def _call_wolfram_llm(self, prompt, temperature=0.2, max_tokens=800):
        if self.wolfram_model == 'text_davinci_003':
            return get_textdavinci003_response(prompt, temperature=temperature, max_tokens=max_tokens)
        if self.wolfram_model == 'gemini':
            return get_gemini_response(prompt)
        messages = [{"role": "user", "content": prompt}]
        return get_chat_response(messages=messages, temperature=temperature, max_tokens=max_tokens)

    def _generate_wolfram_initial_query(self, full_prompt):
        tries = 0
        last_response = None

        while tries < 3:
            last_response = self._call_wolfram_llm(full_prompt, temperature=0.5, max_tokens=600)
            tries += 1
            parsed = parse_wolfram_query_response(last_response)
            if parsed and parsed.get("query"):
                return parsed, last_response

        return None, last_response

    def _plan_wolfram_next_step(self, question_text, trace):
        prompt = build_wolfram_next_step_prompt(question_text, trace)
        tries = 0
        last_response = None

        while tries < 3:
            last_response = self._call_wolfram_llm(prompt, temperature=0.2, max_tokens=700)
            tries += 1
            parsed = parse_wolfram_next_step_response(last_response)
            if parsed is not None:
                return parsed, last_response

        return None, last_response

    def _format_wolfram_response_context(self, trace, final_answer):
        trace_text = format_wolfram_trace(trace)
        if not trace_text:
            return ""

        lines = []
        for step in trace:
            step_number = step.get("step")
            thought = step.get("thought")
            query = step.get("query")
            output = step.get("output")
            if thought:
                lines.append(f"Wolfram Step {step_number} Thought: {thought}")
            if query:
                lines.append(f"Wolfram Step {step_number} Query: {query}")
            if output:
                lines.append(f"Wolfram Step {step_number} Output: {output}")

        if final_answer not in (None, ""):
            lines.append(f"Wolfram Final Answer: {final_answer}")

        return "\n" + "\n".join(lines) + "\n"

    
    def wolfram_alpha_search(self):
        app_id = self._optional_env("WOLFRAM_ALPHA_APPID")
        if app_id is None:
            return self._skip_optional_module("wolfram_alpha_search", "WOLFRAM_ALPHA_APPID is not configured")

        question_text = self.get_question_text()
        response = self.cache["response"] if "response" in self.cache else ""

        if self.dataset == "AQUA":
            demo_prompt = build_option_wolfram_demo_prompt(self._option_letters_for_dataset()).strip()
        elif self.dataset == "MMLU":
            demo_prompt = build_option_wolfram_demo_prompt(self._option_letters_for_dataset()).strip()
        elif self.dataset == "GSM":
            demo_prompt = build_gsm_wolfram_demo_prompt().strip()
        else:
            demo_prompt = prompt_walpha_context_withthought.prompt.strip()

        try:
            ind = self.modules.index('query_generator')
        except:
            ind = None

        if ind is not None:
            mods = self.modules[:ind]
            mods = " ".join(mods)
        else:
            mods = ""

        if response != "" and mods != "":
            test_prompt = f"Question:{question_text}\nModules used till now:[{mods}]\n{response}\nThought:"
        else:
            test_prompt = f"Question: {question_text}\nThought:"

        direct_query_instruction = (
            "\nAdditional instruction: prefer a Wolfram query that directly computes the quantity asked in the question."
            "\nIf you must solve an intermediate system first, expect follow-up Wolfram steps before finalizing."
        )
        full_prompt = demo_prompt + "\n" + test_prompt + direct_query_instruction

        initial_plan, query = self._generate_wolfram_initial_query(full_prompt)
        answer_walpha = None
        q = initial_plan.get("query") if initial_plan else None
        thought = initial_plan.get("thought", "") if initial_plan else ""
        trace = []
        final_query = None
        last_error = None
        resolved_wolfram_option = None

        if q:
            current_query = q
            seen_query_signatures = set()

            for step_index in range(1, 5):
                normalized_query = normalize_wolfram_query_signature(current_query)
                if not normalized_query:
                    last_error = "Empty Wolfram Alpha query"
                    break
                if normalized_query in seen_query_signatures:
                    last_error = "Wolfram query planner repeated a previous query before reaching the requested final quantity"
                    break
                seen_query_signatures.add(normalized_query)

                query_result = query_wolfram_alpha(app_id, current_query, logger=logging)
                executed_query = query_result.get("query") or current_query
                final_query = executed_query
                res = query_result.get("result")
                step_output = query_result.get("answer")
                step_error = query_result.get("error")

                if step_output in (None, "") and res:
                    step_output = self.call_answer_cleaner(executed_query, res)

                step_record = {
                    "step": step_index,
                    "thought": thought,
                    "query": executed_query,
                    "output": step_output,
                    "error": step_error,
                    "source": query_result.get("source"),
                }
                trace.append(step_record)

                if step_error and step_output in (None, ""):
                    logging.warning("Wolfram Alpha query failed: %s", step_error)
                    last_error = step_error
                    break

                decision, _ = self._plan_wolfram_next_step(question_text, trace)
                if decision is None:
                    last_error = "Wolfram planner could not determine whether the requested final quantity had been reached"
                    break

                if decision.get("status") == "FINAL":
                    raw_answer_walpha = decision.get("final_answer") or step_output
                    resolved_wolfram_option = self._resolve_option_crosscheck(
                        step_output,
                        raw_answer_walpha,
                        question_text=question_text,
                    )
                    answer_walpha = raw_answer_walpha
                    if resolved_wolfram_option:
                        explicit_wolfram_option = (
                            extract_final_answer_option_letter(raw_answer_walpha, allowed=self._allowed_option_letters())
                            or extract_option_letter(raw_answer_walpha, allowed=self._allowed_option_letters())
                        )
                        if explicit_wolfram_option != resolved_wolfram_option.get("key"):
                            answer_walpha = self._format_resolved_option_answer(resolved_wolfram_option)
                        elif question_requests_closest_option(question_text):
                            answer_walpha = self._format_resolved_option_answer(resolved_wolfram_option)

                    if decision.get("thought"):
                        trace[-1]["completion_thought"] = decision.get("thought")
                    if resolved_wolfram_option and (
                        question_requests_closest_option(question_text)
                        or trace[-1].get("completion_thought") in (None, "")
                        or (
                            extract_final_answer_option_letter(raw_answer_walpha, allowed=self._allowed_option_letters())
                            or extract_option_letter(raw_answer_walpha, allowed=self._allowed_option_letters())
                        ) != resolved_wolfram_option.get("key")
                    ):
                        trace[-1]["completion_thought"] = self._format_option_crosscheck_completion(resolved_wolfram_option)
                    break

                thought = decision.get("thought", "")
                next_query = decision.get("next_query")
                if not next_query:
                    last_error = "Wolfram planner requested another step without providing a follow-up query"
                    break
                loop_reason = detect_wolfram_loop_reason(trace, next_query=next_query)
                if loop_reason:
                    if not trace[-1].get("error"):
                        trace[-1]["error"] = loop_reason
                    last_error = loop_reason
                    break
                current_query = next_query
        else:
            last_error = "Wolfram query generator did not return a usable query"

        if answer_walpha not in (None, ""):
            resolved_option = resolved_wolfram_option or self._resolve_option_crosscheck(answer_walpha, question_text=question_text)
            if resolved_option:
                self.cache["wolfram_alpha_search:resolved_option"] = resolved_option["key"]
                explicit_wolfram_option = (
                    extract_final_answer_option_letter(answer_walpha, allowed=self._allowed_option_letters())
                    or extract_option_letter(answer_walpha, allowed=self._allowed_option_letters())
                )
                if explicit_wolfram_option != resolved_option.get("key") or question_requests_closest_option(question_text):
                    answer_walpha = self._format_resolved_option_answer(resolved_option)
                elif not explicit_wolfram_option and resolved_option.get("match_type") == "closest-numeric":
                    answer_walpha = (
                        f"{answer_walpha} (closest option {resolved_option['key']}: {resolved_option['label']})"
                    )
            response += self._format_wolfram_response_context(trace, answer_walpha)
            response = response.strip()
            self.cache.pop("wolfram_alpha_search:error", None)
        else:
            self.cache["wolfram_alpha_search:error"] = last_error or "Wolfram Alpha did not reach the requested final answer"

        rendered_input = final_query or q
        if trace:
            rendered_input = "\n".join(f"Step {step['step']}: {step.get('query')}" for step in trace if step.get("query"))

        self.cache["query"] = final_query or q
        self.cache["response"] = response
        self.cache["query_generator:input"] = test_prompt
        self.cache["query_generator:output"] = query
        self.cache["wolfram_alpha_search:input"] = rendered_input
        self.cache["wolfram_alpha_search:output"] = answer_walpha
        self.cache["wolfram_alpha_search:trace"] = trace

        return final_query or q, answer_walpha   
        
    def get_wiki_summary_(self,query):
        import wikipedia
        page = self.get_closest_wikipage(query)
        
        if page == None:
            return ""
        
        summary = page.summary
        summary = summary.split(".")
        
        return summary[:6]
      
    def get_closest_wikipage(query):
        
        import wikipedia
        try: 
                wiki_page = wikipedia.page(query)
                return wiki_page
        except wikipedia.exceptions.DisambiguationError as e:
                wiki_page = wikipedia.page(e.options[0]) 
                return wiki_page
        except wikipedia.exceptions.PageError as e:
                return None  
    
    def wikipedia_search(self):
        
        question_text = self.get_question_text()
        
        # Wiki query
        wiki_query = self.cache["query"] if "query" in self.cache else None
        
        # Response of the pipeline till now 
        response = self.cache["response"] if "response" in self.cache else ""

        # execute the module (call the Bing Search API and get the responses)
        if wiki_query != None and wiki_query != "":
            result = self.get_wiki_summary_(wiki_query)
        else:
            result = None
        

        if len(result) > 0 and result != "" and result!=None:
            response += f"\n\n Wikipedia Search response: {result}"
            response = response.strip()

        # update the cache
        self.cache["response"] = response
        self.cache["wiki_search:input"] = wiki_query
        self.cache["wiki_search:output"] = result
        return wiki_query, result
    

    def knowledge_retrieval(self):
        # get the example
        question_text = self.get_question_text()
        
        # Get the response till now 
        response = self.cache["response"] if "response" in self.cache else ""

        # build the prompt
        if self.dataset == "AQUA":
           demo_prompt = build_option_knowledge_demo_prompt(self._option_letters_for_dataset()).strip()
        elif self.dataset == "GSM":
           demo_prompt = build_gsm_knowledge_demo_prompt().strip()
        elif self.dataset == "MMLU":
           demo_prompt = build_option_knowledge_demo_prompt(self._option_letters_for_dataset()).strip()
        else:
           demo_prompt = build_math_knowledge_demo_prompt().strip()

        test_prompt, full_prompt = build_knowledge_retrieval_prompt(
            demo_prompt,
            question_text,
            response,
        )

        knowledge = self._generate_knowledge_text(full_prompt)
        knowledge = self._maybe_regenerate_cross_problem_knowledge(full_prompt, question_text, knowledge)

        # update the response cache
        if knowledge != "" and knowledge != None:
            response += f"\n\nKnowledge Retrieval:\n{knowledge}"
            response = response.strip()

        # update the cache
        self.cache["response"] = response
        self.cache["knowledge_retrieval:input"] = test_prompt
        self.cache["knowledge_retrieval:output"] = knowledge
        return test_prompt, knowledge
    
    
    def program_generator(self):
        if self.model in {"kr_pg_sg", "kr_pg_walpha_sg"}:
            test_prompt, full_prompt = self.build_prompt_for_kr_pg()
        else:
            test_prompt, full_prompt = self.build_prompt_for_pg()

        response = self.cache["response"] if "response" in self.cache else ""
        question = strip_asy_blocks_for_model_input(self.cache["example"]["problem"])
        program = self._generate_python_program(full_prompt)

        if program is None:
            program = ""

        program = sanitize_generated_python(program)

        if self.dataset == "GSM" and looks_like_gsm_story_leak(question, program):
            leaked_tokens = ", ".join(gsm_story_leak_tokens(question, program)[:6]) or "unrelated story terms"
            retry_prompt = (
                full_prompt
                + "\nIMPORTANT: Your previous draft appears to be copied from a different word problem.\n"
                + f"Unrelated leaked terms: {leaked_tokens}\n"
                + "Regenerate from scratch for ONLY the current question, using only the nouns and numbers in that question.\n"
                + "Code:\n"
            )
            regenerated = self._generate_python_program(retry_prompt)
            if regenerated:
                program = sanitize_generated_python(regenerated)

        program = self._maybe_reverse_check_option_program(full_prompt, question, program)
        program = self._maybe_reverse_check_geometry_program(full_prompt, question, program)
        program = self._maybe_regenerate_cross_problem_program(full_prompt, question, program)
        program = self._retry_python_program_for_repo_echo(full_prompt, question, program)
        if not program:
            program = self._build_last_resort_python_program()
            warning = "program_generator: sanitized program was still empty after recovery; using last-resort stub."
            warnings = self.cache.setdefault("module_warnings", [])
            if warning not in warnings:
                warnings.append(warning)
        
        self.cache["response"] = response
        self.cache["program"] = program
        self.cache["program_generator:input"] = test_prompt
        self.cache["program_generator:output"] = program

        return test_prompt, program
    

    def code_fixer(self,error_program,error_message):
        demo_prompt = prompt_codefixer.prompt
        test_prompt = f"\nIncorrect Python code:\n{error_program}\nError message:{error_message}\n"
        full_prompt = demo_prompt + test_prompt
        system_message = """
        You are an AI assistant skilled in Python programming and debugging. Help users identify errors in their Python code and output the new correct python code. Make sure to optimize the corrected code and follow best practices for writing clean, efficient, and maintainable Python code.
        Here are some common errors that the input python code may have:
        (1) Use of undefined functions or making up function names.
        (2) Forgetting to declare symbols or variables in the python code.
        (3) Use of classes or methods without properly importing required libraries like Sympy, math, etc.
        (4) Wrong way of handling mathematical objects specially in the Sympy library, use of invalid operators with class objects.
        (5) Code has an abrupt end, or code contains natural language sentences instead of python syntax.
        (6) Wrong assumptions about the shape of Sympy solve() output. If the code indexes solve(...) with a Symbol, use dict=True and access the first dict, or use integer indexing if solve() returns a list.
        (7) Do not call .evalf() on plain Python ints or floats. Use the numeric value directly, or use N(...) / expr.evalf(...) only for actual SymPy expressions.
        (8) If solve() is applied to an inequality or a logical And/Or condition, replace it with direct algebra, reduce_inequalities(...), or another valid method.
        (9) If code accesses .free_symbols or other symbolic attributes on a list/tuple returned by solve(), first select the relevant expression or rewrite the computation more directly.
        (10) If SymPy lcm/gcd is being used on plain Python integers and fails, switch to math.lcm/math.gcd or direct integer arithmetic.
        (11) Always make sure the corrected program prints the derived final answer on its final line.
        Preserve helpful comments when possible, and keep any comments in the corrected code short and directly tied to the main computation steps.
        """
        
        code_fixer_response = get_chat_response_code(full_prompt,temperature = 0.7, max_tokens=5000,system_mess=system_message)
        logging.info(f"Code-fixer response {code_fixer_response}")
        
        # Parse output to get new program
        try:
            idx1 = code_fixer_response.index("Corrected Python Code:")
        except:
            repaired_program = sanitize_generated_python(code_fixer_response)
            if repaired_program:
                return repaired_program, None
            return error_program,None
        try:
            idx2 = code_fixer_response.index("Errors fixed:")
        except:
            repaired_program = sanitize_generated_python(code_fixer_response)
            if repaired_program:
                return repaired_program, None
            return error_program,None
        
        new_program = code_fixer_response[idx1+len("Corrected Python Code:"):idx2]
        errors_fixed = code_fixer_response[idx2+len("Errors fixed:"):]
        return sanitize_generated_python(new_program),errors_fixed


    def _execute_candidate_program(self, program):
        cleaned_program = sanitize_generated_python(program)
        output, raw_error = safe_execute(cleaned_program)
        _, missing_package = extract_missing_dependency(raw_error)

        if missing_package and missing_package not in self.dependency_install_attempts:
            self.dependency_install_attempts.add(missing_package)
            dependency_install = install_missing_dependency(raw_error)
            self.cache.setdefault("dependency_installs", []).append(dependency_install)
            if dependency_install.get("installed"):
                output, raw_error = safe_execute(cleaned_program)

        if raw_error in (None, ""):
            output = self._maybe_annotate_option_program_output(self.get_question_text(), output)

        effective_error = execution_failure_reason(self.dataset, output, raw_error)
        return cleaned_program, output, raw_error, effective_error


    def _attempt_program_repairs(self, program):
        def execute_program(candidate_program):
            repaired_program, repaired_output, repaired_raw_error, _ = self._execute_candidate_program(candidate_program)
            return repaired_output, repaired_raw_error

        repaired_program, repaired_output, repaired_raw_error, repaired_error, attempts = repair_program_until_runnable(
            program,
            self.dataset,
            execute_program,
            llm_repair=self.code_fixer,
            max_attempts=3,
        )

        self.cache["program_repair_trace"] = attempts
        if attempts:
            self.cache["program_repair"] = attempts[-1]
        else:
            self.cache.pop("program_repair", None)

        self.cache["program"] = repaired_program
        return repaired_program, repaired_output, repaired_raw_error, repaired_error



    def python_generator_refine_executor(self):
        
        if self.model in {'kr_pg_sg', 'kr_pg_walpha_sg'}:
               test_prompt, full_prompt = self.build_prompt_for_kr_pg()
        else:
               test_prompt, full_prompt = self.build_prompt_for_pg()
        
        messages=[
               {"role": "user", "content": full_prompt},
            ]
        # Get the response till now
        response = self.cache["response"] if "response" in self.cache else ""
        max_iterations=3   # Maximum no of attempts to correct the code
        count=0
        errors_fixed = None
        
        while True and count<max_iterations:
            copy_messages = messages.copy()
            count=count+1
            if count>1:
                
                # Extract the program and error message of last round
                error_program = self.cache["refine_round"+str(count-1)]['code']
                error_message = self.cache["refine_round"+str(count-1)]['error']
                
                # Feed the error message and program to an independent code_fixer module
                # Make changes to the code using error message
                program,errors_fixed = self.code_fixer(error_program,error_message)
            
            if count<=1:
                # Generate code 1st time
                program = self._generate_python_program(copy_messages[0]["content"])

            program = self._maybe_reverse_check_option_program(full_prompt, self.get_question_text().strip(), program)
            program = self._maybe_reverse_check_geometry_program(full_prompt, self.get_question_text().strip(), program)
            program = self._maybe_regenerate_cross_problem_program(full_prompt, self.get_question_text().strip(), program)
            program = self._retry_python_program_for_repo_echo(full_prompt, self.get_question_text().strip(), program)
            if not program:
                program = self._build_last_resort_python_program()
            
            # Check if the code is executable
            program, output, _, error_message = self._execute_candidate_program(program)
             
            if error_message is None:
               response += f"\nPython generator:\n{program}"
               response += f"\nPython output:\n{output}"
               response = response.strip()
               self.cache["refine_round"+str(count)] ={'code':program, 'error':error_message, 'output':output,'errors_fixed':errors_fixed}
               
               '''
               print("Code\n",self.cache["refine_round"+str(count)]['code'])
               print("Error\n",self.cache["refine_round"+str(count)]['error'])
               print("Output\n",self.cache["refine_round"+str(count)]['output'])
               print("Errors fixed\n",self.cache["refine_round"+str(count)]['errors_fixed'])
               '''
               break
            
            else:
               '''
               # Add the error message and the program to the messages context
                messages[0]["content"] += f"\n{program}"
                messages[0]["content"] += f"\nOutput:{output}"
                messages[0]["content"] += f"\nError message:\n{error_message}"
               '''
               # Store the code for this round
               self.cache["refine_round"+str(count)] ={'code':program, 'error':error_message, 'output':output,'errors_fixed':errors_fixed}
               
               '''
               print("Code\n",self.cache["refine_round"+str(count)]['code'])
               print("Error\n",self.cache["refine_round"+str(count)]['error'])
               print("Output\n",self.cache["refine_round"+str(count)]['output'])
               print("Errors fixed\n",self.cache["refine_round"+str(count)]['errors_fixed'])
               '''
        
        # update the cache
        self.cache["response"] = response
        self.cache["program"] = program
        # Store the messages of refine
        self.cache["messages_refine"] = messages
        # Store the no of steps of refinement
        self.cache["num_refines"] = count
        self.cache["program_generator:input"] = test_prompt
        self.cache["program_generator:output"] = program
        return test_prompt, program

    def program_executor(self):
        response = self.cache["response"] if "response" in self.cache else ""

        program = sanitize_generated_python(self.cache.get("program"))
        recovery_note = None
        if not program:
            program, recovery_note = self._recover_missing_program_for_execution()
            program = sanitize_generated_python(program)

        if not program:
            error_message = "No program found in cache"
            if recovery_note:
                error_message = f"{error_message}. {recovery_note}"
            self.cache["program_executor:output"] = None
            self.cache["program_executor:error"] = error_message
            if recovery_note:
                self.cache["program_executor:recovery"] = recovery_note
            response += f"\n\nPython execution error:\n{error_message}"
            self.cache["response"] = response.strip()
            return None, f"Execution failed: {error_message}"

        self.cache["program"] = program
        if recovery_note:
            self.cache["program_executor:recovery"] = recovery_note

        program, ans, raw_error, error_message = self._attempt_program_repairs(program)

        # Store results
        self.cache["program_executor:output"] = ans
        self.cache["program_executor:error"] = error_message

        if ans is not None and ans != "":
            response += f"\n\nPython generator:\n{program}"
            response += f"\n\nPython output:\n{ans}"
            response = response.strip()
            self.cache["response"] = response
            return program, ans

        response += f"\n\nPython generator:\n{program}"
        response += f"\n\nPython execution error:\n{error_message}"
        response = response.strip()
        self.cache["response"] = response

        return program, f"Execution failed: {error_message}"

    def solution_generator(self):
        # get the module input
        response = self.cache["response"] if "response" in self.cache else ""

        prompt_family = solution_prompt_family(self.model)

        if prompt_family == "cot":
            test_prompt, full_prompt = self.build_prompt_for_sg_cot()

        elif prompt_family == "pot":
            test_prompt, full_prompt = self.build_prompt_for_pot()

        elif prompt_family == "kr_sg":
            test_prompt, full_prompt = self.build_prompt_for_kr_sg()

        elif prompt_family == "kr_pg_sg":
            test_prompt, full_prompt = self.build_prompt_for_kr_pg_sg()

        else:
            test_prompt, full_prompt = self.build_prompt_for_kr_walpha_sg()

        # excute the module
        success = False
        patience = self.sg_patience
        count = 0
        while count < patience and not success:
            if self.sg_temperature < 0.1 and count > 0:
                _temperature = min(self.sg_temperature + 0.1, 1.0)
            else:
                _temperature = self.sg_temperature
            
            solution = self._generate_solution_text(full_prompt, _temperature)

            if self.dataset == "GSM" and looks_like_gsm_story_leak(self.get_question_text(), solution):
                leaked_tokens = ", ".join(gsm_story_leak_tokens(self.get_question_text(), solution)[:6]) or "unrelated story terms"
                retry_prompt = (
                    full_prompt
                    + "\nIMPORTANT: The previous draft appears grounded in a different word problem.\n"
                    + f"Unrelated leaked terms: {leaked_tokens}\n"
                    + "Rewrite the solution for ONLY the current question and ignore any stray unrelated context.\n"
                    + "Solution: "
                )
                regenerated = self._generate_solution_text(retry_prompt, _temperature)
                if regenerated:
                    solution = regenerated

            solution = self._maybe_regenerate_cross_problem_solution(
                full_prompt,
                self.get_question_text(),
                solution,
                _temperature,
            )
            solution = self._maybe_reverse_check_option_solution(
                self.get_question_text(),
                response,
                solution,
                _temperature,
            )

            
            #pattern = re.compile(r"[Tt]he answer is ([A-Z])")      # "The answer is XXXXX.",
            #res = pattern.findall(solution)
            
            if self.dataset == "AQUA":
                success = extract_option_letter(solution, allowed="ABCDE") is not None

            elif self.dataset == "GSM":
                success = extract_tagged_answer(solution) is not None or extract_numeric_answer(solution) is not None
            
            elif self.dataset == "MMLU":
                success = extract_option_letter(solution, allowed="ABCD") is not None

            else:
                success = bool(solution and ("boxed" in solution or extract_tagged_answer(solution)))
            
            count += 1
        
        response = response + "\nSolution:\n" + solution
        # update the cache
        self.cache["response"] = response
        self.cache["solution"] = solution
        self.cache["solution_generator:input"] = test_prompt
        self.cache["solution_generator:output"] = solution
        return test_prompt, solution


    def bing_search(self):
        
        # Set up Bing credentials
        endpoint = self._optional_env('BING_API_ENDPOINT')
        count = os.getenv('BING_API_COUNT', '5')
        bing_api_key = self._optional_env('BING_API_KEY')
        if endpoint is None or bing_api_key is None:
            return self._skip_optional_module("bing_search", "BING_API_ENDPOINT or BING_API_KEY is not configured")

        
        # Get the question and context
        question_text = self.get_question_text()
        response = self.cache["response"] if "response" in self.cache else ""
        

        try:
            ind = (self.modules).find('bing_search')
        except:  
            ind = None

        if ind!=None:
            mods = (self.modules)[:ind]
            mods = ' '.join(mods)
        else:
            mods = ""    


        # Use LLM to set up query based on question and context (response)
        if self.dataset == "AQUA":
            demo_prompt = prompt_bing_query.prompt_AQUA
        elif self.dataset == "MMLU":
            demo_prompt = prompt_bing_query.prompt_MMLU
        else:
            demo_prompt = prompt_bing_query.prompt
        
        if response != "" and mods!="":
            test_prompt = f"Question:{question_text}\nModules used till now:[{mods}]\n{response}\nThought:"
        else:
            test_prompt = f"Question: {question_text}\nThought:"
        
        full_prompt = demo_prompt + test_prompt
        
        messages=[
            {"role": "user", "content": full_prompt},
        ]

        # Query for Bing concept search using LLM-generated query
        
        num_tries = 3
        f = 0
        while(f<num_tries):
            f+=1
            if self.bing_model == 'text_davinci_003': # Check text-davinci-003
                query_output = get_textdavinci003_response(full_prompt,temperature=0.5, max_tokens=5000)
            elif self.bing_model == 'gemini':
                query_output = get_gemini_response(full_prompt)
            else:    
                query_output = get_chat_response(messages, temperature=0.5, max_tokens=5000)
            
            if query_output.find("Query:")!= -1:
                break
        
        
        # Extract the queries and call api 
        query1= question_text  # Query for similar questions search is the input question
        ind = query_output.find("Query:")
        query2= query_output[ind+len("Query:"):]
        
        result1 = call_bing_search(endpoint, bing_api_key, query1, count)

        # execute the module (call the Bing Search API and get the responses for query2)
        if query2 != None and query2 != "":
            result2 = call_bing_search(endpoint, bing_api_key, query2, count)
        else:
            result2 = None
        
        
        # Get all the response snippets retrieved
        responses1 = parse_bing_result(result1)
        responses2 = parse_bing_result(result2)
        
        logging.info(f"Bing response 1 {responses1}")
        logging.info(f"Bing response 2 {responses2}")

        
        # Use LLM to extract useful information from responses
        if self.dataset == "AQUA":
            demo_prompt_extract = prompt_bing_answer_extractor.prompt_AQUA
        elif self.dataset == "MMLU":
            demo_prompt_extract = prompt_bing_answer_extractor.prompt_MMLU 
        else:   
            demo_prompt_extract = prompt_bing_answer_extractor.prompt

        test_prompt_extract1 = f"Question:{question_text}\nBing Search API result:{responses1}\nUseful_information:\n"
        test_prompt_extract2= f"Question:{question_text}\nBing Search API result:{responses2}\nUseful_information:\n"

        full_prompt_extract1 = demo_prompt_extract + test_prompt_extract1
        full_prompt_extract2 = demo_prompt_extract + test_prompt_extract2

        
        messages1=[
            {"role": "user", "content": full_prompt_extract1},
        ]

        
        if self.bing_model == 'text_davinci_003':  # Check text-davinci-003
            info_bing1 = get_textdavinci003_response(full_prompt_extract1,temperature=0.5, max_tokens=5000)
        elif self.bing_model == 'gemini':
            info_bing1 = get_gemini_response(full_prompt_extract1)
        else:
            info_bing1 = get_chat_response(messages1, temperature=0.5, max_tokens=500)
        
            
        messages2=[
            {"role": "user", "content": full_prompt_extract2},
        ]

        
        if self.bing_model == 'text_davinci_003': # Check text-davinci-003
            info_bing2 = get_textdavinci003_response(full_prompt_extract2,temperature=0.5, max_tokens=5000)
        elif self.bing_model == 'gemini':
            info_bing2 = get_gemini_response(full_prompt_extract2)
        else:
            info_bing2 = get_chat_response(messages2, temperature=0.5, max_tokens=5000)
        
        
        # Concatenate bing responses from query 'question' and 'query2' using context
        
        info_bing = info_bing1 + "\n" + info_bing2
        
        if  info_bing!="" and info_bing is not None:
            response += f"\n\nBing search response:\n{info_bing}"
            response = response.strip()

        # update the cache
        self.cache["response"] = response
        self.cache["bing_query2"] = query2
        self.cache["bing_query2_output"] = responses2
        self.cache["bing_query1"] = query1
        self.cache["bing_query1_output"] = responses1
        self.cache["bing_search:input"] = query_output
        self.cache["bing_search:output"] = info_bing
        return query_output, info_bing


    def build_prompt_for_pot(self):
        
        question_text = self.get_question_text()
        
        #metadata = self.get_metadata()
        
        response = self.cache["response"] if "response" in self.cache else ""

        # build the prompt
        if self.dataset == "AQUA":
            demo_prompt = prompt_pot.prompt_pot_AQUA.strip() 
        else:   
            demo_prompt = prompt_pot.prompt_pot.strip() 

        return self._build_solution_prompt(demo_prompt, question_text, response)


    def _build_solution_prompt(self, demo_prompt, question_text, response):
        if response != "":
            test_prompt = f"Question: {question_text}\n\n{response}\n\nSolution: "
        else:
            test_prompt = f"Question: {question_text}\n\nSolution: "

        full_prompt = demo_prompt + "\n\n" + test_prompt
        return test_prompt, full_prompt


    def build_prompt_for_kr_walpha_sg(self):
        
        question_text = self.get_question_text()
        #metadata = self.get_metadata()
        response = self.cache["response"] if "response" in self.cache else ""
        
        #logging.info(f"Response context: {response}")
        flag = response.find("Solution:")
        if flag!=-1:
            response = response[:flag]
        #logging.info(f"After removing Solution - Response context: {response}")


        # build the prompt
        if self.dataset == "AQUA" and self.model=='sg':
            demo_prompt = build_option_solution_demo_prompt(self._option_letters_for_dataset()).strip()
        
        elif self.dataset == "AQUA":
            demo_prompt = build_option_solution_demo_prompt(self._option_letters_for_dataset()).strip()
        
        elif self.dataset == "GSM":
            demo_prompt = build_gsm_solution_demo_prompt().strip()
        elif self.dataset == "MMLU" :
            demo_prompt = build_option_solution_demo_prompt(self._option_letters_for_dataset()).strip()
        else:
            demo_prompt = prompt_walpha_kr_sg.prompt.strip() 
        


        return self._build_solution_prompt(demo_prompt, question_text, response)

    def build_prompt_for_kr_sg(self):
        
        question_text = self.get_question_text()
        
        #metadata = self.get_metadata()
        
        response = self.cache["response"] if "response" in self.cache else ""

        # build the prompt
        if self.dataset == "GSM":
            demo_prompt = build_gsm_solution_demo_prompt().strip()
        elif self.dataset in {"AQUA", "MMLU"}:
            demo_prompt = build_option_solution_demo_prompt(self._option_letters_for_dataset()).strip()
        else:
            demo_prompt = prompt_kr_sg.prompt.strip() # WARNING: this is the prompt for kr_sg
        
        return self._build_solution_prompt(demo_prompt, question_text, response)


    def build_prompt_for_pg(self):
        question = self.get_question_text().strip()
        response = self.cache["response"] if "response" in self.cache else ""

        try:
            ind = self.modules.index('program_generator')
        except:
            ind = self.modules.index("python_generator_refine_executor")

        mods = self.modules[:ind]
        mods = " ".join(mods)

        if self.extra_python_libraries == 'no':
            if self.dataset == "AQUA":
                demo_prompt = build_option_program_demo_prompt(self._option_letters_for_dataset()).strip()
            elif self.dataset == "MMLU":
                demo_prompt = build_option_program_demo_prompt(self._option_letters_for_dataset()).strip()
            elif self.dataset == "GSM":
                demo_prompt = build_gsm_program_demo_prompt().strip()
            else:
                demo_prompt = prompt_pg.prompt.strip()
        else:
            if self.dataset == "AQUA":
                demo_prompt = build_option_program_demo_prompt(self._option_letters_for_dataset()).strip()
            elif self.dataset == "MMLU":
                demo_prompt = build_option_program_demo_prompt(self._option_letters_for_dataset()).strip()
            elif self.dataset == "GSM":
                demo_prompt = build_gsm_program_demo_prompt().strip()
            else:
                demo_prompt = prompt_pg.prompt2.strip()

        python_rules = """
            Python generator:
            # Write ONLY executable Python code.
            # Do not include any explanation, markdown, or code fences.
            # The first line must be exactly: from sympy import *
            # Rules:
            # 1. Always define every symbol before using it, for example: x = symbols('x')
            # 2. Never hardcode the final answer unless it was derived in code.
            # 3. Prefer direct arithmetic/combinatorics/algebra over overly symbolic approaches when possible.
            # 4. Do not use solve() for domain/interval logic unless necessary.
            # 5. Do not sort symbolic expressions.
            # 6. Print the final answer clearly.
            # 7. Print useful intermediate values only if they help verify correctness.
            # 8. The code must run as-is with no placeholders.
            # 9. Be careful with SymPy solve(): for one variable, solve(...) usually returns a list, so use sol[0].
            # 10. For systems of equations, use solve(..., dict=True), then access the first dict like solutions[0][x].
            # 11. Never index a plain solve() list with a Symbol like solution[x] unless you explicitly requested dict=True.
            # 12. Include a few crisp Python comments that explain the main steps.
            # 13. Keep comments short and useful; do not comment every line.
            # 14. Every comment line must begin with #. Never write plain English prose as a bare code line.
            # 15. Only call .evalf() on SymPy expressions. If a value is already numeric, use it directly instead of numeric_value.evalf().
            """

        if self.dataset == "MATH":
            python_rules += "# For domain questions, use Interval / Union / intersection logic explicitly.\n"
            python_rules += "# For polynomial remainder questions, define the variable symbols before using div().\n"
            python_rules += "# For geometry, do not assign coordinates or use analytic coordinate geometry unless the problem explicitly gives coordinates.\n"
            python_rules += "# Never use hidden coordinates from a diagram or drawing code.\n"
            python_rules += "# Prefer synthetic geometric relationships and stated facts over invented coordinate setups.\n"
        elif self.dataset == "GSM":
            python_rules += "# For GSM word problems, prefer direct arithmetic with named variables before symbolic solving.\n"
            python_rules += "# Use solve() only when a short equation is genuinely needed.\n"
            python_rules += "# For time/rate or distance problems, prefer direct rate arithmetic over unrelated symbolic systems.\n"
            python_rules += "# Keep arithmetic exact when needed; prefer Rational(...) over unnecessary decimal literals.\n"
            python_rules += "# The final printed line must be the derived numeric answer.\n"
            python_rules += "# Do not copy prompt text, instructions, or prose into the code.\n"
        elif self.dataset in {"AQUA", "MMLU"} and self._current_options() and question_requests_closest_option(question):
            python_rules += "# This is a closest/nearest/estimate multiple-choice question.\n"
            python_rules += "# Compute the exact target quantity asked in the question before comparing options.\n"
            python_rules += "# Evaluate every option expression numerically and choose the smallest absolute difference.\n"
            python_rules += "# Do not round intermediate quantities unless the problem text explicitly instructs that rounding.\n"
            python_rules += "# Do not invent a per-term rounding rule from the options.\n"
            python_rules += "# Print the computed target quantity and the numeric value of each option before printing the final answer letter.\n"

        test_prompt = (
            f"Question: {question}\n"
            f"Modules used till now: [{mods}]\n"
            f"{response}\n\n"
            f"{python_rules}\n"
            "Code:\n"
        )

        full_prompt = demo_prompt + "\n\n" + test_prompt
        return test_prompt, full_prompt


    def build_prompt_for_kr_pg(self):
        
        question_text = self.get_question_text()
        
        #metadata = self.get_metadata()
        
        response = self.cache["response"] if "response" in self.cache else ""

        # build the prompt
        if self.dataset == "GSM":
            demo_prompt = build_gsm_program_demo_prompt().strip()
        elif self.dataset in {"AQUA", "MMLU"}:
            demo_prompt = build_option_program_demo_prompt(self._option_letters_for_dataset()).strip()
        else:
            demo_prompt = prompt_kr_pg.prompt.strip()  # WARNING: this is the prompt for kr_pg_sg
        
        python_instructions = (
            "Write executable Python only. The first line must be exactly 'from sympy import *'. "
            "Print the final answer and only helpful intermediate values. "
            "Every comment line must begin with #, for systems solved with SymPy use solve(..., dict=True) before symbol-key access, "
            "and never call .evalf() on plain Python numbers."
        )
        if self.dataset in {"AQUA", "MMLU"} and self._current_options() and question_requests_closest_option(question_text):
            python_instructions += (
                " This is a closest/estimate multiple-choice question, so compute the exact target quantity first, "
                "evaluate every option numerically, choose the smallest absolute difference, and do not round "
                "intermediate quantities unless the problem explicitly instructs that rounding."
            )
        if self._geometry_forbids_coordinate_methods(question_text):
            python_instructions += (
                " This is a geometry problem without explicit coordinates, so do not assign coordinates, use Point objects, "
                "or use analytic coordinate geometry; rely only on stated geometric relationships."
            )
        if response != "":
            test_prompt = f"Question: {question_text}\n\n{response}\n\nInstructions: {python_instructions}\n\nPython code:\n"
        else:
            test_prompt = f"Question: {question_text}\n\nInstructions: {python_instructions}\n\nPython code:\n"
        
        full_prompt = demo_prompt + "\n\n" + test_prompt # full prompt
        return test_prompt, full_prompt


    def build_prompt_for_kr_pg_sg (self):
        
        question_text = self.get_question_text()
        #metadata = self.get_metadata()
        response = self.cache["response"] if "response" in self.cache else ""

        # build the prompt
        if self.dataset == "AQUA":
            demo_prompt = build_option_solution_demo_prompt(self._option_letters_for_dataset()).strip()
        elif self.dataset == "GSM":
            demo_prompt = build_gsm_solution_demo_prompt().strip()
        elif self.dataset == "MMLU":
            demo_prompt = build_option_solution_demo_prompt(self._option_letters_for_dataset()).strip()
        else:   
            demo_prompt = prompt_kr_pg_sg.prompt.strip() # WARNING: this is the prompt for kr_sg
        
        return self._build_solution_prompt(demo_prompt, question_text, response)


    def build_prompt_for_sg_cot(self):

        question_text = self.get_question_text()
        #metadata = self.get_metadata()
        response = self.cache["response"] if "response" in self.cache else ""

        # build the prompt
        demo_prompt = prompt_for_cot.prompt.strip()        # WARNING: this is the prompt for cot
        
        return self._build_solution_prompt(demo_prompt, question_text, response)

