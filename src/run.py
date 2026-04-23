import os
import sys
import json
import argparse
import random
import time
import copy
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

# add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

from core.app_support import (
    GLOBAL_MODEL_CHOICES,
    apply_global_model_overrides,
    print_module_plan,
    print_module_result,
    print_problem_header,
    print_problem_summary,
    print_run_header,
    print_run_summary,
    print_warning,
)
from core.pipeline_shared import (
    build_initial_question_response,
    normalize_module_sequence,
    rebuild_cached_response,
    resolve_modules_for_model,
)
from utilities import *
from model import solver
from presentation.reporting import generate_report, normalize_record
from presentation.benchmarking import append_run_benchmark, build_run_benchmark_row, update_aggregate_benchmark


MODEL_ALIASES = {
    "wa_sg": "walpha_sg",
    "wa_pg_sg": "walpha_pg_sg",
    "pg_wa_sg": "pg_walpha_sg",
    "kr_wa_sg": "kr_walpha_sg",
    "kr_pg_wa_sg": "kr_pg_walpha_sg",
}

MODEL_CHOICES = [
    "cot",
    "pot",
    "planner",
    "kr_sg",
    "kr_walpha_sg",
    "kr_pg_sg",
    "kr_pg_walpha_sg",
    "walpha_sg",
    "pg_sg",
    "walpha_pg_sg",
    "pg_walpha_sg",
    "bing_sg",
    "bing_pg_sg",
    "pg_bing_sg",
    "bing_walpha_sg",
    "walpha_bing_sg",
    "bing_pg_walpha_sg",
    "sg",
]

def dataset_display_name(dataset):
    if dataset == "ALL":
        return "Mixed"
    return dataset


def default_task_name(dataset):
    if dataset == "ALL":
        return "mixed"
    return str(dataset).lower()


def normalize_model_name(model_name):
    return MODEL_ALIASES.get(model_name, model_name)


def data_root_note(data_root):
    if data_root:
        return f"--data_root is compatibility-only; dataset file paths still come from environment configuration. Requested: {data_root}"
    return "Dataset file paths come from environment configuration."


def _default_chat_backend_label():
    return globals().get("MODEL_NAME") or "default chat backend"


def _gemini_backend_label():
    return globals().get("GEMINI_MODEL_NAME") or "gemini"


def _configured_or_default_backend(configured_value):
    if configured_value == "gemini":
        return _gemini_backend_label()
    if configured_value not in (None, "", "no"):
        return str(configured_value)
    return _default_chat_backend_label()


def _module_backend_label(solver_instance, module_name):
    if module_name == "knowledge_retrieval":
        return _configured_or_default_backend(getattr(solver_instance, "knowledge_model", "no"))
    if module_name == "bing_search":
        return f"Bing API + {_configured_or_default_backend(getattr(solver_instance, 'bing_model', 'no'))}"
    if module_name == "wolfram_alpha_search":
        return f"Wolfram Alpha + {_configured_or_default_backend(getattr(solver_instance, 'wolfram_model', 'no'))}"
    if module_name in {"program_generator", "python_generator_refine_executor"}:
        return _configured_or_default_backend(getattr(solver_instance, "python_model", "no"))
    if module_name == "program_executor":
        return "Local Python"
    if module_name == "solution_generator":
        return _configured_or_default_backend(getattr(solver_instance, "sg_model", "no"))
    if module_name == "answer_generator":
        return _configured_or_default_backend(getattr(solver_instance, "sg_model", "no"))
    return _default_chat_backend_label()


def call_solver_module(solver_instance, module_name):
    module_fn = getattr(solver_instance, module_name, None)
    backend_label = _module_backend_label(solver_instance, module_name)
    started_at = time.perf_counter()

    if module_fn is None or not callable(module_fn):
        message = f"Unknown module '{module_name}'"
        solver_instance.cache.setdefault("module_errors", []).append(message)
        elapsed_seconds = round(time.perf_counter() - started_at, 4)
        solver_instance.cache.setdefault("module_timings_seconds", {})[module_name] = elapsed_seconds
        solver_instance.cache.setdefault("module_backends", {})[module_name] = backend_label
        return None, f"Skipped: {message}", message, elapsed_seconds, backend_label

    try:
        module_input, module_output = module_fn()
        error = None
    except Exception as exc:
        message = f"{module_name} failed: {exc}"
        solver_instance.cache.setdefault("module_errors", []).append(message)
        module_input, module_output, error = None, f"Execution failed: {exc}", message

    elapsed_seconds = round(time.perf_counter() - started_at, 4)
    solver_instance.cache.setdefault("module_timings_seconds", {})[module_name] = elapsed_seconds
    solver_instance.cache.setdefault("module_backends", {})[module_name] = backend_label
    return module_input, module_output, error, elapsed_seconds, backend_label


def resolve_run_modules(args, solver_instance):
    # Keep pipeline selection in one place so normal runs and future recovery
    # modes do not drift apart as more model variants are added.
    try:
        return resolve_modules_for_model(
            args.model,
            refine=args.refine,
            custom_modules=args.modules,
            planner_callable=solver_instance.predict_modules,
        )
    except ValueError as exc:
        print_warning(str(exc))
        return ["solution_generator"]


def execute_modules(solver_instance, module_names, debug=False, emit_logs=True):
    module_names = normalize_module_sequence(module_names)
    executed_solution_generator = False
    for index, module_name in enumerate(module_names):
        module_input, module_output, error, elapsed_seconds, backend_label = call_solver_module(solver_instance, module_name)
        if module_name == "solution_generator" and error is None:
            executed_solution_generator = True
        if emit_logs:
            print_module_result(module_name, module_input, module_output)
        if error and emit_logs:
            print_warning(error)
        if debug and emit_logs:
            print(f"======== [Module]: solver.{module_name} ========\n")
            print(f"# [Input]\n{module_input}\n")
            print(f"# [Output]\n{module_output}\n")
            print(f"# [Backend]\n{backend_label}\n")
            print(f"# [Elapsed Seconds]\n{elapsed_seconds}\n")
            print(f"======== End module========\n")

        if error and "solution_generator" not in module_names[index + 1:] and not executed_solution_generator:
            fallback_input, fallback_output, fallback_error, fallback_elapsed_seconds, fallback_backend_label = call_solver_module(solver_instance, "solution_generator")
            print_module_result("solution_generator", fallback_input, fallback_output)
            if fallback_error:
                print_warning(fallback_error)
            if debug:
                print(f"======== [Module]: solver.solution_generator ========\n")
                print(f"# [Input]\n{fallback_input}\n")
                print(f"# [Output]\n{fallback_output}\n")
                print(f"# [Backend]\n{fallback_backend_label}\n")
                print(f"# [Elapsed Seconds]\n{fallback_elapsed_seconds}\n")
                print(f"======== End module========\n")
            executed_solution_generator = fallback_error is None
            break


def prepare_example_for_dataset(example, dataset):
    example = dict(example)
    if dataset == "AQUA":
        example["problem"] = example['question'] + " Options:" + str(example['options'])
    elif dataset == "GSM":
        example["problem"] = example['question']
    elif dataset == "MMLU":
        example["problem"] = (
            "\n"
            + example['Question']
            + "\n"
            + 'Option A:' + example['Option A']
            + "\n"
            + 'Option B:' + example['Option B']
            + "\n"
            + 'Option C:' + example['Option C']
            + "\n"
            + 'Option D:' + example['Option D']
        )

    return example


def build_problem_preview(example):
    return (
        example.get("problem")
        or example.get("question")
        or example.get("Question")
        or str(example)
    )


def build_question_signature(example, dataset, pid=None):
    problem = (
        example.get("problem")
        or example.get("question")
        or example.get("Question")
        or ""
    )
    payload = "\n".join(
        [
            str(dataset or ""),
            str(pid if pid is not None else ""),
            str(problem),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def reset_run_artifacts(paths):
    for path in paths:
        if path and os.path.exists(path):
            os.remove(path)


def initialize_question_state(solver_instance, pid, example, requested_dataset):
    example_copy = prepare_example_for_dataset(copy.deepcopy(example), example.get("dataset", requested_dataset))
    current_dataset = example_copy.get("dataset", requested_dataset)

    solver_instance.cache = {"pid": pid}
    solver_instance.modules = []
    solver_instance.dependency_install_attempts = set()
    solver_instance.dataset = current_dataset
    solver_instance.cache["example"] = example_copy
    solver_instance.cache["dataset"] = current_dataset
    solver_instance.cache["requested_dataset"] = requested_dataset
    solver_instance.cache["question_signature"] = build_question_signature(example_copy, current_dataset, pid)

    initial_response = build_initial_question_response(example_copy, current_dataset)
    if initial_response:
        solver_instance.cache["response"] = initial_response

    return current_dataset


def clone_solver_for_problem(base_solver):
    cloned_solver = copy.copy(base_solver)
    cloned_solver.cache = {}
    cloned_solver.modules = []
    cloned_solver.dependency_install_attempts = set()
    return cloned_solver


def enrich_cache_with_final_answer(cache_row):
    normalized_record = normalize_record(cache_row)
    cache_row["answer"] = normalized_record.get("final_answer")
    cache_row["final_answer"] = normalized_record.get("final_answer")
    cache_row["final_answer_value"] = normalized_record.get("final_answer_value")
    cache_row["final_answer_option"] = normalized_record.get("final_answer_option")
    return normalized_record


def persist_problem_outputs(cache_row, normalized_record, cache_file, cache_jsonl, readable_jsonl):
    with open(cache_file, "a", encoding="utf-8") as f:
        try:
            f.write(json.dumps(cache_row, indent=2, separators=(',', ': '), ensure_ascii=False) + "\n")
        except Exception as e:
            print_warning(str(e))

    with open(cache_jsonl, "a", encoding="utf-8") as f:
        try:
            json.dump(cache_row, f, ensure_ascii=False)
            f.write('\n')
        except Exception as e:
            print_warning(str(e))

    with open(readable_jsonl, "a", encoding="utf-8") as f:
        try:
            json.dump(normalized_record, f, ensure_ascii=False)
            f.write("\n")
        except Exception as e:
            print_warning(str(e))


def run_problem_once(base_solver, args, pid, example, *, debug=False, emit_logs=True):
    problem_solver = clone_solver_for_problem(base_solver)
    initialize_question_state(problem_solver, pid, example, args.dataset)

    modules = resolve_run_modules(args, problem_solver)
    module_names = list(modules)
    problem_solver.modules = list(module_names)
    problem_solver.cache["modules"] = list(module_names)
    problem_solver.cache["run_model"] = args.model
    if emit_logs:
        print_module_plan(module_names)

    question_started_at = time.perf_counter()
    try:
        execute_modules(problem_solver, module_names, debug=debug, emit_logs=emit_logs)
    except Exception as exc:
        message = f"Problem {pid} failed: {exc}"
        problem_solver.cache.setdefault("module_errors", []).append(message)
        if emit_logs:
            print_warning(message)
    finally:
        problem_solver.cache["question_elapsed_seconds"] = round(time.perf_counter() - question_started_at, 4)

    normalized_record = enrich_cache_with_final_answer(problem_solver.cache)
    return {
        "pid": pid,
        "cache": problem_solver.cache,
        "normalized_record": normalized_record,
        "module_names": module_names,
    }


def iter_problem_batches(problem_ids, batch_size):
    ids = list(problem_ids)
    if not ids:
        return
    if batch_size is None or batch_size <= 0:
        batch_size = len(ids)
    for start in range(0, len(ids), batch_size):
        yield ids[start:start + batch_size]


def prepare_error_mode_state(solver_instance, cached_row, rerun_modules, requested_dataset):
    pid = cached_row.get("pid", solver_instance.current_index)
    example = cached_row.get("example") or {}
    current_dataset = initialize_question_state(solver_instance, pid, example, requested_dataset)

    preserved_modules = normalize_module_sequence(
        module_name
        for module_name in (cached_row.get("modules") or [])
        if module_name not in set(rerun_modules)
    )

    preserved_prefixes = {
        "knowledge_retrieval",
        "bing_search",
        "wolfram_alpha_search",
        "query_generator",
    }
    for key, value in cached_row.items():
        if any(key.startswith(f"{prefix}:") for prefix in preserved_prefixes):
            solver_instance.cache[key] = copy.deepcopy(value)

    for extra_key in [
        "query",
        "bing_query1",
        "bing_query1_output",
        "bing_query2",
        "bing_query2_output",
        "wolfram_query",
        "wolfram_output",
        "wolfram_error",
        "wolfram_trace",
        "run_model",
    ]:
        if extra_key in cached_row:
            solver_instance.cache[extra_key] = copy.deepcopy(cached_row[extra_key])

    if "program" in cached_row:
        solver_instance.cache["program"] = copy.deepcopy(cached_row["program"])

    rebuilt_response = rebuild_cached_response(cached_row, modules=preserved_modules)
    if rebuilt_response:
        solver_instance.cache["response"] = rebuilt_response
    else:
        solver_instance.cache.pop("response", None)

    solver_instance.cache["dataset"] = current_dataset
    solver_instance.cache["requested_dataset"] = requested_dataset
    return current_dataset


def parse_args():

    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/MATH', choices=['data/MATH', 'data/GSM-8K', 'data/AQUA', 'data/MMLU'])
    parser.add_argument('--data_file', type=str, default='no')
    parser.add_argument('--dataset', type=str, default='ALL', choices=['ALL', 'MATH', 'GSM', 'AQUA', 'MMLU'])
    parser.add_argument(
        '--mixed_dataset_strategy',
        type=str,
        default='balanced',
        choices=['balanced', 'proportional'],
        help='When --dataset ALL, sample datasets uniformly at random or keep size-proportional sampling.',
    )
    parser.add_argument('--output_root', type=str, default='output_MATHSENSEI')
    parser.add_argument('--model', type=str, default='pg_walpha_sg', choices=MODEL_CHOICES + sorted(MODEL_ALIASES.keys()))
    parser.add_argument('--label', type=str, default='MATHSENSEI_outfile')
    parser.add_argument('--task_name', type=str, default=None, choices=["math", "gsm", "aqua", "mmlu", "mixed"])
    parser.add_argument('--test_split', type=str, default='test', 
                        choices=['train', 'val', 'test', 'minitrain', 'minival', 'minitest'])

    parser.add_argument('--test_number', type=int, default=None)    # Set to number of example in dataset
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--global_model', type=str, default='no', choices=GLOBAL_MODEL_CHOICES)
    parser.add_argument('--python_model', type=str, default='no', choices=['code_llama7b_python', 'code_llama13b_python','code_davinci002','code_llama34b','wizardcoder_34B','code_llama34b_pythonV1','gemini'])
    parser.add_argument('--extra_python_libraries', type=str, default='no')
    parser.add_argument('--knowledge_model', type=str, default='no', choices=['text_davinci_002','text_davinci_003','llama2_13b','llama2_7b','gemini'])
    parser.add_argument('--bing_model', type=str, default='no', choices=['text_davinci_002','text_davinci_003','llama2_13b','llama2_7b','gemini'])
    parser.add_argument('--sg_model', type=str, default='no', choices=['text_davinci_002','text_davinci_003','llama2_13b','llama2_7b','gemini'])
    parser.add_argument('--wolfram_model', type=str, default='no', choices=['gemini','text_davinci_002','text_davinci_003','llama2_13b','llama2_7b'])

    
    # module prediction
    parser.add_argument('--modules', nargs='+', default=None, help='default modules')
    parser.add_argument('--policy_engine', type=str, default=os.getenv('POLICY_ENGINE') or os.getenv('DEFAULT_ENGINE') or "gpt-5-nano", help='engine for module prediction')
    parser.add_argument('--policy_temperature', type=float, default=0, help='temperature for module prediction')
    parser.add_argument('--policy_max_tokens', type=int, default=1000, help='max tokens for module prediction')
    
    # program generation
    parser.add_argument('--pg_engine', type=str, default=os.getenv('PG_ENGINE') or os.getenv('DEFAULT_ENGINE') or "gpt-5-nano", help='engine for program generation')
    parser.add_argument('--pg_temperature', type=float, default=0.5, help='temperature for program generation')
    parser.add_argument('--pg_max_tokens', type=int, default=1500, help='max tokens for program generation')
    
    # knowledge retrieval
    parser.add_argument('--kr_engine', type=str, default=os.getenv('KR_ENGINE') or os.getenv('DEFAULT_ENGINE') or "gpt-5-nano", help='engine for knowledge retrieval')
    parser.add_argument('--kr_temperature', type=float, default=0.5, help='temperature for knowledge retrieval')
    parser.add_argument('--kr_max_tokens', type=int, default=1000, help='max tokens for knowledge retrieval')
   
    # query generator
    parser.add_argument('--qg_engine', type=str, default=os.getenv('QG_ENGINE') or os.getenv('DEFAULT_ENGINE') or "gpt-5-nano", help='engine for query generator')
    parser.add_argument('--qg_temperature', type=float, default=0., help='temperature for query generator')
    parser.add_argument('--qg_max_tokens', type=int, default=1000, help='max tokens for query generator')
    parser.add_argument('--qg_patience', type=int, default=5, help='patience for query generator')

    # solution_generator
    parser.add_argument('--sg_engine', type=str, default=os.getenv('SG_ENGINE') or os.getenv('DEFAULT_ENGINE') or "gpt-5-nano", help='engine for solution generator')
    parser.add_argument('--sg_temperature', type=float, default=0.5, help='temperature for solution generator')
    parser.add_argument('--sg_max_tokens', type=int, default=3000, help='max tokens for solution generator')
    parser.add_argument('--sg_patience', type=int, default=2, help='patience for solution generator')
    
    
    parser.add_argument('--current_index', type=int, default=0, help='index to start')  # Index in dataset to start from
    parser.add_argument('--refine',type=str,default='no',help="Whether to include the refinement of code using error message")
    parser.add_argument('--error_mode',type=str,default='no',help="Finishing the examples which had error (None or '') output in 1st run")
    parser.add_argument('--bing_count', type=int, default=5, help='no of results returned for bing')
    parser.add_argument(
        '--parallel_workers',
        type=int,
        default=1,
        help='Number of problems to solve concurrently in normal mode. 1 keeps sequential execution.',
    )
    parser.add_argument(
        '--parallel_batch_size',
        type=int,
        default=0,
        help='Problems submitted per batch when parallel mode is enabled. 0 means use parallel_workers.',
    )

    

    # debug
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    if args.parallel_workers < 1:
        parser.error("--parallel_workers must be at least 1")
    if args.parallel_batch_size < 0:
        parser.error("--parallel_batch_size cannot be negative")
    args.model = normalize_model_name(args.model)
    args = apply_global_model_overrides(args)
    if args.task_name is None:
        args.task_name = default_task_name(args.dataset)

    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))
    return args


if __name__ == "__main__":

    args = parse_args()
    random.seed(args.seed)

    # Build the solver
    solver = solver(args)
    print(f"# Number of test examples: {solver.test_number}\n")
   
    # Get the result file
    result_root = os.path.abspath(os.path.join(args.output_root, args.task_name))
    os.makedirs(result_root, exist_ok=True)
    cache_file = f"{result_root}/{args.label}_{args.test_split}_cache.json"
    cache_jsonl = f"{result_root}/{args.label}_{args.test_split}_cache.jsonl"
    result_file = f"{result_root}/{args.label}_{args.test_split}.json"
    readable_jsonl = f"{result_root}/{args.label}_{args.test_split}_readable.jsonl"
    report_file = f"{result_root}/{args.label}_{args.test_split}_report.html"
    run_benchmark_file = f"{result_root}/{args.label}_{args.test_split}_benchmark.json"

    if args.error_mode == "no" and args.current_index == 0:
        reset_run_artifacts([
            cache_file,
            cache_jsonl,
            result_file,
            readable_jsonl,
            report_file,
            run_benchmark_file,
        ])
        print_warning(
            "Starting a fresh run at index 0, so previous cache/report artifacts for this label were cleared."
        )
    
    print_run_header(args, result_file, solver.test_number, data_note=data_root_note(args.data_root))
    
    # Running in error model 
    if args.error_mode != 'no':  
        indices = []
        with open(cache_jsonl,'r', encoding="utf-8") as file :
            indices = []
            count = 0 
            for line in file:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    print_warning(f"Skipping malformed cache row {count}: {exc}")
                    count += 1
                    continue
                example_code = data.get('program')
                if not example_code:
                    count += 1
                    continue
                output, error_message = safe_execute(example_code)
                if output is not None and data.get('program_executor:output') != output:
                    indices.append(count)
                count += 1  
    
        print("No of Indices",len(indices))
    
    error_mode_cache_jsonl_file = f"{result_root}/{args.label}_{args.test_split}_cache_error_mode_{args.model}.jsonl"
    
    
    if args.error_mode !='no':
        with open(cache_jsonl,'r', encoding="utf-8") as infile:
             with open(error_mode_cache_jsonl_file,'a', encoding="utf-8") as outfile:
                 count = 0
                 for line in infile:
                    
                    if count<args.current_index:
                        count=count+1
                        continue

                     
                    if count in indices:
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError as exc:
                            print_warning(f"Skipping malformed cache row {count}: {exc}")
                            count += 1
                            continue
                        pid = count  

                        if args.debug:
                            print("\n\n===================================\n")
                            print(f"# [Pid]: {pid}\n") # problem id
                        
                        solver.current_index+= 1                         # number of current results
                        rerun_modules = ["program_executor","solution_generator"]
                        prepare_error_mode_state(solver, data, rerun_modules, args.dataset)
                        

                        modules = rerun_modules   # Set to setting of error_mode 
                        
                        '''
                        if args.modules is not None:
                            modules = args.modules
                            print(f"# [Modules]\n{modules}\n")
                        else:
                            if args.model == 'cot':
                                modules = ["solution_generator"]

                            elif args.model == 'pot':
                                if args.refine == "no":
                                    modules = ["program_generator","program_executor","solution_generator","answer_generator"]

                                else:
                                    modules = ["python_generator_refine_executor","solution_generator","answer_generator"]    
                            
                            elif args.model == 'pg_sg':
                                modules = ["program_generator","program_executor","solution_generator"]

                            elif args.model == 'kr_sg':
                                modules = ["knowledge_retrieval","solution_generator","answer_generator"]

                            elif args.model == 'kr_walpha_sg':
                                modules = ["knowledge_retrieval","wolfram_alpha_search","solution_generator"]

                            elif args.model == 'kr_pg_sg':
                                if args.refine == "no":
                                   modules = ["knowledge_retrieval","program_generator","program_executor","solution_generator","answer_generator"] 
                                else:
                                   modules = ["knowledge_retrieval","python_generator_refine_executor","solution_generator","answer_generator"] 

                            elif args.model == 'walpha_sg':
                                modules = ["wolfram_alpha_search","solution_generator"]

                            elif args.model == 'planner':    
                                modules = solver.predict_modules()
                        '''
                        module_names = list(modules)
                        solver.modules = list(module_names)
                        solver.cache["modules"] = list(module_names)
                        solver.cache["run_model"] = args.model
                        # [2] Execute the modules 
                        if args.debug:
                            print(f"# [Modules]\n{module_names}\n")
                        
                        print("Cache:", solver.cache)
                        question_started_at = time.perf_counter()
                        execute_modules(solver, module_names, debug=args.debug)
                        solver.cache["question_elapsed_seconds"] = round(time.perf_counter() - question_started_at, 4)

                        try:
                            json.dump(solver.cache, outfile, ensure_ascii=False)
                            outfile.write('\n')   

                        except Exception as e:
                            print(e)
                            print(solver.cache)


                    
                    elif count not in indices:
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError as exc:
                            print_warning(f"Skipping malformed cache row {count}: {exc}")
                            count += 1
                            continue
                        outfile.write(json.dumps(data, ensure_ascii=False)+'\n')

                    count+=1     # Increment count 

                    
        sys.exit()    # Exit the program [no need to continue]

                                        
    # If error_mode is "no" (Default run)           
    complete_count = 0
    needs_review_count = 0
    run_records = []
    problem_ids = list(range(solver.current_index, solver.test_number))
    parallel_workers = min(args.parallel_workers, len(problem_ids)) if problem_ids else 1

    if parallel_workers > 1:
        batch_size = args.parallel_batch_size or parallel_workers
        print_warning(
            f"Parallel mode enabled: {parallel_workers} workers; batch size={batch_size}. "
            "Problem-level results remain isolated and are written in pid order."
        )
        if args.debug:
            print_warning("Parallel mode suppresses per-module debug output to avoid interleaved logs.")

        for batch_ids in iter_problem_batches(problem_ids, batch_size):
            batch_results = {}
            max_workers = min(parallel_workers, len(batch_ids))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_by_pid = {}
                for pid in batch_ids:
                    problem_preview = build_problem_preview(solver.examples[pid])
                    print_problem_header(pid, solver.test_number, problem_preview)
                    future = executor.submit(
                        run_problem_once,
                        solver,
                        args,
                        pid,
                        solver.examples[pid],
                        debug=False,
                        emit_logs=False,
                    )
                    future_by_pid[future] = pid

                for future in as_completed(future_by_pid):
                    pid = future_by_pid[future]
                    try:
                        batch_results[pid] = future.result()
                    except Exception as exc:
                        fallback_cache = {
                            "pid": pid,
                            "dataset": args.dataset,
                            "modules": [],
                            "run_model": args.model,
                            "module_errors": [f"Parallel worker crashed: {exc}"],
                        }
                        normalized_record = enrich_cache_with_final_answer(fallback_cache)
                        batch_results[pid] = {
                            "pid": pid,
                            "cache": fallback_cache,
                            "normalized_record": normalized_record,
                            "module_names": [],
                        }
                        print_warning(f"Problem {pid} crashed in parallel worker: {exc}")

            for pid in batch_ids:
                result = batch_results.get(pid)
                if result is None:
                    continue
                solver.current_index = max(solver.current_index, pid + 1)
                cache_row = result["cache"]
                normalized_record = result["normalized_record"]
                persist_problem_outputs(cache_row, normalized_record, cache_file, cache_jsonl, readable_jsonl)
                run_records.append(normalized_record)

                if normalized_record["status"] == "complete":
                    complete_count += 1
                else:
                    needs_review_count += 1

                print_module_plan(result.get("module_names") or [])
                print_problem_summary(cache_row)
    else:
        for pid in problem_ids:
            if args.debug:
                print("\n\n===================================\n")
                print(f"# [Pid]: {pid}\n")

            problem_preview = build_problem_preview(solver.examples[pid])
            print_problem_header(pid, solver.test_number, problem_preview)
            solver.current_index += 1
            result = run_problem_once(
                solver,
                args,
                pid,
                solver.examples[pid],
                debug=args.debug,
                emit_logs=True,
            )
            cache_row = result["cache"]
            normalized_record = result["normalized_record"]

            persist_problem_outputs(cache_row, normalized_record, cache_file, cache_jsonl, readable_jsonl)
            run_records.append(normalized_record)

            if normalized_record["status"] == "complete":
                complete_count += 1
            else:
                needs_review_count += 1

            print_problem_summary(cache_row)

    benchmark_row = build_run_benchmark_row(args, run_records)
    with open(run_benchmark_file, "w", encoding="utf-8") as handle:
        json.dump(benchmark_row, handle, indent=2, ensure_ascii=False)
    append_run_benchmark(args.output_root, benchmark_row)
    _, aggregate_row = update_aggregate_benchmark(args.output_root, benchmark_row)

    try:
        generate_report(
            cache_jsonl,
            output_path=report_file,
            title="MathSensei",
            run_benchmark=benchmark_row,
            aggregate_benchmark=aggregate_row,
        )
    except Exception as e:
        print_warning(f"Failed to generate HTML report at {os.path.normpath(report_file)}: {e}")

    print_run_summary(
        solver.test_number,
        complete_count,
        needs_review_count,
        result_root,
        report_path=report_file,
        benchmark=benchmark_row,
    )


        
