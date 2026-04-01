import os
import sys
import json
import argparse
import random

# add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

from console_ui import (
    print_module_plan,
    print_module_result,
    print_problem_header,
    print_problem_summary,
    print_run_header,
    print_run_summary,
    print_warning,
)
from utilities import *
from model import solver
from reporting import generate_report, normalize_record

def parse_args():

    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/MATH', choices=['data/MATH', 'data/GSM-8K', 'data/AQUA', 'data/MMLU'])
    parser.add_argument('--data_file', type=str, default='no')
    parser.add_argument('--dataset', type=str, default='MATH', choices=['MATH', 'GSM', 'AQUA', 'MMLU'])
    parser.add_argument('--output_root', type=str, default='output_MATHSENSEI')
    parser.add_argument('--model', type=str, default='no', choices=['cot', 'pot','planner','kr_sg','kr_walpha_sg','kr_pg_sg','walpha_sg','pg_sg', 'walpha_pg_sg','walpha_sg','pg_walpha_sg','bing_sg','bing_pg_sg','pg_bing_sg','bing_walpha_sg','walpha_bing_sg','bing_pg_walpha_sg','sg'])
    parser.add_argument('--label', type=str, default='MATHSENSEI_outfile')
    parser.add_argument('--task_name', type=str, default='math', choices=["math", "gsm", "aqua", "mmlu"])
    parser.add_argument('--test_split', type=str, default='test', 
                        choices=['train', 'val', 'test', 'minitrain', 'minival', 'minitest'])

    parser.add_argument('--test_number', type=int, default=None)    # Set to number of example in dataset
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--python_model', type=str, default='no', choices=['code_llama7b_python', 'code_llama13b_python','code_davinci002','code_llama34b','wizardcoder_34B','code_llama34b_pythonV1','gemini'])
    parser.add_argument('--extra_python_libraries', type=str, default='no')
    parser.add_argument('--knowledge_model', type=str, default='no', choices=['text_davinci_002','text_davinci_003','llama2_13b','llama2_7b'])
    parser.add_argument('--bing_model', type=str, default='no', choices=['text_davinci_002','text_davinci_003','llama2_13b','llama2_7b'])
    parser.add_argument('--sg_model', type=str, default='no', choices=['text_davinci_002','text_davinci_003','llama2_13b','llama2_7b','gemini'])
    parser.add_argument('--wolfram_model', type=str, default='no', choices=['gemini','text_davinci_002','text_davinci_003','llama2_13b','llama2_7b'])

    
    # module prediction
    parser.add_argument('--modules', nargs='+', default=None, help='default modules')
    parser.add_argument('--policy_engine', type=str, default="gpt4o", help='engine for module prediction')
    parser.add_argument('--policy_temperature', type=float, default=0, help='temperature for module prediction')
    parser.add_argument('--policy_max_tokens', type=int, default=1000, help='max tokens for module prediction')
    
    # program generation
    parser.add_argument('--pg_engine', type=str, default="gpt4o", help='engine for program generation')
    parser.add_argument('--pg_temperature', type=float, default=0.5, help='temperature for program generation')
    parser.add_argument('--pg_max_tokens', type=int, default=1500, help='max tokens for program generation')
    
    # knowledge retrieval
    parser.add_argument('--kr_engine', type=str, default="gpt4o", help='engine for knowledge retrieval')
    parser.add_argument('--kr_temperature', type=float, default=0.5, help='temperature for knowledge retrieval')
    parser.add_argument('--kr_max_tokens', type=int, default=1000, help='max tokens for knowledge retrieval')
   
    # query generator
    parser.add_argument('--qg_engine', type=str, default="gpt4o", help='engine for query generator')
    parser.add_argument('--qg_temperature', type=float, default=0., help='temperature for query generator')
    parser.add_argument('--qg_max_tokens', type=int, default=1000, help='max tokens for query generator')
    parser.add_argument('--qg_patience', type=int, default=5, help='patience for query generator')

    # solution_generator
    parser.add_argument('--sg_engine', type=str, default="gpt4o", help='engine for solution generator')
    parser.add_argument('--sg_temperature', type=float, default=0.5, help='temperature for solution generator')
    parser.add_argument('--sg_max_tokens', type=int, default=3000, help='max tokens for solution generator')
    parser.add_argument('--sg_patience', type=int, default=2, help='patience for solution generator')
    
    
    parser.add_argument('--current_index', type=int, default=0, help='index to start')  # Index in dataset to start from
    parser.add_argument('--refine',type=str,default='no',help="Whether to include the refinement of code using error message")
    parser.add_argument('--error_mode',type=str,default='no',help="Finishing the examples which had error (None or '') output in 1st run")
    parser.add_argument('--bing_count', type=int, default=5, help='no of results returned for bing')

    

    # debug
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

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
    result_root = f"{args.output_root}/{args.task_name}"
    os.makedirs(result_root, exist_ok=True)
    cache_file = f"{result_root}/{args.label}_{args.test_split}_cache.json"
    cache_jsonl = f"{result_root}/{args.label}_{args.test_split}_cache.jsonl"
    result_file = f"{result_root}/{args.label}_{args.test_split}.json"
    readable_jsonl = f"{result_root}/{args.label}_{args.test_split}_readable.jsonl"
    report_file = f"{result_root}/{args.label}_{args.test_split}_report.html"
    
    print_run_header(args, result_file, solver.test_number)
    
    # Running in error model 
    if args.error_mode != 'no':  
        
        with open(cache_jsonl,'r') as file :
            indices = []
            count = 0 
            for line in file:
                data = json.loads(line)
                example_code = (data['program'])
                output, error_message = safe_execute(example_code)
                if output!=None:
                       if data['program_executor:output'] != output :
                          indices.append(count)
                          count +=1  
    
        print("No of Indices",len(indices))
    
    error_mode_cache_jsonl_file = f"{result_root}/{args.label}_{args.test_split}_cache_error_mode_pg_sg.jsonl"
    
    
    if args.error_mode !='no':
        with open(cache_jsonl,'r') as infile:
             with open(error_mode_cache_jsonl_file,'a') as outfile:
                 count = 0
                 for line in infile:
                    
                    if count<args.current_index:
                        count=count+1
                        continue

                     
                    if count in indices:
                        data = json.loads(line)
                        pid = count  
                        solver.cache = data

                        if args.debug:
                            print("\n\n===================================\n")
                            print(f"# [Pid]: {pid}\n") # problem id
                        
                        solver.current_index+= 1                         # number of current results
                        example_code = (solver.cache['program'])
                        

                        modules = ["program_executor","solution_generator"]   # Set to setting of error_mode 
                        
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
                                modules = ["knowledge_retrieval","walpha","solution_generator"]

                            elif args.model == 'kr_pg_sg':
                                if args.refine == "no":
                                   modules = ["knowledge_retrieval","program_generator","program_executor","solution_generator","answer_generator"] 
                                else:
                                   modules = ["knowledge_retrieval","python_generator_refine_executor","solution_generator","answer_generator"] 

                            elif args.model == 'walpha_sg':
                                modules = ["walpha","solution_generator"]

                            elif args.model == 'planner':    
                                modules = solver.predict_modules()
                        '''
                        modules = [f"solver.{module}" for module in modules]

                            
                        # [2] Execute the modules 
                        if args.debug:
                            print(f"# [Modules]\n{modules}\n")
                        
                        solver.modules = solver.modules + modules
                        print("Cache:", solver.cache)

                        for module in modules:
                            input, output = eval(module)()     # eval the module and update the cache
                            if args.debug:
                                print(f"======== [Module]: {module} ========\n")
                                print(f"# [Input]\n{input}\n")
                                print(f"# [Output]\n{output}\n")
                                print(f"======== End module========\n")

                        try:
                            json.dump(solver.cache, outfile)
                            outfile.write('\n')   

                        except Exception as e:
                            print(e)
                            print(solver.cache)


                    
                    elif count not in indices:
                        data = json.loads(line)
                        outfile.write(json.dumps(data)+'\n')

                    count+=1     # Increment count 

                    
        sys.exit()    # Exit the program [no need to continue]

                                        
    # If error_mode is "no" (Default run)           
    complete_count = 0
    partial_count = 0
    needs_review_count = 0

    for pid in range(solver.current_index, solver.test_number):

 
        solver.cache = {"pid": pid}      # clear the cache
       

        if args.debug:
            print("\n\n===================================\n")
            print(f"# [Pid]: {pid}\n") # problem id
        
        solver.current_index+= 1                         # number of current results
        solver.cache["example"] = solver.examples[pid]   # get one example 
        
        if args.dataset == "AQUA":
            solver.cache["example"]["problem"] =  solver.cache["example"]['question'] + " Options:" +  str(solver.cache["example"]['options']) 
        
        if args.dataset == "GSM": # Convert to key problem
            solver.cache["example"]["problem"] =  solver.cache["example"]['question']

        if args.dataset == "MMLU": # Convert to key problem
            solver.cache["example"]["problem"] =  "\n" + solver.cache["example"]['Question'] + "\n" + 'Option A:' + solver.cache["example"]['Option A'] + "\n" + 'Option B:' + solver.cache["example"]['Option B'] + "\n" + 'Option C:' + solver.cache["example"]['Option C'] + "\n" + 'Option D:' + solver.cache["example"]['Option D']

        if args.dataset == "MATH":
            try: 
                typ = str(solver.cache["example"]["type"])
            except:
                typ = str(solver.cache["example"]['subject'])

            lvl = str(solver.cache["example"]["level"])
            solver.cache["response"] = f"\nMathematics Problem Type:{typ}\nLevel of Problem:{lvl}"
        
        problem_preview = (
            solver.cache["example"].get("problem")
            or solver.cache["example"].get("question")
            or solver.cache["example"].get("Question")
            or str(solver.cache["example"])
        )
        print_problem_header(pid, solver.test_number, problem_preview)
        
        if args.modules is not None:
            modules = args.modules
        else:
            if args.model == 'cot' or  args.model =='sg':
                modules = ["solution_generator"]

            elif args.model == 'pg_sg':
                if args.refine == "no":
                     modules = ["program_generator","program_executor","solution_generator"]

                else:
                     modules = ["python_generator_refine_executor","solution_generator"]    
            
                  
            elif args.model == 'bing_sg':
                modules = ["bing_search","solution_generator"]

            elif args.model == 'bing_pg_sg':
                modules = ["bing_search","program_generator","program_executor","solution_generator"]
            
            elif args.model == 'bing_walpha_sg':
                modules = ["bing_search",'wolfram_alpha_search',"solution_generator"]
            
            elif args.model == 'bing_pg_walpha_sg':
                modules = ["bing_search","program_generator","program_executor",'wolfram_alpha_search',"solution_generator"]
            
            elif args.model == 'walpha_bing_sg' :
                 modules = ['wolfram_alpha_search',"bing_search","solution_generator"]


            elif args.model == 'pg_bing_sg':
                modules = ["program_generator","program_executor","bing_search","solution_generator"]

            elif args.model == 'kr_sg':
                modules = ["knowledge_retrieval","solution_generator"]

            elif args.model == 'kr_walpha_sg':
                modules = ["knowledge_retrieval",'wolfram_alpha_search',"solution_generator"]
            
            elif args.model == 'walpha_pg_sg':
                 modules = ['wolfram_alpha_search',"program_generator","program_executor","solution_generator"]

            elif args.model == 'pg_walpha_sg':
                 modules = ["program_generator","program_executor",'wolfram_alpha_search',"solution_generator"]     


            elif args.model == 'kr_pg_sg':
                if args.refine == "no":
                   modules = ["knowledge_retrieval","program_generator","program_executor","solution_generator"] 
                else:
                   modules = ["knowledge_retrieval","python_generator_refine_executor","solution_generator"] 
            
            elif args.model == 'walpha_sg':
                            ##modules = ['wolfram_alpha_search',"wolfram_alpha_search","solution_generator"]
                            modules = ['wolfram_alpha_search',"solution_generator"]

            elif  args.model == 'planner':
              modules = solver.predict_modules()
        
        module_names = list(modules)
        print_module_plan(module_names)
        solver.modules = solver.modules + module_names
        module_calls = [f"solver.{module}" for module in module_names]
        
       
            
        # [2] Execute the modules 
        if args.debug:
            print(f"# [Modules]\n{module_calls}\n")
            
        for module_name, module in zip(module_names, module_calls):
            input, output = eval(module)()     # eval the module and update the cache
            print_module_result(module_name, input, output)
            if args.debug:
                print(f"======== [Module]: {module} ========\n")
                print(f"# [Input]\n{input}\n")
                print(f"# [Output]\n{output}\n")
                print(f"======== End module========\n")
        

        with open(cache_file, "a") as f:
            try:
                f.write(json.dumps(solver.cache, indent=2, separators=(',', ': ')) + "\n")
            except Exception as e:
                print(e)
                print(solver.cache)
        
        with open(cache_jsonl, "a") as f:
            try:
                json.dump(solver.cache, f)
                f.write('\n')
            except Exception as e:
                print_warning(str(e))

        normalized_record = normalize_record(solver.cache)
        with open(readable_jsonl, "a", encoding="utf-8") as f:
            try:
                json.dump(normalized_record, f, ensure_ascii=False)
                f.write("\n")
            except Exception as e:
                print_warning(str(e))

        if normalized_record["status"] == "complete":
            complete_count += 1
        elif normalized_record["status"] == "needs-review":
            needs_review_count += 1
        else:
            partial_count += 1

        print_problem_summary(solver.cache)

    try:
        generate_report(
            readable_jsonl,
            output_path=report_file,
            title=f"MathSensei ({args.dataset})",
        )
    except Exception as e:
        print_warning(f"Failed to generate HTML report: {e}")

    print_run_summary(
        solver.test_number,
        complete_count + partial_count,
        needs_review_count,
        report_file,
    )


        
