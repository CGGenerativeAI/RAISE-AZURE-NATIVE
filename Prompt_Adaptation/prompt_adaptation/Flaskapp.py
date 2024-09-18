from flask import Flask, request, jsonify, redirect, url_for
import numpy as np
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer, util
import evaluate

app = Flask(__name__)

# HuggingFace credentials
HF_TOKEN = "" 
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

# Cost per 1k tokens for different models
cost_output = {
    "gpt-4": 0.03 / 1000,
    "gpt-3.5-turbo-0125": 0.0005 / 1000
}

def context_compressor(input_context, input_type, percent_reduction):
    input_text_content = ""

    if input_type == "Rewrite":
        input_text_content += f"Please re-write the following text in {percent_reduction}% fewer words without losing context: {input_context}"
    elif input_type == "Summarize":
        input_text_content += f"Summarize the following text in {percent_reduction}% fewer words, maintaining context: {input_context}"

    torch.random.manual_seed(0)

    model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct", device_map="auto", torch_dtype="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    generation_args = {"max_new_tokens": 500, "return_full_text": False, "temperature": 0.0, "do_sample": False}
    output = pipe([{"role": "user", "content": input_text_content}], **generation_args)
    result = output[0]['generated_text']

    return f"{input_type}: {result}"

def evaluate_context(original_text, compressed_text, model_name):
    metrics = {}
    model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

    original_embedding = model.encode(original_text, convert_to_tensor=True)
    compressed_embedding = model.encode(compressed_text, convert_to_tensor=True)

    cosine_similarity = util.dot_score(original_embedding, compressed_embedding).item()
    metrics["cosine_similarity"] = np.round((cosine_similarity * 100), 3)

    rouge = evaluate.load('rouge')
    result = rouge.compute(predictions=[compressed_text], references=[original_text])
    metrics["rouge_score"] = {key: np.round((val * 100), 2) for key, val in result.items()}

    token_count_dict = token_counts(original_text, compressed_text)
    metrics["total_token_count"] = token_count_dict

    token_cost_pt = token_cost(token_count_dict["input_tokens"], token_count_dict["compressed_tokens"], model_name)
    metrics["cost_reduction_percentage"] = token_cost_pt

    return metrics

def token_counts(in_prompt, compressed_prompt):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
    input_prompt_tokens = len(tokenizer.tokenize(in_prompt))
    compressed_prompt_tokens = len(tokenizer.tokenize(compressed_prompt))

    return {"input_tokens": input_prompt_tokens, "compressed_tokens": compressed_prompt_tokens}

def token_cost(original_tokens, compressed_tokens, model):
    original_cost = cost_output[model] * original_tokens
    compressed_cost = cost_output[model] * compressed_tokens
    cost_difference = original_cost - compressed_cost
    cost_reduction_percentage = (cost_difference / original_cost) * 100
    return f"{model}: {cost_reduction_percentage:.2f}%"

@app.route('/', methods=['GET', 'POST'])
@app.route('/compress', methods=['POST'])
def compress_text():
    if request.method == 'GET':
        return redirect(url_for('compress_text'))  # Redirects to compress_text POST logic
    
    data = request.json
    input_text = data.get('input_text', '')
    model = data.get('model', 'gpt-4')
    compression_method = data.get('compression_method', 'Rewrite')
    compression_percent = data.get('compression_percent', 50)

    compressed_text = context_compressor(input_context=input_text, input_type=compression_method, percent_reduction=compression_percent)
    evaluation = evaluate_context(original_text=input_text, compressed_text=compressed_text, model_name=model)

    response = {
        "compressed_text": compressed_text,
        "evaluation": evaluation
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


#######OLD-CODE#############
# from flask import Flask, request, jsonify
# import numpy as np
# import os
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from sentence_transformers import SentenceTransformer, util
# import evaluate

# app = Flask(__name__)

# # HuggingFace credentials
# HF_TOKEN = "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"
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

# # Cost per 1k tokens for different models
# cost_output = {
#     "gpt-4": 0.03 / 1000,
#     "gpt-3.5-turbo-0125": 0.0005 / 1000
# }

# def context_compressor(input_context, input_type, percent_reduction):
#     input_text_content = ""

#     if input_type == "Rewrite":
#         input_text_content += f"Please re-write the following text in {percent_reduction}% fewer words without losing context: {input_context}"
#     elif input_type == "Summarize":
#         input_text_content += f"Summarize the following text in {percent_reduction}% fewer words, maintaining context: {input_context}"

#     torch.random.manual_seed(0)

#     model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct", device_map="auto", torch_dtype="auto", trust_remote_code=True)
#     tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

#     pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
#     generation_args = {"max_new_tokens": 500, "return_full_text": False, "temperature": 0.0, "do_sample": False}
#     output = pipe([{"role": "user", "content": input_text_content}], **generation_args)
#     result = output[0]['generated_text']

#     return f"{input_type}: {result}"

# def evaluate_context(original_text, compressed_text, model_name):
#     metrics = {}
#     model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

#     original_embedding = model.encode(original_text, convert_to_tensor=True)
#     compressed_embedding = model.encode(compressed_text, convert_to_tensor=True)

#     cosine_similarity = util.dot_score(original_embedding, compressed_embedding).item()
#     metrics["cosine_similarity"] = np.round((cosine_similarity * 100), 3)

#     rouge = evaluate.load('rouge')
#     result = rouge.compute(predictions=[compressed_text], references=[original_text])
#     metrics["rouge_score"] = {key: np.round((val * 100), 2) for key, val in result.items()}

#     token_count_dict = token_counts(original_text, compressed_text)
#     metrics["total_token_count"] = token_count_dict

#     token_cost_pt = token_cost(token_count_dict["input_tokens"], token_count_dict["compressed_tokens"], model_name)
#     metrics["cost_reduction_percentage"] = token_cost_pt

#     return metrics

# def token_counts(in_prompt, compressed_prompt):
#     tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
#     input_prompt_tokens = len(tokenizer.tokenize(in_prompt))
#     compressed_prompt_tokens = len(tokenizer.tokenize(compressed_prompt))

#     return {"input_tokens": input_prompt_tokens, "compressed_tokens": compressed_prompt_tokens}

# def token_cost(original_tokens, compressed_tokens, model):
#     original_cost = cost_output[model] * original_tokens
#     compressed_cost = cost_output[model] * compressed_tokens
#     cost_difference = original_cost - compressed_cost
#     cost_reduction_percentage = (cost_difference / original_cost) * 100
#     return f"{model}: {cost_reduction_percentage:.2f}%"

# @app.route('/compress', methods=['POST'])
# def compress_text():
#     data = request.json
#     input_text = data.get('input_text', '')
#     model = data.get('model', 'gpt-4')
#     compression_method = data.get('compression_method', 'Rewrite')
#     compression_percent = data.get('compression_percent', 50)

#     compressed_text = context_compressor(input_context=input_text, input_type=compression_method, percent_reduction=compression_percent)
#     evaluation = evaluate_context(original_text=input_text, compressed_text=compressed_text, model_name=model)

#     response = {
#         "compressed_text": compressed_text,
#         "evaluation": evaluation
#     }

#     return jsonify(response)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)
