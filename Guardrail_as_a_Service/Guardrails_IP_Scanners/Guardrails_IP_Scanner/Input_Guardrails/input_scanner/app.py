from flask import Flask, request, jsonify
from config import ServiceConfig

config = ServiceConfig()
Config_data = config.config_data

class Build_Service:
    def __init__(self):
        self.default_kwargs = Config_data["Input"]

        self.model_input_default = {}
        for key, val in self.default_kwargs.items():
            for pkey, pval in val.items():
                self.model_input_default[f"{key}_{pkey}"] = pval

        from llm_guard.input_scanners import (PromptInjection, Toxicity, BanTopics, CustomGuard)
        self.input_scanners = [BanTopics(), PromptInjection(), Toxicity(), CustomGuard()]

    def _check_prompt(self, prompt, params):
        """
        Checks the incoming prompt against set scanners.
        """
        from llm_guard.evaluate import scan_prompt
        sanitized_prompt, results_valid, results_score = scan_prompt(self.input_scanners, prompt, **params)
        return results_valid, results_score

    def format_params(self, model_input, default_kwargs, model_input_default):
        params = {key: model_input.get(key, model_input_default[key]) for key in model_input_default}

        scanner_kwargs = {}
        for key in default_kwargs.keys():
            for key1, val1 in params.items():
                if key in key1:
                    if key in scanner_kwargs:
                        scanner_kwargs[key].update({key1.replace(f'{key}_', ""): val1})
                    else:
                        scanner_kwargs[key] = {key1.replace(f'{key}_', ""): val1}

        return scanner_kwargs

    def predict(self, model_input, params):
        """
        Generates prediction for the given input.
        """
        outputs = []

        kwargs = self.format_params(params, self.default_kwargs, self.model_input_default)
        for i in range(len(model_input["prompt"])):
            prompt = model_input["prompt"][i]
            results_valid, results_score = self._check_prompt(prompt, kwargs)

            if any(not result for result in results_valid.values()):
                violations = [key for key in results_valid.keys() if results_valid[key] is False]
                output = f"I cannot assist with your request as it violates {', '.join(violations)} policies. <input_scan_result> {results_score}"
                outputs.append(output)
            else:
                output = f"Your prompt is valid. <input_scan_result> {results_score}"
                outputs.append(output)
        
        return {"text": outputs}


service = Build_Service()

app = Flask(__name__)

@app.route('/input', methods=['POST'])

def input():
    data = request.json
    model_input = data.get("model_input", {})
    params = data.get("params", {})

    if not model_input:
        return jsonify({"error": "Invalid input"}), 400

    try:
        result = service.predict(model_input, params)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=8080,debug=True)