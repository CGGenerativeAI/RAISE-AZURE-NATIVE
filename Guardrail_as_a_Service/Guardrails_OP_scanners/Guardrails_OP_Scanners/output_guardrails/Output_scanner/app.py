from flask import Flask, request, jsonify
from config import ServiceConfig

config = ServiceConfig()
Config_data = config.config_data

class Build_Service: 

    def __init__(self):

        self.default_kwargs = Config_data["Output"]

        self.model_input_default = {}
        for key,val in self.default_kwargs.items():
            for pkey,pval in val.items():
                self.model_input_default[f"{key}_{pkey}"] = pval

        from llm_guard.output_scanners import BanTopics,Toxicity,Bias,Relevance,CustomGuard
        self.output_scanners  = [BanTopics(), Toxicity(), Bias(), Relevance(), CustomGuard()]       


    def _check_response(self, prompt, model_output, params):
        """
        This method checks the incoming prompt against set scanners.
        """
        from llm_guard.evaluate import scan_output
        sanitized_prompt, results_valid, results_score = scan_output(self.output_scanners, prompt, model_output, **params)
        return results_valid, results_score
    
    def format_params(self, model_input, default_kwargs, model_input_default):
        params = {key:model_input.get(key,model_input_default[key]) for key in model_input_default}

        scanner_kwargs = {}
        for key in default_kwargs.keys():
            for key1, val1 in params.items():
                if key in key1:
                    if key in scanner_kwargs:
                        scanner_kwargs[key].update({key1.replace(f'{key}_',""):val1})
                    else:
                        scanner_kwargs[key] = {key1.replace(f'{key}_',""):val1}

        return scanner_kwargs
    
    def predict(self,model_input,params):
        """
        This method generates prediction for the given input.
        """
        outputs = []

        kwargs = self.format_params(params, self.default_kwargs, self.model_input_default)

        for i in range(len(model_input["prompt"])):

            model_output = model_input["model_output"][i]
            prompt = model_input["prompt"][i]

            results_valid,results_score = self._check_response(prompt, model_output ,kwargs)

            if any(not result for result in results_valid.values()):
                print("if is satisfied")
                viaolations = [key for key in results_valid.keys() if results_valid[key] is False]
                output = f"Warning: We have found out the generated output is violating our {', '.join(viaolations)} Policies. <output_scan_result>: {results_score}"
                outputs.append(output)

            else:
                 outputs.append(f"The generated output is valid <output_scan_result>: {results_score}""")

        return {"text": outputs}
    
service = Build_Service()

app = Flask(__name__)

@app.route('/output', methods=['POST'])

def output():
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
    app.run( port=8080, debug=True)