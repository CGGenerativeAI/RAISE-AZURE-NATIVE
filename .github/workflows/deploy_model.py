import time
from azure.ai.ml import MLClient
from azure.identity import AzureCliCredential
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment
from azure.identity import DefaultAzureCredential

# Load config from config.txt
def load_config(file_path):
    config = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                name, value = line.strip().split("=")
                config[name.strip()] = value.strip()
    except Exception as e:
        print(f"Error loading configuration: {e}")
    return config

# Load configuration from the config file
config = load_config("config.txt")
credential = DefaultAzureCredential()

# Authenticate with Azure
# credential = AzureCliCredential()

# Initialize ML client for your workspace
ml_client = MLClient(
    credential=credential,
    subscription_id=config["subscription_id"],
    resource_group_name=config["resource_group_name"],
    workspace_name=config["workspace_name"]
)

# Initialize ML client for the registry
registry_ml_client = MLClient(
    credential=credential, 
    registry_name=config["registry_name"]
)

# Define the model from the registry
# model_name = "Phi-3.5-mini-instruct"
model_name = config["model_name"]
foundation_model = registry_ml_client.models.get(model_name, version=config["version"])
print(
    "\n\nUsing model name: {0}, version: {1}, id: {2} for inferencing".format(
        foundation_model.name, foundation_model.version, foundation_model.id
    )
)

# Create online endpoint with a unique name using timestamp
timestamp = int(time.time())
online_endpoint_name = "SS-phi-generation-v1-" + str(timestamp)

# Store the endpoint name for future reference
with open("endpoint_name.txt", "w") as f:
    f.write(online_endpoint_name)

# Create the endpoint
endpoint = ManagedOnlineEndpoint(
    name=online_endpoint_name,
    description="Online endpoint for " + foundation_model.name + ", for text-generation task",
    auth_mode="key",
)
ml_client.begin_create_or_update(endpoint).wait()

# Create the deployment
demo_deployment = ManagedOnlineDeployment(
    name="demo",
    endpoint_name=online_endpoint_name,
    model=foundation_model.id,
    # instance_type="Standard_NC24ads_A100_v4", #config
    instance_type = config["instance_type"],
    instance_count=1,
)
ml_client.online_deployments.begin_create_or_update(demo_deployment).wait()

# Set traffic to 100% for the demo deployment
endpoint.traffic = {"demo": 100}
ml_client.begin_create_or_update(endpoint).result()

print("Model deployed successfully.")
