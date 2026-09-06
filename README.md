# Ripdude

Fully structured agent harness based on Kubernetes, aimed to load in local or cloud environment.

# Kustomize

This is a Kustomize version of Ripdude deployment.

# Installation

### Prerequisites: fully operating kubernetes cluster with **nvidia-ctk** runtime configured as default and single gpu device plugin installed.

https://github.com/NVIDIA/nvidia-container-toolkit

```
sudo nvidia-ctk runtime configure --runtime=containerd --set-as-default
```

https://github.com/nvidia/k8s-device-plugin

```
helm upgrade --install nvdp nvdp/nvidia-device-plugin \
   --namespace nvidia-device-plugin with helm \
   --set nodeSelector=null \
   --set mps.enabled=false \
   --set affinity.nodeAffinity=null
```

Verify allocated gpu: \
`kubectl get node -o jsonpath='{.items[\*].status.allocatable}'`

### Create a Cognee container image with FalkorDB adapter package injection in *k8s/base/cognee/falkor-adapter/*

Cognee does not work with FalkorDB out-of-the-box. The official image does not contain *cognee_community_hybrid_adapter_falkor* package. The provided Dockerfile creates an image extension with the package injected.

```
./k8s/tools/falkor-adapter/BuildContainer.sh
```

### Set values in config files:

- k8s/config/config.yaml
- k8s/config/namespace.yaml
- k8s/base/ripdude/ripdude-cfg.yaml - environment variables
- k8s/base/inference/llama-cfg.yaml and inference-depl.yaml (limits)
- k8s/base/playwright/playwright-cfg.yaml
- k8s/base/cognee/cognee-cfg.yaml - mainly *LLM_MODEL* and *LLM_ENDPOINT*

### Files and directory structures

Following are requeted by PV and PVCs:

```
cluster/ 
   harness/ (HARNESS_ROOT)
   workspace/ (WORKSPACE_ROOT)
```

```
data/
   cognee/
      data/
      fastembed/
      system/
   falkordb/
   llama/
   redis/
```

### Agents and project context setup

The HARNESS_ROOT is where the context for project and agents are setup. \
Each agent requires a directory with an agent.md file. \
Directory name must correspond to agent name (AGENT_NAME). \
Ripdude's Chainlit ui creates agents selector based on their directories.

A single agent directory is meant to be used by single agent container. \
To create additional agent, deploy another ripdude pod with its own env vars setup.

`agent.md` and `context.md` files are injected to the system messages. Describe the harness identity here.

```
harness/
   context.md
   agent1_name/
      agent.md
   agent2_name/
      agent.md
   (...)
```

### Default ports exposed

* :30000 - ripdude ui - the main ui for agent
* :30100 - falkordb ui
* :30200 - cognee ui
* :30300 - cognee backend exposed for ui, not directly usable
* :30400 - redis ui
* :30500 - prometheus ui
* :30600 - grafana ui
* :30700 - llama-benchy ui
* :30800 - playwright vnc
* :30900 - mcp-inspector

### Check out the additional tools for testing `./k8s/tools/`

If you need Prometheus, install it separately with Helm chart.
Then you can deploy additional prometheus metrics under `./k8s/tools/prometheus`

You can also apply these containers separately on-demand:

- llama-benchy - useful for testing max context vs vram for llama.cpp
- MCP Inspector - test MCP tools from ui interface