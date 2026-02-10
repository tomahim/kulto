# n8n + Ollama

This folder runs [n8n](https://n8n.io/) and [Ollama](https://ollama.com/) with Docker Compose so n8n can call local LLMs (e.g. Llama 3.2) via Ollama.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose

## Start the services

From the **n8n** folder:

```bash
cd n8n
docker compose up -d
```

- **n8n**: http://localhost:5679  
- **Ollama**: http://localhost:11434  

Use **localhost** or **127.0.0.1** in the browser (not `0.0.0.0`, which can hang or fail).

Pull the Llama 3.2 model (once) so the workflow can use it:

```bash
docker exec ollama pull llama3.2
```

## Stop the services

From the **n8n** folder:

```bash
cd n8n
docker compose down
```

To remove volumes as well (n8n data and Ollama models):

```bash
docker compose down -v
```

## Workflow: Ollama llama3.2

- **File**: `workflows/ollama.workflow.json`
- **Behavior**: Sends a prompt to Ollama’s `llama3.2` model and returns the reply.

### Using the workflow

1. Open n8n at http://localhost:5679.
2. Import the workflow: **Workflows** → **Import from File** → choose `workflows/ollama.workflow.json`.
3. Run the workflow with **Execute Workflow**.
4. To set your prompt: open the **Manual Trigger** node, use **Add input data** and add JSON like:
   ```json
   { "prompt": "Your question or instruction here" }
   ```
5. The **Ollama Generate** node calls `http://ollama:11434/api/generate`. The model reply is in the **response** field of that node’s output.

If you don’t add input data, the workflow uses the default prompt: `"Hello, respond briefly."`

## Data persistence

- **n8n**: workflows and config are stored in the `n8n_data` volume.
- **Ollama**: pulled models are stored in the `ollama_data` volume.

Both volumes persist across `docker compose down` unless you use `docker compose down -v`.
