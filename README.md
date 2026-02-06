## Download and Run OpenSearch

### Using Docker Compose (Recommended)

```bash
docker-compose up -d
```

To stop OpenSearch:

```bash
docker-compose down
```

To stop and remove data:

```bash
docker-compose down -v
```

### Using Docker directly

```bash
docker pull opensearchproject/opensearch:latest && docker run -it -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" -e "DISABLE_SECURITY_PLUGIN=true" -e "cluster.routing.allocation.disk.threshold_enabled=false" -e "cluster.blocks.create_index=false" opensearchproject/opensearch:latest
```


Better model options (if you have more resources):
- google/flan-t5-large - Good balance of quality and speed
- mistralai/Mistral-7B-Instruct-v0.1 - Higher quality but slower