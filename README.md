# Azure BatchAI workaround for TF_CONFIG environment variable

Currently there is an [issue](https://github.com/Azure/BatchAI/issues/15) with the `TF_CONFIG` environment variable provided by Azure Batch AI.

This shows how to modify the provided variable such that distributed training with estimators works. Predominately this code:

```python
def remap_tfconfig(is_master):
  tf_config = json.loads(os.environ['TF_CONFIG'])
  master_worker = tf_config['cluster']['worker'][0]
  tf_config['cluster']['worker'] = tf_config['cluster']['worker'][1:]
  tf_config['cluster']['chief'] = [master_worker]
  if is_master:
    tf_config['task']['type'] = 'chief'
    tf_config['task']['index'] = 0
  elif tf_config['task']['type'] == 'worker':
    tf_config['task']['index'] -= 1
  
  os.environ['TF_CONFIG'] = json.dumps(tf_config)
```

Run this from the CLI like:

```bash
JOB_ID=$(date "+%Y-%m-%d-%H-%M-%S")
az batchai job create \
  --resource-group batch \
  --workspace BatchWorkspace \
  --cluster dsvm \
  --experiment mnist \
  --name "mnist-$JOB_ID" \
  --config-file config.json
```