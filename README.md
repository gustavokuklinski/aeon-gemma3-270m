# Gemma-3-270m-it

Allows the AI to "see" and interpret images. By linking visual data with language, it can describe scenes, answer visual queries, and identify objects within pictures.

Original model: [google/gemma-3-270m-it](https://huggingface.co/google/gemma-3-270m-it)

---
## Plugin Config
```yaml
aeon_plugin:
  plugin_name: gemma-3-270m-it
  type: text-text
  model_path: ./model/
  command: /gemma3
  parameters: <PROMPT>
  desc: Used to embed Gemma3 results to RAG
```


