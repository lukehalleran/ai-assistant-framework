#!/usr/bin/env bash
set -euo pipefail

echo '== Repo scan: topic categorization references (py only) =='
grep -RIn --include='*.py' --exclude-dir=.git --exclude-dir=.venv --exclude-dir=data --exclude-dir=chroma_db --exclude-dir=embedded_parquet \
    -E '(TopicManager|topic_manager|class[[:space:]]+Topic|topic[^\n]{0,40}categor|categor[^\n]{0,40}topic|topic[[:space:]]*[:=])' core utils processing knowledge || true

echo
echo '== Repo scan: model/LLM usage references (py only) =='
grep -RIn --include='*.py' --exclude-dir=.git --exclude-dir=.venv --exclude-dir=data --exclude-dir=chroma_db --exclude-dir=embedded_parquet \
    -E '(model[[:space:]]*[:=]|OPENAI_?MODEL|MODEL_NAME|CHAT_MODEL|gpt-|openai|anthropic|groq|llm|client\.|ChatOpenAI|OpenAI)' core utils processing knowledge models || true

echo
echo '== Repo scan: files likely of interest =='
ls -1 core || true
ls -1 models || true
ls -1 utils || true
ls -1 knowledge || true

echo
echo '== Repo scan: model defaults/switch =='
grep -RIn --include='*.py' --exclude-dir=.git --exclude-dir=.venv \
    -E '(switch_model\(|active_model_name|load_openai_model\(|api_models\[|gpt-4|gpt-3\.5)' models core || true

echo
echo '== Preview: utils/topic_manager.py (lines 1-120) =='
nl -ba utils/topic_manager.py 2>/dev/null | sed -n '1,120p' || true
echo
echo '== Preview: utils/topic_manager.py (lines 121-260) =='
nl -ba utils/topic_manager.py 2>/dev/null | sed -n '121,260p' || true

echo
echo '== Preview: processing/topic_manager.py (first 240 lines) =='
nl -ba processing/topic_manager.py 2>/dev/null | sed -n '1,240p' || true

echo
echo '== Preview: knowledge/topic_manager.py (first 60 lines) =='
nl -ba knowledge/topic_manager.py 2>/dev/null | sed -n '1,60p' || true
