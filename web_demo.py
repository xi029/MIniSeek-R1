"""用 Streamlit 与本地模型或兼容 Chat Completions 的接口对话。"""

from pathlib import Path
from threading import Thread

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from model.model_miniseek import MiniMindConfig, MiniMindForCausalLM


@st.cache_resource
def load_local(path, hidden_size, num_layers, use_moe, modified_time):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if path.is_file():
        config = MiniMindConfig(hidden_size=hidden_size, num_hidden_layers=num_layers, use_moe=use_moe)
        model = MiniMindForCausalLM(config)
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("model", state))
        tokenizer = AutoTokenizer.from_pretrained(Path(__file__).parent / "model")
    else:
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    return model.eval().to(device), tokenizer


def stream_local(model, tokenizer, messages, max_tokens, temperature, stage):
    if stage == "pretrain":
        prompt = tokenizer.bos_token + messages[-1]["content"]
    else:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=stage in ("reason", "grpo"))
    batch = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    errors = []

    def generate():
        try:
            with torch.inference_mode():
                model.generate(**batch, max_new_tokens=max_tokens, temperature=temperature,
                               do_sample=True, top_p=0.85, streamer=streamer,
                               pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        except Exception as error:
            errors.append(error)
            streamer.end()

    thread = Thread(target=generate, daemon=True)
    thread.start()
    yield from streamer
    thread.join()
    if errors:
        raise errors[0]


def stream_api(url, api_key, model_id, messages, max_tokens, temperature):
    from openai import OpenAI

    with OpenAI(base_url=url, api_key=api_key or "none") as client:
        chunks = client.chat.completions.create(model=model_id, messages=messages, stream=True,
                                                max_tokens=max_tokens, temperature=temperature)
        for chunk in chunks:
            if chunk.choices:
                yield chunk.choices[0].delta.content or ""


st.set_page_config(page_title="MiniSeek-R1", page_icon="💬")
st.title("MiniSeek-R1")
source = st.sidebar.radio("模型来源", ["本地模型", "API"])
if source == "本地模型":
    path = Path(st.sidebar.text_input("权重文件或 Transformers 目录", "out/grpo_512_moe.pth")).expanduser()
    with st.sidebar.expander("模型配置"):
        width = st.number_input("隐藏维度", min_value=16, value=512, step=16)
        layers = st.number_input("层数", min_value=1, value=8)
        moe = st.checkbox("使用 MoE", value=True)
else:
    url = st.sidebar.text_input("API 地址", "http://127.0.0.1:8000/v1")
    model_id = st.sidebar.text_input("模型 ID", "minimind")
    key = st.sidebar.text_input("API Key", type="password")
history = st.sidebar.slider("携带历史对话轮数", 0, 6, 0)
max_tokens = st.sidebar.slider("最多生成 token 数", 1, 8192, 256)
temperature = st.sidebar.slider("采样温度", 0.1, 1.5, 0.85)
messages = st.session_state.setdefault("messages", [])
if st.sidebar.button("清空对话"):
    messages.clear()

for i, message in enumerate(messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and st.button("从此轮删除", key=f"delete_{i}"):
            del messages[i-1:]
            st.rerun()

if source == "本地模型" and not (path.is_file() or (path / "config.json").is_file()):
    st.info("请在侧栏填写已训练的 .pth 文件路径或导出的模型目录。训练步骤见仓库 README。")
    st.stop()

if prompt := st.chat_input("给模型发一条消息"):
    with st.chat_message("user"):
        st.write(prompt)
    user_message = {"role": "user", "content": prompt}
    context = (messages[-2 * history:] if history else []) + [user_message]
    with st.chat_message("assistant"):
        try:
            if source == "本地模型":
                model, tokenizer = load_local(path.resolve(), width, layers, moe, path.stat().st_mtime_ns)
                stream = stream_local(model, tokenizer, context, max_tokens, temperature, path.stem.split("_")[0])
            else:
                stream = stream_api(url, key, model_id, context, max_tokens, temperature)
            answer = st.write_stream(stream)
        except Exception as error:
            st.error(f"生成失败：{error}")
            st.stop()
    messages.extend([user_message, {"role": "assistant", "content": answer}])
