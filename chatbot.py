# set CMAKE_ARGS=-DLLAMA_CUBLAS=on
# set FORCE_CMAKE=1
# pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
# python 3.10.6
# Copy files from: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\extras\visual_studio_integration\MSBuildExtensions
# to C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Microsoft\VC\v170\BuildCustomizations
# https://medium.com/@piyushbatra1999/installing-llama-cpp-python-with-nvidia-gpu-acceleration-on-windows-a-short-guide-0dfac475002d
# give a python code to balance achievements points based on the gamification configured levels

# pip freeze > requirements.txt - Cria o arquivo com as dependências do projeto


import time
from llama_cpp import Llama

llm = Llama(model_path="amethyst-13b-mistral.Q6_K.gguf", n_gpu_layers=2048, n_ctx=512, n_batch=126)


def generate_text(
    prompt,
    max_tokens=2048,
    temperature=0.1,
    top_p=0.5,
    echo=False,
    stop=["#"],
):
    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        echo=echo,
        stop=stop,
        repeat_penalty=1.2
    )

    output_text = output["choices"][0]["text"].strip()

    return output_text


def generate_prompt_from_template(question):
    chat_prompt_template = f"""<|im_start|>system
You are a helpful chatbot.<|im_end|>
<|im_start|>user
{question}<|im_end|>"""

    return chat_prompt_template


def generate_prompt_from_template_amethyst(question):
    chat_prompt_template = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{question}

### Response:"""

    return chat_prompt_template


def chat():
    question = input('Whats you question dumbass? ')

    if question == 'exit':
        exit()
    else:
        start_time = time.time()

        # generate_prompt_from_template(question),
        answear = generate_text(
            generate_prompt_from_template_amethyst(question),
            max_tokens=2048
        )

        print(answear)
        print("--- %s seconds ---" % (time.time() - start_time))


while True:
    chat()

