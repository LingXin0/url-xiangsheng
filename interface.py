import gradio as gr
from langchain_helper import *

with gr.Blocks() as demo:
    url = gr.Textbox()
    chatbot = gr.Chatbot(height=700)
    submit_btn = gr.Button("Submit")


    def generate_conversation(url):
        xs: XianSheng = get_xs(url)
        print(f"返回最终结果：{xs}")
        chat_history = []

        def parse_line(line: Line):
            if line is None:
                return ""
            return f"{line.character}: {line.content}"

        for i in range(0, len(xs.script), 2):
            line1 = xs.script[i]
            line2 = xs.script[i + 1] if (i + 1) < len(xs.script) else None
            chat_history.append([parse_line(line1), parse_line(line2)])
        return chat_history


    submit_btn.click(fn=generate_conversation, inputs=url, outputs=chatbot)

if __name__ == "__main__":
    demo.launch(share=True)
