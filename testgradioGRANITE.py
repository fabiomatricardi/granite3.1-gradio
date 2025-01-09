import gradio as gr
gr.load_chat("http://localhost:8080/v1/", model="bartowski_granite-3.1-2b-instruct-GGUF", token='note-required').launch()
