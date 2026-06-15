"""
Vision AI - 多模态图片理解应用
基于 Qwen-VL 模型，支持图片问答、内容分析、描述生成
"""

import gradio as gr
import torch
from PIL import Image
import os

# 模型相关
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

# 系统提示词
SYSTEM_PROMPT = """你是一个专业的图片理解助手，能够准确分析图片内容并回答用户问题。
请用中文回答，语言简洁专业。如果图片内容无法回答问题，请如实说明。"""

def load_model():
    """加载模型"""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    
    print(f"正在加载模型: {MODEL_NAME}")
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    
    print("模型加载完成!")
    return model, processor

def chat_with_image(image, question, model, processor, history=[]):
    """处理图片问答"""
    if image is None:
        return "请先上传一张图片", history
    
    if not question:
        return "", history
    
    # 构建消息
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }
    ]
    
    # 处理输入
    from qwen_vl_utils import process_vision_info
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info([{"role": "user", "content": [messages[1]["content"]]}])
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # 生成回答
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False
        )
    
    # 解码
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    answer = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    history.append((question, answer))
    return "", history

def analyze_image(image, model, processor):
    """分析图片内容"""
    if image is None:
        return "请先上传一张图片"
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "请详细分析这张图片，包括：1. 图片主体是什么 2. 图片场景/背景 3. 图片风格/色调 4. 图片中的文字或数据（如有）5. 其他值得注意的细节"}
            ]
        }
    ]
    
    from qwen_vl_utils import process_vision_info
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info([{"role": "user", "content": [messages[1]["content"]]}])
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=768,
            do_sample=False
        )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    result = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    return result

def describe_image(image, model, processor):
    """生成图片描述"""
    if image is None:
        return "请先上传一张图片"
    
    messages = [
        {"role": "system", "content": "你是一个专业的图片描述生成器，请生成简洁准确的中文图片描述。"},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "请用一句话描述这张图片的主要内容"}
            ]
        }
    ]
    
    from qwen_vl_utils import process_vision_info
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info([{"role": "user", "content": [messages[1]["content"]]}])
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False
        )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    result = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    return result

# 全局模型
model = None
processor = None

def initialize():
    """初始化模型"""
    global model, processor
    try:
        model, processor = load_model()
        return True
    except Exception as e:
        print(f"模型加载失败: {e}")
        return False

# 自定义CSS
CSS = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}
"""

def create_interface():
    """创建Gradio界面"""
    with gr.Blocks(title="Vision AI - 多模态图片理解", css=CSS) as demo:
        gr.Markdown("""
        # 🤖 Vision AI - 多模态图片理解助手
        
        基于 **Qwen2-VL** 模型，支持图片问答、内容分析、描述生成
        
        ---
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="📷 上传图片",
                    type="pil",
                    height=400
                )
                
                with gr.Row():
                    analyze_btn = gr.Button("🔍 分析图片", variant="primary")
                    describe_btn = gr.Button("📝 图片描述", variant="secondary")
                
                example_images = gr.Examples(
                    examples=[
                        ["examples/demo1.jpg", "这张图片里有什么？"],
                        ["examples/demo2.jpg", "描述这张图片的场景"],
                    ],
                    inputs=[image_input],
                    label="📌 示例"
                )
            
            with gr.Column(scale=1):
                with gr.Tab("💬 问答模式"):
                    question_input = gr.Textbox(
                        label="❓ 提问",
                        placeholder="对图片提问，例如：这张图片里有什么？",
                        lines=2
                    )
                    ask_btn = gr.Button("🚀 提问", variant="primary")
                    answer_output = gr.Chatbot(label="对话历史", height=300)
                    
                    ask_btn.click(
                        fn=chat_with_image,
                        inputs=[image_input, question_input, gr.State(model), gr.State(processor), answer_output],
                        outputs=[question_input, answer_output]
                    )
                    question_input.submit(
                        fn=chat_with_image,
                        inputs=[image_input, question_input, gr.State(model), gr.State(processor), answer_output],
                        outputs=[question_input, answer_output]
                    )
                
                with gr.Tab("📊 分析结果"):
                    analyze_output = gr.Textbox(label="详细分析", lines=15, show_label=True)
                    analyze_btn.click(
                        fn=analyze_image,
                        inputs=[image_input, gr.State(model), gr.State(processor)],
                        outputs=analyze_output
                    )
                
                with gr.Tab("📝 图片描述"):
                    describe_output = gr.Textbox(label="图片描述", lines=5, show_label=True)
                    describe_btn.click(
                        fn=describe_image,
                        inputs=[image_input, gr.State(model), gr.State(processor)],
                        outputs=describe_output
                    )
        
        gr.Markdown("""
        ---
        ### 💡 使用说明
        
        1. **上传图片**：点击左侧上传按钮或拖拽图片
        2. **提问问答**：在问答模式输入问题，获取图片相关信息
        3. **一键分析**：点击"分析图片"获取详细分析
        4. **生成描述**：点击"图片描述"获取简短描述
        
        ---
        """)
        
        gr.Markdown("""
        <div style='text-align: center; color: #666;'>
        <p>Powered by <a href='https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct'>Qwen2-VL</a> | 
        Built with <a href='https://gradio.app/'>Gradio</a></p>
        </div>
        """)
    
    return demo

def main():
    """主函数"""
    print("=" * 50)
    print("Vision AI - 多模态图片理解应用")
    print("=" * 50)
    
    # 初始化模型
    if not initialize():
        print("警告: 模型加载失败，尝试使用Hugging Face推理API...")
    
    # 创建界面
    demo = create_interface()
    
    # 启动服务
    print("\n启动服务中...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )

if __name__ == "__main__":
    main()
