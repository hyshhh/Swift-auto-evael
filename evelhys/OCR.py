"""
弦号识别 Pipeline
功能：通过大模型识别图片中船体侧面的弦号位置，输出可视化结果

使用方法：
    python OCR.py --image_dir /media/ddc/新加卷/hys/hysnew3/OCR-peddle-v2.0/crops
    python OCR.py --image_dir /media/ddc/新加卷/hys/hysnew3/OCR-peddle-v2.0/crops --output_dir ./results
"""

import os
import re
import json
import time
import base64
import argparse
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


# ==================== 配置 ====================
# API 模式（远程调用）
API_KEY = "sk-dllnqafnfsrvpxtttcepjrbyljrjausbjblbjbodtpvlmmri"
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen3.5-VL-2B-Instruct"

# 本地模式（vLLM 本地服务）
LOCAL_MODE = True  # 设为 True 使用本地模型
LOCAL_API_URL = "http://localhost:7890/v1/chat/completions"
LOCAL_MODEL_NAME = "Qwen/Qwen3-VL-4B-AWQ"
LOCAL_API_KEY = "abc123"

# 根据模式选择配置
if LOCAL_MODE:
    API_URL = LOCAL_API_URL
    MODEL_NAME = LOCAL_MODEL_NAME
    API_KEY = LOCAL_API_KEY

# 提示词
PROMPT = """你是一个专业的弦号识别专家。

任务：识别图片中船体侧面的弦号位置，输出其中心点的归一化坐标。

要求：
1. 弦号位于船体侧面（船舷外侧），为数字/字母组合标识
2. 优先关注船体侧面区域，忽略背景文字、水印、标牌等干扰
3. 即使模糊、遮挡、低分辨率，也必须尽量估计其中心位置
4. 每张图像中必定存在且仅存在一个弦号

输出格式（严格遵守）：
{"x": float, "y": float}

其中 x 和 y 是归一化坐标（0-1之间），表示弦号中心点在图片中的位置。
x=0 表示图片最左边，x=1 表示图片最右边
y=0 表示图片最上边，y=1 表示图片最下边

只输出 JSON，不要输出其他任何内容。"""


def parse_args():
    parser = argparse.ArgumentParser(description='弦号识别 Pipeline')
    parser.add_argument('--image_dir', type=str, required=True, help='图片目录路径')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录路径')
    parser.add_argument('--api_key', type=str, default=API_KEY, help='API Key')
    parser.add_argument('--api_url', type=str, default=API_URL, help='API URL')
    parser.add_argument('--model', type=str, default=MODEL_NAME, help='模型名称')
    parser.add_argument('--box_size', type=int, default=100, help='标注框大小')
    parser.add_argument('--line_width', type=int, default=3, help='标注线宽度')
    return parser.parse_args()


def encode_image(image_path):
    """将图片编码为 base64"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def call_vlm_api(args, image_path):
    """调用 VLM API 识别弦号位置"""
    # 编码图片
    img_base64 = encode_image(image_path)

    # 构建请求
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": PROMPT
                }
            ]
        }
    ]

    payload = {
        "model": args.model,
        "messages": messages,
        "max_tokens": 256,
        "temperature": 0,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }

    # 调用 API
    response = requests.post(
        args.api_url,
        json=payload,
        headers=headers,
        timeout=60,
    )
    response.raise_for_status()

    result = response.json()
    return result['choices'][0]['message']['content']


def parse_coord(response_text):
    """解析模型输出的坐标"""
    # 尝试从 JSON 格式提取
    try:
        match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            x = float(data.get('x', 0.5))
            y = float(data.get('y', 0.5))
            # 确保在 0-1 范围内
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            return x, y
    except (json.JSONDecodeError, KeyError, ValueError):
        pass

    # 尝试从文本中提取坐标
    match = re.search(r'"x"\s*:\s*([\d.]+).*?"y"\s*:\s*([\d.]+)', response_text, re.DOTALL)
    if match:
        try:
            x = float(match.group(1))
            y = float(match.group(2))
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            return x, y
        except ValueError:
            pass

    # 默认返回图片中心
    return 0.5, 0.5


def draw_result(image_path, x, y, box_size=100, line_width=3):
    """在图片上绘制识别结果"""
    # 打开图片
    img = Image.open(image_path)
    width, height = img.size

    # 转换归一化坐标为像素坐标
    pixel_x = int(x * width)
    pixel_y = int(y * height)

    # 创建绘图对象
    draw = ImageDraw.Draw(img)

    # 绘制中心点
    draw.ellipse(
        [pixel_x - 5, pixel_y - 5, pixel_x + 5, pixel_y + 5],
        fill='red',
        outline='red',
    )

    # 绘制矩形框
    half_size = box_size // 2
    draw.rectangle(
        [pixel_x - half_size, pixel_y - half_size,
         pixel_x + half_size, pixel_y + half_size],
        outline='red',
        width=line_width,
    )

    # 绘制交叉线
    draw.line(
        [pixel_x - half_size, pixel_y, pixel_x + half_size, pixel_y],
        fill='red',
        width=line_width,
    )
    draw.line(
        [pixel_x, pixel_y - half_size, pixel_x, pixel_y + half_size],
        fill='red',
        width=line_width,
    )

    # 添加坐标文字
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    coord_text = f"({x:.3f}, {y:.3f})"
    draw.text(
        [pixel_x + half_size + 10, pixel_y - 10],
        coord_text,
        fill='red',
        font=font,
    )

    return img


def process_single_image(args, image_path, output_dir):
    """处理单张图片，返回结果和耗时"""
    # 计时开始
    start_time = time.time()

    # 调用 API
    response_text = call_vlm_api(args, image_path)

    # 解析坐标
    x, y = parse_coord(response_text)

    # 绘制结果
    result_img = draw_result(image_path, x, y, args.box_size, args.line_width)

    # 保存结果
    image_name = Path(image_path).stem
    output_path = os.path.join(output_dir, f"{image_name}_result.jpg")
    result_img.save(output_path, quality=95)

    # 计时结束
    end_time = time.time()
    latency = end_time - start_time

    return {
        'image': image_path,
        'x': x,
        'y': y,
        'output': output_path,
        'raw_response': response_text,
        'latency': latency,
        'fps': 1.0 / latency if latency > 0 else 0,
    }


def main():
    args = parse_args()

    # 创建输出目录
    output_dir = args.output_dir or os.path.join(args.image_dir, 'results')
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有图片
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = [
        os.path.join(args.image_dir, f)
        for f in os.listdir(args.image_dir)
        if Path(f).suffix.lower() in image_extensions
    ]

    print('=' * 60)
    print('弦号识别 Pipeline')
    print('=' * 60)
    print(f'图片目录: {args.image_dir}')
    print(f'输出目录: {output_dir}')
    print(f'图片数量: {len(image_files)}')
    print(f'模型: {args.model}')
    print('=' * 60)

    # 处理每张图片
    results = []
    for image_path in tqdm(image_files, desc='识别中'):
        try:
            result = process_single_image(args, image_path, output_dir)
            results.append(result)

            # 打印结果（含单帧速度）
            print(f'\n✓ {Path(image_path).name}')
            print(f'  坐标: ({result["x"]:.3f}, {result["y"]:.3f})')
            print(f'  耗时: {result["latency"]:.2f}s | FPS: {result["fps"]:.2f}')
            print(f'  输出: {result["output"]}')

        except Exception as e:
            print(f'\n✗ {Path(image_path).name}: {e}')
            continue

    # 保存汇总结果
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print('\n' + '=' * 60)
    print('识别完成')
    print('=' * 60)
    print(f'成功: {len(results)}/{len(image_files)}')
    print(f'结果保存到: {output_dir}')
    print(f'汇总文件: {summary_path}')

    # 速度统计
    if results:
        latencies = [r['latency'] for r in results]
        avg_latency = sum(latencies) / len(latencies)
        avg_fps = 1.0 / avg_latency if avg_latency > 0 else 0
        min_latency = min(latencies)
        max_latency = max(latencies)

        print(f'\n【速度统计】')
        print(f'平均耗时: {avg_latency:.2f}s')
        print(f'平均 FPS: {avg_fps:.2f} fps')
        print(f'最快: {min_latency:.2f}s ({1.0/min_latency:.2f} fps)')
        print(f'最慢: {max_latency:.2f}s ({1.0/max_latency:.2f} fps)')


if __name__ == '__main__':
    main()
