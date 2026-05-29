"""
GPU 推理速度测试
用法：python test_gpu_speed.py
"""
import cv2
import time
import torch
from paddleocr import TextDetection


def get_gpu_info():
    """获取 GPU 信息"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"GPU: {gpu_name}")
        print(f"显存: {gpu_memory:.1f} GB")
    else:
        print("未检测到 CUDA GPU")


def benchmark(model, img_path, warmup=5, rounds=50):
    """
    测试推理速度
    warmup: 预热轮数
    rounds: 正式测试轮数
    """
    # 预热
    print(f"\n预热 {warmup} 次...")
    for _ in range(warmup):
        _ = model.predict(img_path)

    # 正式测试
    print(f"开始测试 ({rounds} 轮)...")
    times = []
    for i in range(rounds):
        start = time.perf_counter()
        _ = model.predict(img_path)
        end = time.perf_counter()
        times.append(end - start)
        if (i + 1) % 10 == 0:
            print(f"  完成 {i+1}/{rounds}")

    # 统计结果
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    fps = 1.0 / avg_time

    print(f"\n{'='*40}")
    print(f"测试结果 ({rounds} 轮)")
    print(f"{'='*40}")
    print(f"平均耗时: {avg_time*1000:.2f} ms")
    print(f"最快耗时: {min_time*1000:.2f} ms")
    print(f"最慢耗时: {max_time*1000:.2f} ms")
    print(f"帧率 (FPS): {fps:.2f}")
    print(f"{'='*40}")

    return avg_time, fps


def main():
    # GPU 信息
    get_gpu_info()

    # 初始化模型
    print("\n加载 TextDetection 模型...")
    model = TextDetection()

    # 测试图片
    img_path = "/media/ddc/新加卷/hys/hysnew2/boatdetect5/output/frame_000750.jpg"

    # 速度测试
    avg_time, fps = benchmark(model, img_path, warmup=5, rounds=50)

    # 单次推理并保存结果
    print("\n执行单次推理并保存结果...")
    output = model.predict(img_path)
    img = cv2.imread(img_path)

    for res in output:
        boxes = res['dt_polys']
        for box in boxes:
            box = box.astype(int)
            cv2.polylines(
                img,
                [box],
                isClosed=True,
                color=(0, 255, 0),
                thickness=2
            )

    # 保存结果
    import os
    os.makedirs("./output", exist_ok=True)
    cv2.imwrite("./output/demo.jpg", img)
    print("demo图已保存到 ./output/demo.jpg")


if __name__ == '__main__':
    main()
