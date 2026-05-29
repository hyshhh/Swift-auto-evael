"""
修复 vLLM 和 transformers 的 Qwen3_5Config 类型冲突
运行一次即可：python patch_vllm_qwen35.py
"""
import os
import sys


def find_vllm_context_file():
    """找到 vllm 的 context.py 文件"""
    try:
        import vllm
        vllm_path = os.path.dirname(vllm.__file__)
        context_file = os.path.join(vllm_path, 'multimodal', 'processing', 'context.py')
        if os.path.exists(context_file):
            return context_file
    except ImportError:
        pass
    return None


def patch_context_file(context_file):
    """修改 context.py，让 get_hf_config 接受两种类型的 config"""
    with open(context_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查是否已经打过补丁
    if 'ALLOWED_CONFIG_TYPES' in content:
        print('已经打过补丁，跳过')
        return

    # 找到 get_hf_config 方法中的类型检查
    old_code = '''    def get_hf_config(self, config_type):
        config = self.get_model_config().hf_config
        if not isinstance(config, config_type):
            raise TypeError(
                f"Invalid type of HuggingFace config. "
                f"Expected type: {config_type}, but found type: {type(config)}")
        return config'''

    new_code = '''    # ms-swift patch: 允许 transformers 和 vllm 的 config 类型兼容
    ALLOWED_CONFIG_TYPES = set()

    def get_hf_config(self, config_type):
        config = self.get_model_config().hf_config
        ALLOWED_CONFIG_TYPES.add(config_type)
        # 如果类型匹配，直接返回
        if isinstance(config, config_type):
            return config
        # 如果是同名的不同模块的 config（如 transformers vs vllm），也允许
        config_class_name = type(config).__name__
        expected_class_name = config_type.__name__
        if config_class_name == expected_class_name:
            return config
        # 否则报错
        raise TypeError(
            f"Invalid type of HuggingFace config. "
            f"Expected type: {config_type}, but found type: {type(config)}")'''

    if old_code in content:
        content = content.replace(old_code, new_code)
        with open(context_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'成功修改: {context_file}')
    else:
        print(f'未找到目标代码，可能版本不同')
        print(f'请手动修改: {context_file}')
        print(f'找到 get_hf_config 方法，将 isinstance 检查改为兼容同名类型')


def main():
    context_file = find_vllm_context_file()
    if context_file is None:
        print('未找到 vLLM，请先安装: pip install vllm')
        sys.exit(1)

    print(f'找到 vLLM context 文件: {context_file}')
    patch_context_file(context_file)
    print('\n修复完成！现在可以运行 swift infer 了')


if __name__ == '__main__':
    main()
