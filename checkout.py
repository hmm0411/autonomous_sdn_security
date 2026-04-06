import os
import glob
import re

files = glob.glob('**/*.py', recursive=True)
files = [f for f in files if '.venv' not in f]
modules = {}
for f in files:
    name = os.path.basename(f).replace('.py', '')
    if name != '__init__':
        modules[name] = f

usage_counts = {m: 0 for m in modules}

for f in files:
    with open(f, 'r', encoding='utf-8') as file:
        try:
            content = file.read()
            for m in modules:
                # Bỏ qua chính bản thân file đó
                if m == os.path.basename(f).replace('.py', ''):
                    continue
                # Tìm xem tên module có xuất hiện trong file này dưới dạng import không (vd: from rl_engine.agent.agent import ...)
                if re.search(r'\b' + m + r'\b', content):
                    usage_counts[m] += 1
        except:
            pass

unused = [(m, modules[m]) for m in usage_counts if usage_counts[m] == 0]
for name, path in unused:
    print(f'Unused/Unimported: {path}')
