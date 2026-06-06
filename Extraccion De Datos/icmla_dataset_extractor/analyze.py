import re
with open('sample_ieee.html', 'r', encoding='utf-8') as f:
    html = f.read()

matches = re.findall(r'metadata\s*=\s*({.*?});', html, re.DOTALL)
for i, m in enumerate(matches):
    print(f"Match {i} First 200 chars:")
    print(m[:200])
