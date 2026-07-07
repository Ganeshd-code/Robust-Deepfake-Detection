import json

with open('paper_texts.json', 'r', encoding='utf-8') as f:
    texts = json.load(f)

# Print first 4000 chars of each paper
for key in ['p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12']:
    print(f'\n\n{"="*70}')
    print(f'PAPER {key}')
    print('='*70)
    print(texts[key][:4000])
    print('--- [END FIRST 4000] ---')
