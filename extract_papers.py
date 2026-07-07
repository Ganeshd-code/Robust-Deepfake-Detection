import fitz
import json

def extract_text(path, max_pages=30):
    doc = fitz.open(path)
    text = ''
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        text += page.get_text()
    doc.close()
    return text

papers = {
    'p1': 'Research paper/A Comprehensive Survey from theReliability Perspective.pdf',
    'p2': 'Research paper/A Novel Deep Learning Approach for Deepfake Image Detection.pdf',
    'p3': 'Research paper/Deep learning model for deep fake face recognition and detection.pdf',
    'p4': 'Research paper/Deepfake_Generation_and_Detection_Case_Study_and_Challenges.pdf',
    'p5': 'Research paper/DeepfakeBench A Comprehensive Benchmark of deepfake detection.pdf',
    'p6': 'Research paper/Detecting_Deepfake_Images_Using_Deep_Lea.pdf',
    'p7': 'Research paper/Face Recognition from Video using Deep Learning Models.pdf',
    'p8': 'Research paper/jcp-02-00007.pdf',
    'p9': 'Research paper/NeurIPS-2024-df40-toward-next-generation-deepfake-detection-Paper-Datasets_and_Benchmarks_Track.pdf',
    'p10': 'Research paper/S_M_Jaybhaye_itmconf_icdsac2023_03005.pdf',
    'p11': 'Research paper/WIREs Data Min   Knowl - 2023 - Heidari - Deepfake detection using deep learning methods  A systematic and comprehensive.pdf',
    'p12': 'Research paper/Zhao_Multi-Attentional_Deepfake_Detection_CVPR_2021_paper.pdf',
}

all_texts = {}
for key, path in papers.items():
    text = extract_text(path, 30)
    all_texts[key] = text
    print(f'Extracted {key}: {len(text)} chars')

# Save to file for inspection
with open('paper_texts.json', 'w', encoding='utf-8') as f:
    json.dump(all_texts, f, ensure_ascii=False, indent=2)

print("Done. Saved to paper_texts.json")
