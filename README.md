# UniDocR: Universal Multimodal Document Retrieval

## Overview
UniDocR is a retrieval system designed to enhance document search capabilities by incorporating both text and image-based queries. It extends traditional text-based retrieval methods by integrating Vision-Language Models (VLMs) and Optical Character Recognition (OCR) to improve the understanding and retrieval of multimodal documents.

## Features
- **Multimodal Query Support**: Supports text-only, image-only, and combined text-image queries.
- **Vision-Language Integration**: Utilizes Qwen2-VL for enhanced document understanding.
- **Late-Interaction Mechanism**: Employs ColBERT and ColPali for efficient and precise retrieval.
- **Optimized for M2KR Benchmark**: Evaluated on Q2A, I2A, and IQ2A tasks.

## Technical Details
- **Retrieval Architecture**: Uses a hybrid approach combining neural retrieval (ColBERT) and vision-language processing.
- **OCR Integration**: Extracts textual information from images for improved search accuracy.
- **Neural Ranking**: Enhances document ranking through multimodal feature fusion.
- **Frameworks Used**: ColBERT, ColPali, Vision Transformers (ViTs), and Qwen2-VL.

## Installation
To set up the UniDocR environment, follow these steps:
```bash
# Clone the repository
git clone https://github.com/yourusername/unidocr.git
cd unidocr
```

## Usage
1. **Indexing Documents**
```bash
python index.py --data_path /path/to/documents
```
2. **Querying**
```bash
python search.py --query "Your search text" --image_path /path/to/query/image
```

## Dataset
UniDocR is tested on the M2KR benchmark, supporting:
- **Q2A (Text-to-Text Retrieval)**
- **I2A (Image-to-Text Retrieval)**
- **IQ2A (Image + Text to Text Retrieval)**

## Contributions
If you'd like to contribute, feel free to submit a pull request or open an issue.

## Contact
For questions, reach out at tungduong0708@gmail.com.