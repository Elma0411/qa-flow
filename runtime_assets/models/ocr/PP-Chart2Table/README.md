---
license: apache-2.0
library_name: PaddleOCR
language:
- en
- zh
pipeline_tag: image-to-text
tags:
- OCR
- PaddlePaddle
- PaddleOCR
- chart_parsing
---

# PP-Chart2Table

## Introduction

PP-Chart2Table is a SOTA multimodal model developed by the PaddlePaddle team, specializing in chart parsing for both Chinese and English. Its high performance is driven by a novel "Shuffled Chart Data Retrieval" training task, which, combined with a refined token masking strategy, significantly improves its efficiency in converting charts to data tables. The model is further strengthened by an advanced data synthesis pipeline that uses high-quality seed data, RAG, and LLMs persona design to create a richer, more diverse training set. To address the challenge of large-scale unlabeled, out-of-distribution (OOD) data, the team implemented a two-stage distillation process, ensuring robust adaptability and generalization on real-world data. In-house benchmarks demonstrate that PP-Chart2Table not only outperforms models of a similar scale but also achieves performance on par with 7-billion parameter Vision Language Models (VLMs) in critical application scenarios.

<img src="https://cdn-uploads.huggingface.co/production/uploads/684acf07de103b2d44c85531/IsYzsgw5f8ehK4zn9IP1x.png"/>


## Quick Start

### Installation

1. PaddlePaddle

Please refer to the following commands to install PaddlePaddle using pip:

```bash
# for CUDA11.8
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# for CUDA12.6
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# for CPU
python -m pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

For details about PaddlePaddle installation, please refer to the [PaddlePaddle official website](https://www.paddlepaddle.org.cn/en/install/quick).

2. PaddleX

Install the latest version of the PaddleX inference package from PyPI:

```bash
python -m pip install paddlex && python -m pip install "paddlex[multimodal]"
```

### Model Usage

You can integrate the model inference of PP-Chart2Table into your project. Before running the following code, please download the sample image to your local machine.

```python
from paddlex import create_model
model = create_model('PP-Chart2Table')
results = model.predict(
    input={"image": "https://cdn-uploads.huggingface.co/production/uploads/684acf07de103b2d44c85531/OrlFuIXQUhO3Fg1G9_H1u.png"},
    batch_size=1
)
for res in results:
    res.print()
    res.save_to_json(f"./output/res.json")
```

After running, the obtained result is as follows:

```bash
{'res': {'image': 'https://cdn-uploads.huggingface.co/production/uploads/684acf07de103b2d44c85531/OrlFuIXQUhO3Fg1G9_H1u.png', 'result': 'Agency | Favorable | Not Sure | Unfavorable\nNational Park Service | 81% | 12% | 7%\nU.S. Postal Service | 77% | 3% | 20%\nNASA | 74% | 17% | 9%\nSocial Security Administration | 61% | 12% | 28%\nCDC | 56% | 6% | 38%\nVeterans Affairs | 56% | 16% | 28%\nEPA | 55% | 14% | 31%\nHealth and Human Services | 55% | 15% | 30%\nFBI | 52% | 12% | 36%\nDepartment of Transportation | 52% | 12% | 36%\nDepartment of Homeland Security | 51% | 18% | 35%\nDepartment of Justice | 49% | 10% | 41%\nCIA | 46% | 21% | 33%\nDepartment of Education | 45% | 8% | 47%\nFederal Reserve | 43% | 20% | 37%\nIRS | 42% | 7% | 51%'}}
```

The visualized result is as follows:

<img src="https://cdn-uploads.huggingface.co/production/uploads/684acf07de103b2d44c85531/vxlQiD7IGA4n9U7eJFUJo.png"/>

For details about usage command and descriptions of parameters, please refer to the [Document](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/vlm_modules/chart_parsing.html#iii-quick-integration).

### Pipeline Usage

The ability of a single model is limited. But the pipeline consists of several models can provide more capacity to resolve difficult problems in real-world scenarios.

#### PP-StructureV3

Layout analysis is a technique used to extract structured information from document images. PP-StructureV3 includes the following seven modules:
* Layout Detection Module
* Chart Recognition Module（Optional）
* General OCR Sub-pipeline
* Document Image Preprocessing Sub-pipeline （Optional）
* Table Recognition Sub-pipeline （Optional）
* Seal Recognition Sub-pipeline （Optional）
* Formula Recognition Sub-pipeline （Optional）

You can quickly experience the PP-StructureV3 pipeline with a single command.

```bash
paddleocr pp_structurev3 --chart_recognition_model_name PP-Chart2Table \
    --use_chart_recognition True \
    -i https://cdn-uploads.huggingface.co/production/uploads/684acf07de103b2d44c85531/Mk1PKgszCEEutZukT3FPB.png
```

You can experience the inference of the pipeline with just a few lines of code. Taking the PP-StructureV3 pipeline as an example:

```python
from paddleocr import PPStructureV3

pipeline = PPStructureV3(chart_recognition_model_name="PP-Chart2Table", use_chart_recognition=True)
# ocr = PPStructureV3(use_doc_orientation_classify=True) # Use use_doc_orientation_classify to enable/disable document orientation classification model
# ocr = PPStructureV3(use_doc_unwarping=True) # Use use_doc_unwarping to enable/disable document unwarping module
# ocr = PPStructureV3(use_textline_orientation=True) # Use use_textline_orientation to enable/disable textline orientation classification model
# ocr = PPStructureV3(device="gpu") # Use device to specify GPU for model inference
output = pipeline.predict("./Mk1PKgszCEEutZukT3FPB.png", use_chart_recognition=True)
for res in output:
    res.print() ## Print the structured prediction output
    res.save_to_json(save_path="output") ## Save the current image's structured result in JSON format
    res.save_to_markdown(save_path="output") ## Save the current image's result in Markdown format
```

The default model used in pipeline is `PP-Chart2Table`, so you don't have to specify `PP-Chart2Table` for the `chart_recognition_model_name argument`, but you can use the local model file by argument `chart_recognition_model_dir`.
For details about usage command and descriptions of parameters, please refer to the [Document](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/PP-StructureV3.html#2-quick-start).

## Links

[PaddleOCR Repo](https://github.com/paddlepaddle/paddleocr)

[PaddleOCR Documentation](https://paddlepaddle.github.io/PaddleOCR/latest/en/index.html)

[PaddleX Repo](https://github.com/paddlepaddle/paddlex)

[PaddleX Documentation](https://paddlepaddle.github.io/PaddleX/latest/en/index.html)