import fitz
import io
from PIL import Image
from typing import List, Dict, Any
from src.processors.image_captioner import MultimodalCaptioner

class PDFProcessor:
    def __init__(self, file_bytes: bytes, filename: str):
        self.filename = filename
        self.doc = fitz.open(stream=file_bytes, filetype="pdf")

    def process(self) -> List[Dict[str, Any]]:
        extracted_chunks = []
        raw_images = []
        image_metadata = []

        for page_idx, page in enumerate(self.doc):
            page_num = page_idx + 1
            # 1. Text extraction with layout preservation
            page_text = page.get_text("text").strip()
            if page_text:
                extracted_chunks.append({
                    "content": page_text,
                    "metadata": {
                        "source": self.filename,
                        "page": page_num,
                        "type": "text"
                    }
                })

            # 2. Embedded images & visual diagrams
            image_list = page.get_images(full=True)
            for img_idx, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = self.doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    # Ignore tiny decorative icons (e.g. < 60x60)
                    if pil_img.width > 60 and pil_img.height > 60:
                        raw_images.append(pil_img)
                        image_metadata.append({
                            "source": self.filename,
                            "page": page_num,
                            "index": img_idx + 1,
                            "type": "image"
                        })
                except Exception:
                    continue

        # 3. Captioning embedded visuals batch-wise
        if raw_images:
            captioner = MultimodalCaptioner.get_instance()
            captions = captioner.caption_images(raw_images)
            for cap, meta in zip(captions, image_metadata):
                visual_content = f"[Visual element / Figure from page {meta['page']}]: {cap}"
                extracted_chunks.append({
                    "content": visual_content,
                    "metadata": meta
                })

        return extracted_chunks
