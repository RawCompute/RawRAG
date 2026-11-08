import re
import threading

import numpy as np
import PyPDF2

_ocr_engine = None
_ocr_engine_lock = threading.Lock()
_pdfium_lock = threading.Lock()


def has_readable_text(text):
    tokens = text.split()
    if not tokens:
        return False
    encoded_tokens = sum(
        re.fullmatch(r"/(?:i)?\d+(?:/\d+)*", token) is not None
        for token in tokens
    )
    return encoded_tokens / len(tokens) < 0.5


def _get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        with _ocr_engine_lock:
            if _ocr_engine is None:
                from rapidocr import LangDet, LangRec, ModelType, OCRVersion, RapidOCR

                _ocr_engine = RapidOCR(
                    params={
                        "Det.ocr_version": OCRVersion.PPOCRV6,
                        "Det.lang_type": LangDet.EN,
                        "Det.model_type": ModelType.TINY,
                        "Rec.ocr_version": OCRVersion.PPOCRV6,
                        "Rec.lang_type": LangRec.EN,
                        "Rec.model_type": ModelType.TINY,
                        "Global.use_cls": False,
                        "EngineConfig.onnxruntime.intra_op_num_threads": 1,
                        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
                    }
                )
    return _ocr_engine


def _ocr_page(pdf_document, page_number, engine, scale):
    with _pdfium_lock:
        page = pdf_document[page_number - 1]
        bitmap = None
        try:
            bitmap = page.render(scale=scale)
            rendered_image = bitmap.to_pil()
            image = rendered_image.convert("RGB")
            rendered_image.close()
        finally:
            if bitmap is not None:
                bitmap.close()
            page.close()

    try:
        result = engine(image)
        return " ".join(result.txts or ())
    finally:
        image.close()


def extract_pages_from_pdf(pdf_path):
    reader = PyPDF2.PdfReader(pdf_path)
    pdfium_document = None
    engine = None
    pages = []
    try:
        for page in reader.pages:
            page_number = len(pages) + 1
            page_text = (page.extract_text() or "").replace("\n", " ")
            if not has_readable_text(page_text):
                if pdfium_document is None:
                    import pypdfium2

                    with _pdfium_lock:
                        pdfium_document = pypdfium2.PdfDocument(pdf_path)
                    engine = _get_ocr_engine()
                scale = 1 if page_text.strip() else 2
                page_text = _ocr_page(pdfium_document, page_number, engine, scale)
            pages.append((page_number, page_text))
    finally:
        if pdfium_document is not None:
            with _pdfium_lock:
                pdfium_document.close()
    return pages


def extract_text_from_pdf(pdf_path):
    return " ".join(text for _, text in extract_pages_from_pdf(pdf_path))


def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []

    Stride = chunk_size - overlap
    for i in range(0, len(words), Stride):
        chunk_text = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk_text)

        if i + chunk_size >= len(words):
            break
    return chunks
