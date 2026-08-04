"""Dependency-optional OCR adapters with one canonical result schema."""

from __future__ import annotations

from pathlib import Path
import re
import unicodedata
from typing import Iterable, Protocol

from agentforce.data.schemas import BoundingBox, OCRFrame, OCRItem, write_jsonl


class OCRDependencyError(RuntimeError):
    pass


def normalize_ocr_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.casefold()


class OCREngine(Protocol):
    name: str

    def recognize(self, image_path: str | Path) -> list[OCRItem]: ...


class TesseractOCREngine:
    name = "tesseract"

    def __init__(self, *, language: str = "vie+eng", config: str = "--psm 11") -> None:
        self.language = language
        self.config = config

    def recognize(self, image_path: str | Path) -> list[OCRItem]:
        try:
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            raise OCRDependencyError(
                "Tesseract OCR is optional. Install pytesseract and Pillow, "
                "plus the tesseract binary."
            ) from exc
        with Image.open(image_path) as image:
            width, height = image.size
            data = pytesseract.image_to_data(
                image,
                lang=self.language,
                config=self.config,
                output_type=pytesseract.Output.DICT,
            )
        items: list[OCRItem] = []
        for index, raw_text in enumerate(data.get("text", [])):
            text = " ".join(str(raw_text).split())
            try:
                confidence = float(data["conf"][index]) / 100.0
            except (KeyError, IndexError, TypeError, ValueError):
                confidence = 0.0
            if not text or confidence < 0:
                continue
            left = float(data["left"][index])
            top = float(data["top"][index])
            box_width = float(data["width"][index])
            box_height = float(data["height"][index])
            items.append(
                OCRItem(
                    text=text,
                    normalized_text=normalize_ocr_text(text),
                    confidence=min(1.0, confidence),
                    bbox=BoundingBox(
                        x_min=left / width,
                        y_min=top / height,
                        x_max=(left + box_width) / width,
                        y_max=(top + box_height) / height,
                    ),
                )
            )
        return items


class EasyOCREngine:
    name = "easyocr"

    def __init__(self, *, languages: tuple[str, ...] = ("vi", "en"), gpu: bool = False) -> None:
        self.languages = languages
        self.gpu = gpu
        self._reader = None

    def _get_reader(self):
        if self._reader is None:
            try:
                import easyocr
            except ImportError as exc:
                raise OCRDependencyError(
                    "EasyOCR is optional. Install the project OCR extra before preprocessing."
                ) from exc
            self._reader = easyocr.Reader(list(self.languages), gpu=self.gpu)
        return self._reader

    def recognize(self, image_path: str | Path) -> list[OCRItem]:
        results = self._get_reader().readtext(str(image_path))
        items: list[OCRItem] = []
        for points, raw_text, raw_confidence in results:
            text = " ".join(str(raw_text).split())
            if not text:
                continue
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            items.append(
                OCRItem(
                    text=text,
                    normalized_text=normalize_ocr_text(text),
                    confidence=float(raw_confidence),
                    bbox=BoundingBox(
                        x_min=min(xs),
                        y_min=min(ys),
                        x_max=max(xs),
                        y_max=max(ys),
                        normalized=False,
                    ),
                )
            )
        return items


def create_ocr_engine(name: str, **options: object) -> OCREngine:
    """Construct an OCR backend from configuration without importing it yet."""

    normalized = name.strip().casefold()
    if normalized in {"tesseract", "pytesseract"}:
        return TesseractOCREngine(**options)  # type: ignore[arg-type]
    if normalized in {"easyocr", "easy"}:
        return EasyOCREngine(**options)  # type: ignore[arg-type]
    raise ValueError(f"Unsupported OCR engine: {name!r}")


class OCRProcessor:
    def __init__(self, engine: OCREngine, *, min_confidence: float = 0.3) -> None:
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be in [0, 1]")
        self.engine = engine
        self.min_confidence = min_confidence

    def process(self, image_path: str | Path, keyframe_uid: str) -> OCRFrame:
        items = [
            item
            for item in self.engine.recognize(image_path)
            if item.confidence >= self.min_confidence and item.normalized_text
        ]
        full_text = "\n".join(dict.fromkeys(item.text for item in items))
        return OCRFrame(
            keyframe_uid=keyframe_uid,
            items=items,
            full_text=full_text,
            normalized_text=normalize_ocr_text(full_text),
            engine=self.engine.name,
            source_path=str(image_path),
        )

    def process_many(
        self,
        items: Iterable[tuple[str, str | Path]],
        output_path: str | Path,
    ) -> None:
        write_jsonl((self.process(path, uid) for uid, path in items), output_path)
