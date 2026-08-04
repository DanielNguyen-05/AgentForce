"""Dependency-optional OCR adapters with one canonical result schema."""

from __future__ import annotations

import re
import unicodedata
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol

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

    def __init__(
        self,
        *,
        languages: tuple[str, ...] = ("vi", "en"),
        gpu: bool = False,
        batch_size: int = 1,
        suppress_pin_memory_warnings: bool = True,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.languages = languages
        self.gpu = gpu
        self.batch_size = batch_size
        self.suppress_pin_memory_warnings = suppress_pin_memory_warnings
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

    def _suppress_repetitive_backend_warnings(self) -> None:
        # EasyOCR 1.7.x hardcodes ``pin_memory=True`` for every recognition
        # DataLoader. PyTorch then disables it on CPU/MPS and emits the same
        # harmless warning for nearly every detected text box. Suppress only
        # those two upstream messages locally; all other warnings remain
        # visible and inference behavior is unchanged.
        if not self.suppress_pin_memory_warnings:
            return
        warnings.filterwarnings(
            "ignore",
            message=r"'pin_memory' argument is set as true but no accelerator is found,.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"'pin_memory' argument is set as true but not supported on MPS now,.*",
            category=UserWarning,
        )

    @staticmethod
    def _items_from_results(results: Iterable[object]) -> list[OCRItem]:
        items: list[OCRItem] = []
        for result in results:
            points, raw_text, raw_confidence = result  # type: ignore[misc]
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

    def recognize(self, image_path: str | Path) -> list[OCRItem]:
        with warnings.catch_warnings():
            self._suppress_repetitive_backend_warnings()
            reader = self._get_reader()
            if self.batch_size == 1:
                results = reader.readtext(str(image_path))
            else:
                results = reader.readtext(str(image_path), batch_size=self.batch_size)
        return self._items_from_results(results)

    def recognize_many(
        self, image_paths: Sequence[str | Path]
    ) -> list[list[OCRItem]]:
        """Recognize ordered, equal-sized keyframes with EasyOCR's batched detector.

        ``readtext_batched`` preserves input order. Chunking bounds detector memory,
        while ``batch_size=1`` deliberately retains the established serial path.
        """

        if self.batch_size == 1:
            return [self.recognize(path) for path in image_paths]

        reader = self._get_reader()
        outputs: list[list[OCRItem]] = []
        for start in range(0, len(image_paths), self.batch_size):
            chunk = image_paths[start : start + self.batch_size]
            with warnings.catch_warnings():
                self._suppress_repetitive_backend_warnings()
                result_batch = reader.readtext_batched(
                    [str(path) for path in chunk],
                    batch_size=self.batch_size,
                )
            if len(result_batch) != len(chunk):
                raise RuntimeError(
                    "EasyOCR returned a different number of batches than input images"
                )
            outputs.extend(self._items_from_results(results) for results in result_batch)
        return outputs


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

    def _build_frame(
        self,
        image_path: str | Path,
        keyframe_uid: str,
        recognized_items: Iterable[OCRItem],
    ) -> OCRFrame:
        items = [
            item
            for item in recognized_items
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

    def process(self, image_path: str | Path, keyframe_uid: str) -> OCRFrame:
        return self._build_frame(
            image_path,
            keyframe_uid,
            self.engine.recognize(image_path),
        )

    def process_many(
        self,
        items: Iterable[tuple[str, str | Path]],
        output_path: str | Path,
    ) -> None:
        entries = list(items)
        recognize_many = getattr(self.engine, "recognize_many", None)
        if not callable(recognize_many):
            write_jsonl((self.process(path, uid) for uid, path in entries), output_path)
            return

        recognized_batches = recognize_many([path for _, path in entries])
        if len(recognized_batches) != len(entries):
            raise RuntimeError("OCR engine returned a different number of results than inputs")
        write_jsonl(
            (
                self._build_frame(path, uid, recognized)
                for (uid, path), recognized in zip(entries, recognized_batches, strict=True)
            ),
            output_path,
        )
