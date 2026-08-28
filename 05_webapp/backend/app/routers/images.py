"""Figure Studio — single-image serving (format conversion + thumbnails)."""
from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query, Request, Response

from .. import gallery, imagesvc

router = APIRouter(prefix="/api/figure", tags=["figure-images"])

_ALLOWED_FMT = {"png", "jpeg", "webp"}
_MIN_WIDTH = 16
_MAX_WIDTH = 4096


@router.api_route("/img", methods=["GET", "HEAD"])
def get_img(
    request: Request,
    kind: Literal["pred", "input", "gt"] = Query(...),
    dataset: str = Query(...),
    image: str = Query(...),
    stage: Optional[str] = Query(None),
    experiment: Optional[str] = Query(None),
    run_name: Optional[str] = Query(None),
    w: Optional[int] = Query(None),
    fmt: str = Query("png"),
):
    fmt = fmt.lower()
    if fmt not in _ALLOWED_FMT:
        raise HTTPException(status_code=400, detail=f"unsupported fmt: {fmt!r}")
    if w is not None and not (_MIN_WIDTH <= w <= _MAX_WIDTH):
        raise HTTPException(status_code=400, detail=f"w out of range [{_MIN_WIDTH}, {_MAX_WIDTH}]: {w}")

    try:
        path = gallery.resolve_panel(
            kind, dataset, image, stage=stage, experiment=experiment, run_name=run_name
        )
    except gallery.PanelNotFound:
        raise HTTPException(
            status_code=404,
            detail=f"panel not found: kind={kind} dataset={dataset} image={image}",
        ) from None
    except Exception as exc:  # unexpected input shape, not a resolution failure
        raise HTTPException(
            status_code=400, detail=f"invalid panel request: {type(exc).__name__}"
        ) from exc

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"panel not found: kind={kind} dataset={dataset} image={image}",
        )

    etag = imagesvc.etag_for(path, w, fmt, kind)
    headers = {
        "Cache-Control": "public, max-age=31536000, immutable",
        "ETag": etag,
    }

    if_none_match = request.headers.get("if-none-match")
    if if_none_match and etag in (tok.strip() for tok in if_none_match.split(",")):
        return Response(status_code=304, headers=headers)

    try:
        content, mime = imagesvc.encoded(path, kind, width=w, fmt=fmt)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"panel not found: kind={kind} dataset={dataset} image={image}",
        ) from None
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"image conversion failed: {type(exc).__name__}"
        ) from exc

    body = b"" if request.method == "HEAD" else content
    return Response(content=body, media_type=mime, headers=headers)
