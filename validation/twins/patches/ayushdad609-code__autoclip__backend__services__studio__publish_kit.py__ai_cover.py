def ai_cover(project_id: str, job_id: str, strategy_id: str) -> bool:
    """Generate the AI cover for one variant from its designed cover's frame and title; True when stored."""
    for _part in (project_id, job_id, strategy_id):
        if not _part or Path(str(_part)).name != str(_part) or str(_part) in (".", ".."):
            return False
    from PIL import Image
    from backend.core.image_providers import ImageRequest, generate_image
    from backend.services import cover
    from backend.services.studio import cover_design as cd
    cfg = cover.load_config()
    frame_file, meta_file = frame_path(project_id, job_id, strategy_id), cd_meta_path(project_id, job_id, strategy_id)
    if not (cfg.enabled and cfg.configured) or not frame_file.is_file() or not meta_file.is_file():
        return False
    if not cfg.allow_send_frame:
        # Without the frame the model would invent a face, then carry the real guest's nameplate.
        return False
    meta = json.loads(meta_file.read_text(encoding='utf-8'))
    width, height = cd.size_for(strategy_id)
    request = ImageRequest(prompt=ai_prompt(strategy_id, meta['lines'], meta['accent'], meta.get('name', ''), meta.get('palette')), width=width, height=height,
                           reference=frame_file.read_bytes(), model=cfg.model)
    image = None
    for attempt in range(2):
        try:
            image = generate_image(provider=cfg.provider, api_key=cfg.api_key, base_url=cfg.base_url, request=request)
        except Exception as error:  # noqa: BLE001 - the designed cover stays
            logger.warning('AI cover failed: %s', type(error).__name__)
            return False
        if _title_ok(image, meta['lines']) is not False:
            break
        request = ImageRequest(prompt=request.prompt + ' The previous attempt misspelled the headline: render it character by character exactly.', width=width, height=height,
                               reference=request.reference, model=cfg.model)
    else:
        return False  # the title stayed wrong twice: keep the designed cover
    generated = Image.open(io.BytesIO(image)).convert('RGB')
    fitted = _fit(generated, (width, height))
    if meta.get('guest') and meta.get('name'):
        fitted = cd.nameplate(fitted, meta['name'], meta.get('role', ''), meta.get('palette'))
    out = io.BytesIO()
    fitted.save(out, format='JPEG', quality=92)
    _atomic_write(kit_cover_path(project_id, job_id, strategy_id), out.getvalue())
    if height != 1440:  # the publish slot is 9:16 / 16:9; Xiaohongshu's 3:4 stays kit-only
        slot = _publish_slot(strategy_id)
        _atomic_write(cover.cover_path(project_id, clip_id(job_id), slot), out.getvalue())
        cover.write_cover_meta(project_id, clip_id(job_id), slot, {'method': 'model', 'title': ' '.join(meta['lines'])})
    return True
