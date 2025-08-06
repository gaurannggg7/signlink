import ffmpeg

def stitch_clips(paths: list[str], output_path: str = "output.mp4") -> str:
    """
    Concatenate arbitrary video clips into one MP4 by:
      1) probing the first clip's resolution,
      2) scaling all clips to that resolution,
      3) filter-concat then encode to H.264.
    """
    if not paths:
        raise ValueError("No clips to stitch.")

    # 1) Probe first clip to get target width/height
    info = ffmpeg.probe(paths[0], select_streams='v')
    stream = next(s for s in info['streams'] if s['codec_type']=='video')
    width, height = int(stream['width']), int(stream['height'])

    # 2) Load and scale each clip
    scaled = []
    for p in paths:
        inp = ffmpeg.input(p)
        # scale to match the first clip's resolution
        scaled.append(inp.filter('scale', width, height))

    # 3) Concat video-only streams
    joined = ffmpeg.concat(*scaled, v=1, a=0)

    # 4) Re-encode to baseline-compatible MP4
    out = ffmpeg.output(
        joined,
        output_path,
        vcodec='libx264',
        pix_fmt='yuv420p'
    )
    out.run(overwrite_output=True)
    return output_path
