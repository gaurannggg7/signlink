import os
import subprocess
import tempfile

def stitch_clips(paths: list[str], output_path: str = "output.mp4") -> str:
    if not paths:
        raise ValueError("No clips to stitch.")

    TARGET_W = 1280
    TARGET_H = 720
    pid = os.getpid()
    normalized = []
    concat_list_path = None

    try:
        # Pass 1: normalize each clip individually
        for i, p in enumerate(paths):
            if not os.path.exists(p):
                print(f"⚠️ Skipping missing file: {p}")
                continue

            tmp_out = f"/tmp/norm_{pid}_{i}.mp4"
            vf = (
                f"scale={TARGET_W}:{TARGET_H}:"
                f"force_original_aspect_ratio=decrease,"
                f"pad={TARGET_W}:{TARGET_H}:(ow-iw)/2:(oh-ih)/2,"
                f"setsar=1"
            )
            cmd = [
                "ffmpeg", "-y",
                "-i", p,
                "-vf", vf,
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "ultrafast",
                "-crf", "28",
                "-an",  # no audio
                tmp_out
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"⚠️ Normalize failed for {p}: {result.stderr[-300:]}")
                continue

            if os.path.exists(tmp_out) and os.path.getsize(tmp_out) > 0:
                normalized.append(tmp_out)
            else:
                print(f"⚠️ Normalize produced empty file for {p}")

        if not normalized:
            raise ValueError("No clips were successfully normalized.")

        if len(normalized) == 1:
            # Single clip — just copy it
            import shutil
            shutil.copy(normalized[0], output_path)
            return output_path

        # Pass 2: write concat list
        concat_fd, concat_list_path = tempfile.mkstemp(suffix='.txt', dir='/tmp')
        with os.fdopen(concat_fd, 'w') as f:
            for p in normalized:
                f.write(f"file '{p}'\n")

        # Verify list was written
        with open(concat_list_path) as f:
            content = f.read()
        print(f"Concat list:\n{content}")

        # Pass 3: concat
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_path,
            "-c", "copy",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg concat failed:\n{result.stderr[-500:]}")

        return output_path

    finally:
        for p in normalized:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        if concat_list_path and os.path.exists(concat_list_path):
            try:
                os.remove(concat_list_path)
            except Exception:
                pass
