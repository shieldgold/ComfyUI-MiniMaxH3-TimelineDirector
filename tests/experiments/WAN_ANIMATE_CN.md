# Wan Animate 目标人物替换复核工具

这些是离线实验工具，不改变 H3 节点或正在运行的 ComfyUI 服务。
素材、模型权重和生成视频不纳入仓库。

## 提交与恢复

`comfy_job_receipt.py` 对同一 API 文件和 endpoint 保存提交凭据与 prompt ID。
恢复时沿用原凭据；提交结果不明确时停止自动重试，避免重复排队。

```sh
python3 tests/experiments/comfy_job_receipt.py \
  --endpoint http://127.0.0.1:8190 \
  --api /absolute/path/segment_api.json \
  --receipt /absolute/path/segment-dispatch.json --wait
```

新参数或新配方使用新的 API 文件和凭据文件。脚本不取消任务、清理模型或重启服务。

## 保留背景与旁人

`ops/wan_psy_composite.py` 使用素材 manifest 中的 `crop_xywh`、缩放和留白变换，
将生成区域还原到原片坐标。原人物语义遮罩用于跟踪；原旁人遮罩优先保护。
生成区域外与旁人区域在编码前按原帧逐像素检查，交付的 H.264 视频会发生有损编码。

默认只做预检；实际合成需要 `--compose`、输出路径和匹配该 RAW 的人工质量记录。
最终视频仍必须单独逐帧复核并播放，不能把任务成功或结构初筛当作成片通过。

```sh
python3 ops/wan_psy_composite.py \
  --assets /absolute/path/segment_assets.json \
  --raw /absolute/path/segment_raw.mp4 \
  --quality-report /absolute/path/segment_qc.json \
  --audio-source /absolute/path/original_full_audio.wav \
  --preflight-report /absolute/path/segment_preflight.json \
  --compose --output /absolute/path/segment_protected1080p.mp4 \
  --segmentation-model /absolute/path/yolo11m-seg.pt
```

依赖已有 ffmpeg/ffprobe、OpenCV、NumPy、PyTorch、Ultralytics 和本地分割权重。
`source_frame_indices` 使用完整原片的绝对帧号，音频也从完整音源按该帧号切片。
24 fps 的 C2 起点若为原片第83帧，则音频起点是 `83/24` 秒，不能再叠加局部偏移。
镜头切换应拆开生成；续段去掉用于延续动作的重复帧后再合成。

## 验证

```sh
python3 -m unittest discover -s tests/experiments -p 'test_comfy_job_receipt.py' -v
python3 -m unittest discover -s tests/experiments -p 'test_wan_psy_composite.py' -v
```

测试覆盖提交恢复、拒绝重复提交、历史身份匹配、坐标变换、切镜头帧映射、
质量记录绑定，以及按原帧号精确截取 PCM 音频。
