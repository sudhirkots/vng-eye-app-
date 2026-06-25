# EyeVNG Data Flow

```text
Upload MP4
  -> Ingestion and metadata validation
  -> Frame-by-frame tracking
  -> Face landmarks + eye landmarks + iris centres
  -> Head pose estimation
  -> Raw measurements stored
  -> Quality assessment
  -> Eye ROI export per frame
  -> Overlay video generation
  -> CSV export
  -> Analysis report generation
```

## Key design points
- Raw measurements are stored separately from any future processed values.
- The tracking pipeline remains independent from calibration.
- Head pose is handled by a dedicated module for future vHIT and VOR work.
- Overlay and ROI outputs are generated for visual verification and torsional analysis readiness.
