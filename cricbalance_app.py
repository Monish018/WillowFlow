"""
WillowFlow — Web Interface (Gradio 6.0 Compatible)
=====================================================
Run: python cricbalance_app.py
"""

import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO
from collections import deque
import tempfile, os

# ── Load model once ───────────────────────────────────────────────────────────
print("Loading YOLOv8-Pose model...")
model = YOLO("yolov8n-pose.pt")
print("Model ready.\n")

# ── Keypoint indices ──────────────────────────────────────────────────────────
KP_LEFT_HIP    = 11
KP_RIGHT_HIP   = 12
KP_LEFT_ANKLE  = 15
KP_RIGHT_ANKLE = 16

# ── Colours ───────────────────────────────────────────────────────────────────
C_HIP_DOT  = (0,   255, 255)
C_HIP_LINE = (255, 180,   0)
C_COG_LINE = (50,  255,  50)
C_TRAIL    = (255,  80, 200)
C_FRONT    = (0,   220, 120)
C_BACK     = (0,   140, 255)
C_TEXT     = (255, 255, 255)
C_BG       = (10,   10,  10)
C_POS      = (50,  255, 100)
C_NEG      = (50,  150, 255)

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_kp(xy, conf, idx, thr):
    if float(conf[idx]) < thr:
        return None
    return (int(float(xy[idx][0])), int(float(xy[idx][1])))

def mid(a, b):
    return ((a[0]+b[0])//2, (a[1]+b[1])//2)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def ankle_sides(la, ra, facing):
    if facing == "Auto":
        return (la, ra) if la[0] < ra[0] else (ra, la)
    elif "Right" in facing:
        return la, ra
    else:
        return ra, la

def foot_pct(cog_x, front, back):
    span = front[0] - back[0]
    if abs(span) < 5:
        return 50.0, 50.0
    r = clamp((cog_x - back[0]) / span, 0.0, 1.0)
    return round(r*100,1), round((1-r)*100,1)

def get_writer(path, fps, w, h):
    """Try codecs in order until one works."""
    for codec in ["avc1", "mp4v", "XVID"]:
        ext = ".mp4" if codec != "XVID" else ".avi"
        out = path.replace(".mp4", ext)
        writer = cv2.VideoWriter(
            out, cv2.VideoWriter_fourcc(*codec), fps, (w, h)
        )
        if writer.isOpened():
            return writer, out
    return None, path

# ── Drawing ───────────────────────────────────────────────────────────────────
def draw_trail(frame, trail):
    if len(trail) < 2:
        return
    pts = np.array(list(trail), dtype=np.int32)
    for i in range(1, len(pts)):
        a = i / len(pts)
        cv2.line(frame, tuple(pts[i-1]), tuple(pts[i]),
                 tuple(int(c*a) for c in C_TRAIL),
                 max(1, int(3*a)), cv2.LINE_AA)

def draw_overlays(frame, lh, rh, cog, la, ra, fa, ba, trail):
    fh = frame.shape[0]
    draw_trail(frame, trail)
    cv2.line(frame, lh, rh, C_HIP_LINE, 2, cv2.LINE_AA)
    cv2.line(frame, cog, (cog[0], fh), C_COG_LINE, 2, cv2.LINE_AA)
    for y in range(0, cog[1], 14):
        cv2.line(frame, (cog[0],y), (cog[0],y+7), (25,127,25), 1, cv2.LINE_AA)

    if la != (0,0) and ra != (0,0):
        cv2.line(frame, la, ra, (160,160,160), 1, cv2.LINE_AA)
        for ankle, label in [(la,"L.Ankle"),(ra,"R.Ankle")]:
            is_f = (ankle == fa)
            col  = C_FRONT if is_f else C_BACK
            cv2.circle(frame, ankle, 8, col, -1, cv2.LINE_AA)
            cv2.circle(frame, ankle, 8, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{label}({'F' if is_f else 'B'})",
                        (ankle[0]-12, ankle[1]+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

    for hip, lbl in [(lh,"L.Hip"),(rh,"R.Hip")]:
        cv2.circle(frame, hip, 8, C_HIP_DOT, -1, cv2.LINE_AA)
        cv2.circle(frame, hip, 8, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(frame, lbl, (hip[0]-10, hip[1]-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_HIP_DOT, 1, cv2.LINE_AA)

    cv2.circle(frame, cog, 10, C_COG_LINE, -1, cv2.LINE_AA)
    cv2.circle(frame, cog, 10, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame, "CoG", (cog[0]+12, cog[1]+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_COG_LINE, 1, cv2.LINE_AA)

def draw_bar(frame, fp, bp):
    h, w = frame.shape[:2]
    by, bx, bw, bh = h-46, 15, w-30, 36
    ov = frame.copy()
    cv2.rectangle(ov,(bx,by),(bx+bw,by+bh),(20,20,20),-1)
    cv2.addWeighted(ov,0.65,frame,0.35,0,frame)
    fw = int(bw*fp/100)
    if fw > 0:
        cv2.rectangle(frame,(bx,by+4),(bx+fw,by+bh-4),C_FRONT,-1)
    if bw-fw > 0:
        cv2.rectangle(frame,(bx+fw,by+4),(bx+bw,by+bh-4),C_BACK,-1)
    cv2.line(frame,(bx+fw,by),(bx+fw,by+bh),(255,255,255),2)
    cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),(180,180,180),1)
    ft = f"FRONT FOOT  {fp:.0f}%"
    bt = f"{bp:.0f}%  BACK FOOT"
    for txt, x in [(ft, bx+8)]:
        cv2.putText(frame,txt,(x,by+24),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(frame,txt,(x,by+24),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1,cv2.LINE_AA)
    btw = cv2.getTextSize(bt,cv2.FONT_HERSHEY_SIMPLEX,0.55,1)[0][0]
    cv2.putText(frame,bt,(bx+bw-btw-8,by+24),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(frame,bt,(bx+bw-btw-8,by+24),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1,cv2.LINE_AA)

def draw_hud(frame, shift, base_x, cog_x, fnum, fps, fp, bp, ff, bf):
    hx,hy,lh,pad = 15,15,28,8
    if shift is None:    ss,sc = "Acquiring...", C_TEXT
    elif shift > 0:      ss,sc = f"+{shift}px  Off-side >>", C_POS
    elif shift < 0:      ss,sc = f"{shift}px  << Leg-side", C_NEG
    else:                ss,sc = "0px  Balanced", C_TEXT

    total = ff+bf
    if total > 0:
        pf,pb = ff/total*100, bf/total*100
        dom   = "FRONT FOOT" if pf>=50 else "BACK FOOT"
        dc    = C_FRONT if pf>=50 else C_BACK
        shot  = f"{dom}  ({max(pf,pb):.0f}%)"
    else:
        shot,dc = "Acquiring...", C_TEXT

    fw_str = f"Front {fp:.0f}%  |  Back {bp:.0f}%" if fp is not None else "---"

    lines = [
        ("WillowFlow  |  IPCV Analytics",          C_TEXT, 0.55, 1),
        (f"Frame {fnum}        FPS {fps:.1f}",       C_TEXT, 0.45, 1),
        (f"Baseline CoG-X : {base_x or '---'}",     C_TEXT, 0.48, 1),
        (f"Current CoG-X  : {cog_x or '---'}",      C_TEXT, 0.48, 1),
        (f"Weight Shift   : {ss}",                  sc,     0.50, 2),
        ("- - - - - - - - - - - - - - - - - -",     (70,70,70), 0.38, 1),
        (f"Frame Weight   : {fw_str}",              C_TEXT, 0.48, 1),
        (f"Shot Type      : {shot}",                dc,     0.52, 2),
    ]
    bw2, bh2 = 355, len(lines)*lh+pad*2
    ov = frame.copy()
    cv2.rectangle(ov,(hx-pad,hy-pad),(hx+bw2,hy+bh2),C_BG,-1)
    cv2.addWeighted(ov,0.6,frame,0.4,0,frame)
    cv2.rectangle(frame,(hx-pad,hy-pad),(hx+bw2,hy+bh2),C_HIP_LINE,1)
    for i,(txt,col,sc2,th) in enumerate(lines):
        cv2.putText(frame,txt,(hx,hy+pad+i*lh+16),
                    cv2.FONT_HERSHEY_SIMPLEX,sc2,col,th,cv2.LINE_AA)

# ── Main processing ───────────────────────────────────────────────────────────
def analyze_video(video_path, facing, conf_thresh, trail_len, progress=gr.Progress()):
    if video_path is None:
        return None, "⚠️  Please upload a video first."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "⚠️  Could not open video. Please try again."

    fw   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tot  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Temp output file
    tmp      = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()

    writer, out_path = get_writer(tmp_path, fps, fw, fh)
    if writer is None:
        return None, "⚠️  Could not create output video writer."

    trail     = deque(maxlen=int(trail_len))
    base_x    = None
    fnum      = 0
    ff_frames = 0
    bf_frames = 0
    log       = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fnum += 1
        if tot > 0:
            progress(fnum/tot, desc=f"Processing frame {fnum} / {tot}")

        shift  = None
        cog_x  = None
        fp_val = None
        bp_val = None
        fa     = (0,0)
        ba     = (0,0)
        la_pt  = (0,0)
        ra_pt  = (0,0)

        results = model(frame, verbose=False)

        for result in results:
            if result.keypoints is None or len(result.keypoints) == 0:
                break

            if result.boxes is not None and len(result.boxes) > 1:
                areas = [(b[2]-b[0])*(b[3]-b[1]) for b in result.boxes.xyxy]
                pidx  = int(np.argmax(areas))
            else:
                pidx = 0

            xy   = result.keypoints.xy[pidx]
            conf = result.keypoints.conf[pidx]

            lh = get_kp(xy, conf, KP_LEFT_HIP,  conf_thresh)
            rh = get_kp(xy, conf, KP_RIGHT_HIP, conf_thresh)
            if lh is None or rh is None:
                break

            cog   = mid(lh, rh)
            cog_x = cog[0]
            trail.append(cog)

            if base_x is None:
                base_x = cog_x
            shift = cog_x - base_x

            la = get_kp(xy, conf, KP_LEFT_ANKLE,  conf_thresh)
            ra = get_kp(xy, conf, KP_RIGHT_ANKLE, conf_thresh)

            if la and ra:
                la_pt, ra_pt = la, ra
                fa, ba = ankle_sides(la, ra, facing)
                fp_val, bp_val = foot_pct(cog[0], fa, ba)
                if fp_val >= 50:
                    ff_frames += 1
                else:
                    bf_frames += 1
                log.append({"frame":fnum,"cog_x":cog_x,
                             "shift":shift,"front":fp_val,"back":bp_val})

            draw_overlays(frame, lh, rh, cog, la_pt, ra_pt, fa, ba, trail)
            if fp_val is not None:
                draw_bar(frame, fp_val, bp_val)
            break

        draw_hud(frame, shift, base_x, cog_x,
                 fnum, fps, fp_val, bp_val, ff_frames, bf_frames)
        writer.write(frame)

    cap.release()
    writer.release()

    # ── Report ────────────────────────────────────────────────────────────
    total = ff_frames + bf_frames
    if total > 0:
        ff_p   = ff_frames/total*100
        bf_p   = bf_frames/total*100
        dom    = "FRONT FOOT" if ff_p>=50 else "BACK FOOT"
        shifts = [f["shift"] for f in log]
        fronts = [f["front"] for f in log]
        report = f"""
╔══════════════════════════════════════╗
   WILLOWFLOW — SHOT ANALYSIS REPORT
╚══════════════════════════════════════╝

 Video Info
    Frames processed  : {fnum}
    Resolution        : {fw} x {fh}
    FPS               : {fps:.1f}

 Foot Weight Distribution
    Front Foot frames : {ff_frames}  ({ff_p:.1f}%)
    Back Foot frames  : {bf_frames}  ({bf_p:.1f}%)

 Shot Classification
    --> {dom} DOMINANT  ({max(ff_p,bf_p):.1f}%)

 Center of Gravity (CoG) Shift
    Baseline CoG-X    : {base_x}px
    Min shift         : {min(shifts)}px
    Max shift         : {max(shifts)}px
    Peak displacement : {max(abs(s) for s in shifts)}px

 Average Frame Weight
    Avg Front Foot    : {sum(fronts)/len(fronts):.1f}%
    Avg Back Foot     : {100-sum(fronts)/len(fronts):.1f}%

 Settings Used
    Facing direction  : {facing}
    Confidence thresh : {conf_thresh}
    Trail length      : {int(trail_len)} frames
"""
    else:
        report = (
            "No keypoints detected.\n\n"
            "Try:\n"
            "  - Lower the Confidence Threshold slider\n"
            "  - Use a clearer side-on cricket video\n"
            "  - Make sure the batsman is fully in frame"
        )

    return out_path, report

# ── Gradio UI (Gradio 6.0 compatible) ────────────────────────────────────────
CSS = """
    body { background: #0a0f0a; }
    .gradio-container { max-width: 1100px !important; margin: 0 auto; }
    #header { text-align:center; padding:28px 0 18px;
              border-bottom:1px solid #1e3a1e; margin-bottom:20px; }
    #header h1 { font-size:2.4rem; font-weight:700; color:#4ade80;
                 letter-spacing:2px; margin:0; }
    #header p  { color:#6b8f6b; font-size:0.95rem; margin:5px 0 0; }
    #analyze-btn { background:linear-gradient(135deg,#16a34a,#15803d) !important;
                   border:none !important; color:white !important;
                   font-size:1.05rem !important; font-weight:700 !important;
                   letter-spacing:2px !important; height:50px !important;
                   border-radius:8px !important; margin-top:10px !important; }
    #analyze-btn:hover { background:linear-gradient(135deg,#15803d,#166534) !important;
                         box-shadow:0 4px 20px rgba(74,222,128,0.3) !important; }
    #report textarea  { font-family:'Courier New',monospace !important;
                        font-size:0.8rem !important; background:#060e06 !important;
                        color:#86efac !important; border:1px solid #1e3a1e !important;
                        line-height:1.7 !important; }
    #footer { text-align:center; padding:16px 0 8px; color:#3a5a3a;
              font-size:0.72rem; letter-spacing:1px;
              border-top:1px solid #1e3a1e; margin-top:20px; }
"""

THEME = gr.themes.Base(
    primary_hue="green",
    secondary_hue="emerald",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Rajdhani"), "sans-serif"],
)

with gr.Blocks(title="WillowFlow") as app:

    gr.HTML("""
        <div id="header">
            <h1>🏏 WILLOWFLOW</h1>
            <p>Cricket Batsman Biomechanics & Weight Transfer Analyzer
             · IPCV Project · BNMIT AIML</p>
        </div>
    """)

    with gr.Row():
        # Left — controls
        with gr.Column(scale=1):
            gr.HTML("<b style='color:#4ade80'>📂 Upload Video</b>")
            video_input = gr.Video(label="Drag & drop your cricket video", height=200)

            gr.HTML("<br><b style='color:#4ade80'>⚙️ Settings</b>")
            facing = gr.Radio(
                choices=["Auto",
                         "Facing Left (bowler on left)",
                         "Facing Right (bowler on right)"],
                value="Auto",
                label="Batsman Facing Direction",
                info="Change if FRONT/BACK labels appear swapped."
            )
            conf_thresh = gr.Slider(0.1, 0.9, value=0.5, step=0.05,
                                    label="Keypoint Confidence Threshold")
            trail_len   = gr.Slider(10, 120, value=60, step=5,
                                    label="CoG Trail Length (frames)")
            btn = gr.Button("▶  ANALYSE VIDEO",
                            elem_id="analyze-btn", variant="primary")

        # Right — output
        with gr.Column(scale=2):
            gr.HTML("<b style='color:#4ade80'>🎬 Processed Output</b>")
            video_output = gr.Video(label="Annotated Video", height=360)

            gr.HTML("<br><b style='color:#4ade80'>📊 Analysis Report</b>")
            report_out = gr.Textbox(label="", lines=20,
                                    elem_id="report",
                                    placeholder="Report appears here after analysis...")

    gr.HTML("""
        <div id="footer">
            BNMIT · B.E. AIML · 6th Semester · IPCV Project ·
            YOLOv8-Pose + OpenCV + Gradio
        </div>
    """)

    btn.click(fn=analyze_video,
              inputs=[video_input, facing, conf_thresh, trail_len],
              outputs=[video_output, report_out])

if __name__ == "__main__":
    app.launch(
        theme=THEME,
        css=CSS,
        inbrowser=True,
        server_port=7860,
        share=False,
    )
