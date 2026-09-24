# Fast YOLO retrain at a practice round (pillars + parking lot)

From "field is free" to "new model on the car" in about 40 minutes. The only slow step
is checking labels in Roboflow, and two people halve it.

| Step | Where | Time | Command / action |
| --- | --- | --- | --- |
| 1. Record | Car + web page | 10 min | Tune: `RECORD_RUNS` on, drive laps, then the Record button and push the car by hand |
| 2. Harvest | PC | 2-3 min | `python tools/yolo/yolo.py harvest` |
| 3. Review labels | Roboflow | 10-15 min | Fix the pre-drawn boxes, add to dataset |
| 4. New version | Roboflow | 1 min | Generate, no resize, no augmentation |
| 5. Train | Colab | 8-12 min | Runtime → Run all |
| 6. Deploy | PC | 1 min | `python tools/yolo/yolo.py deploy` |
| 7. Check | Car | 2 min | One lap; page shows `YOLO ncnn pillars_<date>` |

Start the Colab install (cells 1-2) while you review labels in step 3.

---

## A. One-time setup at home (do it all before the event)

**PC**

```
cd BlueprintPi
git pull
pip install -r tools/yolo/requirements.txt
ssh suntzu@192.168.137.45 "echo ok"
```

Stop `ssh` asking for the Pi password every time (Windows, once):

```
ssh-keygen -t ed25519
type %USERPROFILE%\.ssh\id_ed25519.pub | ssh suntzu@192.168.137.45 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

If the Pi's address changes at the venue, edit `pi_host` in `tools/yolo/config.json`.

**Roboflow**

1. Open the existing pillar project. In **Classes**, add `PARKING LOT`. The three names must be exactly
   `GREEN PILLAR`, `RED PILLAR`, `PARKING LOT`.
2. Put the workspace and project IDs in `tools/yolo/config.json`: they are the two parts of the project URL,
   `app.roboflow.com/<workspace>/<project>`.
3. Get the **private** API key (Settings → API Keys) and store it on the PC. Do this yourself, and never commit it:
   ```
   setx ROBOFLOW_API_KEY "paste-your-key-here"
   ```
   Open a new terminal afterwards. Without a key everything still works; `harvest` tells you which zip to
   drag into Roboflow by hand.

**Colab**

1. colab.research.google.com → File → Open notebook → GitHub → `josef-ami/BlueprintPi` →
   `tools/yolo/train_colab.ipynb`. Then File → Save a copy in Drive, so you can reopen it offline from Drive.
2. Runtime → Change runtime type → **T4 GPU**.
3. Key icon on the left → Add new secret: `ROBOFLOW_API_KEY`, with **Notebook access** on.
4. Fill cell 1 (workspace, project) and do one full Run all on the current dataset version.
   This proves the chain works and saves `wro_yolo/latest.pt` to Drive.

**Pi**

`git pull`, restart `obstacleRound.py`, and check the page: detector `YOLO ncnn pillars26` and a Tune group
"Recording". If you have magenta parking walls at home, record and train a first parking model now, so the
venue retrain only adapts it.

---

## B. At the venue

### 1. Record (about 10 min of field time)

- Tune tab → Recording → `RECORD_RUNS` on. Every run from GO to FINISHED/STOP is saved to `take/`, 3 frames/s,
  still frames skipped.
- Drive **2-3 laps each direction**, with the pillar layouts the field allows.
- Then press **Record** and push the car by hand for 2-3 min. This gets the views laps never produce:
  - every pillar from 1.5 m down to 10 cm, from both sides, and from a corner;
  - the **parking lot** from every distance and angle, including entering and leaving it;
  - the car yawed 30-45° (how it sees pillars mid-swerve);
  - pillars half out of the frame, and against the venue background, spectators and lights.
- Press **Stop recording**, then turn `RECORD_RUNS` **off** before any scored run.
- Aim for 800-1500 raw frames. The page shows the count, e.g. `take/run_... 412 saved`.

### 2. Harvest (PC, 2-3 min)

```
python tools/yolo/yolo.py harvest
```

This does, in one go:

- **pull:** new sessions from the Pi in one tar stream over ssh (already-pulled ones are skipped);
- **prep:**
  - drops dark, motion-blurred and near-duplicate frames;
  - pre-labels pillars with the current model and draws a first parking-lot box around the magenta walls;
  - keeps about 10% empty frames and caps the batch at 400 (`--max 600` for more);
- **upload:** sends the batch to Roboflow, or prints the zip to drag in by hand.

Open `datasets/upload/batch_..._preview.jpg` for a 5-second look at the pre-labels.

### 3. Review in Roboflow (10-15 min: the only slow step)

Annotate → the new batch. For each image: accept, move, add or delete boxes. Most images only need a glance.
With two people, split the batch with **Assign** (half each).

**Labelling rules** (consistency beats everything):

| Class | Box | Skip |
| --- | --- | --- |
| `GREEN PILLAR` / `RED PILLAR` | Tight around the visible pillar, top to bottom. Every pillar you can recognise, other lanes and far ones too. Cut by the frame edge: box the visible part | More than ~70% hidden, or smaller than ~12 px tall |
| `PARKING LOT` | **One box for the whole lot**: from the outer edge of one magenta wall to the outer edge of the other, full wall height. Only one wall visible: box what you see of the lot | Magenta tape or reflections on the mat, anything not a wall |

Frames with nothing in them stay empty; don't delete them. They teach the model "nothing here".

When done: **Add to dataset** with split **80 / 15 / 5** (train / valid / test).

### 4. Generate a version (1 min)

Versions → Generate:

- **Preprocessing:** Auto-Orient only. **Remove Resize.** The Pi letterboxes 640x480 frames to 416 keeping
  the aspect ratio; a "stretch to 640x640" version trains on squashed pillars.
- **Augmentation:** none (YOLO augments during training).

### 5. Train (Colab, 8-12 min)

Cell 1:

| Field | Setting |
| --- | --- |
| `VERSION` | 0 (the newest version) |
| `BASE` | `last model saved to Drive` (the second retrain onward), or `pillars26` for the first |
| `EPOCHS` | 60 (stops early when it stops improving) |

Then Runtime → **Run all**.

- **Cell 6** prints precision / recall / mAP50 per class. Pillars should be ≥ 0.9; `PARKING LOT` ≥ 0.8 after
  a few hundred labelled lot frames. Under 0.85 overall it warns: look for label mistakes before you trust it.
- **Cell 7** exports ncnn + onnx, checks the output format the Pi decoder needs, saves to Drive and downloads
  `pillars_<date>_<time>.zip`.

### 6. Deploy (PC, 1 min)

```
python tools/yolo/yolo.py deploy
```

It takes the newest zip in Downloads (or `deploy path\to\file.zip`), then:

- refuses a model without RED and GREEN classes;
- prints old-vs-new detections on the test frames and your latest batch;
- copies the model to the Pi and makes it active.

The running `obstacleRound.py` switches within 2 s, with no restart. The page's detector line shows the new name.

### 7. Check, or roll back

Drive one lap and watch the camera view: boxes on every pillar, and the lot boxed in magenta.

- Worse than before: `python tools/yolo/yolo.py activate pillars26`, or the previous model's name.
- `python tools/yolo/yolo.py models` lists what is on the Pi (`*` = active).

Repeat from step 1 at the next practice slot: new frames add to the same Roboflow project, and training starts
from the last model, so each round gets better.

---

## What the car does with each class

- **GREEN PILLAR / RED PILLAR:** unchanged; the largest one goes to the STM32 exactly as before.
- **PARKING LOT:** detected and drawn on the page, but **not used for driving yet** (there is no parking
  logic in the firmware). The detector matches classes by name, so adding it changes nothing for the pillars.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ssh failed` | Is the Pi on? `ssh suntzu@<ip>` by hand; update `pi_host` in `tools/yolo/config.json` |
| `Nothing recorded on the Pi` | `RECORD_RUNS` was off during the laps, or the Record button was never pressed |
| Recording stops by itself | SD card has less than `RECORD_MIN_FREE_MB` free. After pulling: `ssh suntzu@<ip> "rm -rf ~/BlueprintPi/take/*"` |
| Roboflow "Missing Labelmap" | Drag the **zip**, not the folder: the zip carries `data.yaml` |
| Roboflow boxes are the wrong class | A class name in Roboflow differs from `GREEN PILLAR`, `RED PILLAR`, `PARKING LOT` |
| Colab: no GPU | Runtime → Change runtime type → T4. If the free GPU quota is used up, another Google account works |
| Colab: cell 3 no key | Secret name must be exactly `ROBOFLOW_API_KEY`, with Notebook access on; or upload a YOLOv8 export zip when asked |
| No venue internet | Keep driving on the current model. Roboflow and Colab both need internet; everything on the Pi keeps working |
| Page still shows the old model | `yolo.py models`: is the `*` on the new one? Else restart `obstacleRound.py` |
| New model misses pillars that the old one saw | Roll back (step 7). Usually labels got missed in step 3: look at the Roboflow batch again |

## Files

| File | What |
| --- | --- |
| `obstacleRound.py` | Recorder (Tune → Recording, Run tab Record button); loads `models/<ACTIVE>` |
| `tools/yolo/yolo.py` | pull / prep / upload / harvest / deploy / activate / models / status |
| `tools/yolo/config.json` | Pi address, Roboflow workspace and project, batch settings |
| `tools/yolo/train_colab.ipynb` | The Colab notebook: 1 form, then Run all |
| `datasets/` (PC, not in git) | `raw/` pulled sessions, `upload/` batches and previews, `used.txt` |
| `take/` (Pi, not in git) | Recorded sessions |
