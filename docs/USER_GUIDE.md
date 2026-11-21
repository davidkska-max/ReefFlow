# ReefFlow — using the browser UI

The UI runs entirely on your own computer. Nothing is sent to the internet.

## 1. Start it

Open a terminal in the ReefFlow folder and run:

    reefflow ui                  (or:  python -m reefflow.ui)

Your web browser opens at an address like `http://127.0.0.1:8765/`. Keep the terminal window open while you work; closing it, or pressing Ctrl+C in it, stops ReefFlow.

Your projects and results are saved in the folder `reefflow_workspace` next to where you started it. To use another folder, start with `reefflow ui --workspace D:\Reef\MyTanks`.

## 2. Open or create a project

The **Projects** button lists your tanks. **From 75-gal demo** creates the reference tank, a good place to start learning. **Blank tank** creates an empty one. **Import** opens an existing project file (YAML) together with its rock files (STL).

**Save** (or Ctrl+S) writes the project file. A yellow "● unsaved" mark means you have changes that are not saved yet. **Undo** and **Redo** (Ctrl+Z / Ctrl+Y) step through your edits. **YAML** downloads the project file.

The badge at top right is green ("valid") when the project is ready to run. Click it to see what needs fixing.

**US / SI** switches every number in the UI between inches/GPH and millimetres/L/h. The calculation is the same either way.

**Coordinates.** Positions are measured from the inside front-left-bottom corner of the glass:

- x runs along the length, left to right as seen from the front;
- y runs from the front glass to the back glass;
- z runs up.

The view shows a red x and a green y axis and the word FRONT.

**Tabs:** 1 Tank · 2 Photos (existing tank) · 3 Rockscape designer (new tank) · 4 Pumps & lights · 5 Run · 6 Results · 7 Compare.

**Mouse in the 3-D view.** Left-drag rotates, right-drag pans, and the wheel zooms. The **3D / Top / Front / Side** buttons jump to standard views.

**Finding your way.** The tabs along the top are the steps, in order. A green tick shows a step that is done for this project; Photos and Compare are optional. Click any section heading in the left panel to fold it away; ReefFlow remembers what you folded.

## 3. Tank (tab 1)

Enter the external size, the glass thickness, how far the water sits below the rim, and the sand depth. The internal size and the water volumes are calculated for you. Default simulation times and the dead-zone speed threshold are also set here.

## 4. Rockscape from photos (tab 2): existing tanks

Use this tab if your tank is already set up. It builds the 3-D rock from photos. For a new tank, use the Rockscape designer (tab 3) instead.

**Taking the photos**

- **Top (required):** turn the pumps off and wait until the surface is still. Shoot from above, so the whole sand bed and all four top corners of the tank are in the picture. Lift or swing the lights out of the way first: corners hidden behind a fixture cannot be clicked, and a top photo with a fit error over 6 px is not used to guide the side photos' masks.
- **Sides:** take a front photo and at least one end (left or right), from a metre or two away, with the whole tank in the picture: the top of the frame and where the tank sits on the stand.
- **Extra photo (recommended):** a fourth photo from an angle, for example a front photo from higher up and to one side, makes the result noticeably better. Add it as another "front" (it becomes "front-2").
- Use JPEG or PNG; HEIC from some phones must be exported as JPEG first. Large photos are reduced to 2000 px on the long side.
- If your camera stores its focal length (EXIF), ReefFlow offers to use it. Without it (photos sent from a phone browser lose it), the calibration starts from a typical phone main camera and adjusts it.

**Step 1 — Add the photos.** Choose the side (top, front, back, left, right), press **+ Add photo** and pick the file.

**Or take them with your phone (📱 Use my phone).** This is the easiest way when the rock is already in the tank.

1. The computer and the phone must be on the same home Wi-Fi or router.
2. Press **📱 Use my phone**. A QR code appears. Point the phone's camera at it and tap the link.
3. The phone shows a shot list: Front and Top (required), Left, Right, a second front photo from higher up (recommended: it stops the 3-D rock coming out too fat), and Back (optional). Each shot has a small diagram and tips for a **filled** or a **dry** tank; switch between them at the top.
4. Tap **Take photo** on a card. The photo goes straight to the computer and appears in the Photos list here within a couple of seconds. **Retake** replaces it.
5. Back on the computer, continue with Step 2 (click the tank corners).

Notes:
- The first time, Windows may ask whether Python may use the network. Allow **Private networks**. If the phone cannot open the page, check that Windows treats your Wi-Fi as a Private network.
- The link works only from devices on your home network (private addresses), only for this project, and only with the random code in the QR link. It can only add photos, and delete the ones the phone sent. It stops after an hour without use, when you press **Stop the link**, or when ReefFlow closes.
- iPhone HEIC photos are converted to JPEG on the phone. The phone's browser removes most of the photo's lens data (EXIF) when it uploads, so the focal length is worked out from your clicks, starting from a typical phone camera (26 mm equivalent).

**Step 2 — Calibrate (click the tank corners).**

First check the tank on the Tank tab, then press **Save**:

- **Tank style:** *Rimmed* (black frame at the top and bottom), *Rimless top, with a base frame*, or *Rimless* (no frame). It decides whether you click the corners of the frame or of the glass.
- **Outside size:** length, width and height, measured with a tape, over the frame if there is one (height from the stand to the top edge). A wrong height alone can put every click 20–50 px off.

The Photos tab uses the **saved** project. If the Tank tab has unsaved size changes, a red box says so, with a **Save project now** button. After a size change, every photo is calibrated again with your clicks.

1. Select a photo. The panel shows a small sketch of where each point is, and the next one is pink (in the sketch, the list and the "Next:" line).
2. Click that point on the photo. A magnifier in the corner of the photo helps you hit it exactly. The list moves on by itself.
3. Skip any point you can't see; click its row (or its dot in the sketch) later to place or re-place it. Mouse wheel zooms, right-drag (or Alt-drag) pans, and **F** fits the photo to the window.

For a front, back or side photo the points are:

- **Top corners** (2):
  - rimmed tank: the outside top corner of the black frame;
  - rimless top: the top corner of the glass itself, where the front glass meets the side glass.
- **Bottom corners** (2):
  - with a base frame: the bottom outside corner of the frame, where it sits on the stand (not where the glass enters the frame);
  - rimless: the bottom corner of the glass, on the stand or leveling mat.
- **Water line at the back corners** (2): through the water, where the water surface meets the back corner.
- **Back corner joints** (2): click anywhere on the vertical joint between the back glass and the side glass, low down is best. These are lines, not points, so any spot on the joint works.

For the top photo: the 4 outside top corners and the 4 corners where the sand meets the glass. The old inside corners (rim, bottom) are still under **More points**; they suit rimless tanks, but a frame or the sand usually hides them.

After 4 or more corner points, ReefFlow works out where the camera was, allowing for the light bending through the water and glass.

- Coloured lines show where it now thinks the tank is: the outside box, the water and sand lines, and the corner joints (purple). Check that they sit on the real edges.
- Before you click a point, a dashed pink ring shows where the current fit expects it.
- A red line from a click to its small circle is that click's fit error. An average under 3 px is good. A large one usually means a click is on the wrong point, or the tank size is wrong.
- **If the tank size looks wrong**, a red box says so, with the size that fits the photo better, for example "about 22.8 in tall, not the 21 in entered". Measure and correct it on the Tank tab, then Save; your clicks are kept.
  - One photo judges the height well but the depth only roughly (glass thickness alone shifts it by about an inch). If your tape disagrees, press **My measurements are right: stop asking**.
- **Clear all clicks** starts the photo again. If the photo still has clicks from the old corner list, a yellow box says so.

**Step 3 — Rock mask.**

1. Press **Suggest mask** (or **Suggest all**). ReefFlow colours the rock green; the top photo is done first and guides the side photos.
   - On front, back and side photos it looks for **texture**: porous rock is rough, while water, glass and the back wall are smooth. Tiny bubbles in the water are filtered out first. Low rubble on the sand counts when it is darker than the sand around it. The outline is only refined close to the rough areas it found, so it does not spread across open water. Under blue reef lights the rock and the water are almost the same colour, so colour alone does not work there. Only rough areas that reach the sand count, so equipment hanging from the rim (heaters, pumps, overflow boxes) is left out.
   - On the top photo it uses **colour**, because from above the sand is rough too.
   - **Find rock by** lets you switch to colour (or texture) for every photo.
   - **Sensitivity** (0–100, default 50) works like a selection tolerance. Move it up when the outline stops short, for example at rock resting on the sand; move it down when it takes in too much. It only moves the outline near the rock, so it cannot fill the open water, and openings such as arches stay open. Each step keeps everything the lower step had. It updates in about a second, but it replaces brush edits, so set it before you brush.
2. Check the green area carefully and correct it:
   - **Brush +** / **Brush −**: drag to paint rock in or out.
   - **Polygon + / −**: click the corners of an area, then double-click or press Enter to close it.
   - **Fill holes**, **Remove specks**, **Smooth**: quick clean-ups. **Undo** (Ctrl+Z) reverses the last change.

Typical things to fix: pumps or heaters in front of the rock or resting on the sand, corals sticking up from the rock (add them in if you want them in the model), coralline growth on the glass, and the sand being taken for rock.

**Step 4 — Carve.**

1. Choose a voxel size. 5 mm takes about half a minute; use no larger than the cell size of the simulations you plan.
2. Press **Carve**. The progress bar runs while ReefFlow removes everything that any photo shows is not rock.
3. When it finishes, the **3D result** opens.

**Step 5 — Fix caves and arches.** Photos only show outlines, so the reconstruction fills caves, arches, overhangs and gaps between rocks that no photo can see into. Open them up:

- **Tunnel:** click two or more points on the rock in the 3D view (an entry and an exit), set the radius, and press **Apply**.
- **Arch opening:** click the floor point in the middle of the opening, then set its width, height, length and direction.
- **Sphere / cube:** carve out a pocket, or switch the action to **add rock** to fill something in.

Edits are listed and can be removed one by one.

**Step 6 — Check against the photos.** Press **Update comparison**. Each photo is shown with the reconstruction drawn over it:

- green = the two agree;
- red = the reconstruction has rock the photo does not show (usually a filled cave, or a mask or calibration problem in that photo);
- blue = rock in the photo that the reconstruction is missing (another photo's mask cut it away).

The **silhouette IoU** (1.0 = perfect match) summarises each photo.

**Step 7 — Volume and weight.** ReefFlow reports the rock volume and an estimated dry weight. Set the bulk density to match your rock: dry reef rock is typically 1.1–1.8 kg/L, and the default is 1.4.

Treat this volume as an upper bound, because the hidden cavities are filled. In testing on a known rockscape, three photos overestimated the volume by about half (+47 %). Compare it with what you know you bought. If it is much higher, open hidden caves or add an angled photo.

**Step 8 — Use it.** **Use as rockscape** makes the reconstruction the project's rock, so you can go straight to Pumps and Run. It appears in the Rockscape designer as one imported piece, and you can add designer pieces on top and rebuild. **Download STL** saves the mesh.

Everything (photos, clicks, masks, edits and the result) is saved with the project and is still there after a restart. Changing the tank dimensions resets the calibrations: press the calibrate step again, and your clicks are kept.

## 5. Rockscape designer (tab 3)

1. Add pieces with **+ Rock**, **+ Slab / shelf**, **+ Pillar**, **+ Arch** and **+ Bridge**, bring in a phone scan of a real rock with **📱 Import phone scan…** (see below), or a finished model with **+ Import model…** (choose the unit the file was drawn in first).
2. Click a piece to select it. Drag the coloured arrows to move it. **Move / Rotate / Scale** (keys W / E / R) change what dragging does. You can also type exact numbers in the panel.
3. **Snap down** drops the piece onto the sand or the rock below it. **Duplicate** (Ctrl+D) copies it, and **Delete** (Del) removes it. The 🎲 button gives a rock a new random shape.
4. Press **Build rockscape**. This merges all pieces into one solid rock model, which is what the simulation uses. It reports the rock volume and an estimated dry weight; set the density to match your rock.
5. **Geometry checks** flag rock that touches the glass (no flow behind it, and no room to clean), rock that floats, rock that rises above the water, and groups whose centre of mass overhangs their footprint on the sand. These are warnings only, not a structural analysis.
6. **Export STL** downloads the merged rock, for example to 3-D print or to keep.

If you change the rock after building, the badge shows "rock not built". Build again before running.

### Importing a phone 3-D scan (dry rock)

If you are setting up a new tank, you can scan each dry rock with a phone app (Polycam, Scaniverse, 3d Scanner App, KIRI Engine and others) and bring it in as a real shape.

**Scanning**
- Put the rock on a plain table. Walk all the way round it, then go round again lower and higher. Phones with LiDAR (iPhone Pro, iPad Pro) give the true size; photo-only scans need a tape measurement (below).
- Crop the scan to the rock in the app if it caught the room. The table under the rock is fine: ReefFlow removes it.
- Export **OBJ** or **GLB** (PLY and STL also work, as does the app's ZIP export). USDZ is not supported.

**Clean-up (Rockscape tab → 📱 Import phone scan…)**
1. **Size.** ReefFlow guesses the unit (phone apps write metres) and shows the size of the rock. If it is wrong, or the scan had no LiDAR, measure the real rock, type one dimension under **Real size** and press **Scale to this**.
2. **Upright.** **Auto** finds the table under the rock and stands the rock on it. If the rock looks tipped over in the view, press another button (Z up, Y up and so on) until it stands right.
3. **Table or floor.** The part drawn in red is cut away: the table, and a few millimetres of the rock base so it stands flat. Raise or lower the cut with ▲ ▼ if needed.
4. **Clean up.** ReefFlow closes the holes the camera missed, closes the open bottom, drops loose bits (pieces of the room, floating specks), fills sealed hollow pockets, and makes a watertight rock. Tunnels and caves that are open to the outside stay open. It reports the volume, the estimated dry weight, how many holes were closed and how much was removed.
5. **Add to rockscape.** The cleaned rock becomes an imported element, standing on the sand in the middle of the tank. Move and turn it with the gizmo like any other piece; add more scans for the other rocks; then **Build rockscape**.

Detail **Auto** uses voxels of about 1/150 of the rock size (1.5–4 mm). A 26 cm rock takes about a second, and a 1-million-triangle scan about 6 s. The raw scan is kept with the project, so you can redo the clean-up with other settings.

**+ Import model…** (the older button) is for finished, watertight models, for example from CAD or a 3-D print file. It does no clean-up.

### Sculpting the rock (terrain-style brushes)

Press **Sculpt the rockscape** at the bottom of the Rockscape tab. This works on whatever the rock is now: built pieces, an imported scan, or a photo reconstruction. From the Photos tab, **Sculpt it with brushes →** takes you straight here with the reconstruction.

The brushes work like the terrain tools in park-builder games, seen from where you are looking:
- **1 Raise**: the rock under the circle comes toward you, most in the middle and fading to nothing at the edge. Scrub back and forth to build up gradually; each pass adds a little more. Hold still to grow a smooth dome. On the sand it grows rock up out of the sand.
- **2 Lower**: the same, away from you. Scrub to dig gradually; hold still to dig a hollow.
- **3 Smooth** rounds off bumps, steps and sharp edges.
- **4 Flatten** levels everything under the brush to the height where the stroke started. It cuts rock above that height and fills below it, so dragging past an edge builds a flat shelf for frags. With a **Fixed** direction it cuts a flat face square to that direction instead (for example a vertical face with front → back).
- **5 Tunnel** drills a straight, round hole. Hold the button and it goes deeper; it stops by itself once it breaks through the far side. Hold **Ctrl** with Tunnel to extrude an arm or ledge of rock toward you instead.

Speed: at 50 % strength, Raise and Lower move the rock 40 mm (1.6 in) per second at the centre of the brush; the panel shows the speed. Shift + wheel changes it. Raise never removes rock and Lower never adds any, and Raise stops 10 mm short of the glass and the water surface so water can still flow there. To change which way Raise and Lower push, turn the view: looking through the front glass they work front ↔ back, looking down from above they work up ↔ down.

**Tunnel direction.** Tunnel works along a straight line, not along the bumpy surface:
- **Straight in** (the default) uses the tank direction closest to how you are looking: front → back when you look through the front glass, straight down when you look from above. Use the **Front / Top / Side** view buttons to pick it quickly.
- **Fixed** uses a direction you choose: front → back, back → front, left → right, right → left, down or up. Turn it 15° at a time with **Q / E** and tilt it with **W / S**, for tunnels that run at an angle.

**Making an opening (cave or tunnel).**
1. Choose **5 Tunnel** and point at the rock. The arrow shows the direction, and a dashed line and a yellow ring show where the hole will come out. The label says how far that is ("through at 6.8 in").
2. Hold the left button until the label says **through ✓**, then let go. Or tick **Click bores through** (key **B**) and a single click drills all the way.
3. To make the opening bigger, use a bigger brush (mouse wheel) and drill again. To make it wiggle, drill again from the other end with a turned direction.

Dragging while you hold Tunnel cuts a groove (Extrude: a ridge) that follows the rock's original surface.

| Control | Effect |
|---|---|
| Left drag on the rock | sculpt; scrub to build up, hold still to keep going |
| Left drag off the rock, or right drag | orbit the view |
| Middle drag | pan |
| Mouse wheel, or `[` and `]` | brush size |
| Shift + wheel | strength (speed) |
| Ctrl + wheel | zoom |
| 1 2 3 4 5 | Raise, Lower, Smooth, Flatten, Tunnel |
| Hold Ctrl | Raise ↔ Lower; Tunnel → Extrude |
| Hold Shift | Smooth |
| Q / E, W / S | Tunnel: turn / tilt the direction 15° |
| V | Tunnel: Straight in |
| B | Tunnel: Click bores through on or off |
| Ctrl+Z / Ctrl+Y | undo / redo strokes |

The mouse pointer shows the tool in use: a crosshair with a coloured badge (+ Raise, − Lower, wave Smooth, level mark Flatten, ring Tunnel, arrow Extrude); over empty space it turns into a hand, meaning a drag orbits the view. The circle on the rock shows where the brush will act. Caves and holes are drawn darker, so you can see the openings from any angle.

**Apply to project** turns the sculpt into the project's rock model. Runs and lighting use it from then on, and the panel shows the same geometry checks as Build. **Save draft** keeps your work without changing the project. Either way, the next sculpt session picks up exactly where you left off. **Start over** reloads the project's current rock.

Two things to know:
- **Resolution.** The sculpt resolution (4 mm by default) only sets how fine you can sculpt. The flow model always sees the rock at its own cell size.
- **Small caves and gaps.** A cave or gap narrower than about 3 flow cells barely carries water in the simulation: that is 15 mm at Standard and 30 mm at Preview.

## 6. Pumps and lights (tab 4)

**Real equipment.** Every **Model** list (pumps, overflow, lights) has real products grouped by brand, plus the generic types at the top. Picking a product fills in its published specs, and a card under the list shows each number, where it came from (the maker, a retailer, read from the maker's chart, or estimated, and how), when it was checked, and links to the sources. **Notes on the data** opens the details. Discontinued models are marked. You can still type over any number.

- Powerheads: the rated flow is the maker's figure. Most makers do not publish the outlet size; the card says when it is estimated (A-844).
- Return pumps: when the maker publishes a head-flow curve, ReefFlow uses it to get the flow at your head (static head + fittings allowance). DC pumps without a curve use a straight-line curve between the maximum flow and the maximum head, which matches the published DC curves better than the old generic shape (D-845). If you type a different maximum flow, the published curve no longer applies.
- Lights: no maker publishes the total light output, so ReefFlow works it out from published PAR readings where there are any, or estimates it from the wattage; the card says which. Measured PAR readings (from the maker or independent tests) are listed so you can compare. If you have a PAR meter, calibrating beats all of this.
- Overflow: a product sets the box width (and depth for internal boxes) and its **drain capacity**. ReefFlow warns when the return flow is higher than that capacity; **Typical drain capacity by pipe size** lists commonly quoted figures (they disagree by up to 2×).

**Pumps**

- Choose a model (or a generic type) and press **+ Add**.
- Click a pump to select it. In Move mode you drag its outlet; in Rotate mode you drag to aim it. You can also type the aim: azimuth is the direction in plan view (0° = toward the right glass, 90° = toward the back), and elevation is up (+) or down (−).
- **Snap to glass** puts the outlet 45 mm off the nearest side glass.
- **Rated flow**: use the manufacturer's number. **Derating** (default 0.7) accounts for the fact that rated flows are free-flow figures and overstate what the pump really delivers. The delivered flow, nozzle velocity and jet momentum are shown beneath.
- **Mode**: constant, sine (smooth wave) or pulse. Two pumps with the same settings and phases 0 and 0.5 alternate.
- **Return nozzle**: enter the pump's zero-head flow and the head (sump to tank, plus fittings). ReefFlow estimates the real flow from a generic pump curve. A return needs an **overflow**. ReefFlow warns when the water leaves a nozzle so fast that its exit head alone is more than your fittings allowance (then the real flow would be lower).
- **Several return nozzles on one pump**: select the return pump and press **+ Add nozzle**. Each nozzle is its own arrow in the view; click one (or its row in the nozzle table) to move or aim it. The pump's total flow is shared between the nozzles: by default in proportion to their outlet areas (equal nozzles share equally). You can give a nozzle its own diameter or a fixed share of the flow. The table shows each nozzle's flow, exit speed and exit head. A dashed line links the nozzles of one pump.
  - Adding nozzles does not change the total flow by itself: the nozzles' exit losses are assumed to be part of the **fittings allowance**. If more nozzles really lower your losses, lower the allowance.

**Overflow**

- Tick **Tank has an overflow**, then pick the glass it sits on (back, front, left or right) and where along that glass its centre is.
- To move it with the mouse, press **Drag in view** (or click the box in the view): the arrow slides it along its glass only, and it stops at the corners.
- Buttons put it flush in either corner, in the middle of the glass, or make it **coast to coast** (full length of that glass).
- Also set its width along the glass, how far it sticks out, and how far the weir sits below the water.
- Pick a **Model** to use a real overflow box, or type its **Drain capacity** to get a warning when the return pump sends more water than it can take.

**Lights**

Add fixtures (pick a real model or a generic type), then set their position over the tank, height above the water, tilt and intensity. The yellow cone shows the beam. Finer settings (channels, photoperiod, water clarity, calibration) live in the project file for now.

## 7. Run (tab 5)

1. Choose a **Resolution preset**:
   - **Preview** (10 mm cells, minutes) for quick checks;
   - **Standard** (5 mm) for design decisions;
   - **Fine** (3 mm, can take hours) for final confirmation.
2. Press **Estimate** to see the grid size, memory and expected run time, and notes on how each pump is modelled. Very small outlets are modelled as slightly larger ones that keep the jet's momentum. This is normal at coarse resolution.
3. Tick **Compute PAR** to include lighting when the project has fixtures. Then press **Start run**.
4. Progress shows as a bar with a live log. Runs wait in a queue and use the graphics card one at a time. **Cancel** stops a run. After the flow simulation, the analysis (dead zones, water age, sand shear, coral scores, report) runs automatically. Then press **Open results**.

## 8. Results (tab 6)

Pick a run. Then use these layers:

- **Slices**: horizontal (plan), front-elevation and side-elevation cuts through the water. Slide them to any position. **Slice field** picks what is coloured: speed, turbulence, water age (how long water has been in the tank since the return brought it in; high age means stagnant), or PAR when the run included lighting. Point at a slice to read the value at that spot.
- **Velocity vectors** show the flow direction on the slices.
- **Dead-zone overlay** colours red the water that moves slower than your threshold.
- **Sand bed shear** shows where detritus settles (low shear) and where the sand gets blown around (high shear).
- **Rock display: coral score** colours the rock for the chosen coral profile (SPS / LPS / soft), from poor to ideal. Numbered pink markers are the best spots; click a row in the table to look at one. Grey rock (undersides) is not scored.

The panel lists the key numbers, each with its statistical uncertainty. It also lists the largest dead zones with their locations, the sand and detritus figures, and coral scores for each profile.

**Open report** gives the full printable report with all assumptions. **Re-analyze** repeats the analysis, for example after changing thresholds.

## 9. Compare (tab 7)

This is how you make design decisions:

1. Run a baseline.
2. Copy the project (Projects → Copy) or change one thing: move or re-aim a pump, or edit the rock.
3. Run again.
4. In the Compare tab, choose the two runs and press **Compare**. The table shows each metric for A and B, the change, and whether the change is larger than the statistical noise. The image shows where the water got faster or slower and where dead zones appeared or disappeared.

## Good to know

- The coral profiles and thresholds are starting points, not husbandry rules. Edit them in the project file.
- The model's assumptions (rigid flat water surface, solid rock by default, generic pump curves, and so on) are listed in the report for every run.
