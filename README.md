# Battery Counter

Counts AA, AAA, CR2016 and LR44 batteries from one photo taken from above, measures each one
in millimetres against a credit card, and gives the count per battery and the grand total.
Classical image processing only (NumPy, SciPy, OpenCV), no machine learning.

## Run

Double-click `run.bat`, or:

    .venv\Scripts\python -m uvicorn app:app --host 127.0.0.1 --port 8300

then open http://127.0.0.1:8300

## Taking the photo

- Put the batteries on a sheet of white paper, spread out, with a credit card beside them.
  Any bank, debit or ID card works: they are all 85.60 x 53.98 mm (ISO/IEC 7810 ID-1).
- AA and AAA lie on their side, coin and button cells face up.
- Photograph from above. The table round the paper may show; it is cropped away.
- A coloured card works best. Even daylight is better than a lamp straight above.

## Pipeline (`ivp/steps.py`)

| Step | What it does |
|---|---|
| Load and check | reads the photo, Laplacian-variance sharpness check |
| Auto-crop | whiteness V(1-S), Otsu, closing/opening, largest component, convex hull, Douglas-Peucker 4 corners, Huber line fit per side, perspective warp |
| Clean | median and Gaussian filters |
| Light | background fitted and divided out |
| Find | colour distance from background, Otsu, shadow test (HSV) |
| Tidy | region filling, opening/closing, neck cuts |
| Scale | card found by aspect ratio and fill; sides fitted as lines; its 4 corners give a pixel-to-mm homography |
| Split | distance transform, h-maxima, watershed for touching objects |
| Measure | cylinder or round cell by elongation and circularity; sub-pixel edges; robust length and diameter from the two straight sides; equivalent diameter for cells |
| Identify | nearest catalogue size in tolerance units, ambiguity margin, Gaussian confidence; touching cylinders counted by size multiples |
| Leakage check | per battery, at the ends (cylinders) or face and rim (cells): white crust = bright + grey + rough (RMS of the positive 7x7 median residual), glare removed by grey-scale top-hat + "brighter than paper", green/rust corrosion by HSV colour slicing minus the body colour; opening, closing, connected components; deposit area in mm^2 -> OK / check / leaking |
| Count and total | count x price per battery, grand total; PDF, CSV and JPG |

## Catalogue

`data/battery_catalogue.csv`: `type` is `cylinder` (size_mm = length, second_mm = diameter)
or `button` (size_mm = diameter). Edit the prices there or upload a CSV on the Catalogue page.

## Tests

    .venv\Scripts\python tools\make_battery_scenes.py   # draws 3 test photos with known contents
    .venv\Scripts\python tools\test_batteries.py        # runs the pipeline and compares with the truth
    .venv\Scripts\python tools\tune_leakage.py          # picks the leak thresholds on tuning photos only
    .venv\Scripts\python tools\test_leakage.py          # leak accuracy on held-out photos -> data/tests/leak_report.json

The leakage check also has its own page (Leakage check) for close-ups, with or without a card.

Current result on the drawn photos: 3 of 3 exact (12 batteries, total 260.00 each),
confidence 79 to 99%. Not yet measured on real phone photos.
