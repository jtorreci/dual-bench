# -*- coding: utf-8 -*-
"""Reference results from Paper A (Torre-Cifuentes, 2026).

Experimental design: 5 problems × 5 formulations × 6 algorithms × 20 reps
= 3,000 runs.  NFE = 25,000 per run.  Platypus engine, pop_size = 100.

Metrics
-------
- ``hv_mean`` / ``hv_std``: 2-D hypervolume (ref = [1.1, 1.1]).
- ``df_mean`` / ``df_std``: Δf* = (f_best - f*) / f*  among feasible
  solutions only.  ``NaN`` means no feasible solution was found.

Two levels of aggregation are provided:

1. :func:`by_algorithm` — per (problem, formulation, algorithm), n=20.
2. :func:`by_formulation` — per (problem, formulation), n=120 (pooled).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# ======================================================================
# Constants
# ======================================================================

PROBLEMS = ["Spring", "PressureVessel", "WeldedBeam",
            "SpeedReducer", "CantileverBeam"]

FORMULATIONS = ["MO", "2OB", "2OB-REST", "FOB", "FOB-REST"]

ALGORITHMS = ["NSGAII", "GDE3", "SPEA2", "SMPSO", "NSGAIII", "MOEAD"]

NFE = 25_000
POP_SIZE = 100
N_REPS = 20
ENGINE = "Platypus"

# ======================================================================
# Per-algorithm data  (problem, formulation, algorithm) -> metrics
# ======================================================================
# Each value: (hv_mean, hv_std, df_mean, df_std, n)
# df_mean/df_std = None when no feasible solution was found in any rep.

_BY_ALGORITHM: Dict[Tuple[str, str, str], Tuple[float, float, Optional[float], Optional[float], int]] = {
    # --- Spring ---
    ("Spring", "MO", "NSGAII"):      (0.6903, 0.0107, 0.0837, 0.1059, 20),
    ("Spring", "MO", "GDE3"):        (0.7114, 0.0085, 0.0479, 0.0374, 20),
    ("Spring", "MO", "SPEA2"):       (0.6956, 0.0054, 0.0696, 0.0669, 20),
    ("Spring", "MO", "SMPSO"):       (0.6969, 0.0037, 0.0181, 0.0362, 20),
    ("Spring", "MO", "NSGAIII"):     (0.6785, 0.0141, 0.1995, 0.1389, 20),
    ("Spring", "MO", "MOEAD"):       (0.6826, 0.0131, 0.1602, 0.1297, 20),
    ("Spring", "2OB", "NSGAII"):     (0.8176, 0.0056, 0.0895, 0.0504, 20),
    ("Spring", "2OB", "GDE3"):       (0.8174, 0.0018, 0.0526, 0.0142, 20),
    ("Spring", "2OB", "SPEA2"):      (0.8105, 0.0056, 0.4936, 0.3023, 20),
    ("Spring", "2OB", "SMPSO"):      (0.8210, 0.0003, 0.0727, 0.0325, 20),
    ("Spring", "2OB", "NSGAIII"):    (0.8160, 0.0034, 0.0623, 0.0368, 20),
    ("Spring", "2OB", "MOEAD"):      (0.7896, 0.0026, 0.1602, 0.0478, 20),
    ("Spring", "2OB-REST", "NSGAII"):  (0.7537, 0.0037, 0.0489, 0.0350, 20),
    ("Spring", "2OB-REST", "GDE3"):    (0.7533, 0.0050, 0.0568, 0.0477, 20),
    ("Spring", "2OB-REST", "SPEA2"):   (0.7428, 0.0117, 0.1530, 0.1160, 20),
    ("Spring", "2OB-REST", "SMPSO"):   (0.7555, 0.0018, 0.0368, 0.0177, 20),
    ("Spring", "2OB-REST", "NSGAIII"): (0.7529, 0.0028, 0.0499, 0.0237, 20),
    ("Spring", "2OB-REST", "MOEAD"):   (0.7451, 0.0077, 0.1278, 0.0732, 20),
    ("Spring", "FOB", "NSGAII"):     (0.7600, 0.0169, 2.1903, 1.7488, 20),
    ("Spring", "FOB", "GDE3"):       (0.7655, 0.0140, 3.1334, 3.0307, 20),
    ("Spring", "FOB", "SPEA2"):      (0.7774, 0.0076, 1.5799, 0.7271, 20),
    ("Spring", "FOB", "SMPSO"):      (0.7663, 0.0139, 2.3507, 1.5648, 20),
    ("Spring", "FOB", "NSGAIII"):    (0.7977, 0.0042, 0.1923, 0.2172, 20),
    ("Spring", "FOB", "MOEAD"):      (0.7952, 0.0060, 0.6996, 0.3645, 20),
    ("Spring", "FOB-REST", "NSGAII"):  (0.7455, 0.0052, 0.0441, 0.0439, 20),
    ("Spring", "FOB-REST", "GDE3"):    (0.7443, 0.0047, 0.0629, 0.0447, 20),
    ("Spring", "FOB-REST", "SPEA2"):   (0.7272, 0.0101, 0.2196, 0.1009, 20),
    ("Spring", "FOB-REST", "SMPSO"):   (0.7458, 0.0047, 0.0492, 0.0424, 20),
    ("Spring", "FOB-REST", "NSGAIII"): (0.7366, 0.0111, 0.1398, 0.1077, 20),
    ("Spring", "FOB-REST", "MOEAD"):   (0.7450, 0.0029, 0.0272, 0.0258, 20),
    # --- PressureVessel (df* recomputed with f*=5885.33, continuous formulation) ---
    ("PressureVessel", "MO", "NSGAII"):      (0.6987, 0.0060, 0.0842, 0.0560, 20),
    ("PressureVessel", "MO", "GDE3"):        (0.6908, 0.0075, 0.1952, 0.0499, 20),
    ("PressureVessel", "MO", "SPEA2"):       (0.6988, 0.0052, 0.1081, 0.0562, 20),
    ("PressureVessel", "MO", "SMPSO"):       (0.6979, 0.0042, 0.0884, 0.0394, 20),
    ("PressureVessel", "MO", "NSGAIII"):     (0.6953, 0.0067, 0.1136, 0.0695, 20),
    ("PressureVessel", "MO", "MOEAD"):       (0.6969, 0.0071, 0.1012, 0.0698, 20),
    ("PressureVessel", "2OB", "NSGAII"):     (0.8207, 0.0093, 0.5668, 0.2431, 20),
    ("PressureVessel", "2OB", "GDE3"):       (0.8322, 0.0032, 0.3974, 0.2543, 20),
    ("PressureVessel", "2OB", "SPEA2"):      (0.8489, 0.0028, 0.1289, 0.0367, 20),
    ("PressureVessel", "2OB", "SMPSO"):      (0.8236, 0.0068, 0.5044, 0.2301, 20),
    ("PressureVessel", "2OB", "NSGAIII"):    (0.8369, 0.0048, 0.1866, 0.1218, 20),
    ("PressureVessel", "2OB", "MOEAD"):      (0.6970, 0.0244, 0.1174, 0.0021, 20),
    ("PressureVessel", "2OB-REST", "NSGAII"):  (0.7661, 0.0067, 0.1437, 0.0660, 20),
    ("PressureVessel", "2OB-REST", "GDE3"):    (0.7619, 0.0058, 0.1897, 0.0588, 20),
    ("PressureVessel", "2OB-REST", "SPEA2"):   (0.7668, 0.0060, 0.1432, 0.0615, 20),
    ("PressureVessel", "2OB-REST", "SMPSO"):   (0.7675, 0.0058, 0.1341, 0.0601, 20),
    ("PressureVessel", "2OB-REST", "NSGAIII"): (0.7663, 0.0054, 0.1451, 0.0553, 20),
    ("PressureVessel", "2OB-REST", "MOEAD"):   (0.7019, 0.0082, 0.0985, 0.0547, 20),
    ("PressureVessel", "FOB", "NSGAII"):     (0.6605, 0.0455, 21.587, 12.062, 20),
    ("PressureVessel", "FOB", "GDE3"):       (0.6472, 0.0406, 29.515, 20.399, 20),
    ("PressureVessel", "FOB", "SPEA2"):      (0.3784, 0.1530, 10.733, 5.5673, 20),
    ("PressureVessel", "FOB", "SMPSO"):      (0.6780, 0.0338, 18.400, 12.290, 20),
    ("PressureVessel", "FOB", "NSGAIII"):    (0.7820, 0.0160, 0.9798, 0.6143, 20),
    ("PressureVessel", "FOB", "MOEAD"):      (0.1467, 0.1707, None, None, 20),
    ("PressureVessel", "FOB-REST", "NSGAII"):  (0.7323, 0.0092, 0.2036, 0.0790, 20),
    ("PressureVessel", "FOB-REST", "GDE3"):    (0.7111, 0.0178, 0.3714, 0.1808, 20),
    ("PressureVessel", "FOB-REST", "SPEA2"):   (0.6487, 0.0402, 0.9272, 0.4349, 20),
    ("PressureVessel", "FOB-REST", "SMPSO"):   (0.6827, 0.0268, 0.6866, 0.2690, 20),
    ("PressureVessel", "FOB-REST", "NSGAIII"): (0.6004, 0.0379, 1.6293, 0.3958, 20),
    ("PressureVessel", "FOB-REST", "MOEAD"):   (0.7029, 0.0044, 0.0598, 0.0269, 20),
    # --- WeldedBeam ---
    ("WeldedBeam", "MO", "NSGAII"):      (0.7692, 0.0048, 0.1268, 0.1772, 20),
    ("WeldedBeam", "MO", "GDE3"):        (0.7967, 0.0084, 0.2748, 0.1037, 20),
    ("WeldedBeam", "MO", "SPEA2"):       (0.7810, 0.0051, 0.1717, 0.1383, 20),
    ("WeldedBeam", "MO", "SMPSO"):       (0.7712, 0.0018, 0.0443, 0.0639, 20),
    ("WeldedBeam", "MO", "NSGAIII"):     (0.7617, 0.0060, 0.3880, 0.2192, 20),
    ("WeldedBeam", "MO", "MOEAD"):       (0.7668, 0.0064, 0.2472, 0.2232, 20),
    ("WeldedBeam", "2OB", "NSGAII"):     (0.9136, 0.0032, 0.3811, 0.1350, 20),
    ("WeldedBeam", "2OB", "GDE3"):       (0.9223, 0.0010, 0.1552, 0.0793, 20),
    ("WeldedBeam", "2OB", "SPEA2"):      (0.9232, 0.0017, 0.2697, 0.1381, 20),
    ("WeldedBeam", "2OB", "SMPSO"):      (0.9193, 0.0014, 0.3225, 0.1184, 20),
    ("WeldedBeam", "2OB", "NSGAIII"):    (0.7265, 0.0781, 10.784, 2.2965, 20),
    ("WeldedBeam", "2OB", "MOEAD"):      (0.9192, 0.0020, 0.0400, 0.0199, 20),
    ("WeldedBeam", "2OB-REST", "NSGAII"):  (0.9121, 0.0006, 0.0371, 0.0194, 20),
    ("WeldedBeam", "2OB-REST", "GDE3"):    (0.9091, 0.0020, 0.1275, 0.0602, 20),
    ("WeldedBeam", "2OB-REST", "SPEA2"):   (0.9087, 0.0030, 0.1276, 0.0962, 20),
    ("WeldedBeam", "2OB-REST", "SMPSO"):   (0.9083, 0.0022, 0.1780, 0.0780, 20),
    ("WeldedBeam", "2OB-REST", "NSGAIII"): (0.9098, 0.0011, 0.0900, 0.0370, 20),
    ("WeldedBeam", "2OB-REST", "MOEAD"):   (0.9040, 0.0044, 0.0570, 0.0794, 20),
    ("WeldedBeam", "FOB", "NSGAII"):     (0.7975, 0.0537, 9.8428, 8.3340, 20),
    ("WeldedBeam", "FOB", "GDE3"):       (0.7450, 0.0439, 20.725, 10.889, 20),
    ("WeldedBeam", "FOB", "SPEA2"):      (0.8558, 0.0149, 3.4369, 2.5553, 20),
    ("WeldedBeam", "FOB", "SMPSO"):      (0.7151, 0.0561, 17.272, 9.6703, 20),
    ("WeldedBeam", "FOB", "NSGAIII"):    (0.8849, 0.0107, 0.3516, 0.2880, 20),
    ("WeldedBeam", "FOB", "MOEAD"):      (0.9022, 0.0065, 0.6698, 0.3073, 20),
    ("WeldedBeam", "FOB-REST", "NSGAII"):  (0.8853, 0.0087, 0.0775, 0.0378, 20),
    ("WeldedBeam", "FOB-REST", "GDE3"):    (0.8858, 0.0079, 0.1436, 0.0573, 20),
    ("WeldedBeam", "FOB-REST", "SPEA2"):   (0.8796, 0.0074, 0.4652, 0.2436, 20),
    ("WeldedBeam", "FOB-REST", "SMPSO"):   (0.8714, 0.0116, 0.2903, 0.0807, 20),
    ("WeldedBeam", "FOB-REST", "NSGAIII"): (0.8821, 0.0051, 0.6241, 0.1438, 20),
    ("WeldedBeam", "FOB-REST", "MOEAD"):   (0.8949, 0.0078, 0.0625, 0.0903, 20),
    # --- SpeedReducer ---
    ("SpeedReducer", "MO", "NSGAII"):      (0.5604, 0.0000, 0.0001, 0.0001, 20),
    ("SpeedReducer", "MO", "GDE3"):        (0.5604, 0.0000, 0.0000, 0.0000, 20),
    ("SpeedReducer", "MO", "SPEA2"):       (0.5604, 0.0000, 0.0004, 0.0002, 20),
    ("SpeedReducer", "MO", "SMPSO"):       (0.5597, 0.0004, 0.0033, 0.0017, 20),
    ("SpeedReducer", "MO", "NSGAIII"):     (0.5604, 0.0000, 0.0000, 0.0000, 20),
    ("SpeedReducer", "MO", "MOEAD"):       (0.5604, 0.0000, 0.0001, 0.0000, 20),
    ("SpeedReducer", "2OB", "NSGAII"):     (0.6134, 0.0007, 0.0051, 0.0021, 20),
    ("SpeedReducer", "2OB", "GDE3"):       (0.6136, 0.0004, 0.0068, 0.0019, 20),
    ("SpeedReducer", "2OB", "SPEA2"):      (0.6124, 0.0012, 0.0056, 0.0037, 20),
    ("SpeedReducer", "2OB", "SMPSO"):      (0.6136, 0.0000, 0.0100, 0.0028, 20),
    ("SpeedReducer", "2OB", "NSGAIII"):    (0.6135, 0.0003, 0.0072, 0.0027, 20),
    ("SpeedReducer", "2OB", "MOEAD"):      (0.5930, 0.0062, None, None, 20),
    ("SpeedReducer", "2OB-REST", "NSGAII"):  (0.5639, 0.0002, 0.0018, 0.0007, 20),
    ("SpeedReducer", "2OB-REST", "GDE3"):    (0.5639, 0.0002, 0.0019, 0.0008, 20),
    ("SpeedReducer", "2OB-REST", "SPEA2"):   (0.5635, 0.0002, 0.0033, 0.0010, 20),
    ("SpeedReducer", "2OB-REST", "SMPSO"):   (0.5636, 0.0003, 0.0028, 0.0014, 20),
    ("SpeedReducer", "2OB-REST", "NSGAIII"): (0.5638, 0.0002, 0.0020, 0.0007, 20),
    ("SpeedReducer", "2OB-REST", "MOEAD"):   (0.5614, 0.0007, 0.0001, 0.0001, 20),
    ("SpeedReducer", "FOB", "NSGAII"):     (0.6031, 0.0035, 0.5154, 0.3225, 20),
    ("SpeedReducer", "FOB", "GDE3"):       (0.6030, 0.0026, 0.5418, 0.4925, 20),
    ("SpeedReducer", "FOB", "SPEA2"):      (0.5901, 0.0082, 0.6513, 0.1175, 20),
    ("SpeedReducer", "FOB", "SMPSO"):      (0.6015, 0.0026, 0.9484, None, 20),
    ("SpeedReducer", "FOB", "NSGAIII"):    (0.5453, 0.0169, 0.1628, 0.0643, 20),
    ("SpeedReducer", "FOB", "MOEAD"):      (0.5858, 0.0097, None, None, 20),
    ("SpeedReducer", "FOB-REST", "NSGAII"):  (0.5629, 0.0007, 0.0049, 0.0026, 20),
    ("SpeedReducer", "FOB-REST", "GDE3"):    (0.5618, 0.0010, 0.0100, 0.0044, 20),
    ("SpeedReducer", "FOB-REST", "SPEA2"):   (0.5578, 0.0034, 0.0259, 0.0142, 20),
    ("SpeedReducer", "FOB-REST", "SMPSO"):   (0.5603, 0.0012, 0.0164, 0.0052, 20),
    ("SpeedReducer", "FOB-REST", "NSGAIII"): (0.5571, 0.0023, 0.0293, 0.0098, 20),
    ("SpeedReducer", "FOB-REST", "MOEAD"):   (0.5620, 0.0007, 0.0001, 0.0000, 20),
    # --- CantileverBeam ---
    ("CantileverBeam", "MO", "NSGAII"):      (0.6923, 0.0004, 0.0051, 0.0037, 20),
    ("CantileverBeam", "MO", "GDE3"):        (0.7204, 0.0139, 0.0053, 0.0027, 20),
    ("CantileverBeam", "MO", "SPEA2"):       (0.7003, 0.0028, 0.0064, 0.0034, 20),
    ("CantileverBeam", "MO", "SMPSO"):       (0.6917, 0.0010, 0.0104, 0.0093, 20),
    ("CantileverBeam", "MO", "NSGAIII"):     (0.6916, 0.0008, 0.0112, 0.0079, 20),
    ("CantileverBeam", "MO", "MOEAD"):       (0.6936, 0.0027, 0.0085, 0.0061, 20),
    ("CantileverBeam", "2OB", "NSGAII"):     (0.8263, 0.0127, 0.3992, 0.1830, 20),
    ("CantileverBeam", "2OB", "GDE3"):       (0.8106, 0.0153, 0.5100, 0.1985, 20),
    ("CantileverBeam", "2OB", "SPEA2"):      (0.7235, 0.0595, 1.0614, 0.4157, 20),
    ("CantileverBeam", "2OB", "SMPSO"):      (0.8545, 0.0095, 0.1563, 0.1076, 20),
    ("CantileverBeam", "2OB", "NSGAIII"):    (0.2794, 0.2665, 16.811, 7.0408, 20),
    ("CantileverBeam", "2OB", "MOEAD"):      (0.8771, 0.0003, 0.0094, 0.0044, 20),
    ("CantileverBeam", "2OB-REST", "NSGAII"):  (0.8504, 0.0007, 0.0095, 0.0059, 20),
    ("CantileverBeam", "2OB-REST", "GDE3"):    (0.8341, 0.0062, 0.1485, 0.0534, 20),
    ("CantileverBeam", "2OB-REST", "SPEA2"):   (0.6829, 0.0964, 1.3103, 0.7405, 20),
    ("CantileverBeam", "2OB-REST", "SMPSO"):   (0.8450, 0.0049, 0.0585, 0.0421, 20),
    ("CantileverBeam", "2OB-REST", "NSGAIII"): (0.8418, 0.0041, 0.0744, 0.0421, 20),
    ("CantileverBeam", "2OB-REST", "MOEAD"):   (0.8482, 0.0004, 0.0061, 0.0037, 20),
    ("CantileverBeam", "FOB", "NSGAII"):     (0.8773, 0.0005, 0.0319, 0.0169, 20),
    ("CantileverBeam", "FOB", "GDE3"):       (0.8518, 0.0210, 0.1939, 0.1520, 20),
    ("CantileverBeam", "FOB", "SPEA2"):      (0.6997, 0.0890, 1.2475, 0.6996, 20),
    ("CantileverBeam", "FOB", "SMPSO"):      (0.8788, 0.0004, 0.0282, 0.0097, 20),
    ("CantileverBeam", "FOB", "NSGAIII"):    (0.8769, 0.0011, 0.0206, 0.0087, 20),
    ("CantileverBeam", "FOB", "MOEAD"):      (0.8769, 0.0002, 0.0069, 0.0020, 20),
    ("CantileverBeam", "FOB-REST", "NSGAII"):  (0.8502, 0.0009, 0.0112, 0.0076, 20),
    ("CantileverBeam", "FOB-REST", "GDE3"):    (0.8339, 0.0087, 0.1523, 0.0730, 20),
    ("CantileverBeam", "FOB-REST", "SPEA2"):   (0.6572, 0.0830, 1.5119, 0.6296, 20),
    ("CantileverBeam", "FOB-REST", "SMPSO"):   (0.8469, 0.0036, 0.0416, 0.0289, 20),
    ("CantileverBeam", "FOB-REST", "NSGAIII"): (0.8417, 0.0045, 0.0874, 0.0411, 20),
    ("CantileverBeam", "FOB-REST", "MOEAD"):   (0.8484, 0.0005, 0.0039, 0.0036, 20),
}

# ======================================================================
# Per-formulation data  (problem, formulation) -> metrics
# ======================================================================
# Each value: (hv_mean, hv_std, hv_median, df_mean, df_std, df_median, n)

_BY_FORMULATION: Dict[Tuple[str, str], Tuple[float, float, float, float, float, float, int]] = {
    ("Spring", "MO"):       (0.6925, 0.0145, 0.6966, 0.0965, 0.1129, 0.0472, 120),
    ("Spring", "2OB"):      (0.8120, 0.0112, 0.8169, 0.1552, 0.2003, 0.0811, 120),
    ("Spring", "2OB-REST"): (0.7506, 0.0079, 0.7534, 0.0789, 0.0755, 0.0503, 120),
    ("Spring", "FOB"):      (0.7770, 0.0185, 0.7779, 1.6746, 1.8447, 1.1355, 120),
    ("Spring", "FOB-REST"): (0.7407, 0.0098, 0.7447, 0.0905, 0.0957, 0.0456, 120),
    # PressureVessel df* recomputed with f*=5885.33 (continuous formulation)
    ("PressureVessel", "MO"):       (0.6964, 0.0067, 0.6977, 0.1151, 0.0679, 0.0994, 120),
    ("PressureVessel", "2OB"):      (0.8099, 0.0527, 0.8304, 0.3498, 0.2582, 0.2581, 120),
    ("PressureVessel", "2OB-REST"): (0.7551, 0.0248, 0.7647, 0.1424, 0.0640, 0.1405, 120),
    ("PressureVessel", "FOB"):      (0.5488, 0.2386, 0.6524, 16.243, 15.436, 13.794, 120),
    ("PressureVessel", "FOB-REST"): (0.6797, 0.0511, 0.6991, 0.6463, 0.5937, 0.4792, 120),
    ("WeldedBeam", "MO"):       (0.7744, 0.0129, 0.7717, 0.2088, 0.1958, 0.1647, 120),
    ("WeldedBeam", "2OB"):      (0.8873, 0.0788, 0.9193, 1.9921, 4.0564, 0.2686, 120),
    ("WeldedBeam", "2OB-REST"): (0.9087, 0.0035, 0.9096, 0.1029, 0.0811, 0.0819, 120),
    ("WeldedBeam", "FOB"):      (0.8167, 0.0791, 0.8408, 8.7163, 10.493, 5.1504, 120),
    ("WeldedBeam", "FOB-REST"): (0.8832, 0.0108, 0.8839, 0.2772, 0.2439, 0.1840, 120),
    ("SpeedReducer", "MO"):       (0.5603, 0.0003, 0.5604, 0.0006, 0.0014, 0.0001, 120),
    ("SpeedReducer", "2OB"):      (0.6099, 0.0080, 0.6136, 0.0070, 0.0032, 0.0068, 120),
    ("SpeedReducer", "2OB-REST"): (0.5633, 0.0010, 0.5637, 0.0020, 0.0013, 0.0019, 120),
    ("SpeedReducer", "FOB"):      (0.5881, 0.0221, 0.5985, 0.5279, 0.3150, 0.5939, 120),
    ("SpeedReducer", "FOB-REST"): (0.5603, 0.0029, 0.5610, 0.0144, 0.0130, 0.0123, 120),
    ("CantileverBeam", "MO"):       (0.6983, 0.0119, 0.6926, 0.0078, 0.0064, 0.0063, 120),
    ("CantileverBeam", "2OB"):      (0.7286, 0.2345, 0.8182, 3.1578, 6.7572, 0.4117, 120),
    ("CantileverBeam", "2OB-REST"): (0.8171, 0.0718, 0.8442, 0.2679, 0.5568, 0.0603, 120),
    ("CantileverBeam", "FOB"):      (0.8436, 0.0748, 0.8769, 0.2548, 0.5336, 0.0303, 120),
    ("CantileverBeam", "FOB-REST"): (0.8131, 0.0778, 0.8458, 0.3014, 0.6022, 0.0474, 120),
}


# ======================================================================
# Public API
# ======================================================================

def by_algorithm(problem: str = None, formulation: str = None,
                 algorithm: str = None):
    """Return Paper A results at the per-algorithm level.

    Each result is a dict with keys: ``problem``, ``formulation``,
    ``algorithm``, ``hv_mean``, ``hv_std``, ``df_mean``, ``df_std``, ``n``.

    Parameters
    ----------
    problem, formulation, algorithm : str, optional
        Filter results.  If ``None``, all values are returned.

    Returns
    -------
    list of dict
    """
    rows = []
    for (p, f, a), (hv_m, hv_s, df_m, df_s, n) in _BY_ALGORITHM.items():
        if problem and p != problem:
            continue
        if formulation and f != formulation:
            continue
        if algorithm and a != algorithm:
            continue
        rows.append({
            "problem": p, "formulation": f, "algorithm": a,
            "hv_mean": hv_m, "hv_std": hv_s,
            "df_mean": df_m, "df_std": df_s, "n": n,
        })
    return rows


def by_formulation(problem: str = None, formulation: str = None):
    """Return Paper A results pooled across algorithms.

    Each result is a dict with keys: ``problem``, ``formulation``,
    ``hv_mean``, ``hv_std``, ``hv_median``, ``df_mean``, ``df_std``,
    ``df_median``, ``n``.

    Parameters
    ----------
    problem, formulation : str, optional
        Filter results.

    Returns
    -------
    list of dict
    """
    rows = []
    for (p, f), (hv_m, hv_s, hv_med, df_m, df_s, df_med, n) in _BY_FORMULATION.items():
        if problem and p != problem:
            continue
        if formulation and f != formulation:
            continue
        rows.append({
            "problem": p, "formulation": f,
            "hv_mean": hv_m, "hv_std": hv_s, "hv_median": hv_med,
            "df_mean": df_m, "df_std": df_s, "df_median": df_med,
            "n": n,
        })
    return rows


def experimental_setup() -> dict:
    """Return the experimental design parameters used in Paper A."""
    return {
        "problems": list(PROBLEMS),
        "formulations": list(FORMULATIONS),
        "algorithms": list(ALGORITHMS),
        "nfe": NFE,
        "pop_size": POP_SIZE,
        "n_reps": N_REPS,
        "engine": ENGINE,
        "total_runs": len(PROBLEMS) * len(FORMULATIONS) * len(ALGORITHMS) * N_REPS,
        "reference": "Torre-Cifuentes (2026), Engineering Optimization",
    }
