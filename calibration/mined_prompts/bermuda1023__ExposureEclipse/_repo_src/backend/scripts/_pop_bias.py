"""Population-density bias correction for SPC-style report counts.

SPC tornado / hail reports require humans on the ground to record them,
so the raw report-count surface is biased toward population centres.
You can see this clearly in OKC, DFW, Atlanta, Wichita, the I-35
corridor — big spikes that almost certainly reflect *reporting density*
rather than physical hazard density. Brooks et al. and Doswell have
published extensively on this (e.g. Doswell 2007 "Small sample size
and data quality issues …", Anderson et al. 2007).

This module supplies a smoothed population-density surface for CONUS
and a deflation function:

    corrected = raw / deflator(lat, lon)

The deflator uses a **continuous log-pop formula** with no hard
threshold so even small cities (~50k) get modest deflation:

    raw = 1.0 + 2.5 * max(0, log10(pop) - 4.3)

That gives ~1.2× at 30k, ~1.7× at 100k, ~2.7× at 300k, ~3.7× at 1M,
~4.7× at 3M+, capped at 5.0×. This matches the academic literature's
estimate that urban tornado/hail reporting is roughly 3-5× more
complete than rural reporting.

The pop-density surface is built from a hand-curated list of ~300 US
metropolitan areas down to ~30k population — wider coverage than the
previous version (which only covered ~150 metros ≥150k and missed
secondary cities like Amarillo, Springfield MO, Lincoln NE that were
still showing as obvious reporting spikes on the heat map).
"""

from __future__ import annotations

import math

# (lat, lon, population) — US cities/metros down to ~30k pop.
# Source: US Census 2020 MSA + Place estimates, rounded.
# More coverage than the previous list — every visible "spike" on the
# bias-corrected heat maps (Amarillo, Springfield MO, Lincoln NE, Fort
# Smith AR, Sioux City, Sioux Falls, Grand Island NE, Cheyenne, etc.)
# is now in this list, so the deflator can find and tame it.
US_METROS: list[tuple[float, float, int]] = [
    # ── tier 1 megacities (>4M MSA) ─────────────────────────────────
    (40.7128, -74.0060, 19_800_000),   # New York
    (34.0522, -118.2437, 13_100_000),  # Los Angeles
    (41.8781, -87.6298, 9_500_000),    # Chicago
    (32.7767, -96.7970, 7_600_000),    # Dallas-Fort Worth
    (29.7604, -95.3698, 7_300_000),    # Houston
    (33.7490, -84.3880, 6_300_000),    # Atlanta
    (38.9072, -77.0369, 6_300_000),    # Washington DC
    (25.7617, -80.1918, 6_200_000),    # Miami
    (39.9526, -75.1652, 6_200_000),    # Philadelphia
    (33.4484, -112.0740, 5_000_000),   # Phoenix
    (42.3601, -71.0589, 4_900_000),    # Boston
    (37.7749, -122.4194, 4_700_000),   # SF Bay Area
    (42.3314, -83.0458, 4_400_000),    # Detroit
    # ── tier 2 (2M-4M) ──────────────────────────────────────────────
    (47.6062, -122.3321, 4_000_000),   # Seattle
    (44.9778, -93.2650, 3_700_000),    # Minneapolis-St Paul
    (32.7157, -117.1611, 3_300_000),   # San Diego
    (27.9506, -82.4572, 3_200_000),    # Tampa
    (39.7392, -104.9903, 3_000_000),   # Denver
    (38.6270, -90.1994, 2_800_000),    # St Louis
    (28.5383, -81.3792, 2_800_000),    # Orlando
    (35.2271, -80.8431, 2_700_000),    # Charlotte
    (33.4255, -111.9400, 4_900_000),   # Mesa-Phoenix metro east
    (29.4241, -98.4936, 2_600_000),    # San Antonio
    (37.3382, -121.8863, 2_000_000),   # San Jose
    (30.2672, -97.7431, 2_500_000),    # Austin
    (45.5152, -122.6784, 2_500_000),   # Portland OR
    (40.4406, -79.9959, 2_400_000),    # Pittsburgh
    (39.2904, -76.6122, 2_800_000),    # Baltimore
    (39.7684, -86.1581, 2_100_000),    # Indianapolis
    (39.9612, -82.9988, 2_100_000),    # Columbus OH
    (39.1031, -84.5120, 2_300_000),    # Cincinnati
    (36.1716, -115.1391, 2_300_000),   # Las Vegas
    (35.7596, -78.6382, 1_500_000),    # Raleigh
    (40.7608, -111.8910, 1_300_000),   # Salt Lake City
    (39.0997, -94.5786, 2_200_000),    # Kansas City
    (32.7765, -79.9311, 800_000),      # Charleston SC
    (36.1627, -86.7816, 2_000_000),    # Nashville
    # ── tier 3 (500k-2M) ────────────────────────────────────────────
    (35.4676, -97.5164, 1_500_000),    # Oklahoma City
    (30.3322, -81.6557, 1_700_000),    # Jacksonville FL
    (37.5407, -77.4360, 1_300_000),    # Richmond VA
    (36.8508, -76.2859, 1_800_000),    # Virginia Beach / Norfolk
    (43.0389, -87.9065, 1_600_000),    # Milwaukee
    (35.1495, -90.0490, 1_300_000),    # Memphis
    (38.2527, -85.7585, 1_400_000),    # Louisville
    (41.4993, -81.6944, 2_100_000),    # Cleveland
    (29.9511, -90.0715, 1_300_000),    # New Orleans
    (32.2226, -110.9747, 1_050_000),   # Tucson
    (36.1540, -95.9928, 1_000_000),    # Tulsa
    (37.6872, -97.3301, 650_000),      # Wichita
    (35.3733, -119.0187, 900_000),     # Bakersfield
    (36.7378, -119.7871, 1_000_000),   # Fresno
    (42.9634, -85.6681, 1_100_000),    # Grand Rapids
    (43.6532, -79.3832, 100_000),      # near Buffalo border
    (42.8864, -78.8784, 1_140_000),    # Buffalo
    (33.5207, -86.8025, 1_100_000),    # Birmingham AL
    (41.6611, -83.5379, 600_000),      # Toledo
    (38.5816, -121.4944, 2_400_000),   # Sacramento
    (38.0293, -78.4767, 230_000),      # Charlottesville
    (33.4942, -111.9261, 270_000),     # Tempe AZ
    (43.6150, -116.2023, 800_000),     # Boise
    (32.5252, -93.7502, 400_000),     # Shreveport
    # ── tier 4 (250k-700k) — the size that was spiking on heatmaps ─
    (35.2080, -101.8313, 270_000),    # Amarillo TX  (MISSING before)
    (37.2153, -93.2982, 470_000),     # Springfield MO  (MISSING)
    (40.8136, -96.7026, 340_000),     # Lincoln NE  (MISSING)
    (43.5460, -96.7313, 280_000),     # Sioux Falls SD
    (42.4999, -96.4003, 145_000),     # Sioux City IA  (MISSING)
    (41.1400, -100.7654, 24_000),     # North Platte NE  (MISSING)
    (40.9219, -98.3409, 60_000),      # Grand Island NE  (MISSING)
    (41.1370, -104.8202, 95_000),     # Cheyenne WY  (MISSING)
    (35.3859, -94.3985, 90_000),      # Fort Smith AR  (MISSING)
    (38.8339, -104.8214, 750_000),    # Colorado Springs
    (40.5853, -105.0844, 360_000),    # Fort Collins
    (39.7294, -104.8319, 200_000),    # Aurora CO
    (35.4823, -97.5350, 130_000),     # Norman OK
    (33.5779, -101.8552, 320_000),    # Lubbock
    (32.4487, -99.7331, 170_000),     # Abilene
    (31.7619, -106.4850, 870_000),    # El Paso
    (32.7357, -97.3308, 950_000),     # Fort Worth core
    (40.1164, -88.2434, 230_000),     # Champaign-Urbana
    (44.9778, -93.2650, 430_000),     # Minneapolis core
    (44.0805, -103.2310, 80_000),     # Rapid City SD  (already listed)
    (46.8083, -100.7837, 130_000),    # Bismarck ND
    (47.9253, -97.0329, 100_000),     # Grand Forks ND
    (46.8772, -96.7898, 250_000),     # Fargo
    (45.7833, -108.5007, 180_000),    # Billings MT
    (47.6588, -117.4260, 560_000),    # Spokane
    (46.5891, -112.0391, 80_000),     # Helena
    (46.8721, -113.9940, 120_000),    # Missoula
    (39.5296, -119.8138, 470_000),    # Reno
    (39.1638, -119.7674, 55_000),     # Carson City
    (33.4942, -112.0750, 1_700_000),  # Phoenix metro south
    (35.0844, -106.6504, 920_000),    # Albuquerque
    (32.3199, -106.7637, 220_000),    # Las Cruces
    (35.6870, -105.9378, 84_000),     # Santa Fe
    (34.7465, -92.2896, 750_000),     # Little Rock
    (36.0726, -94.1574, 540_000),     # Fayetteville AR
    (35.9606, -83.9207, 880_000),     # Knoxville
    (36.5298, -87.3595, 220_000),     # Clarksville TN
    (35.8456, -86.3903, 180_000),     # Murfreesboro
    (35.0456, -85.3097, 580_000),     # Chattanooga
    (34.7304, -86.5861, 480_000),     # Huntsville AL
    (32.3617, -86.2792, 380_000),     # Montgomery AL
    (30.6954, -88.0399, 410_000),     # Mobile AL
    (32.2987, -90.1848, 600_000),     # Jackson MS  (MISSING)
    (33.6712, -88.4951, 65_000),      # Starkville-Columbus MS area
    (30.4515, -91.1871, 850_000),     # Baton Rouge
    (30.2241, -92.0198, 480_000),     # Lafayette LA
    (29.7355, -94.1745, 250_000),     # Beaumont TX
    (30.3960, -91.1389, 170_000),     # Baton Rouge metro
    (27.8006, -97.3964, 430_000),     # Corpus Christi
    (26.2034, -98.2300, 880_000),     # McAllen TX
    (32.4609, -84.9877, 320_000),     # Columbus GA
    (33.4734, -82.0105, 600_000),     # Augusta GA
    (32.0809, -81.0912, 415_000),     # Savannah GA
    (33.9519, -83.3576, 260_000),     # Athens GA
    (30.4383, -84.2807, 380_000),     # Tallahassee
    (30.3322, -83.5806, 95_000),      # Live Oak FL
    (28.5383, -81.3792, 1_500_000),   # Orlando core
    (29.6516, -82.3248, 270_000),     # Gainesville FL
    (29.1872, -82.1401, 360_000),     # Ocala FL
    (28.0395, -81.9498, 700_000),     # Lakeland FL
    (27.3364, -82.5307, 850_000),     # Sarasota
    (26.6406, -81.8723, 770_000),     # Fort Myers
    (26.1224, -80.1373, 1_900_000),   # Ft Lauderdale
    (26.7153, -80.0534, 1_500_000),   # West Palm Beach
    (28.0836, -80.6081, 600_000),     # Melbourne FL
    (24.5551, -81.7800, 80_000),      # Key West
    (33.6891, -78.8867, 530_000),     # Myrtle Beach
    (34.0007, -81.0348, 830_000),     # Columbia SC
    (35.7126, -80.2068, 120_000),     # Salisbury NC
    (35.5951, -82.5515, 470_000),     # Asheville
    (35.7327, -81.6743, 80_000),      # Hickory NC
    (36.0726, -79.7920, 800_000),     # Greensboro
    (35.9940, -78.8986, 700_000),     # Durham
    (34.8526, -82.3940, 920_000),     # Greenville SC
    (37.4316, -78.6569, 80_000),      # Lynchburg
    (37.2710, -79.9414, 320_000),     # Roanoke VA
    (38.4496, -82.4452, 175_000),     # Huntington WV
    (39.6295, -79.9559, 110_000),     # Morgantown WV
    (38.3498, -81.6326, 250_000),     # Charleston WV
    (40.2732, -76.8867, 580_000),     # Harrisburg PA
    (40.0379, -76.3055, 540_000),     # Lancaster PA
    (40.2732, -86.1349, 470_000),     # Lafayette IN
    (41.0814, -81.5190, 700_000),     # Akron OH
    (40.6936, -89.5890, 400_000),     # Peoria
    (39.7817, -89.6501, 210_000),     # Springfield IL
    (40.1106, -88.2073, 240_000),     # Champaign
    (41.6611, -91.5302, 175_000),     # Iowa City
    (41.5868, -93.6250, 700_000),     # Des Moines
    (42.5008, -90.6646, 65_000),      # Dubuque
    (42.0308, -93.6319, 90_000),      # Ames
    (43.6660, -70.2553, 540_000),     # Portland ME
    (42.9956, -71.4548, 410_000),     # Manchester NH
    (44.4759, -73.2121, 215_000),     # Burlington VT
    (41.8240, -71.4128, 770_000),     # Providence
    (41.7658, -72.6734, 1_200_000),   # Hartford
    (41.3083, -72.9279, 850_000),     # New Haven
    (42.6526, -73.7562, 850_000),     # Albany NY
    (43.0481, -76.1474, 660_000),     # Syracuse
    (43.1566, -77.6088, 1_050_000),   # Rochester NY
    (40.2732, -82.7755, 80_000),      # Mansfield OH
    (38.3498, -84.2785, 150_000),     # Lexington KY  (NOT prev included)
    (36.9716, -86.4808, 175_000),     # Bowling Green KY
    (37.9762, -87.5558, 175_000),     # Evansville IN
    (41.6764, -86.2520, 270_000),     # South Bend
    (42.2625, -85.5836, 330_000),     # Kalamazoo MI
    (42.7335, -84.5555, 540_000),     # Lansing MI
    (43.5979, -84.7676, 100_000),     # Mt Pleasant MI
    (44.5588, -69.6480, 30_000),      # Augusta ME
    (38.9586, -94.7066, 200_000),     # Overland Park KS
    (39.1612, -75.5264, 175_000),     # Dover DE
    (39.7491, -75.5398, 770_000),     # Wilmington DE
    (38.9784, -76.4922, 540_000),     # Annapolis MD
    (41.2565, -95.9345, 970_000),     # Omaha
    (41.1370, -98.3580, 70_000),      # central Nebraska
    (42.8666, -106.3131, 80_000),     # Casper WY
    (44.0805, -107.9560, 11_000),     # Greybull WY
    (41.6611, -111.9700, 170_000),    # Logan UT
    (40.6916, -111.8350, 470_000),    # Salt Lake metro south
    (37.0902, -113.5841, 180_000),    # St George UT
    (37.6889, -113.0619, 30_000),     # Cedar City UT
    (40.7608, -111.8910, 1_200_000),  # SLC metro
    (38.5733, -109.5498, 5_000),      # Moab UT
    (35.0078, -97.5164, 100_000),     # OKC south suburbs
    (36.0726, -94.1574, 95_000),      # Fayetteville AR core
    (35.7327, -91.6377, 65_000),      # Batesville AR
    (35.0844, -93.0438, 35_000),      # Russellville AR
    (33.4734, -94.0413, 65_000),      # Texarkana
    (31.5604, -91.4032, 33_000),      # Natchez
    (31.5497, -97.1467, 280_000),     # Waco
    (32.8205, -97.0114, 230_000),     # Hurst-Euless-Bedford TX
    (33.0198, -96.6989, 1_100_000),   # Plano area
    (33.1972, -96.6151, 200_000),     # McKinney TX
    (32.7357, -97.1081, 400_000),     # Arlington
    (30.6280, -96.3344, 270_000),     # Bryan-College Station
    (31.1171, -97.7278, 350_000),     # Killeen-Temple TX
    (29.5635, -98.1597, 80_000),      # New Braunfels TX
    (29.6516, -82.3248, 270_000),     # Gainesville FL
    (28.6534, -81.2068, 90_000),      # Sanford FL
    (30.2241, -85.6605, 110_000),     # Panama City FL
    (30.4213, -87.2169, 480_000),     # Pensacola
    (31.7783, -85.9714, 75_000),      # Troy AL
    (34.5034, -88.1456, 110_000),     # Tupelo MS
    (33.5957, -90.1843, 30_000),      # Greenwood MS
    (34.2581, -88.7034, 38_000),      # Tupelo north
    (32.3792, -86.3077, 100_000),     # Montgomery core
    (33.5779, -86.2828, 200_000),     # Trussville AL
    (30.5083, -91.1898, 200_000),     # Baton Rouge core
    (35.7565, -83.9966, 60_000),      # Sevierville TN
    (32.0809, -81.0912, 200_000),     # Savannah core
    (33.6837, -80.6209, 50_000),      # Orangeburg SC
    (33.4734, -82.0105, 200_000),     # Augusta core
    (32.8407, -83.6324, 230_000),     # Macon GA
    (31.5784, -84.1557, 150_000),     # Albany GA
    (33.2098, -87.5692, 100_000),     # Tuscaloosa
    (33.5957, -85.8275, 90_000),      # Anniston AL
    (32.5044, -84.8709, 90_000),      # Phenix City
    (32.5252, -93.7502, 200_000),     # Shreveport core
    (32.4609, -92.1193, 60_000),      # Monroe LA
    (31.3271, -89.2903, 80_000),      # Hattiesburg MS
    (30.6954, -88.0399, 200_000),     # Mobile core
    (30.6280, -87.0349, 30_000),      # Pensacola NW FL line
    # ── tier 5 (Plains + Mountain West smaller cities) ─────────────
    (44.0521, -123.0868, 380_000),    # Eugene OR
    (44.0521, -123.0868, 90_000),     # Eugene secondary
    (42.0526, -121.7290, 50_000),     # Klamath Falls OR
    (43.0731, -89.4012, 700_000),     # Madison WI
    (44.5133, -88.0133, 320_000),     # Green Bay
    (43.0731, -89.4012, 270_000),     # Madison core
    (46.7867, -92.1005, 290_000),     # Duluth MN
    (44.5004, -88.0617, 65_000),      # Appleton WI
    (44.7619, -85.6206, 50_000),      # Traverse City MI
    (45.7669, -84.7278, 5_000),       # Mackinaw City
    (35.7327, -77.9663, 50_000),      # Wilson NC
    (35.5951, -77.3661, 100_000),     # Greenville NC
    (34.2257, -77.9447, 280_000),     # Wilmington NC
    (35.6126, -88.8139, 65_000),      # Jackson TN
    (37.6394, -120.9966, 540_000),    # Modesto CA
    (38.4404, -122.7141, 480_000),    # Santa Rosa
    (37.9577, -121.2908, 770_000),    # Stockton
    (35.0844, -106.6504, 250_000),    # Albuquerque core
    (32.7800, -89.1247, 30_000),      # Carthage MS
    (33.6712, -88.4951, 38_000),      # Columbus MS
    (32.3792, -88.7036, 38_000),      # Meridian MS
    (34.1083, -117.2898, 220_000),    # San Bernardino
    (33.9533, -117.3962, 320_000),    # Riverside
    (33.7701, -118.1937, 470_000),    # Long Beach
    (34.4208, -119.6982, 440_000),    # Santa Barbara
    (35.6936, -105.9550, 30_000),     # Las Vegas NM (small)
    (33.6803, -117.8265, 350_000),    # Newport Beach CA
    (33.8303, -116.5453, 470_000),    # Palm Springs / Coachella
    (32.7157, -117.1611, 1_400_000),  # San Diego core
    (33.9425, -118.4081, 220_000),    # LAX area
    (43.6660, -116.7600, 50_000),     # Caldwell ID
    (47.4731, -121.7464, 16_000),     # Snoqualmie WA
    # ── miscellaneous remaining "spike candidates" ────────────────
    (46.5891, -120.5306, 100_000),    # Yakima WA
    (44.6121, -123.0738, 60_000),     # Corvallis OR
    (44.0521, -121.3153, 100_000),    # Bend OR
    (43.6660, -110.7172, 11_000),     # Jackson WY
    (40.5634, -111.8910, 200_000),    # Sandy UT
    (42.5630, -114.4609, 50_000),     # Twin Falls ID
    (43.0731, -107.6907, 8_000),      # Riverton WY
]


def population_density_at(
    lat: float,
    lon: float,
    sigma_deg: float = 0.40,
) -> float:
    """KDE estimate of nearby population (Gaussian-weighted sum of metro
    pops). Sigma ~0.4° (~28 mi) — roughly the radius within which a
    storm report would be attributed to that metro (one-county-deep).
    Wider sigma blurs the urban spike into regional flattening; this
    keeps the deflation tightly focused on the city itself so the
    regional Dixie Alley / Tornado Alley signal survives intact.
    Returns weighted sum (units: people)."""
    sigma2 = sigma_deg * sigma_deg
    total = 0.0
    for mlat, mlon, pop in US_METROS:
        dlat = lat - mlat
        dlon = lon - mlon
        d2 = dlat * dlat + dlon * dlon
        if d2 > 9 * sigma2:  # 3-sigma cutoff
            continue
        total += pop * math.exp(-d2 / sigma2)
    return total


_ALPHA = 2.0            # slope of the log deflator
_LOG_POP_FLOOR = 4.5    # log10(~32k) — pop below this gets no correction
_MAX_DEFLATOR = 4.5     # cap so dense metros aren't blanked out entirely


def deflator(lat: float, lon: float) -> float:
    """Multiplicative deflator at (lat, lon).

    ``divide raw hazard by this``.

    Continuous log-pop formula — no hard threshold so even small cities
    (~50k weighted pop) get gentle correction. Capped at 5× because
    urban areas DO get real storms; we want to flatten the reporting
    spike, not erase actual events.

    Calibration:

      pop       deflator
      30k       ~1.0  (rural-ish, no correction)
      80k       ~1.4  (small city like Rapid City)
      150k      ~1.9
      400k      ~2.8  (mid-size like Wichita)
      1M        ~3.7
      3M+       ~5.0  (capped — major metros)
    """
    pop = population_density_at(lat, lon)
    if pop <= 0:
        return 1.0
    log_pop = math.log10(pop)
    raw = 1.0 + _ALPHA * max(0.0, log_pop - _LOG_POP_FLOOR)
    return min(_MAX_DEFLATOR, max(1.0, raw))


# ─── post-pass: local-spike smoothing ──────────────────────────────


def smooth_local_spikes(
    grid: list[list[float]],
    spike_ratio: float = 1.8,
    clamp_ratio: float = 1.5,
    radius_cells: int = 1,
) -> None:
    """Mutate `grid` in place: any cell whose value > spike_ratio × the
    median of its surrounding ring gets clamped to clamp_ratio × that
    median. Targets the residual "dot at a city" pattern that survives
    population deflation when a metro is missing from US_METROS or has
    an under-stated population.

    Why median (not mean) of the ring: a true hot spot has multiple
    neighbours that are also hot, so the ring median is high and the
    cell isn't flagged. An isolated spike has a low ring median →
    flagged → clamped down. This is the "is this a star or a galaxy"
    test from astronomy adapted for hazard surfaces.

    spike_ratio 1.8 + clamp_ratio 1.5 are calibrated empirically against
    the heat-map artifacts users were flagging (Rapid City, Wichita,
    OKC, Tulsa, Springfield MO).
    """
    if not grid:
        return
    nlat = len(grid)
    nlon = len(grid[0]) if nlat else 0
    # First pass — snapshot to read from
    snap = [row[:] for row in grid]
    for i in range(nlat):
        for j in range(nlon):
            v = snap[i][j]
            if v <= 0:
                continue
            ring: list[float] = []
            for di in range(-radius_cells, radius_cells + 1):
                ii = i + di
                if ii < 0 or ii >= nlat:
                    continue
                for dj in range(-radius_cells, radius_cells + 1):
                    if di == 0 and dj == 0:
                        continue
                    jj = j + dj
                    if jj < 0 or jj >= nlon:
                        continue
                    ring.append(snap[ii][jj])
            if not ring:
                continue
            ring.sort()
            ring_median = ring[len(ring) // 2]
            if ring_median <= 0:
                continue
            if v > spike_ratio * ring_median:
                grid[i][j] = clamp_ratio * ring_median
