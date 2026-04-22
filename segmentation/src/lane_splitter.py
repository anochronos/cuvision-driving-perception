import numpy as np
import cv2


def clean_da_mask(da_mask, kernel_size=7):
    """Morphological cleanup of the drivable area mask to remove noise."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = (da_mask > 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def estimate_lane_width_per_row(da_mask, vanishing_row_frac=0.35,
                                lane_width_frac=0.20):
    """
    Estimate expected single-lane width at each row using a fixed reference
    scaled by perspective geometry.

    Uses a fixed fraction of image width as the single-lane reference at the
    bottom of the frame (default 20%, ~384px for 1920w). This avoids the
    self-referential problem of measuring lane width from the DA mask itself,
    which fails on multi-lane roads where the mask spans all lanes.

    The reference width shrinks linearly toward the vanishing point.

    Args:
        da_mask: cleaned binary mask (H, W)
        vanishing_row_frac: approximate vanishing point as fraction of image height
        lane_width_frac: single lane width as fraction of image width at bottom

    Returns:
        lane_width: array of shape (H,) with estimated single-lane width per row.
                    -1 for rows above vanishing point.
    """
    H, W = da_mask.shape
    vanishing_row = int(H * vanishing_row_frac)
    bottom_row = H - 1

    # Fixed reference: single lane width at the bottom of the frame
    ref_width = W * lane_width_frac

    # Scale by perspective: width shrinks linearly from bottom to vanishing point
    lane_width = np.full(H, -1, dtype=np.float64)
    for y in range(vanishing_row, H):
        scale = (y - vanishing_row) / max(bottom_row - vanishing_row, 1)
        lane_width[y] = ref_width * scale

    return lane_width


def extract_road_boundaries(da_mask):
    """
    For each row, find the leftmost and rightmost drivable pixel.
    Returns arrays of shape (H,) for left and right boundaries.
    Rows with no drivable pixels get -1.
    """
    H, W = da_mask.shape
    left = np.full(H, -1, dtype=np.int32)
    right = np.full(H, -1, dtype=np.int32)

    for y in range(H):
        row = da_mask[y]
        indices = np.where(row > 0)[0]
        if len(indices) > 0:
            left[y] = indices[0]
            right[y] = indices[-1]

    return left, right


def clamp_boundaries_for_turns(left, right, lane_width_per_row, num_lanes,
                               max_expansion=1.8):
    """
    Clamp road boundaries to prevent turn areas from distorting lane dividers.

    At turns/intersections, the DA mask expands on one side (e.g. right side
    on a right turn). This makes the equal-width split shift toward the turn.

    Fix: compute the expected road width (num_lanes * single_lane_width) and
    if the actual width exceeds it by too much, clamp the boundary that
    expanded. We use the road center from the bottom (stable, straight section)
    as the anchor to decide which side expanded.

    Args:
        left, right: per-row boundaries (H,) arrays
        lane_width_per_row: expected single-lane width per row
        num_lanes: number of ego-direction lanes
        max_expansion: maximum allowed width as multiple of expected width

    Returns:
        clamped_left, clamped_right: adjusted boundary arrays
    """
    H = len(left)
    cl = left.copy()
    cr = right.copy()

    # Compute road center from the bottom 20% of valid rows (stable reference)
    valid = (left >= 0) & (right >= 0)
    valid_rows = np.where(valid)[0]
    if len(valid_rows) < 10:
        return cl, cr

    bottom_start = valid_rows[-1] - int(len(valid_rows) * 0.2)
    bottom_rows = valid_rows[valid_rows >= bottom_start]
    if len(bottom_rows) < 5:
        return cl, cr

    ref_center = np.mean((left[bottom_rows].astype(float) + right[bottom_rows].astype(float)) / 2)

    for y in range(H):
        if left[y] < 0 or right[y] < 0:
            continue
        expected_lane = lane_width_per_row[y]
        if expected_lane <= 0:
            continue

        expected_width = expected_lane * num_lanes * max_expansion
        actual_width = right[y] - left[y]

        if actual_width <= expected_width:
            continue

        # Road is wider than expected -clamp the side that expanded away
        # from the reference center
        center = (left[y] + right[y]) / 2
        excess = actual_width - expected_width

        if center > ref_center + expected_lane * 0.3:
            # Center shifted right → right side expanded (right turn area)
            cr[y] = int(right[y] - excess)
        elif center < ref_center - expected_lane * 0.3:
            # Center shifted left → left side expanded (left turn area)
            cl[y] = int(left[y] + excess)
        else:
            # Symmetric expansion (intersection) -clamp both sides equally
            half_excess = int(excess / 2)
            cl[y] = left[y] + half_excess
            cr[y] = right[y] - half_excess

    return cl, cr


def compute_lane_dividers(left, right, num_lanes, lane_width_per_row,
                          min_road_width=20, lane_ratio_threshold=1.6):
    """
    Given left/right road boundaries per row and the number of lanes,
    compute N-1 lane divider x-positions per row.

    Clamps boundaries at turns to prevent the divider from shifting into
    the turn area, then splits the clamped road width equally.

    Returns a list of (N-1) arrays, each of shape (H,) with x-positions.
    """
    H = len(left)
    num_dividers = num_lanes - 1
    dividers = [np.full(H, -1, dtype=np.int32) for _ in range(num_dividers)]

    # Clamp boundaries to prevent turn distortion
    cl, cr = clamp_boundaries_for_turns(left, right, lane_width_per_row,
                                        num_lanes)

    for y in range(H):
        if cl[y] < 0 or cr[y] < 0:
            continue
        road_width = cr[y] - cl[y]
        if road_width < min_road_width:
            continue

        expected_lane = lane_width_per_row[y]
        if expected_lane > 0:
            # Only split if road is wide enough for multiple lanes
            min_width_for_split = expected_lane * lane_ratio_threshold
            if num_lanes > 2:
                min_width_for_split = expected_lane * (lane_ratio_threshold + (num_lanes - 2) * 0.7)
            if road_width < min_width_for_split:
                continue

        for i in range(num_dividers):
            frac = (i + 1) / num_lanes
            dividers[i][y] = int(cl[y] + frac * road_width)

    return dividers


def smooth_divider(xs, window_size=51):
    """Smooth a divider line using a moving average, skipping invalid rows."""
    smoothed = xs.copy().astype(np.float64)
    valid = xs >= 0

    smoothed[~valid] = np.nan

    half = window_size // 2
    result = np.full_like(smoothed, np.nan)
    for y in range(len(smoothed)):
        lo = max(0, y - half)
        hi = min(len(smoothed), y + half + 1)
        segment = smoothed[lo:hi]
        valid_seg = segment[~np.isnan(segment)]
        if len(valid_seg) > 0:
            result[y] = np.mean(valid_seg)

    output = np.full(len(xs), -1, dtype=np.int32)
    valid_result = ~np.isnan(result)
    output[valid_result] = result[valid_result].astype(np.int32)
    return output


def dividers_to_polylines(dividers, da_mask=None, min_gap=5):
    """
    Convert per-row divider arrays into polyline point lists.
    Subsamples rows for efficiency and skips invalid positions.
    If da_mask is provided, also skip rows where the divider x-position
    falls outside the DA mask (i.e. on top of a car or obstacle).
    """
    polylines = []
    for div in dividers:
        points = []
        for y in range(0, len(div), min_gap):
            if div[y] < 0:
                continue
            x = int(div[y])
            if da_mask is not None and da_mask[y, x] == 0:
                continue
            points.append((x, y))
        polylines.append(points)
    return polylines


class DAMaskAccumulator:
    """
    Rolling EMA accumulator for DA masks across frames.

    Instead of using a single frame's DA mask (which flickers), this
    accumulates masks over time so the boundaries are stable. The
    accumulated mask is thresholded to produce a clean binary output.

    A pixel that has been consistently drivable across recent frames will
    stay active even if one frame drops it. A pixel that only appears
    briefly won't reach the threshold.
    """

    def __init__(self, alpha=0.25, threshold=0.4):
        """
        Args:
            alpha: EMA weight for new frame (0.25 = new frame is 25% of result)
            threshold: accumulated value must exceed this to be considered active
        """
        self.alpha = alpha
        self.threshold = threshold
        self.accum = None

    def update(self, da_mask):
        """
        Feed a new frame's DA mask and return the accumulated stable mask.

        Args:
            da_mask: (H, W) binary mask for current frame

        Returns:
            stable_mask: (H, W) uint8 binary mask from accumulated history
        """
        frame = (da_mask > 0).astype(np.float32)

        if self.accum is None or self.accum.shape != frame.shape:
            self.accum = frame.copy()
        else:
            self.accum = self.alpha * frame + (1 - self.alpha) * self.accum

        # Threshold to get stable binary mask
        stable = (self.accum >= self.threshold).astype(np.uint8)
        return stable


class TemporalSmoother:
    """Exponential moving average smoother for lane divider positions across frames."""

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.prev_dividers = None

    def smooth(self, dividers):
        if self.prev_dividers is None or len(self.prev_dividers) != len(dividers):
            self.prev_dividers = [d.copy().astype(np.float64) for d in dividers]
            return dividers

        smoothed = []
        for i, div in enumerate(dividers):
            prev = self.prev_dividers[i]
            out = div.copy().astype(np.float64)
            for y in range(len(div)):
                if div[y] >= 0 and prev[y] >= 0:
                    out[y] = self.alpha * div[y] + (1 - self.alpha) * prev[y]
                elif div[y] < 0 and prev[y] >= 0:
                    out[y] = prev[y]
            self.prev_dividers[i] = out.copy()
            result = np.full(len(div), -1, dtype=np.int32)
            valid = out >= 0
            result[valid] = out[valid].astype(np.int32)
            smoothed.append(result)
        return smoothed


def draw_lane_lines(img, polylines, color=(0, 255, 255), thickness=4):
    """Draw lane divider polylines on the image."""
    for points in polylines:
        if len(points) < 2:
            continue
        pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness)
    return img


def compute_road_aware_dividers(left, right, num_lanes, ego_width_m,
                                total_width_m, lane_width_m=3.5,
                                min_road_width=20):
    """
    Compute lane dividers using real road dimensions from a map API.

    Instead of splitting the DA mask equally, this uses the known road width
    to determine what fraction of the total road the DA mask covers, then
    places dividers at the correct positions for the ego-direction lanes.

    The DA mask typically covers the ego-direction lanes. If the real ego
    width is known, we can compute the expected pixel width per lane and
    place dividers at exact lane boundaries rather than equal divisions.

    Args:
        left, right: per-row road boundaries from DA mask
        num_lanes: ego-direction lane count (from OSM)
        ego_width_m: ego-direction road width in meters
        total_width_m: total road width in meters
        lane_width_m: standard lane width in meters (3.5m EU, 3.7m US)
        min_road_width: minimum pixel width to consider valid

    Returns:
        list of (N-1) arrays, each of shape (H,) with x-positions
    """
    H = len(left)
    num_dividers = num_lanes - 1
    if num_dividers <= 0:
        return []

    dividers = [np.full(H, -1, dtype=np.int32) for _ in range(num_dividers)]

    # DA mask covers ego-direction lanes. Compute the fraction of
    # the DA mask width that each lane should occupy.
    # If ego has 2 lanes at 3.5m each = 7m, each lane = 50% of DA width.
    # But if there's a shoulder or the DA extends beyond ego lanes,
    # the per-lane fraction adjusts.
    ego_lane_width_m = ego_width_m / num_lanes if ego_width_m else lane_width_m

    for y in range(H):
        if left[y] < 0 or right[y] < 0:
            continue
        da_width_px = right[y] - left[y]
        if da_width_px < min_road_width:
            continue

        # Compute meters-per-pixel at this row using the DA width
        # DA mask ≈ ego_width_m wide in real world
        m_per_px = ego_width_m / da_width_px if da_width_px > 0 else 0
        if m_per_px <= 0:
            continue

        # Place dividers at lane boundaries
        for i in range(num_dividers):
            # Distance from left edge to this divider in meters
            divider_m = ego_lane_width_m * (i + 1)
            # Convert to pixels from left boundary
            divider_px = int(left[y] + divider_m / m_per_px)
            # Clamp to road boundaries
            if left[y] < divider_px < right[y]:
                dividers[i][y] = divider_px

    return dividers


def split_da_into_lanes(da_mask, num_lanes, smooth_window=101, min_road_width=20,
                        cutoff_bottom=0.08, lane_ratio_threshold=1.6,
                        temporal_smoother=None, road_info=None,
                        da_accumulator=None):
    """
    Main function: takes a DA segmentation mask and lane count,
    returns polylines for the lane dividers.

    If da_accumulator is provided, uses the accumulated DA mask across
    multiple frames for stable boundaries instead of the flickering
    single-frame mask. This is the primary stability mechanism.

    If road_info is provided (from OSM/map API), uses real road dimensions
    for more accurate lane placement. Otherwise falls back to equal-width
    splitting with perspective-based width checks.

    Args:
        da_mask: (H, W) numpy array, drivable area binary mask
        num_lanes: int, number of lanes in the ego direction
        smooth_window: int, smoothing kernel size for divider lines
        min_road_width: int, minimum pixel width to consider as valid road
        cutoff_bottom: float, fraction of image height to ignore at the bottom
        lane_ratio_threshold: float, road must be this many times wider than
                              estimated single-lane width to split (default 1.6)
        temporal_smoother: optional TemporalSmoother for divider line EMA
        road_info: optional dict from road_info.get_road_info_for_splitting()
        da_accumulator: optional DAMaskAccumulator for stable DA boundaries

    Returns:
        polylines: list of point lists [(x, y), ...] for each divider
    """
    # Use road_info lane count if available, otherwise use the argument
    if road_info and road_info.get("num_lanes") and road_info["source"] != "error":
        num_lanes = road_info["num_lanes"]

    if num_lanes <= 1:
        return []

    mask = clean_da_mask(da_mask)

    # Zero out the bottom portion to avoid hood/dashboard artifacts
    if cutoff_bottom > 0:
        H = mask.shape[0]
        cutoff_row = int(H * (1 - cutoff_bottom))
        mask[cutoff_row:, :] = 0

    # Use accumulated DA mask for stable boundaries across frames
    if da_accumulator is not None:
        stable_mask = da_accumulator.update(mask)
    else:
        stable_mask = mask

    left, right = extract_road_boundaries(stable_mask)

    # Choose splitting strategy based on available road info
    if (road_info and road_info.get("ego_width_m")
            and road_info["source"] != "error"):
        dividers = compute_road_aware_dividers(
            left, right, num_lanes,
            ego_width_m=road_info["ego_width_m"],
            total_width_m=road_info["total_width_m"],
            lane_width_m=road_info.get("lane_width_m", 3.5),
            min_road_width=min_road_width,
        )
    else:
        lane_width_per_row = estimate_lane_width_per_row(stable_mask)
        dividers = compute_lane_dividers(left, right, num_lanes, lane_width_per_row,
                                         min_road_width, lane_ratio_threshold)

    if not dividers:
        return []

    # Spatial smoothing
    dividers = [smooth_divider(d, smooth_window) for d in dividers]

    # Temporal smoothing of divider positions across frames
    if temporal_smoother is not None:
        dividers = temporal_smoother.smooth(dividers)

    # Convert to polylines, skipping points on top of cars (use current mask)
    polylines = dividers_to_polylines(dividers, da_mask=mask)
    return polylines
