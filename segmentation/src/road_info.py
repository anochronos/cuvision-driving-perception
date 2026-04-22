import requests
import math


# Standard lane widths by region (meters)
LANE_WIDTH_DEFAULTS = {
    "US": 3.7,
    "EU": 3.5,
    "default": 3.5,
}

# Typical road widths by OSM highway type when tags are missing
ROAD_DEFAULTS = {
    "motorway":      {"lanes": 6, "width": 22.2, "lanes_forward": 3},
    "motorway_link": {"lanes": 2, "width": 7.4,  "lanes_forward": 1},
    "trunk":         {"lanes": 4, "width": 14.8, "lanes_forward": 2},
    "trunk_link":    {"lanes": 2, "width": 7.4,  "lanes_forward": 1},
    "primary":       {"lanes": 4, "width": 14.0, "lanes_forward": 2},
    "primary_link":  {"lanes": 2, "width": 7.0,  "lanes_forward": 1},
    "secondary":     {"lanes": 2, "width": 7.0,  "lanes_forward": 1},
    "secondary_link":{"lanes": 2, "width": 7.0,  "lanes_forward": 1},
    "tertiary":      {"lanes": 2, "width": 7.0,  "lanes_forward": 1},
    "tertiary_link": {"lanes": 1, "width": 3.5,  "lanes_forward": 1},
    "residential":   {"lanes": 2, "width": 6.0,  "lanes_forward": 1},
    "living_street": {"lanes": 1, "width": 4.0,  "lanes_forward": 1},
    "unclassified":  {"lanes": 2, "width": 6.0,  "lanes_forward": 1},
    "service":       {"lanes": 1, "width": 3.5,  "lanes_forward": 1},
}


def query_osm_road(lat, lon, radius=20, timeout=5):
    """
    Query OpenStreetMap Overpass API for the nearest road to (lat, lon).

    Returns a dict with:
        - highway: road type (e.g. 'residential', 'primary')
        - lanes: total lane count (int or None)
        - lanes_forward: lanes in forward direction (int or None)
        - lanes_backward: lanes in backward direction (int or None)
        - width: road width in meters (float or None)
        - name: road name (str or None)
        - oneway: bool
        - maxspeed: speed limit string or None
        - source: 'osm_tag' or 'osm_default' or 'error'

    Args:
        lat: latitude
        lon: longitude
        radius: search radius in meters
        timeout: HTTP timeout in seconds
    """
    query = f"""
    [out:json][timeout:{timeout}];
    way(around:{radius},{lat},{lon})["highway"];
    out tags;
    """

    try:
        resp = requests.get(
            "https://overpass-api.de/api/interpreter",
            params={"data": query},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        return {"source": "error", "error": str(e)}

    if not data.get("elements"):
        return {"source": "error", "error": "no roads found nearby"}

    # Pick the most significant road (highest highway rank)
    rank = list(ROAD_DEFAULTS.keys())
    best = None
    best_rank = len(rank)
    for elem in data["elements"]:
        tags = elem.get("tags", {})
        hw = tags.get("highway", "")
        if hw in rank:
            r = rank.index(hw)
            if r < best_rank:
                best_rank = r
                best = tags

    if best is None:
        best = data["elements"][0].get("tags", {})

    return _parse_road_tags(best)


def _parse_road_tags(tags):
    """Parse OSM road tags into a standardized road info dict."""
    hw = tags.get("highway", "unclassified")
    defaults = ROAD_DEFAULTS.get(hw, ROAD_DEFAULTS["unclassified"])

    result = {
        "highway": hw,
        "name": tags.get("name"),
        "oneway": tags.get("oneway", "no") == "yes",
        "maxspeed": tags.get("maxspeed"),
        "source": "osm_tag",
    }

    # Lanes
    lanes_str = tags.get("lanes")
    lanes_fwd = tags.get("lanes:forward")
    lanes_bwd = tags.get("lanes:backward")

    if lanes_str:
        result["lanes"] = int(lanes_str)
    else:
        result["lanes"] = defaults["lanes"]
        result["source"] = "osm_default"

    if lanes_fwd:
        result["lanes_forward"] = int(lanes_fwd)
    elif result["oneway"]:
        result["lanes_forward"] = result["lanes"]
    else:
        result["lanes_forward"] = math.ceil(result["lanes"] / 2)

    if lanes_bwd:
        result["lanes_backward"] = int(lanes_bwd)
    elif result["oneway"]:
        result["lanes_backward"] = 0
    else:
        result["lanes_backward"] = result["lanes"] - result["lanes_forward"]

    # Width
    width_str = tags.get("width")
    if width_str:
        try:
            result["width"] = float(width_str.replace("m", "").strip())
        except ValueError:
            result["width"] = defaults["width"]
    else:
        result["width"] = defaults["width"]
        if result["source"] != "osm_default":
            result["source"] = "osm_tag"  # lanes were from tag, width from default

    return result


def estimate_ego_lanes_and_width(road_info, region="default"):
    """
    From road info, estimate the number of ego-direction lanes and
    the expected ego-direction road width in meters.

    Args:
        road_info: dict from query_osm_road()
        region: 'US', 'EU', or 'default' for lane width standard

    Returns:
        (ego_lanes, ego_width_m, total_width_m)
    """
    if road_info.get("source") == "error":
        return None, None, None

    ego_lanes = road_info.get("lanes_forward", 1)
    total_width = road_info.get("width", 7.0)
    total_lanes = road_info.get("lanes", 2)

    # Estimate ego-direction width as proportion of total
    if total_lanes > 0:
        ego_width = total_width * (ego_lanes / total_lanes)
    else:
        ego_width = total_width / 2

    return ego_lanes, ego_width, total_width


def get_road_info_for_splitting(lat, lon, region="default"):
    """
    Convenience function: query OSM and return lane splitting parameters.

    Returns:
        dict with:
            - num_lanes: int, ego-direction lane count
            - ego_width_m: float, ego-direction road width in meters
            - total_width_m: float, total road width in meters
            - lane_width_m: float, standard lane width for region
            - road_name: str or None
            - highway_type: str
            - source: str
    """
    road_info = query_osm_road(lat, lon)

    if road_info.get("source") == "error":
        return {
            "num_lanes": None,
            "ego_width_m": None,
            "total_width_m": None,
            "lane_width_m": LANE_WIDTH_DEFAULTS[region],
            "road_name": None,
            "highway_type": None,
            "source": "error",
            "error": road_info.get("error", "unknown"),
        }

    ego_lanes, ego_width, total_width = estimate_ego_lanes_and_width(
        road_info, region
    )

    return {
        "num_lanes": ego_lanes,
        "ego_width_m": ego_width,
        "total_width_m": total_width,
        "lane_width_m": LANE_WIDTH_DEFAULTS[region],
        "road_name": road_info.get("name"),
        "highway_type": road_info.get("highway"),
        "source": road_info["source"],
    }
