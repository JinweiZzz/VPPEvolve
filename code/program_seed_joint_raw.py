# Initial implementation - you can completely rewrite these functions
# Feel free to change the structure, add helper functions, use different approaches, etc.

def alpha_score(p, pv, wind, flex_storage=None, current_storage=None, flex_vehicle=None, current_vehicle=None, flex_AC=None, current_AC=None, flex_wash=None, current_wash=None, t=None, n=None, ctx=None):
    """
    Calculate allocation score for a node.
    Higher score means more bid quantity should be allocated to this node.
    
    Note: returned score is processed by softmax to get per-node bid allocation ratio.

    Args:
        p: electricity price (float)
        pv: photovoltaic generation (kW)
        wind: wind generation (kW)
        flex_storage: storage flexibility, total capacity (kW)
        current_storage: current storage level, charged capacity (kW)
        flex_vehicle: vehicle flexibility, total chargeable capacity (kW)
        current_vehicle: current vehicle level, charged capacity (kW)
        flex_AC: AC flexibility, total power (kW)
        current_AC: current AC usage (kW)
        flex_wash: wash total available runs (count; ~40kW per machine)
        current_wash: current wash runs on (count)
        t: time step, 0-23 (int, optional)
        n: node identifier (string, optional)
        ctx: context dictionary (optional)

    Returns:
        Positive score (float)
    """
    # Simple baseline: linear combination of price, PV and Wind
    # Flexibility and current-state parameters are ignored
    x = 1.0 * p + 1.0 * pv + 1.0 * wind
    return max(1e-6, x)  # Ensure positive


def device_allocation(p, bq, pv, wind, flex_storage=None, current_storage=None, flex_vehicle=None, current_vehicle=None, flex_AC=None, current_AC=None, flex_wash=None, current_wash=None, t=None, n=None, ctx=None):
    """
    Calculate device usage ratios for a node.
    Returns a dictionary with usage ratios for different device types.
    
    Args:
        p: electricity price (float, normalized 0-1)
        bq: bid quantity for this node. bq represents the net power injection target to the grid for this node at time t (generation positive, consumption negative).
        pv: photovoltaic generation (kW)
        wind: wind generation (kW)
        flex_storage: storage flexibility, total capacity (kW)
        current_storage: current storage level (kW)
        flex_vehicle: vehicle flexibility (kW)
        current_vehicle: current vehicle level (kW)
        flex_AC: AC flexibility (kW)
        current_AC: current AC usage (kW)
        flex_wash: wash total available units (count; ~40kW per machine)
        current_wash: current wash units on (count)
        t: time step, 0-23 (int, optional)
        n: node identifier (string, optional)
        ctx: context dictionary (optional)

    Returns:
        dict: pv_ratio, wind_ratio [0,1]; storage_ratio, vehicle_ratio [-1,1] (positive=discharge, negative=charge); ac_ratio [0,1]; wash_on_number (int, >=0).
    """
    # Simple version: PV/Wind at half grid connection, storage/EV neutral, AC off, Wash off
    return {
        "pv_ratio": 0.5,
        "wind_ratio": 0.5,
        "storage_ratio": 0.0,
        "vehicle_ratio": 0.0,
        "ac_ratio": 0.0,
        "wash_on_number": 0,
    }






