import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

def parse_donut_content(content: str) -> dict:
    """Helper to safely parse the Donut string output into a dictionary."""
    if not content:
        return {}
    try:
        # Donut outputs are typically JSON strings
        parsed = json.loads(content)
        return parsed.get("gt_parse", parsed)
    except json.JSONDecodeError:
        # Fallback for malformed JSON, though usually Donut constraints help prevent this
        return {"raw_text": content}


def get_bbox_center(bbox: List[float]) -> tuple:
    """Returns (x_center, y_center) for a given bbox [xmin, ymin, xmax, ymax]"""
    if len(bbox) >= 4:
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    return (0, 0)


def is_point_in_bbox(point: tuple, bbox: List[float]) -> bool:
    """Checks if point (x, y) is inside bbox [xmin, ymin, xmax, ymax]"""
    if len(bbox) < 4:
        return False
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


class Rule:
    """Base class for all evaluator rules."""
    def evaluate(self, plan_data: Dict[str, Any]) -> List[str]:
        """
        Evaluates the rule against the plan data.
        Returns a list of violation reasons. Empty list implies pass.
        """
        raise NotImplementedError("Subclasses must implement evaluate()")


class VisualElementsRule(Rule):
    """
    1. Presence Checks (Visual Elements):
    - North Arrow: Must have at least one. If two exist, one must be within the key plan extents.
    - Stamps/Validation: Must have a "dev stamp" and an "anbls valnum".
    - Scale Bar: Must be present.
    - Signatures: Must have at least 2 "signature"s.
    """
    def evaluate(self, plan_data: Dict[str, Any]) -> List[str]:
        reasons = []
        detections = plan_data.get("detections", [])
        
        # Count classes
        counts = {
            "north arrow": 0,
            "dev stamp": 0,
            "anbls valnum": 0,
            "scale bar": 0,
            "signature": 0,
            "key plan": 0
        }
        
        north_arrows = []
        key_plans = []
        
        for det in detections:
            cls_name = det.get("class_name", "").lower()
            if cls_name in counts:
                counts[cls_name] += 1
            if cls_name == "north arrow":
                north_arrows.append(det)
            elif cls_name == "key plan":
                key_plans.append(det)

        # North Arrow Check
        if counts["north arrow"] == 0:
            reasons.append("Missing North Arrow.")
        elif counts["north arrow"] >= 2:
            # If 2 or more, at least one must be within key plan extents
            if not key_plans:
                reasons.append("Multiple North Arrows found, but no Key Plan exists to contain one.")
            else:
                kp_bbox = key_plans[0].get("bbox", [])
                found_inside = False
                for na in north_arrows:
                    na_center = get_bbox_center(na.get("bbox", []))
                    if is_point_in_bbox(na_center, kp_bbox):
                        found_inside = True
                        break
                if not found_inside:
                    reasons.append("Multiple North Arrows found, but none are within the Key Plan extents.")

        # Stamps / Validation
        if counts["dev stamp"] == 0:
            reasons.append("Missing Development Officer Stamp (dev stamp).")
        if counts["anbls valnum"] == 0:
            reasons.append("Missing ANBLS Validation Number (anbls valnum).")

        # Scale Bar
        if counts["scale bar"] == 0:
            reasons.append("Missing Scale Bar.")

        # Signatures
        if counts["signature"] < 2:
            reasons.append(f"Insufficient signatures. Found {counts['signature']}, required at least 2.")

        return reasons


class TextualComplianceRule(Rule):
    """
    2. Textual Content & Compliance (Extracted via Donut):
    - Directions are in "NB grid azimuth".
    - Distances are in "meters”.
    - Coordinates are in "NB Double Stareographic".
    - Identifiers: Must include an appropriate plan title, plan number, and a "purpose of plan" section.
    - Lot Info: Every primary lot number must have an associated area value.
    - Temporal Logic: The "date of survey" must be before the "date registered".
    """
    def evaluate(self, plan_data: Dict[str, Any]) -> List[str]:
        reasons = []
        detections = plan_data.get("detections", [])
        
        all_text = ""
        identifiers = {
            "plan_title": False,
            "plan_number": False,
            "purpose_of_plan": False
        }
        
        # We will collect lot ids and area vals if we find them in parcel info
        lot_ids = set()
        area_vals = set()
        dates_found = {} # to store "date_of_survey", "date_registered"

        for det in detections:
            content = det.get("parsed_content", "")
            if isinstance(content, str):
                parsed = parse_donut_content(content)
            else:
                parsed = content

            if not parsed:
                continue

            # Accumulate text for generic textual searches
            # Donut might nest things, let's flatten briefly for string match
            all_text += " " + str(parsed).lower()
            
            # Check specifically parsed structures if available
            general = parsed.get("general", {})
            if "plan_title" in general:
                identifiers["plan_title"] = True
            if "plan_number" in general or "file_num" in general or "plan_num" in general:
                # Assuming generic names Donut might use
                identifiers["plan_number"] = True
            
            # If "purpose of plan" is a tag in text or general
            if "purpose" in str(general).lower() or "purpose_of_plan" in general:
                identifiers["purpose_of_plan"] = True

            # Lot Info
            parcel_info = parsed.get("parcel_info", {})
            if parcel_info:
                if isinstance(parcel_info, list):
                    for p in parcel_info:
                        if p.get("lot_id"): lot_ids.add(str(p.get("lot_id")))
                        if p.get("area_val"): area_vals.add(str(p.get("area_val")))
                else:
                    if parcel_info.get("lot_id"): lot_ids.add(str(parcel_info.get("lot_id")))
                    if parcel_info.get("area_val"): area_vals.add(str(parcel_info.get("area_val")))

            # Dates
            if isinstance(general, dict):
                if "date_of_survey" in general:
                    dates_found["date_of_survey"] = general["date_of_survey"]
                if "date_registered" in general:
                    dates_found["date_registered"] = general["date_registered"]
            
        # Basic text presence checks
        if "nb grid azimuth" not in all_text and "n.b. grid azimuth" not in all_text:
            reasons.append("Plan does not explicitly state directions are in 'NB grid azimuth'.")
        
        if "meter" not in all_text and "metre" not in all_text:
            reasons.append("Plan does not explicitly state distances are in 'meters'.")
            
        if "nb double stereographic" not in all_text and "n.b. double stereographic" not in all_text and "nb double stareographic" not in all_text:
            reasons.append("Plan does not explicitly state coordinates are in 'NB Double Stereographic'.")
        
        # Identifiers
        # If the model didn't perfectly map them, we can fallback to text matching
        if not identifiers["plan_title"] and "plan" not in all_text:
            reasons.append("Missing appropriate plan title.")
        # We assume if the YOLO class 'text' has 'plan number' / 'purpose of plan', it will be in all_text
        if not identifiers["plan_number"] and "plan number" not in all_text and "plan no" not in all_text:
            reasons.append("Missing plan number.")
        if not identifiers["purpose_of_plan"] and "purpose" not in all_text:
            reasons.append("Missing 'purpose of plan' section.")

        # Lot Info
        # Every primary lot number must have an associated area value.
        # This is a bit abstract. If we detected 3 lots but fewer areas, flag it.
        if len(lot_ids) > 0 and len(area_vals) < len(lot_ids):
            reasons.append(f"Missing area values for lots. Found {len(lot_ids)} lots but only {len(area_vals)} areas.")

        # Temporal Logic
        try:
            # We would need to parse dates if they exist. 
            # If dates aren't cleanly extracted, we might rely on regular expressions over text in a real prod system.
            # Simplified logic if they are in dates_found:
            ds = dates_found.get("date_of_survey")
            dr = dates_found.get("date_registered")
            if ds and dr:
                # Attempt to parse YYYY-MM-DD or similar
                # Just a simple string comparison often works if ISO formatted, but dates can be tricky.
                # Here we do a naive check if we have them.
                if ds > dr:
                    reasons.append(f"Temporal Logic Error: Date of survey ({ds}) is after Date registered ({dr}).")
        except Exception as e:
            logger.warning(f"Could not parse dates for temporal logic comparison: {e}")

        return reasons


class DataIntegrityRule(Rule):
    """
    3. Data Integrity & Geometry:
    - Bearings/Markers: The number of bearings and distances must be greater than or equal to the number of survey markers.
    - Coordinate Verification: Coordinates marked as lot corners in the ASCII upload must exist in the plan's coordinate table (tabular data).
    """
    def evaluate(self, plan_data: Dict[str, Any]) -> List[str]:
        reasons = []
        detections = plan_data.get("detections", [])
        csv_corners = plan_data.get("csv_corners")

        bearings_distances_count = 0
        markers_count = 0
        
        marker_classes = {"ip", "ribf", "smf", "sms", "sqib"}
        bearing_distance_classes = {"azimuth", "distance"}
        
        plan_coordinates = []

        for det in detections:
            cls_name = det.get("class_name", "").lower()
            if cls_name in marker_classes:
                markers_count += 1
            if cls_name in bearing_distance_classes:
                bearings_distances_count += 1
                
            # Extract coordinates from tabular data if present
            content = det.get("parsed_content", "")
            if isinstance(content, str):
                parsed = parse_donut_content(content)
            else:
                parsed = content
                
            if parsed and "tabular_data" in parsed:
                tabular = parsed["tabular_data"]
                rows = tabular.get("row", [])
                if isinstance(rows, list):
                    for r in rows:
                        if isinstance(r, dict):
                            n = r.get("north")
                            e = r.get("east")
                            if n and e:
                                try:
                                    plan_coordinates.append((float(n), float(e)))
                                except ValueError:
                                    pass

        if bearings_distances_count < markers_count:
            reasons.append(f"Geometry integrity failed: The number of bearings and distances ({bearings_distances_count}) "
                           f"is less than the number of survey markers ({markers_count}).")

        # Coordinate Verification
        if csv_corners and plan_coordinates:
            # Check if all CSV lot corners exist in plan_coordinates
            tolerance = 0.05
            for corner in csv_corners:
                cx = corner.get('x') # Easting
                cy = corner.get('y') # Northing
                if cx is None or cy is None: 
                    continue
                found = False
                for pn, pe in plan_coordinates:
                    if abs(pn - cy) < tolerance and abs(pe - cx) < tolerance:
                        found = True
                        break
                if not found:
                    reasons.append(f"Coordinate verification failed: Corner (N:{cy}, E:{cx}) not found in plan's coordinate table.")
        elif csv_corners and not plan_coordinates:
            reasons.append("Coordinate verification failed: CSV corners provided, but no coordinate table found on the plan.")

        return reasons


class PlanEvaluator:
    def __init__(self):
        self.rules = [
            VisualElementsRule(),
            TextualComplianceRule(),
            DataIntegrityRule()
        ]

    def evaluate(self, detections: List[Dict[str, Any]], csv_corners: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Main entry point to evaluate a plan.
        """
        plan_data = {
            "detections": detections,
            "csv_corners": csv_corners
        }
        
        all_reasons = []
        for rule in self.rules:
            reasons = rule.evaluate(plan_data)
            all_reasons.extend(reasons)
            
        status = "Pass"
        if len(all_reasons) > 0:
            status = "Fail"
            
        confidence = self._compute_confidence(detections)
        
        return {
            "status": status,
            "reasons": all_reasons,
            "confidence": confidence
        }

    def _compute_confidence(self, detections: List[Dict[str, Any]]) -> float:
        """
        Compute an aggregate confidence score based on YOLO detection confidences.
        0.0 to 1.0
        """
        if not detections:
            return 0.0
        
        # Simple average of YOLO detection confidences
        confs = [det.get("confidence", 0) for det in detections if "confidence" in det]
        if confs:
            return sum(confs) / len(confs)
        return 0.0
