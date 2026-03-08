import ezdxf
import math
from typing import List, Dict, Tuple, Set, Optional

class CADValidator:
    def __init__(self, dxf_path: str):
        self.dxf_path = dxf_path
        self.doc = ezdxf.readfile(dxf_path)
        self.msp = self.doc.modelspace()
        self.tolerance = 0.001  # 0.001m = 1mm
        self.allowed_layers = {
            "NBBDRY", "NBSBDRY", "NBBDRYNAT", "NBTIE", "NBEVIDENCE"
        }
        # Pre-calculate uppercase for faster case-insensitive matching
        self._allowed_layers_up = {l.upper() for l in self.allowed_layers}

    def validate_all(self, csv_corners: List[Dict] = None) -> Dict:
        """Runs all validation checks."""
        layer_results = self.verify_layers()
        topology_results = self.verify_topology()
        
        corner_results = None
        if csv_corners:
            corner_results = self.verify_corners(csv_corners)
            
        valid = layer_results["valid"] and topology_results["valid"] and (corner_results["valid"] if corner_results else True)
        
        reasons = []
        if not layer_results["valid"]: reasons.append(f"Layer issues: {layer_results.get('unauthorized_layers')}")
        if not topology_results["valid"]: reasons.append(f"Topology issues: {topology_results.get('intersections_count')} intersects, {topology_results.get('connectivity_gaps_count')} gaps")
        if corner_results and not corner_results["valid"]: reasons.append(f"Corner mismatch: {corner_results.get('missing_count')} points")
        
        summary_message = "All checks passed." if valid else " | ".join(reasons)
        print(f"[DEBUG] CAD Validation Result: {valid} - {summary_message}")
        
        return {
            "valid": valid,
            "layers": layer_results,
            "topology": topology_results,
            "corners": corner_results,
            "message": summary_message
        }

    def verify_layers(self) -> Dict:
        """Verifies that ONLY the allowed layers exist and contain geometry."""
        all_layers = [layer.dxf.name for layer in self.doc.layers]
        found_geometry_layers = set()
        
        # Check all entities to see which layers are actually used for geometry
        for entity in self.msp:
            # Skip empty layers or decorative layers if we ever decide to, but for now be strict
            found_geometry_layers.add(entity.dxf.layer)
            
        print(f"[DEBUG] Found geometry on layers: {found_geometry_layers}")
            
        unauthorized_layers = [l for l in found_geometry_layers if l.upper() not in self._allowed_layers_up]
        missing_layers = [l for l in self.allowed_layers if l.upper() not in {fl.upper() for fl in found_geometry_layers}]
        
        valid = len(unauthorized_layers) == 0
        message = "Only allowed layers contain geometry." if valid else f"Unauthorized layers found: {unauthorized_layers}"
        
        # SPECIAL CASE: Layer '0' is often default and may contain invisible stuff. 
        # If '0' is in unauthorized but has NO real geometry (e.g. empty), it shouldn't fail.
        # However, our loop above already only adds layers if they have entities.
        
        return {
            "valid": valid,
            "found_layers": list(found_geometry_layers),
            "unauthorized_layers": unauthorized_layers,
            "missing_layers": missing_layers,
            "message": message
        }

    def verify_topology(self) -> Dict:
        """Checks for intersections and connectivity."""
        segments = self._extract_segments()
        intersections = self._find_intersections(segments)
        connectivity_gaps = self._check_connectivity(segments)
        
        valid = len(intersections) == 0 and len(connectivity_gaps) == 0
        
        return {
            "valid": valid,
            "intersections_count": len(intersections),
            "intersections": intersections,
            "connectivity_gaps_count": len(connectivity_gaps),
            "connectivity_gaps": connectivity_gaps
        }

    def verify_corners(self, csv_corners: List[Dict]) -> Dict:
        """
        Cross-references CSV corner points with CAD vertices.
        Expects csv_corners to be a list of dicts with 'x' and 'y' (and maybe 'name').
        """
        cad_vertices = self._get_all_vertices()
        matches = []
        missing = []
        
        for corner in csv_corners:
            cx, cy = corner.get('x'), corner.get('y')
            if cx is None or cy is None: continue
            
            found = False
            for vx, vy in cad_vertices:
                if math.isclose(cx, vx, abs_tol=self.tolerance) and \
                   math.isclose(cy, vy, abs_tol=self.tolerance):
                    matches.append({"corner": corner, "cad_vertex": (vx, vy)})
                    found = True
                    break
            if not found:
                missing.append(corner)
                
        return {
            "valid": len(missing) == 0,
            "matches_count": len(matches),
            "missing_count": len(missing),
            "missing_corners": missing
        }

    def _extract_segments(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Extracts all line segments from LINE, POLYLINE, LWPOLYLINE."""
        segments = []
        for entity in self.msp:
            if entity.dxftype() == 'LINE':
                # Use absolute XY to ensure it's a 2-tuple
                p1 = (entity.dxf.start.x, entity.dxf.start.y)
                p2 = (entity.dxf.end.x, entity.dxf.end.y)
                segments.append((p1, p2))
            elif entity.dxftype() in ('LWPOLYLINE', 'POLYLINE'):
                points = list(entity.get_points())
                # Normalize points to (x, y) tuples
                pts = [(float(p[0]), float(p[1])) for p in points]
                for i in range(len(pts) - 1):
                    segments.append((pts[i], pts[i+1]))
                if entity.is_closed:
                    segments.append((pts[-1], pts[0]))
        return segments

    def _get_all_vertices(self) -> Set[Tuple[float, float]]:
        """Collects all unique vertices from supported entities."""
        vertices = set()
        for entity in self.msp:
            if entity.dxftype() == 'LINE':
                vertices.add((round(entity.dxf.start.x, 6), round(entity.dxf.start.y, 6)))
                vertices.add((round(entity.dxf.end.x, 6), round(entity.dxf.end.y, 6)))
            elif entity.dxftype() in ('LWPOLYLINE', 'POLYLINE'):
                for p in entity.get_points():
                    vertices.add((round(p[0], 6), round(p[1], 6)))
        return vertices

    def _find_intersections(self, segments: List[Tuple]) -> List[Dict]:
        """Detects segments that cross each other (not at endpoints)."""
        intersections = []
        # Simple O(n^2) check. For large files, use spatial indexing.
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                s1, s2 = segments[i], segments[j]
                intersect_pt = self._intersect(s1, s2)
                if intersect_pt:
                    # Check if the intersection is exactly at one of the endpoints
                    # If it is, it's not a "crossing" but a "connection"
                    if not self._is_endpoint(intersect_pt, s1) or not self._is_endpoint(intersect_pt, s2):
                        intersections.append({
                            "point": intersect_pt,
                            "segments": [s1, s2]
                        })
        return intersections

    def _check_connectivity(self, segments: List[Tuple]) -> List[Dict]:
        """Ensures every endpoint is shared with at least one other segment."""
        if not segments: return []
        
        endpoints = []
        for s in segments:
            endpoints.append(s[0])
            endpoints.append(s[1])
            
        gaps = []
        for i, pt in enumerate(endpoints):
            # A vertex is "connected" if there is at least one OTHER segment endpoint at the same location
            found_connection = False
            for j, opt in enumerate(endpoints):
                if i == j: continue
                if math.isclose(pt[0], opt[0], abs_tol=self.tolerance) and \
                   math.isclose(pt[1], opt[1], abs_tol=self.tolerance):
                    found_connection = True
                    break
            if not found_connection:
                gaps.append({"point": pt, "message": "Dangling endpoint found."})
        return gaps

    def _intersect(self, s1, s2) -> Optional[Tuple[float, float]]:
        """Returns the intersection point of two segments, or None."""
        # Standard line intersection formula
        # Being very explicit about unpacking to avoid ValueError
        try:
            p1, p2 = s1
            p3, p4 = s2
            x1, y1 = p1[0], p1[1]
            x2, y2 = p2[0], p2[1]
            x3, y3 = p3[0], p3[1]
            x4, y4 = p4[0], p4[1]
        except (TypeError, IndexError, ValueError):
             return None
        
        denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if denom == 0: return None  # Parallel
        
        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
        
        # Check if ua and ub are withing segment ranges [0, 1]
        # Use a small epsilon to exclude endpoints
        eps = 1e-9
        if eps < ua < 1 - eps and eps < ub < 1 - eps:
            return (x1 + ua * (x2 - x1), y1 + ua * (y2 - y1))
        return None

    def _is_endpoint(self, pt, seg) -> bool:
        """Returns True if point is at one of the segment's endpoints within tolerance."""
        return (math.isclose(pt[0], seg[0][0], abs_tol=self.tolerance) and math.isclose(pt[1], seg[0][1], abs_tol=self.tolerance)) or \
               (math.isclose(pt[0], seg[1][0], abs_tol=self.tolerance) and math.isclose(pt[1], seg[1][1], abs_tol=self.tolerance))
