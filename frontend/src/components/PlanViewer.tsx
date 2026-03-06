import { useEffect, useRef } from 'react';
import OpenSeadragon from 'openseadragon';

interface Detection {
    bbox?: number[];
    label?: string;
    score?: number;
    angle?: number;
    rect_w?: number;
    rect_h?: number;
    parsed_content?: string;
}

interface PlanViewerProps {
    imageUrl?: string;
    detections?: Detection[];
}

export default function PlanViewer({ imageUrl, detections }: PlanViewerProps) {
    const containerRef = useRef<HTMLDivElement>(null);
    const viewerRef = useRef<OpenSeadragon.Viewer | null>(null);

    // Create and destroy the viewer once on mount/unmount
    useEffect(() => {
        if (!containerRef.current) return;

        const viewer = OpenSeadragon({
            element: containerRef.current,
            prefixUrl: 'https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/images/',
            animationTime: 0.5,
            constrainDuringPan: true,
            maxZoomPixelRatio: 3,
            minZoomImageRatio: 0.5,
            showNavigationControl: true,
            navigationControlAnchor: OpenSeadragon.ControlAnchor.TOP_LEFT,
        });
        viewerRef.current = viewer;

        return () => {
            try { viewer.destroy(); } catch (_) { /* ignore */ }
            viewerRef.current = null;
        };
    }, []);

    // Load image when URL changes
    useEffect(() => {
        const viewer = viewerRef.current;
        if (!viewer) return;
        if (imageUrl) {
            try {
                viewer.open({ type: 'image', url: imageUrl });
            } catch (e) {
                console.error('[PlanViewer] Error opening image:', e);
            }
        } else {
            try { viewer.close(); } catch (_) { /* ignore */ }
        }
    }, [imageUrl]);

    // Draw overlays - waits for image to fully load via polling
    useEffect(() => {
        const viewer = viewerRef.current;
        if (!viewer || !imageUrl) return;

        const safeDetections = Array.isArray(detections) ? detections : [];
        if (safeDetections.length === 0) return;

        let cancelled = false;
        let attempts = 0;
        const MAX_ATTEMPTS = 50; // wait up to ~5 seconds

        const tryDraw = () => {
            if (cancelled) return;
            if (!viewerRef.current) return;

            const tiledImage = viewer.world.getItemAt(0);
            if (!tiledImage || viewer.world.getItemCount() === 0) {
                // Image not loaded yet - retry shortly
                if (attempts++ < MAX_ATTEMPTS) {
                    setTimeout(tryDraw, 100);
                }
                return;
            }

            // Clear existing overlays
            try { viewer.clearOverlays(); } catch (_) { /* ignore */ }

            // Draw each bounding box
            safeDetections.forEach((det, idx) => {
                try {
                    if (!det || !Array.isArray(det.bbox) || det.bbox.length < 4) return;
                    const [xmin, ymin, xmax, ymax] = det.bbox;
                    if (typeof xmin !== 'number') return;

                    const rect = tiledImage.imageToViewportRectangle(
                        xmin, ymin, xmax - xmin, ymax - ymin
                    );

                    const el = document.createElement('div');
                    el.id = `det-ov-${idx}`;

                    // Apply CSS rotation for OBB detections.
                    // OpenCV minAreaRect returns angles in the range (-90, 0].
                    // We negate to get the visual clockwise rotation expected by CSS.
                    const angle = typeof det.angle === 'number' ? det.angle : 0;
                    const rotateDeg = angle !== 0 ? -angle : 0;

                    el.style.cssText = [
                        'border: 2px solid rgba(34,197,94,0.9)',
                        'background: rgba(34,197,94,0.07)',
                        'position: relative',
                        'box-sizing: border-box',
                        'transform-origin: center center',
                        rotateDeg !== 0 ? `transform: rotate(${rotateDeg}deg)` : '',
                    ].filter(Boolean).join(';');

                    const score = typeof det.score === 'number'
                        ? `${(det.score * 100).toFixed(1)}%` : '';
                    el.title = [det.label, score, det.parsed_content]
                        .filter(Boolean).join(' | ');

                    // small label chip above the box
                    if (det.label) {
                        const chip = document.createElement('span');
                        chip.innerText = det.label;
                        chip.style.cssText = [
                            'position:absolute',
                            'top:-16px',
                            'left:-2px',
                            'background:rgba(34,197,94,1)',
                            'color:#fff',
                            'font-size:9px',
                            'font-weight:700',
                            'padding:1px 4px',
                            'border-radius:2px 2px 0 0',
                            'white-space:nowrap',
                            'line-height:14px',
                            'pointer-events:none',
                        ].join(';');
                        el.appendChild(chip);
                    }

                    viewer.addOverlay({ element: el, location: rect });
                } catch (e) {
                    console.warn(`[PlanViewer] Skipped overlay ${idx}:`, e);
                }
            });
        };

        // Start polling
        tryDraw();

        return () => {
            cancelled = true;
            try {
                if (viewerRef.current) viewerRef.current.clearOverlays();
            } catch (_) { /* ignore */ }
        };
    }, [imageUrl, detections]);

    return (
        <div style={{
            width: '100%', height: '100%',
            position: 'relative',
            background: '#1e293b',
            borderRadius: '8px',
            overflow: 'hidden',
        }}>
            <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
            {!imageUrl && (
                <div style={{
                    position: 'absolute', inset: 0,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    pointerEvents: 'none',
                }}>
                    <span style={{ color: '#94a3b8', fontWeight: 500, fontSize: 14 }}>
                        Upload a survey plan on the Upload tab to view it here.
                    </span>
                </div>
            )}
        </div>
    );
}
