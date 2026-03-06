import { useAppContext } from '../context/AppContext';
import PlanViewer from '../components/PlanViewer';

export default function PlanViewerPage() {
    const { state } = useAppContext();
    const { planImageUrl, detections } = state;

    return (
        <div className="bg-white shadow rounded-lg p-6 min-h-[500px] flex flex-col h-[calc(100vh-10rem)]">
            <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-bold text-gray-900">Interactive Plan Viewer</h2>
                <div className="text-sm text-gray-500">Zoom and pan to inspect details.</div>
            </div>
            <div className="flex-1 w-full bg-gray-100 rounded-lg overflow-hidden relative">
                <PlanViewer imageUrl={planImageUrl || undefined} detections={detections} />
                {/* Render Bounding Boxes */}
                {planImageUrl && detections.length > 0 && (
                    <div className="absolute top-4 right-4 bg-white/90 p-2 text-xs rounded shadow border border-gray-200">
                        Found {detections.length} AI Detections
                    </div>
                )}
            </div>
        </div >
    );
}
