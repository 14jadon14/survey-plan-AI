import { useEffect, useState } from 'react';
import { useAppContext } from '../context/AppContext';
import PlanViewer from '../components/PlanViewer';
import { AlertCircle, CheckCircle2, Loader2 } from 'lucide-react';

interface EvaluationResult {
    status: 'Pass' | 'Fail' | 'Flag';
    reasons: string[];
    confidence: number;
}

export default function PlanViewerPage() {
    const { state } = useAppContext();
    const { planImageUrl, detections } = state;
    const [evalResult, setEvalResult] = useState<EvaluationResult | null>(null);
    const [isEvaluating, setIsEvaluating] = useState(false);

    useEffect(() => {
        if (!planImageUrl || !detections || detections.length === 0) {
            setEvalResult(null);
            return;
        }

        const runEvaluation = async () => {
            setIsEvaluating(true);
            try {
                // In a real app we might also pass csv_corners if they are in context
                const res = await fetch('/api/evaluate-plan', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ detections, csv_corners: null }),
                });

                if (res.ok) {
                    const data = await res.json();
                    setEvalResult(data);
                } else {
                    console.error('Evaluation failed:', await res.text());
                }
            } catch (err) {
                console.error('Error hitting evaluation endpoint', err);
            } finally {
                setIsEvaluating(false);
            }
        };

        runEvaluation();
    }, [planImageUrl, detections]);

    return (
        <div className="bg-white shadow rounded-lg p-6 min-h-[500px] flex flex-col h-[calc(100vh-10rem)]">
            <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-bold text-gray-900">Interactive Plan Viewer</h2>
                <div className="text-sm text-gray-500">Zoom and pan to inspect details.</div>
            </div>
            
            <div className="flex flex-1 gap-4 min-h-0">
                {/* Main Viewer Area */}
                <div className="flex-1 bg-gray-100 rounded-lg overflow-hidden relative">
                    <PlanViewer imageUrl={planImageUrl || undefined} detections={detections} />
                    {planImageUrl && detections.length > 0 && (
                        <div className="absolute top-4 left-4 bg-white/90 px-3 py-1.5 text-xs font-semibold rounded shadow border border-gray-200">
                            Found {detections.length} AI Detections
                        </div>
                    )}
                </div>

                {/* Right Side Evaluation Panel */}
                <div className="w-80 flex flex-col bg-gray-50 border border-gray-200 rounded-lg p-4 overflow-y-auto shadow-inner">
                    <h3 className="text-lg font-semibold text-gray-800 border-b border-gray-200 pb-2 mb-4">
                        Plan Assessment
                    </h3>

                    {!planImageUrl ? (
                        <div className="text-sm text-gray-500 text-center mt-10">
                            Upload a plan to view its assessment.
                        </div>
                    ) : isEvaluating ? (
                        <div className="flex flex-col items-center justify-center flex-1 text-gray-500">
                            <Loader2 className="w-8 h-8 animate-spin mb-2" />
                            <span className="text-sm">Running validation ruleset...</span>
                        </div>
                    ) : evalResult ? (
                        <div className="flex flex-col space-y-4">
                            {/* Status Badge */}
                            <div className={`flex items-center p-3 rounded-md border ${
                                evalResult.status === 'Pass' 
                                ? 'bg-green-50 border-green-200 text-green-700' 
                                : 'bg-red-50 border-red-200 text-red-700'
                            }`}>
                                {evalResult.status === 'Pass' ? (
                                    <CheckCircle2 className="w-6 h-6 mr-2 flex-shrink-0" />
                                ) : (
                                    <AlertCircle className="w-6 h-6 mr-2 flex-shrink-0" />
                                )}
                                <div>
                                    <div className="font-bold text-base uppercase tracking-wider">
                                        {evalResult.status}
                                    </div>
                                    <div className="text-xs opacity-80 mt-0.5">
                                        Confidence: {(evalResult.confidence * 100).toFixed(1)}%
                                    </div>
                                </div>
                            </div>

                            {/* Reasons List */}
                            {evalResult.status === 'Fail' && evalResult.reasons.length > 0 && (
                                <div className="flex-1">
                                    <h4 className="text-sm font-semibold text-gray-700 mb-2">Rule Violations:</h4>
                                    <ul className="space-y-2">
                                        {evalResult.reasons.map((reason, idx) => (
                                            <li key={idx} className="text-sm text-gray-600 bg-white border border-gray-100 p-2 rounded flex items-start">
                                                <span className="text-red-500 mr-2 mt-0.5">•</span>
                                                <span>{reason}</span>
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            )}

                            {evalResult.status === 'Pass' && (
                                <div className="text-sm text-gray-600 text-center mt-4">
                                    All legislative visual and textual checks passed successfully.
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="text-sm text-gray-500 text-center mt-10">
                            No assessment available.
                        </div>
                    )}
                </div>
            </div>
        </div >
    );
}
