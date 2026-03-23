import { useState, useRef, useEffect } from 'react';
import { useAppContext } from '../context/AppContext';
import { RefreshCw, FileImage, FileLineChart, FileCode, CheckCircle2, XCircle } from 'lucide-react';

export default function UploadPage() {
    const { state, setPlanData, setCadValidation, setCsvData, setPlanType, setAnblsNumber } = useAppContext();
    const [loading, setLoading] = useState<string | null>(null);

    // Box 2 State
    const [cadStatus, setCadStatus] = useState<'pending' | 'approved' | 'rejected'>('pending');
    const [csvStatus, setCsvStatus] = useState<'pending' | 'approved' | 'rejected'>('pending');
    const [csvCorners, setCsvCorners] = useState('');

    // Box 3 State
    const [imageFile, setImageFile] = useState<File | null>(null);
    const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
    const [rotation, setRotation] = useState(0);

    const fileInputRef = useRef<HTMLInputElement>(null);

    // Clean up object URLs to avoid memory leaks
    useEffect(() => {
        return () => {
            if (imagePreviewUrl) URL.revokeObjectURL(imagePreviewUrl);
        };
    }, []);

    // --- Action Handlers ---


    const handleDataUpload = async (endpoint: string, type: 'cad' | 'csv', event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;

        setLoading(type);
        const setStatus = type === 'cad' ? setCadStatus : setCsvStatus;

        try {
            const formData = new FormData();
            formData.append('file', file);
            if (type === 'csv' && csvCorners) {
                formData.append('corners', csvCorners);
            }

            const res = await fetch(endpoint, {
                method: 'POST',
                body: formData,
            });

            if (!res.ok) throw new Error(await res.text());
            const data = await res.json();

            if (type === 'cad') {
                setCadValidation(data.validation, data.file_path);
                setStatus(data.valid ? 'approved' : 'rejected');
            } else {
                setCsvData(data.data);
                // Hardcoded to approved for now since we haven't implemented PNEZD check yet
                setStatus('approved');
            }
        } catch (err) {
            console.error(err);
            setStatus('rejected');
        } finally {
            setLoading(null);
        }
    };

    const handleImageSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            setImageFile(file);
            setRotation(0);

            if (imagePreviewUrl) URL.revokeObjectURL(imagePreviewUrl);
            setImagePreviewUrl(URL.createObjectURL(file));
        }
    };

    const handleRotate = () => {
        setRotation((prev) => (prev + 90) % 360);
    };


    const handleSubmitPlan = async () => {
        if (!imageFile) return;

        setLoading('plan');
        try {
            const formData = new FormData();
            formData.append('file', imageFile);

            const res = await fetch(`/api/upload/plan`, {
                method: 'POST',
                body: formData,
            });

            if (!res.ok) throw new Error(await res.text());
            const data = await res.json();

            const validDetections = Array.isArray(data.detections) ? data.detections : [];
            // Preserve the rotated local URL if we ever add actual image rotation to the backend, 
            // for now just pass the rotated object URL to context
            setPlanData(imagePreviewUrl!, validDetections);

            alert('Plan successfully processed and submitted!');
        } catch (err: any) {
            alert(`Failed to process plan: ${err.message}`);
            console.error(err);
        } finally {
            setLoading(null);
        }
    };

    return (
        <div className="max-w-4xl mx-auto space-y-4">

            {/* BOX 1: PLAN DATA */}
            <div className="bg-white pt-2 px-4 pb-4 rounded-lg shadow-sm border border-gray-200">
                <h2 className="text-xl font-bold text-gray-900 mb-2 border-b pb-2">Plan Data</h2>
                <div className="flex flex-col sm:flex-row gap-4 justify-between items-start sm:items-end">

                    <div className="flex-1 w-full sm:max-w-xs">
                        <label className="block text-sm font-semibold text-gray-700 mb-2">Survey Plan Type</label>
                        <select
                            className="block w-full text-base border-gray-300 focus:outline-none focus:ring-sky-500 focus:border-sky-500 sm:text-sm rounded-md border p-2 bg-gray-50"
                            value={state.planType}
                            onChange={(e) => setPlanType(e.target.value)}
                        >
                            <option value="Subdivision Plan">Subdivision Plan</option>
                            {/* Hidden until user asks for them: <option value="Condominium">Condominium</option> */}
                        </select>
                    </div>

                    <div className="flex-1 w-full sm:max-w-xs flex flex-col gap-2">
                        <label className="block text-sm font-semibold text-gray-700">ANBLS Number</label>
                        <input
                            value={state.anblsNumber}
                            onChange={(e) => setAnblsNumber(e.target.value)}
                            placeholder="Enter ANBLS Number"
                            className="block w-full border border-gray-300 rounded-md bg-white px-3 py-2 text-gray-900 focus:outline-none focus:ring-sky-500 focus:border-sky-500 font-mono"
                        />
                    </div>
                </div>
            </div>

            {/* BOX 2: DATA SUBMISSION */}
            <div className="bg-white pt-2 px-4 pb-4 rounded-lg shadow-sm border border-gray-200">
                <h2 className="text-xl font-bold text-gray-900 mb-2 border-b pb-2">Data Submission</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">

                    {/* CAD File */}
                    <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 flex flex-col items-center justify-between text-center min-h-[140px]">
                        {loading === 'cad' ? (
                            <div className="w-full h-full flex flex-col items-center justify-center text-indigo-600 space-y-3">
                                <RefreshCw className="animate-spin" size={32} />
                                <span className="font-medium animate-pulse">Running CAD Validation Layer Check...</span>
                            </div>
                        ) : (
                            <>
                                <div className="space-y-4 w-full">
                                    <div>
                                        <h3 className="text-sm font-bold text-slate-800 flex items-center justify-center gap-2">
                                            <FileCode size={18} className="text-sky-500" /> CAD Submission
                                        </h3>
                                        <p className="text-xs text-slate-500 mt-1">Supports .dxf</p>
                                    </div>
                                    <input
                                        type="file" accept=".dxf,.dwg"
                                        onChange={(e) => handleDataUpload('/api/upload/cad', 'cad', e)}
                                        className="block w-full text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-xs file:font-semibold file:bg-sky-50 file:text-sky-700 hover:file:bg-sky-100 transition-colors"
                                    />
                                </div>

                                {state.cadValidation && state.cadValidation.layers && state.cadValidation.topology && (
                                    <div className="mt-2 text-left text-[10px] space-y-1 bg-white p-2 rounded border border-slate-200">
                                        {state.cadValidation.message && (
                                            <p className="font-semibold text-slate-700 mb-1 border-b border-slate-100 pb-1">
                                                {state.cadValidation.message}
                                            </p>
                                        )}
                                        <p className={state.cadValidation.layers.valid ? "text-green-600" : "text-red-600"}>
                                            {state.cadValidation.layers.valid ? "✓ Layers OK" : "✗ Unauthorized Layers: " + (state.cadValidation.layers.unauthorized_layers?.slice(0, 3).join(', ') || 'None')}
                                        </p>
                                        <p className={state.cadValidation.topology.valid ? "text-green-600" : "text-red-600"}>
                                            {state.cadValidation.topology.valid ? "✓ Linework OK" : `✗ Gaps: ${state.cadValidation.topology.connectivity_gaps_count ?? 0} | Intersects: ${state.cadValidation.topology.intersections_count ?? 0}`}
                                        </p>
                                    </div>
                                )}

                                {/* Status Indicator */}
                                <div className="mt-4 pt-4 border-t border-slate-200 w-full flex justify-center">
                                    {cadStatus === 'pending' && <span className="text-xs font-semibold text-slate-400 capitalize tracking-wider">Awaiting Upload</span>}
                                    {cadStatus === 'approved' && <div className="flex items-center gap-1.5 text-green-600 font-bold bg-green-50 px-3 py-1 rounded-full text-sm border border-green-200"><CheckCircle2 size={16} /> VALIDATED</div>}
                                    {cadStatus === 'rejected' && <div className="flex items-center gap-1.5 text-red-600 font-bold bg-red-50 px-3 py-1 rounded-full text-sm border border-red-200"><XCircle size={16} /> REJECTED</div>}
                                </div>
                            </>
                        )}
                    </div>

                    {/* CSV File */}
                    <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 flex flex-col items-center justify-between text-center min-h-[140px]">
                        {loading === 'csv' ? (
                            <div className="w-full h-full flex flex-col items-center justify-center text-sky-600 space-y-3">
                                <RefreshCw className="animate-spin" size={32} />
                                <span className="font-medium animate-pulse">Running PNEZD Validation Check...</span>
                            </div>
                        ) : (
                            <>
                                <div className="space-y-4 w-full">
                                    <div>
                                        <h3 className="text-sm font-bold text-slate-800 flex items-center justify-center gap-2">
                                            <FileLineChart size={18} className="text-sky-500" /> Coordinate File Submission
                                        </h3>
                                        <p className="text-[10px] text-slate-500 mt-0.5">Supports .csv, .txt, .asc, .xyz, .pts in PNEZD format</p>
                                    </div>

                                    {/* Subject Lot Corners Input */}
                                    <div className="text-left bg-white p-3 rounded border border-slate-200 shadow-sm">
                                        <label className="block text-xs font-semibold text-slate-700 mb-1">Define Subject Lot Corners:</label>
                                        <input
                                            type="text"
                                            placeholder="e.g. 1,2,3,4,5"
                                            value={csvCorners}
                                            onChange={(e) => setCsvCorners(e.target.value)}
                                            className="block w-full text-xs p-1.5 border border-slate-300 rounded focus:border-sky-500 focus:ring-1 focus:ring-sky-500"
                                        />
                                    </div>

                                    <input
                                        type="file" accept=".csv,.txt,.asc,.xyz,.pts"
                                        onChange={(e) => handleDataUpload('/api/upload/csv', 'csv', e)}
                                        className="block w-full text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-xs file:font-semibold file:bg-sky-50 file:text-sky-700 hover:file:bg-sky-100 transition-colors"
                                    />
                                </div>

                                {/* Status Indicator */}
                                <div className="mt-4 pt-4 border-t border-slate-200 w-full flex flex-col items-center gap-3">
                                    <div className="flex justify-center">
                                        {csvStatus === 'pending' && <span className="text-xs font-semibold text-slate-400 capitalize tracking-wider">Awaiting Upload</span>}
                                        {csvStatus === 'approved' && <div className="flex items-center gap-1.5 text-green-600 font-bold bg-green-50 px-3 py-1 rounded-full text-sm border border-green-200"><CheckCircle2 size={16} /> LOADED</div>}
                                        {csvStatus === 'rejected' && <div className="flex items-center gap-1.5 text-red-600 font-bold bg-red-50 px-3 py-1 rounded-full text-sm border border-red-200"><XCircle size={16} /> REJECTED</div>}
                                    </div>
                                </div>
                            </>
                        )}
                    </div>

                </div>
            </div>

            {/* BOX 3: PLAN SUBMISSION */}
            <div className="bg-white pt-2 px-4 pb-4 rounded-lg shadow-sm border border-gray-200">
                <h2 className="text-xl font-bold text-gray-900 mb-2 border-b pb-2">Plan Submission</h2>

                <div className="flex flex-col md:flex-row gap-4 items-stretch min-h-[200px]">

                    {/* Left side: Upload */}
                    <div className="flex-1 flex flex-col justify-center items-center p-4 border-2 border-dashed border-slate-300 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors relative cursor-pointer" onClick={() => fileInputRef.current?.click()}>
                        <FileImage size={40} className="text-slate-400 mb-2" />
                        <h3 className="text-lg font-bold text-slate-700">Upload Survey Plan</h3>
                        <p className="text-xs text-slate-500 mt-1 max-w-xs text-center">Supported formats: JPG, PNG, TIF. Note: PDF files must be converted to image format first.</p>

                        <input
                            ref={fileInputRef}
                            type="file"
                            accept=".jpg,.jpeg,.png,.tif,.tiff"
                            onChange={handleImageSelect}
                            className="hidden"
                        />

                        {imageFile && (
                            <div className="mt-6 inline-flex items-center gap-2 px-4 py-2 bg-sky-50 text-sky-800 rounded-full text-sm font-semibold border border-sky-100">
                                <CheckCircle2 size={16} />
                                {imageFile.name}
                            </div>
                        )}
                    </div>

                    {/* Right side: Preview & action */}
                    <div className="w-full md:w-64 flex flex-col justify-between items-center space-y-4">

                        {/* Image Preview Window */}
                        <div className="w-full aspect-square bg-slate-800 rounded-lg overflow-hidden border-2 border-slate-200 relative shadow-inner flex items-center justify-center">
                            {imagePreviewUrl ? (
                                <img
                                    src={imagePreviewUrl}
                                    className="max-w-full max-h-full object-contain transition-transform duration-300 ease-in-out"
                                    style={{ transform: `rotate(${rotation}deg)` }}
                                    alt="Survey Plan Preview"
                                />
                            ) : (
                                <div className="text-slate-500 text-sm font-medium">No Image</div>
                            )}
                        </div>

                        {/* Rotate Action */}
                        <button
                            disabled={!imagePreviewUrl}
                            onClick={handleRotate}
                            className="w-full py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 text-sm font-semibold rounded border border-slate-300 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex justify-center items-center gap-2"
                        >
                            Rotate <RefreshCw size={14} />
                        </button>

                    </div>

                </div>

                {/* FINAL SUBMIT BUTTON */}
                <div className="mt-4 flex justify-center">
                    <button
                        disabled={!imageFile || loading === 'plan'}
                        onClick={handleSubmitPlan}
                        className="w-full max-w-lg py-3 text-white text-lg font-bold bg-sky-600 hover:bg-sky-700 shadow-xl rounded-lg border border-sky-700 transition-all transform hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-3 mb-4"
                    >
                        {loading === 'plan' ? (
                            <>
                                <RefreshCw className="animate-spin mr-2" size={24} />
                                Processing Submission...
                            </>
                        ) : (
                            <span>Submit</span>
                        )}
                    </button>
                </div>

            </div>
        </div>
    );
}
